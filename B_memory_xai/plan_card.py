"""plan_card.py (yayuen)
Plan Card Generator: before a HIGH-RISK dispatch, explain the agent's multi-step reasoning
to the blood bank officer, who can approve, modify or reject it.

A card shows: the proposed plan, the policy's preference (NOT a probability of success),
the alternatives, the key factors behind the choice, similar past incidents from memory,
and a counterfactual ("what would change the decision")."""
import os
# Windows: PyTorch and FAISS each ship their own OpenMP runtime; loading both in one
# process triggers "OMP: Error #15". Known workaround; we also limit to 1 thread.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import numpy as np

import log_reader as lr
from fallback_controller import uncertainty_from_obs
from incident_db import IncidentDB

# ---- Settings ----
HIGH_RISK_PREFERENCE = 0.60 # dispatch the policy itself only weakly prefers = high risk
TOP_FACTORS = 3
NEUTRAL_ROAD = 0.5  # "no idea" belief used when testing what a road contributes

# Feature groups used for "key factors" (indices into the 34-value observation, CONTRACT 6.2)
FEATURE_GROUPS = {f"road {i} belief": [i] for i in range(lr.N_ROADS)}
FEATURE_GROUPS.update({"rain": [7], "river level": [8]})
for _h, _name in enumerate(lr.HOSPITALS):
    FEATURE_GROUPS[f"{_name} demand"] = [18 + 4 * _h, 19 + 4 * _h, 20 + 4 * _h]
FEATURE_GROUPS["van 0 state"] = list(range(10, 14))
FEATURE_GROUPS["van 1 state"] = list(range(14, 18))


def action_probs(model, obs, mask):
    """Policy preference over the 9 actions (masked actions get 0)."""
    import torch
    obs_t, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=np.asarray(mask, bool).reshape(1, -1))
    return dist.distribution.probs[0].cpu().numpy()


def is_high_risk(action, probs, uncertainty):
    """High risk = a real dispatch that the policy is unsure about, or made under high uncertainty."""
    if action == 0:
        return False, []
    reasons = []
    if probs[action] < HIGH_RISK_PREFERENCE:
        reasons.append(f"policy preference {probs[action]:.2f} (< {HIGH_RISK_PREFERENCE})")
    if uncertainty > lr.TAU_EXIT:
        reasons.append(f"road uncertainty {uncertainty:.4f} (> tau_exit {lr.TAU_EXIT})")
    return bool(reasons), reasons


def key_factors(model, obs, mask, action, probs):
    """Approximate sensitivity: blank out one feature group, see how much the preference
    for the chosen action drops. Bigger drop = that group mattered more."""
    effects = []
    for name, idx in FEATURE_GROUPS.items():
        o = np.array(obs, dtype=np.float32)
        o[idx] = NEUTRAL_ROAD if name.startswith("road") else 0.0
        delta = action_probs(model, o, mask)[action] - probs[action]
        effects.append((name, float(np.mean(np.asarray(obs)[idx])), delta))
    effects.sort(key=lambda e: abs(e[2]), reverse=True)
    return effects[:TOP_FACTORS]


def counterfactual(model, obs, mask, action):
    """Flip one uncertain road belief (open <-> flooded); report the first flip that changes
    the decision. Most uncertain roads (closest to 0.5) are tried first."""
    beliefs = np.asarray(obs[:lr.N_ROADS], dtype=float)
    for i in np.argsort(np.abs(beliefs - 0.5)):
        o = np.array(obs, dtype=np.float32)
        o[i] = 1.0 if beliefs[i] < 0.5 else 0.0
        new_action = int(np.argmax(action_probs(model, o, mask)))
        if new_action != action:
            state = "open (1.0)" if o[i] == 1.0 else "flooded (0.0)"
            return (f"if road {i} were believed {state} instead of {beliefs[i]:.2f}, "
                    f"the agent would choose: {lr.decode_action(new_action)}")
    return "no single road belief change would alter this decision"


def make_card(model, obs, mask, step, db=None):
    probs = action_probs(model, obs, mask)
    action = int(np.argmax(probs))
    u = uncertainty_from_obs(obs)
    high_risk, reasons = is_high_risk(action, probs, u)
    if not high_risk:
        return None
    precedents, nearest = [], None
    if db is not None and len(db):
        vec = lr.to_vector(obs, u)
        precedents = db.search(vec)
        closest = db.search(vec, k=1, max_distance=1e9)
        nearest = round(closest[0][0], 2) if closest else None
    top = np.argsort(probs)[::-1][:3]
    return {
        "step": step,
        "time": f"{int(obs[9] * 24):02d}:{int(obs[9] * 24 * 60) % 60:02d}",
        "action": lr.decode_action(action),
        "reasons": reasons,
        "preference": [(lr.decode_action(int(a)), round(float(probs[a]), 2)) for a in top if probs[a] > 0],
        "factors": [(n, round(v, 2), round(d, 2)) for n, v, d in key_factors(model, obs, mask, action, probs)],
        "precedents": [{"distance": round(d, 2), "label": r["label"], "action": r.get("action", "?"), 
                        "outcome": round(r["outcome_reward"], 1), "seen": r["count"],
                         "provisional": r.get("provisional", False), "contested": r["contested"]} for d, r in precedents],
        "counterfactual": counterfactual(model, obs, mask, action),
        "nearest": nearest,
    }


def render(card):
    """Human-readable plan card."""
    lines = [f"{'=' * 22} PLAN CARD  (step {card['step']}, {card['time']}) {'=' * 22}",
             f"Proposed action : {card['action']}",
             f"Why flagged     : {'; '.join(card['reasons'])}",
             "Policy preference (how strongly the policy prefers each option, NOT chance of success):"]
    lines += [f"    {a:<24}{p:.2f}" for a, p in card["preference"]]
    lines.append("Key factors (approximate sensitivity: change in preference if the factor were unknown):")
    lines += [f"    {n:<24}value {v:.2f}   effect {d:+.2f}" for n, v, d in card["factors"]]
    lines.append("Similar past incidents (Incident VectorDB):")
    if card["precedents"]:
        for p in card["precedents"]:
            flags = (" [provisional]" if p["provisional"] else "") + (" [CONTESTED]" if p["contested"] else "")
            lines.append(f"    distance {p['distance']:.2f}  {p['label']:<8} outcome {p['outcome']:+.1f}  "
                         f"seen {p['seen']}x  (then: {p['action']}){flags}")
    else:
        lines.append(f"    none close enough (nearest stored: distance {card['nearest']}): no precedent")
    lines += [f"Counterfactual  : {card['counterfactual']}",
              "Officer         : [A]pprove   [M]odify   [R]eject (reason required, logged as override)",
              "=" * 70]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from sb3_contrib import MaskablePPO
    from bloodchain_env import BloodChainEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(lr.ROOT / "logs" / "local" / "best_model.zip"))
    ap.add_argument("--scenario", default="severe", choices=["moderate", "severe"])
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--show", type=int, default=3, help="how many cards to print")
    args = ap.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    db_dir = lr.ROOT / "logs" / "local" / "incident_db"
    db = IncidentDB.load(db_dir) if (db_dir / "incidents.json").exists() else None
    print(f"Incident memory: {len(db) if db else 0} incidents")

    env = BloodChainEnv(args.scenario)
    obs, info = env.reset(seed=args.seed)
    cards, steps, done = [], 0, False
    while not done:
        mask = env.action_masks()
        card = make_card(model, obs, mask, steps + 1, db)
        if card:
            cards.append(card)
        action = int(np.argmax(action_probs(model, obs, mask)))
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc

    for c in cards[:args.show]:
        print(render(c), "\n")
    print(f"High-risk dispatches flagged: {len(cards)} of {steps} steps ({len(cards) / steps:.0%})")
    out = lr.ROOT / "logs" / "local" / "plan_cards.json"
    out.write_text(json.dumps(cards, indent=2, default=float))
    print("All cards saved to", out)