"""log_reader.py (yayuen)
loads weiliang(c) JSONL logs, decodes actions and observations, and finds incidents
for the Memory Gate. Everything else in B_memory_xai builds on this file."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
pd.set_option("display.width", 250)

ROOT = Path(__file__).resolve().parents[1]  #repo root
sys.path.insert(0, str(ROOT/ "C_ml_engine"))
from logger import read_jsonl

HOSPITALS = ["Pasir Mas", "Tanah Merah", "Machang", "Kuala Krai"]
N_ROADS = 7
BELIEF_VAR_MAX = 0.25   # max of p(1-p): used to scale belief_var to 0..1
HIGH_VAR_QUANTILE = 0.90    # "high uncertainty" = top 10% of belief_var
OUTCOME_WINDOW = 8  # steps after an incident (8 x 15 min = 2h = urgent deadline)

def decode_action(a):
    """Action index -> readable text. CONTRACT 6.2: 0 = wait, 1 + v*4 +  k = van v -> hospital k."""
    a = int(a)
    if a == 0:
        return "wait"
    van = (a - 1)//4
    hospital = (a - 1)%4
    return f"van {van} -> {HOSPITALS[hospital]}"


def obs_features(obs):
    """34-value observation -> named features (CONTRACT 6.2 layout)."""
    obs = list(obs)
    feats = {
        "road_beliefs": obs[0:N_ROADS],
        "rain": obs[7],
        "river": obs[8],
        "time_of_day": obs[9],
    }
    for v in range(2):  # 2 vans x 4 values, obs[10:18]
        idle, time_to_free, cargo, logger_temp = obs[10 + 4 * v: 14 + 4 * v]
        feats[f"van{v}"] = {"idle": idle, "time_to_free": time_to_free,
                            "cargo": cargo, "logger_temp": logger_temp}
    for h, name in enumerate(HOSPITALS):    # 4 hospitals x 4 values, obs[18:34]
        pending, urgent, deadline, travel = obs[18 + 4 * h: 22 + 4 * h]
        feats[name] = {"pending": pending, "urgent": urgent,
                       "time_to_deadline": deadline, "travel_time": travel}
    return feats


def to_vector(obs, belief_var):
    """The ONE feature vector used by Memory Gate, Incident VectorDB and Plan Card.
    14 numbers, all 0..1: 7 road beliefs, rain, river, scaled belief_var, urgent units at 4 hospitals."""
    obs = np.asarray(obs, dtype=np.float32)
    road_beliefs = obs[0:N_ROADS]
    weather = obs[7:9]   # rain, river
    var_scaled = np.array([min(belief_var / BELIEF_VAR_MAX, 1.0)], dtype=np.float32)
    urgent = obs[19:34:4]   # indices 19, 23, 27, 31
    return np.concatenate([road_beliefs, weather, var_scaled, urgent]).astype(np.float32)


def load(path):
    """Read one .jsonl log into a pandas table, sorted, with readable actions."""
    df = pd.DataFrame(read_jsonl(str(path)))
    df = df.sort_values(["episode", "step"]).reset_index(drop=True)
    df["action_text"] = df["action"].apply(decode_action)
    return df


def belief_var_stats(df):
    """Distribution of belief_var: evidence for choosing the fallback thresholds (tau)."""
    q = df["belief_var"].quantile([0.5, 0.75, 0.9])
    stats = {"min": df["belief_var"].min(), "p50": q[0.5], "p75": q[0.75],
             "p90": q[0.9], "max": df["belief_var"].max()}
    print("belief_var  " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))
    return stats


def find_incidents(df, var_threshold=None):
    """Mark incident steps, then group consecutive incident steps (same episode) into incidents.
    Incident step = temp breach OR wasted units OR override OR fallback active OR high uncertainty."""
    if var_threshold is None:
        var_threshold = df["belief_var"].quantile(HIGH_VAR_QUANTILE)
    df = df.copy()
    df["is_incident"] = (df["temp_breach"].astype(bool)
                         | (df["units_wasted"] > 0)
                         | df["override"].astype(bool)
                         | (df["fallback_level"] > 0)
                         | (df["belief_var"] > var_threshold))
    # a new block starts when incident status changes OR a new episode starts
    new_block = ((df["is_incident"] != df["is_incident"].shift())
                 | (df["episode"] != df["episode"].shift()))
    df["block"] = new_block.cumsum()
    inc = df[df["is_incident"]]
    if inc.empty:
        return pd.DataFrame()
    summary = inc.groupby("block").agg(
        episode=("episode", "first"),
        start_step=("step", "first"),
        end_step=("step", "last"),
        n_steps=("step", "size"),
        max_belief_var=("belief_var", "max"),
        units_wasted=("units_wasted", "sum"),
        temp_breach=("temp_breach", "any"),
        override=("override", "any"),
        total_reward=("reward", "sum"),
        start_obs=("obs", "first"),
        start_belief_var=("belief_var", "first"),
    ).reset_index(drop=True)
    # Outcome = what happened during the incident AND in the next OUTCOME_WINDOW steps
    outcomes = []
    for _, r in summary.iterrows():
        after = df[(df["episode"] == r["episode"])
                   & (df["step"] > r["end_step"])
                   & (df["step"] <= r["end_step"] + OUTCOME_WINDOW)]
        outcomes.append({
            "outcome_reward": r["total_reward"] + after["reward"].sum(),
            "outcome_wasted": int(r["units_wasted"] + after["units_wasted"].sum()),
            "outcome_breach": bool(r["temp_breach"] or after["temp_breach"].any()),
            "resolved": len(after) == OUTCOME_WINDOW,   # False if the episode ended too soon
        })
    return pd.concat([summary, pd.DataFrame(outcomes)], axis=1)


if __name__ == "__main__":
    path = ROOT / "logs" / "sample_urgent_severe.jsonl"
    df = load(path)
    print(f"Loaded {len(df)} steps from {path.name}\n")
    print(df[["episode", "step", "action_text", "belief_var", "reward"]].head(10).to_string(index=False))
    print()
    belief_var_stats(df)
    incidents = find_incidents(df)
    print(f"\nFound {len(incidents)} incidents")
    if not incidents.empty:
        print(incidents.drop(columns=["start_obs"]).to_string(index=False))
    print("\nFeature vector (step 1):", np.round(to_vector(df.loc[0, "obs"], df.loc[0, "belief_var"]), 3))