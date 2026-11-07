"""Robustness test (Member A) -> Fig. 5 and the numbers for report Section 2.2.2.

Grid:  belief estimator  {NaiveBelief (C's placeholder), BetaBelief (A)}
     x sensor fault rate {0, 0.10, 0.25, 0.50}   (A's SensorNoise wrapper)
     x flood scenario    {moderate (training), severe (unseen)}
Same evaluation seeds in every cell (10000...), so cells are compared episode by episode
(paired test). Fault rate 0 + NaiveBelief = exactly the conditions the PPO was trained in;
every other cell is a train-deploy shift.

Extra per-episode measures:
  trap_steps  - steps where an urgent request waits at a hospital that has NO believed
                route (all routes p < 0.5): the belief-induced masking trap (Section 2.2.2)
  fault_steps - steps where the noise wrapper injected at least one fault

Usage (from the repo root, .venv active):
  python A_uncertainty/robustness_test.py --policy urgent --episodes 50
  python A_uncertainty/robustness_test.py --policy logs/local/best_model.zip --episodes 50
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "C_ml_engine"))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy import stats

from evaluate import run_episodes, make_policy
from bloodchain_env import DEPOT
from belief_estimator import BetaBelief
from noise_wrapper import SensorNoise

FAULT_RATES = [0.0, 0.10, 0.25, 0.50]
SCENARIOS = ["moderate", "severe"]
BELIEFS = {"NaiveBelief": None, "BetaBelief": BetaBelief}
SEED_START = 10000                      # unseen evaluation seeds (CONTRACT 6.2)
METRICS = ["ret", "on_time_rate", "failed_trips", "units_wasted", "temp_breaches",
           "unsafe_use", "expired", "waited_with_urgent", "trap_steps", "fault_steps"]


def make_setup(belief_cls, rate):
    def setup(env):
        u = env.unwrapped
        if belief_cls is not None:
            u.belief = belief_cls()
        u.sensor_hook = SensorNoise(fault_rate=rate)
    return setup


def counting(policy_fn, c):
    """Wrap the policy to count masking-trap and fault steps (does not change actions)."""
    def fn(env, obs, rng):
        u = env.unwrapped
        if any(q.urgent and u._path(DEPOT, q.hospital) is None for q in u.requests):
            c["trap_steps"] += 1
        c["fault_steps"] += int(u.sensor_fault)
        return policy_fn(env, obs, rng)
    return fn


def run_cell(policy_fn, scenario, belief_cls, rate, episodes):
    rows = []
    for ep in range(episodes):
        c = {"trap_steps": 0, "fault_steps": 0}
        df = run_episodes(counting(policy_fn, c), scenario, 1, SEED_START + ep,
                          env_setup=make_setup(belief_cls, rate))
        row = df.iloc[0].to_dict()
        row.update(c, episode=ep, seed=SEED_START + ep)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="urgent", help="urgent | nearest | path/to/best_model.zip")
    ap.add_argument("--episodes", type=int, default=50)
    a = ap.parse_args()

    label, policy_fn = make_policy(a.policy)
    label = "urgent" if a.policy == "urgent" else ("nearest" if a.policy == "nearest" else "ppo")
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    t0, frames = time.time(), []
    for scenario in SCENARIOS:
        for rate in FAULT_RATES:
            for bname, bcls in BELIEFS.items():
                df = run_cell(policy_fn, scenario, bcls, rate, a.episodes)
                df["scenario"], df["fault_rate"], df["belief"] = scenario, rate, bname
                frames.append(df)
                print(f"{scenario:8s} fault={rate:.2f} {bname:11s} return {df.ret.mean():7.1f} "
                      f"on-time {df.on_time_rate.mean():.3f} failed {df.failed_trips.mean():.2f} "
                      f"wasted {df.units_wasted.mean():.2f} trap {df.trap_steps.mean():5.1f}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(os.path.join(out_dir, f"robustness_{label}_raw.csv"), index=False)

    # summary: mean and SD per cell
    summ = raw.groupby(["scenario", "fault_rate", "belief"])[METRICS].agg(["mean", "std"]).round(3)
    summ.to_csv(os.path.join(out_dir, f"robustness_{label}_summary.csv"))

    # paired test: BetaBelief - NaiveBelief on the same seeds, per cell
    tests = []
    for (sc, rate), g in raw.groupby(["scenario", "fault_rate"]):
        n = g[g.belief == "NaiveBelief"].set_index("seed")
        b = g[g.belief == "BetaBelief"].set_index("seed")
        for m in ["ret", "on_time_rate", "failed_trips", "units_wasted", "expired", "trap_steps"]:
            d = b[m] - n[m]
            t, p = stats.ttest_rel(b[m], n[m]) if d.std() > 0 else (0.0, 1.0)
            tests.append(dict(scenario=sc, fault_rate=rate, metric=m, naive=n[m].mean(),
                              beta=b[m].mean(), diff=d.mean(),
                              ci95=1.96 * d.std() / np.sqrt(len(d)), t=t, p=p))
    tests = pd.DataFrame(tests).round(4)
    tests.to_csv(os.path.join(out_dir, f"robustness_{label}_paired.csv"), index=False)

    make_figure(raw, os.path.join(out_dir, f"fig5_robustness_{label}.png"), label)
    print("\nPaired BetaBelief - NaiveBelief (same seeds):")
    print(tests[tests.metric.isin(["ret", "failed_trips", "units_wasted"])].to_string(index=False))
    print(f"\nSaved results and Fig. 5 to {out_dir}  ({time.time() - t0:.0f}s)")


def make_figure(raw, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("on_time_rate", "On-time delivery rate"), ("failed_trips", "Failed trips per episode"),
              ("units_wasted", "Blood units wasted per episode"),
              ("trap_steps", "Masking-trap steps per episode")]
    colours = {"NaiveBelief": "#8a8a8a", "BetaBelief": "#1f6fb2"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for i, sc in enumerate(SCENARIOS):
        for j, (m, title) in enumerate(panels):
            ax = axes[i, j]
            for b in BELIEFS:
                g = raw[(raw.scenario == sc) & (raw.belief == b)].groupby("fault_rate")[m]
                mean, se = g.mean(), g.std() / np.sqrt(g.count())
                ax.errorbar(mean.index * 100, mean.values, yerr=1.96 * se.values, marker="o",
                            capsize=3, color=colours[b], label=b)
            ax.set_title(f"{title} ({sc} floods)", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("Sensor fault rate (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Robustness to sensor faults - policy: {label} "
                 f"(mean, 95% CI; synthetic environment)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
