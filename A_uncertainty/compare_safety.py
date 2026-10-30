"""Compare C's CrispSafety with A's FuzzySafety -> Table for report Section 1.2.2.

Every delivery decision is checked against the box's TRUE temperature (known to the
simulator, never to the agent), giving a confusion matrix per setting:
  unsafe_used    - true temp > 10 C but the supervisor said "use"      (missed: -20 reward)
  warm_caught    - true temp > 10 C and the supervisor blocked it      (correct)
  good_used      - true temp <= 10 C and used                          (correct)
  good_rejected  - true temp <= 10 C but quarantined / discarded       (false alarm: blood wasted)

Grid: safety {CrispSafety, FuzzySafety} x scenario {moderate, severe} x fault rate {0, 0.25, 0.5}
with A's BetaBelief + SensorNoise always on. Same seeds in every cell (paired).

Usage (repo root, .venv active):
  python A_uncertainty/compare_safety.py --policy urgent
  python A_uncertainty/compare_safety.py --policy logs/local/best_model.zip
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
from bloodchain_env import CrispSafety, TEMP_LIMIT
from belief_estimator import BetaBelief
from noise_wrapper import SensorNoise
from fuzzy_safety import FuzzySafety

SAFETY = {"CrispSafety": CrispSafety, "FuzzySafety": FuzzySafety}
SCENARIOS = ["moderate", "severe"]
FAULT_RATES = [0.0, 0.25, 0.50]
SEED_START = 10000
CLASSES = ["unsafe_used", "warm_caught", "good_used", "good_rejected"]


class Recorder:
    """Wraps a safety rule; records each decision against the box's true temperature."""

    def __init__(self, inner, env, log):
        self.inner, self.env, self.log = inner, env.unwrapped, log

    def decide(self, logger_temp_c, minutes_out):
        u = self.env
        temps = np.asarray(u.readings["box_temp"], dtype=float)
        cand = [v for v in range(len(u.vans)) if u.vans[v].cargo and u.vans[v].minutes_out == minutes_out]
        v = min(cand or range(len(u.vans)), key=lambda i: abs(temps[i] - logger_temp_c))
        true_t = u.vans[v].temp
        d = self.inner.decide(logger_temp_c, minutes_out)
        warm, used = true_t > TEMP_LIMIT, d == "use"
        cls = ("unsafe_used" if used else "warm_caught") if warm else ("good_used" if used else "good_rejected")
        self.log.append(dict(true_temp=true_t, logger_temp=logger_temp_c, minutes_out=minutes_out,
                             decision=d, cls=cls, units=sum(q.units for q in u.vans[v].cargo)))
        return d


def run_cell(policy_fn, scenario, safety_cls, rate, episodes):
    rows, decisions = [], []
    for ep in range(episodes):
        log = []

        def setup(env, log=log):
            u = env.unwrapped
            u.belief = BetaBelief()
            u.sensor_hook = SensorNoise(fault_rate=rate)
            u.safety = Recorder(safety_cls(), env, log)

        df = run_episodes(policy_fn, scenario, 1, SEED_START + ep, env_setup=setup)
        row = df.iloc[0].to_dict()
        row.update(seed=SEED_START + ep, **{c: sum(d["cls"] == c for d in log) for c in CLASSES})
        rows.append(row)
        decisions += [dict(d, seed=SEED_START + ep) for d in log]
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="urgent", help="urgent | nearest | path/to/best_model.zip")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    _, policy_fn = make_policy(a.policy)
    label = a.policy if a.policy in ("urgent", "nearest") else "ppo"
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    t0, eps, decs = time.time(), [], []
    for sc in SCENARIOS:
        for rate in FAULT_RATES:
            for name, cls in SAFETY.items():
                df, dd = run_cell(policy_fn, sc, cls, rate, a.episodes)
                for x in (df, dd):
                    x["scenario"], x["fault_rate"], x["safety"] = sc, rate, name
                eps.append(df)
                decs.append(dd)
                print(f"{sc:8s} fault={rate:.2f} {name:11s} return {df.ret.mean():7.1f} "
                      f"wasted {df.units_wasted.mean():.2f} unsafe {df.unsafe_use.mean():.3f}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    eps = pd.concat(eps, ignore_index=True)
    decs = pd.concat(decs, ignore_index=True)
    eps.to_csv(os.path.join(out_dir, f"safety_{label}_episodes.csv"), index=False)
    decs.to_csv(os.path.join(out_dir, f"safety_{label}_decisions.csv"), index=False)

    # confusion matrix per cell (totals over all episodes)
    conf = decs.groupby(["scenario", "fault_rate", "safety"])["cls"].value_counts().unstack(fill_value=0)
    conf = conf.reindex(columns=CLASSES, fill_value=0)
    warm = conf.unsafe_used + conf.warm_caught
    good = conf.good_used + conf.good_rejected
    conf["miss_rate"] = (conf.unsafe_used / warm.replace(0, np.nan)).round(3)
    conf["false_alarm_rate"] = (conf.good_rejected / good.replace(0, np.nan)).round(3)
    conf.to_csv(os.path.join(out_dir, f"safety_{label}_confusion.csv"))

    # paired per-episode tests: Fuzzy - Crisp on the same seeds
    tests = []
    for (sc, rate), g in eps.groupby(["scenario", "fault_rate"]):
        c = g[g.safety == "CrispSafety"].set_index("seed")
        f = g[g.safety == "FuzzySafety"].set_index("seed")
        for m in ["ret", "unsafe_use", "units_wasted", "on_time_rate"]:
            d = f[m] - c[m]
            t, p = stats.ttest_rel(f[m], c[m]) if d.std() > 0 else (0.0, 1.0)
            tests.append(dict(scenario=sc, fault_rate=rate, metric=m, crisp=c[m].mean(),
                              fuzzy=f[m].mean(), diff=d.mean(), t=t, p=p))
    tests = pd.DataFrame(tests).round(4)
    tests.to_csv(os.path.join(out_dir, f"safety_{label}_paired.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nConfusion matrix (deliveries, all episodes):")
    print(conf.to_string())
    print("\nPaired FuzzySafety - CrispSafety (same seeds):")
    print(tests.to_string(index=False))
    print(f"\nSaved to {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
