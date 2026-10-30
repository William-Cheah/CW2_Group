"""Compare C's placeholder NaiveBelief with A's BetaBelief (moderate + severe floods)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "C_ml_engine"))

import pandas as pd
from evaluate import run_episodes, make_policy
from belief_estimator import BetaBelief

N_EPISODES = 50
SEED_START = 10000          # unseen evaluation seeds (CONTRACT 6.2)
label, policy = make_policy("urgent")   # policy held fixed: only the belief changes

setups = {
    "NaiveBelief": lambda e: None,
    "BetaBelief": lambda e: setattr(e.unwrapped, "belief", BetaBelief()),
}
cols = ["ret", "on_time_rate", "expired", "units_wasted",
        "temp_breaches", "unsafe_use", "failed_trips"]

out_dir = os.path.join(HERE, "results")
os.makedirs(out_dir, exist_ok=True)
rows = []
for scenario in ["moderate", "severe"]:
    for name, setup in setups.items():
        df = run_episodes(policy, scenario, N_EPISODES, SEED_START, env_setup=setup)
        df.to_csv(os.path.join(out_dir, f"compare_{name}_{scenario}.csv"), index=False)
        for c in cols:
            rows.append({"scenario": scenario, "belief": name, "metric": c,
                         "mean": df[c].mean(), "std": df[c].std()})

summary = pd.DataFrame(rows).pivot_table(index=["scenario", "metric"],
                                         columns="belief", values=["mean", "std"])
pd.set_option("display.width", 200)
print(summary.round(3))
summary.round(3).to_csv(os.path.join(out_dir, "compare_summary.csv"))
print("\nSaved to", out_dir)