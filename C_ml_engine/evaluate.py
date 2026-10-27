"""
evaluate.py - run a policy (baseline or trained MaskablePPO model) on held-out
seeds and write (1) a JSONL log in the CONTRACT format and (2) a summary.

Examples:
  python evaluate.py --policy urgent  --scenario severe --episodes 100
  python evaluate.py --policy runs/mppo_lr3e-4_s0/best_model.zip --scenario severe --log
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from baselines import BASELINES
from bloodchain_env import BloodChainEnv
from logger import JSONLLogger

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))


def make_policy(name):
    """Return (label, fn(env, obs, rng) -> action)."""
    if name in BASELINES:
        f = BASELINES[name]
        return name, lambda env, obs, rng: f(env, rng)
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(name, device="cpu")
    label = os.path.basename(os.path.dirname(name)) or "model"
    return label, lambda env, obs, rng: int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])


def run_episodes(policy_fn, scenario, episodes, seed0, logger=None, env_setup=None):
    rows = []
    for ep in range(episodes):
        env = BloodChainEnv(scenario)
        if env_setup:                       # e.g. A plugs in noise / belief / fuzzy here
            env_setup(env)
        obs, info = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        done, total, step = False, 0.0, 0
        while not done:
            a = policy_fn(env, obs, rng)
            obs, r, term, trunc, info = env.step(a)
            total += r
            step += 1
            if logger:
                logger.log(ep, step, obs, a, r, info)
            done = term or trunc
        s = info["stats"]
        rows.append(dict(episode=ep, ret=total, on_time_rate=s["on_time"] / max(1, s["requests"]), **s))
    return pd.DataFrame(rows)


def summarise(df, label, scenario):
    m = df.mean(numeric_only=True)
    return (f"{label:24s} {scenario:9s} return {m.ret:7.1f} (sd {df.ret.std():5.1f}) | on-time {m.on_time_rate:.3f} | "
            f"expired {m.expired:5.2f} | wasted units {m.units_wasted:4.2f} | breaches {m.temp_breaches:4.2f} | "
            f"failed trips {m.failed_trips:4.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="urgent", help="random | nearest | urgent | path/to/model.zip")
    p.add_argument("--scenario", default=CFG["env"]["test_scenario"], choices=["moderate", "severe"])
    p.add_argument("--episodes", type=int, default=CFG["seeds"]["eval_episodes"])
    p.add_argument("--log", action="store_true", help="also write logs/<run>.jsonl")
    a = p.parse_args()
    label, fn = make_policy(a.policy)
    logger = JSONLLogger(f"eval_{label}_{a.scenario}", log_dir=os.path.join(HERE, "..", "logs", "local")) if a.log else None
    df = run_episodes(fn, a.scenario, a.episodes, CFG["seeds"]["eval_start"], logger)
    if logger:
        logger.close()
        print("log written:", logger.path)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", f"eval_{label}_{a.scenario}.csv"), index=False)
    print(summarise(df, label, a.scenario))
