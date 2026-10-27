"""
check_setup.py - run this after installing, on EVERY laptop:
    python check_setup.py
It checks versions, that the environment passes SB3's check_env, that action
masking works, and that a contract-format log can be written and read back.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "C_ml_engine"))

ok = True
print(f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
    print("  !! use Python 3.10-3.13 (3.11 or 3.12 recommended)"); ok = False
for mod in ["numpy", "gymnasium", "networkx", "torch", "stable_baselines3", "sb3_contrib",
            "tensorboard", "yaml", "pandas", "matplotlib", "skfuzzy", "faiss"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:18s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {mod:18s} MISSING ({e.__class__.__name__})"); ok = False

import numpy as np
from stable_baselines3.common.env_checker import check_env

from bloodchain_env import BloodChainEnv
from logger import FIELDS, JSONLLogger, read_jsonl

env = BloodChainEnv("moderate")
check_env(env, warn=True)
print("check_env: OK")
obs, info = env.reset(seed=0)
assert env.action_masks()[0], "wait must always be allowed"
log = JSONLLogger("setup_check", log_dir=os.path.join("logs", "local"))
rng = np.random.default_rng(0)
for step in range(96):
    a = int(rng.choice(np.flatnonzero(env.action_masks())))
    obs, r, term, trunc, info = env.step(a)
    log.log(0, step + 1, obs, a, r, info)
    if term or trunc:
        break
log.close()
rows = read_jsonl(log.path)
assert list(rows[0]) == FIELDS and len(rows) == 96
print(f"log format: OK ({len(rows)} lines -> {log.path})")
print("ALL GOOD - you are ready." if ok else "Fix the items marked above, then run again.")
