"""
train.py - train the PPO Dispatch Policy with MaskablePPO (sb3-contrib).

  python train.py                       # uses config.yaml, seed 0
  python train.py --seed 1
  python train.py --lr 1e-3 --run mppo_lr1e-3      # ablation
Watch training:  tensorboard --logdir runs/tb
"""
import argparse
import os
import time

import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from bloodchain_env import BloodChainEnv
from evaluate import run_episodes, summarise
from logger import JSONLLogger

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--run", default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    hp = dict(cfg["ppo"])
    if a.lr:
        hp["learning_rate"] = a.lr
    run = f"{a.run or cfg['run_name']}_s{a.seed}"
    out = os.path.join(HERE, "runs", run)
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump({"run": run, "seed": a.seed, "ppo": hp, "env": cfg["env"]}, open(os.path.join(out, "run_config.yaml"), "w"))

    sc = cfg["env"]["train_scenario"]
    venv = DummyVecEnv([lambda: Monitor(BloodChainEnv(sc)) for _ in range(cfg["env"]["n_envs"])])
    venv.seed(a.seed)
    eval_env = Monitor(BloodChainEnv(sc))
    eval_env.reset(seed=cfg["seeds"]["val_start"])

    model = MaskablePPO("MlpPolicy", venv, seed=a.seed, verbose=1,
                        tensorboard_log=os.path.join(HERE, "runs", "tb"),
                        policy_kwargs=dict(net_arch=cfg["policy"]["net_arch"]), **hp)
    callbacks = [
        MaskableEvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(1, cfg["eval_freq"] // cfg["env"]["n_envs"]),
                             best_model_save_path=out, log_path=out, deterministic=True),
        CheckpointCallback(save_freq=max(1, 100_000 // cfg["env"]["n_envs"]), save_path=out, name_prefix="ckpt"),
    ]
    t0 = time.time()
    model.learn(total_timesteps=a.steps or cfg["total_timesteps"], callback=callbacks, tb_log_name=run)
    model.save(os.path.join(out, "final_model"))
    print(f"training done in {time.time() - t0:.0f}s -> {out}")

    # quick check on train + unseen scenario, and a contract-format log for A and B
    best_path = os.path.join(out, "best_model.zip")
    if not os.path.exists(best_path):            # very short runs never reach an evaluation
        best_path = os.path.join(out, "final_model.zip")
    best = MaskablePPO.load(best_path, device="cpu")
    fn = lambda env, obs, rng: int(best.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])
    for scen in (cfg["env"]["train_scenario"], cfg["env"]["test_scenario"]):
        logger = JSONLLogger(f"{run}_{scen}", log_dir=os.path.join(HERE, "..", "logs", "local"))
        df = run_episodes(fn, scen, 20, cfg["seeds"]["eval_start"], logger)
        logger.close()
        print(summarise(df, run, scen))


if __name__ == "__main__":
    main()
