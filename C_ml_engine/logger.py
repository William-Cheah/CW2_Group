"""
logger.py - JSONL logger that follows CONTRACT.md Section 6.3 EXACTLY.
Shared by everyone: C writes logs, A adds sensor_fault, B reads them.

One JSON object per line, saved as logs/<run_name>.jsonl
Do NOT add or rename fields without agreement in the group chat.
"""
import json
import os

FIELDS = ["run", "episode", "step", "obs", "belief_var", "action", "fallback_level",
          "reward", "pending_urgent", "units_wasted", "temp_breach", "sensor_fault", "override"]


class JSONLLogger:
    def __init__(self, run_name, log_dir="logs", round_obs=3):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_name}.jsonl")
        self.run = run_name
        self.round_obs = round_obs
        self.f = open(self.path, "w", encoding="utf-8")

    def log(self, episode, step, obs, action, reward, info, fallback_level=0, override=False):
        rec = {
            "run": self.run,
            "episode": int(episode),
            "step": int(step),
            "obs": [round(float(x), self.round_obs) for x in obs],
            "belief_var": round(float(info["belief_var"]), 4),
            "action": int(action),
            "fallback_level": int(fallback_level),      # set by B's Fallback Controller (0-3)
            "reward": round(float(reward), 4),
            "pending_urgent": int(info["pending_urgent"]),
            "units_wasted": int(info["units_wasted"]),
            "temp_breach": bool(info["temp_breach"]),
            "sensor_fault": bool(info["sensor_fault"]),  # set by A's noise wrapper
            "override": bool(override),                  # human override active
        }
        assert list(rec) == FIELDS
        self.f.write(json.dumps(rec) + "\n")

    def close(self):
        self.f.close()


def read_jsonl(path):
    """Tiny reader so anyone can load a log quickly (B builds the full log reader)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
