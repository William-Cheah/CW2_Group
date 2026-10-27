# A_uncertainty/  (Member A: Uncertainty, Belief & Safety)

Your code plugs into C's environment through **three hooks**, so you never edit `C_ml_engine/`.
The defaults in `bloodchain_env.py` (`no_sensor_noise`, `NaiveBelief`, `CrispSafety`) are placeholders for you to replace.

## The three hooks (from CONTRACT.md)
```python
import sys; sys.path.insert(0, "../C_ml_engine")
from bloodchain_env import BloodChainEnv

env = BloodChainEnv("moderate")
env.sensor_hook = my_noise_fn          # (readings, rng) -> (readings, sensor_fault: bool)
env.belief      = MyBetaBernoulli()    # .reset(n_roads); .update(readings) -> (p_passable[7], var[7])
env.safety      = MyFuzzySupervisor()  # .decide(logger_temp_c, minutes_out) -> "use"|"quarantine"|"discard"
```
`readings` is a dict: `rain_level` (0–3), `river_level` (0–1), `road_reports` (7 values: 1 passable, 0 flooded, −1 no report), `box_temp` (logger °C per van), `van_seen_flooded` (road ids a van just hit; treat these as reliable).

**Order every step:** the Sensor Layer builds readings, then `sensor_hook` corrupts them, then `belief.update` gives road beliefs. Beliefs feed the observation AND the action mask (p < 0.5 means the road is treated as closed). At delivery, `safety.decide` uses the **logger** temperature, not the true one, so a bad rule can cause "unsafe use" (−20 reward).

## Suggested files (yours to write)
| File | What |
|---|---|
| `noise_wrapper.py` | Logger drift/dropout, false road reports, report blackout. Return `sensor_fault=True` when you inject a fault |
| `belief_estimator.py` | Beta–Bernoulli per road: prior from rainfall/river level, update with reports, decay over time. Return mean and variance (variance = `belief_var` in the logs) |
| `fuzzy_safety.py` | scikit-fuzzy: temperature × time out of storage → use / quarantine / discard |
| `robustness_test.py` | Run `evaluate.run_episodes(..., env_setup=...)` at several noise levels × moderate/severe; make Fig. 5 |

Testing before C's model is ready: use `baselines.most_urgent_first` as the policy.
```python
from evaluate import run_episodes, make_policy
label, policy = make_policy("urgent")
df = run_episodes(policy, "severe", 50, 10000, env_setup=lambda e: setattr(e, "belief", MyBetaBernoulli()))
print(df.mean(numeric_only=True))
```
