# CONTRACT (brief Section 6): PROPOSED v0

> **Status: PROPOSED by C on 21 Oct, based on A's minimum requirements.**
> Reply "agree" in the group chat, or suggest changes **before Fri 23 Oct**.
> After that this file is frozen. Any change needs agreement in the group chat,
> and C updates `C_ml_engine/bloodchain_env.py` to match.

## 6.1 Component names (use exactly these everywhere)
Sensor Layer · Belief Estimator · PPO Dispatch Policy · Fuzzy Safety Supervisor · Fallback Controller · Short-Term Memory · Memory Gate · Incident VectorDB · Plan Card Generator

## 6.2 Environment (custom Gymnasium, built by C)

| Setting | Value (v0) |
|---|---|
| Environment file | `C_ml_engine/bloodchain_env.py` (`BloodChainEnv`, id `BloodChain-v0`) |
| Map | 1 blood bank (Kota Bharu) + **4 hospitals** (Pasir Mas, Tanah Merah, Machang, Kuala Krai), **7 roads**. Synthetic, inspired by Kelantan. Every hospital has at least 2 routes, so a flood forces a re-route instead of ending the game |
| Vans / cold-box capacity | **2 vans / 6 units each** |
| Time per step / episode length | **15 minutes / 96 steps (24 h)** |
| Flood model | Hidden rainfall level 0–3 (none, light, heavy, extreme) changes hourly (Markov chain). Each step a road floods with probability `[0, 0.02, 0.08, 0.20][rain] × susceptibility` (0.2–0.85 per road) and a flooded road reopens with `[0.30, 0.15, 0.05, 0.01][rain]`. Rain also slows travel by +25% per level |
| Sensors (Sensor Layer) | Field report per road with P = 0.5 each step, 90% correct (`-1` = no report). River level = 0.25 × rain + noise. Cold-box logger = true temperature + N(0, 0.3 °C). A van that hits a flooded road reports it reliably at the next step |
| Demand | Poisson requests per hospital (0.04 per step × (1 + 0.3 × rain)), 30% urgent, 1–4 units. Deadline: urgent 2 h, routine 6 h. Expires 4 h after the deadline |
| Cold chain | Box packed at +4 °C, warms 0.2–0.6 °C per step while carrying blood (hidden per trip). Transport limit +10 °C. At delivery the Fuzzy Safety Supervisor decides use / quarantine / discard from the **logger** reading |
| Actions | Discrete(9): `0` = wait; `1 + v*4 + k` = send van v (0–1) to hospital k (0–3: Pasir Mas, Tanah Merah, Machang, Kuala Krai). **Masked:** van busy, hospital has nothing pending, or no route that the Belief Estimator thinks is passable (p ≥ 0.5) |
| Observation | Box(34) in [0,1]: 7 road passability beliefs, rain, river level, time of day, 4 per van (idle, time to free, cargo, logger temp), 4 per hospital (pending units, urgent units, time to nearest deadline, believed travel time) |
| Reward terms (C finalises) | + on-time urgent delivery (+10) · + on-time routine (+4) · − late (−3 urgent / −1 routine) · − expired request (−8 / −3) · − wasted units (−5 each) · − temperature breach (−10) · − unsafe use of warm blood (−20) · − failed trip (−2) · − each waiting urgent request (−0.1 per step, which stops the "wait forever" reward hack) |
| Training / unseen test scenarios | **moderate** floods (rain starts mostly 0–2) / **severe** floods (rain starts 1–3 and rises faster) |
| Training seeds / evaluation seeds | Algorithm seeds **0, 1, 2**; validation episode seeds 5000–5019; **evaluation episode seeds 10000–10099** (never used in training) |

### Plug-in points (so A and B never edit C's file)
| Hook on `env.unwrapped` | Owner | Signature | Default (placeholder) |
|---|---|---|---|
| `sensor_hook` | A | `(readings: dict, rng) -> (readings, sensor_fault: bool)` | no noise |
| `belief` | A | object with `reset(n_roads)` and `update(readings) -> (p_passable[7], var[7])` | `NaiveBelief` (latest report) |
| `safety` | A | object with `decide(logger_temp_c, minutes_out) -> "use" \| "quarantine" \| "discard"` | `CrispSafety` (hard thresholds) |
| Fallback Controller | B | wraps the policy: `choose(env, obs) -> (action, fallback_level, override)`, outside the env | none (always L0) |

`readings` keys: `rain_level` (int), `river_level` (float), `road_reports` (array of 7: 1/0/−1), `box_temp` (array of 2, logger °C), `van_seen_flooded` (list of road ids).

## 6.3 Log format (unchanged from the brief)
One JSON object per line → `logs/<run_name>.jsonl`. Written by `C_ml_engine/logger.py` (`JSONLLogger`).

| Field | Type | Meaning |
|---|---|---|
| run | str | Run name |
| episode | int | Episode number |
| step | int | Step within episode |
| obs | list | Observation vector (34 values) |
| belief_var | float | Belief Estimator uncertainty (mean over 7 roads) |
| action | int | Chosen action index (0–8) |
| fallback_level | int | 0–3 |
| reward | float | Step reward |
| pending_urgent | int | Urgent requests not yet served |
| units_wasted | int | Blood units discarded this step |
| temp_breach | bool | A cold box exceeded its temperature limit |
| sensor_fault | bool | Fault injected by A's wrapper |
| override | bool | Human override active |

Example (real line from `logs/sample_urgent_moderate.jsonl`) is in that file.
