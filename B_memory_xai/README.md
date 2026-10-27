# B_memory_xai/  (Member B: Memory, Fallback & XAI)

You do not change the environment. You work from **logs** and from a wrapper around the policy.

## Logs you can use today
`logs/sample_urgent_moderate.jsonl` and `logs/sample_urgent_severe.jsonl` are **real** logs from the v0 environment (Most-Urgent-First policy, 3 episodes each), in the exact CONTRACT 6.3 format. Trained-model logs follow from C by 28 Oct, with the same format, so your code will not need to change.

```python
import sys; sys.path.insert(0, "../C_ml_engine")
from logger import read_jsonl, FIELDS
rows = read_jsonl("../logs/sample_urgent_severe.jsonl")
```
Action index meaning: `0` = wait; `1 + v*4 + k` = van v → hospital k (0 Pasir Mas, 1 Tanah Merah, 2 Machang, 3 Kuala Krai).
Observation layout (34 values) is in `CONTRACT.md` 6.2.

## Fallback Controller interface (agreed in CONTRACT)
```python
def choose(env, obs) -> (action, fallback_level, override):
    # L0 PPO -> L1 Most-Urgent-First (baselines.most_urgent_first) -> L2 manual -> L3 hold (action 0)
```
Pass `fallback_level` and `override` into `JSONLLogger.log(...)`; the logger already has those fields.

## Suggested files (yours to write)
| File | What |
|---|---|
| `log_reader.py` | Load `.jsonl` into a pandas table; decode actions; find incidents (temp_breach, units_wasted > 0, high belief_var) |
| `memory_gate.py` | Store only high-confidence incidents; deduplicate near-identical ones; handle conflicts |
| `incident_db.py` | FAISS store + retrieval of similar past incidents |
| `fallback_controller.py` | L0–L3 with measurable triggers (belief_var threshold, API timeout, operator override) |
| `plan_card.py` | Plan, confidence (policy action probabilities), key factors, counterfactual |

Tip for plan-card confidence: once C shares a model, `model.policy.get_distribution(obs_tensor, action_masks=mask).distribution.probs` gives the action probabilities. C will show you how.
