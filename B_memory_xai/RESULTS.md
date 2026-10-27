# Member B results: Fallback Controller, Memory Gate, Plan Cards

Model: best_model.zip (C's MaskablePPO, from Drive, downloaded 27 Oct 2026)
SHA256: 4e940b3aa5a242479fd8ba29968f0f03f5e2cb0d067daeea9505cc5640016da5
If C shares a new model, check this hash; if it differs, all PPO results must be rerun.

## 1. Fallback Controller
L0 policy: C's MaskablePPO. Scenario: severe floods.
Tuning on validation seeds 5000-5019 (20 days); final test on eval seeds 10000-10099 (100 days).

### Exploratory run (default config, eval seeds, BEFORE tuning)
Return diff vs PPO: -7.7 ± 8.0; L0 share only 29% (fallback in control 71% of the time).
This diagnosis motivated tuning. Disclosure: the test seeds were therefore seen once before the
final run, so the final test is not fully untouched.

### Ablation (validation, 20 days; paired return difference vs PPO alone, 95% CI)
| # | Setting | Return diff | L0 share | Failed trips | Wasted | On-time |
|---|---|---|---|---|---|---|
| - | PPO alone | - | 100% | 4.15 | 0.40 | 0.262 |
| 1 | Default (tau 0.0688/0.0525, enter 2, L2 on) | -12.6 ± 23.7 | 29% | 2.20 | 1.10 | 0.242 |
| 2 | Stricter tau (0.09/0.07) | -11.3 ± 21.5 | 37% | 2.00 | 0.90 | 0.234 |
| 3 | Enter steps 4 | -3.5 ± 21.3 | 38% | 2.15 | 0.65 | 0.246 |
| 4 | L2 fully off | -2.1 ± 15.6 | 51% | 3.20 | 0.50 | 0.271 |
| 5a | tau + enter 4 + L2 recovery only (CHOSEN) | +1.7 ± 21.2 | 50% | 2.15 | 0.45 | 0.273 |
| 5b | tau + enter 4 + L2 fully off | -0.7 ± 15.1 | 59% | 3.60 | 0.40 | 0.255 |

### Final test (config 5a, 100 days, run once after tuning)
| Metric | PPO alone | PPO + fallback |
|---|---|---|
| Return | -41.3 (sd 47.8) | -47.4 (sd 52.3) |
| Paired difference | | -6.1 ± 7.9 (95% CI); fallback better on 41% of days |
| On-time rate | 0.265 | 0.245 |
| Expired | 8.94 | 9.54 |
| Wasted units | 0.29 | 0.61 |
| Temperature breaches | 0.09 | 0.24 |
| Failed trips | 4.67 | 3.21 (-31%) |
| Time at levels | | L0 44%, L1 8%, L2 9%, L3 39%; critical alerts 400 |

### Findings
- Validation gain (+1.7) did not transfer to test (-6.1): best of 6 configs chosen on 20 noisy days (winner's curse). Fix: more validation days.
- Every configuration reduced failed trips (31-52%) but none improved return over trained PPO.
- Trade-off: fewer failed trips vs more wasted units / breaches.
- Bug found during ablation: --no-l2 first disabled only escalation, not recovery (L2 share was not 0). Fixed; both variants (5a, 5b) tested.
- Next (integration): same comparison under A's sensor noise (distribution shift).

## 2. Memory Gate
### PPO logs (severe, 20 days, eval seeds 10000-10019), TAU_EXIT 0.07
125 candidates -> 82 stored, 5 stored provisional, 2 merged, 25 rejected insignificant, 11 rejected unresolved.
87 incidents in memory (~4 per day). 29% of candidates filtered. No contested incidents (duplicates rare at DUP_DISTANCE 0.2).

### Sample logs (Most-Urgent-First baseline), TAU_EXIT 0.0525 (before tuning)
39 candidates -> 18 stored, 8 stored provisional, 8 rejected insignificant, 5 rejected unresolved.
First design REJECTED uncertain incidents: discarded 3 of the 4 worst outcomes (-36.4, -28.2, -18.3) -> changed to provisional flag.

### Design notes
- L2 distance chosen over cosine: cosine rated "road 0 open" vs "road 0 flooded" 0.955 similar; L2 = 0.79.
- Coupling: TAU_EXIT is shared by Fallback and Memory Gate; raising it to 0.07 cut provisional incidents 8 -> 1 on sample logs.

## 3. Plan Card Generator (PPO, severe)
- Seed 10000: 5 of 96 steps flagged (5%). Seed 10050: 2 of 96 (2%). All flagged for low policy preference (< 0.6).
- Precedents found at distance 0.46-0.48 (weak matches, just under MAX_DISTANCE 0.5).
- Fig. 6 = seed 10050, step 56: preference 0.47; precedent "harmful -10.3, then: wait" supports dispatching.
- Limitation: key-factor sensitivity can be implausible (seed 10000 step 70: blanking Machang demand raised preference to go to Machang, +0.41) because perturbed inputs are off-distribution.
- Limitation: counterfactual changes beliefs but not the action mask.
- Windows: KMP_DUPLICATE_LIB_OK workaround (PyTorch + FAISS OpenMP conflict), 1 thread.

## How to reproduce (from repo root)
cd C_ml_engine
python evaluate.py --policy ../logs/local/best_model.zip --scenario severe --episodes 20 --log
cd ..
python B_memory_xai/memory_gate.py logs/local/eval_local_severe.jsonl
python B_memory_xai/fallback_controller.py --policy logs/local/best_model.zip --episodes 100
python B_memory_xai/plan_card.py --seed 10050