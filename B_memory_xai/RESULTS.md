# Member B results: Fallback Controller

L0 policy: C's MaskablePPO (best_model.zip). Scenario: severe floods.
Tuning on validation seeds 5000-5019 (20 days); final test on eval seeds 10000-10099 (100 days, run once).

## Ablation (validation, 20 days; paired return difference vs PPO alone, 95% CI)
| # | Setting | Return diff | L0 share | Failed trips | Wasted | On-time |
|---|---|---|---|---|---|---|
| - | PPO alone | - | 100% | 4.15 | 0.40 | 0.262 |
| 1 | Default (tau 0.0688/0.0525, enter 2, L2 on) | -12.6 ± 23.7 | 29% | 2.20 | 1.10 | 0.242 |
| 2 | Stricter tau (0.09/0.07) | -11.3 ± 21.5 | 37% | 2.00 | 0.90 | 0.234 |
| 3 | Enter steps 4 | -3.5 ± 21.3 | 38% | 2.15 | 0.65 | 0.246 |
| 4 | L2 fully off | -2.1 ± 15.6 | 51% | 3.20 | 0.50 | 0.271 |
| 5a | tau + enter 4 + L2 recovery only (CHOSEN) | +1.7 ± 21.2 | 50% | 2.15 | 0.45 | 0.273 |
| 5b | tau + enter 4 + L2 fully off | -0.7 ± 15.1 | 59% | 3.60 | 0.40 | 0.255 |

## Final test (config 5a, 100 unseen days)
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

## Findings
- Validation gain (+1.7) did not transfer to test (-6.1): selecting the best of 6 configs on 20 noisy days (winner's curse). Fix: more validation days.
- Every configuration reduced failed trips (31-52%) but none improved return over trained PPO.
- Trade-off: fewer failed trips vs more wasted units / breaches.
- Bug found during ablation: --no-l2 first disabled only escalation, not recovery (L2 share was not 0). Fixed; both variants (5a, 5b) tested.
- Next (integration): same comparison under A's sensor noise (distribution shift).

## Memory Gate (sample logs, NaiveBelief)
39 candidate incidents -> 18 stored, 8 stored provisional, 8 rejected insignificant, 5 rejected unresolved.
First design rejected uncertain incidents: discarded 3 of the 4 worst outcomes (-36.4, -28.2, -18.3) -> changed to provisional flag.