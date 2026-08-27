# Retraining cost, in deterministic units

Wall-clock per retrain varies by up to 3x across these studies purely from CPU contention, so it cannot carry a cost claim. Retrains and base learners fitted are exact.


## Per study

| Study | Retrains | Trees fitted | Wall-clock (s) | s/retrain (indicative) |
|---|---|---|---|---|
| main factorial (scratch) | 5.8 | 462 | 95.9 | 16.59 |
| frozen baseline | 0.0 | 0 | 0.0 | n/a |
| minimax alternation | 4.0 | 320 | 159.3 | 39.84 |
| replay cap 800 | 8.0 | 640 | 381.3 | 47.66 |
| fine-tune, cost-matched | 6.3 | 507 | 78.9 | 12.46 |
| fine-tune, capacity-matched | 8.0 | 3200 | 230.4 | 28.80 |

Across studies doing identically-sized refits (80 trees each), measured s/retrain spans **12.46--47.66** — a **3.8x** spread with no difference in the work performed. That spread is contention, and it is larger than several of the policy differences it would otherwise be used to argue for.

## Cadence cost, stated exactly

| Cadence | Retrains | Trees fitted | Relative cost |
|---|---|---|---|
| every_n | 3.0 | 240 | 0.38x |
| every_round | 8.0 | 640 | 1.00x |
| threshold | 6.3 | 507 | 0.79x |

The vendor-facing claim rests on this table, not on wall-clock: retraining every third round performs **3** refits against **8**, a 2.67x reduction in retraining work that holds on any machine.

## Minimax cost

Minimax performs **4** refits against every-round's **8**: it is *cheaper* by a factor of 2.00x in retraining work, while settling at roughly twice the evasion (0.088 vs 0.042). Its higher measured wall-clock was contention: it ran alongside two other studies.

## Fine-tuning cost

| Arm | Retrains | Trees fitted | Settled evasion |
|---|---|---|---|
| refit from scratch | 5.8 | 462 | 0.2454 |
| fine-tune, cost-matched | 6.3 | 507 | 0.7239 |
| fine-tune, capacity-matched | 8.0 | 3200 | 0.7106 |

Cost-matched fine-tuning fits the *same* number of trees per retrain as a scratch refit, so it is not the more expensive option -- the earlier wall-clock reading that suggested otherwise was contention. Capacity-matched fine-tuning fits 5x the trees and is genuinely more expensive, in fitting and in the prediction cost the attacker pays on every query, without recovering the deficit.