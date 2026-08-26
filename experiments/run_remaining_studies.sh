#!/usr/bin/env bash
# Queue the three remaining studies, in cheapest-first order, once the main
# factorial sweep has finished. Each writes to its own results directory and is
# independently resumable: re-running skips cells already recorded, so an
# interrupted queue is picked up rather than restarted.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "=== waiting for the main sweep to finish ==="
while pgrep -f "run_multiseed.py --config configs/ember_factorial.yaml --seeds 5" >/dev/null 2>&1; do
  sleep 20
done
echo "main sweep done"

# ---------------------------------------------------------------------------
# 1. Frozen-classifier baseline (cadence: never).
# The lower bound the single-round evasion literature reports against, and the
# control without which "retraining helps" has nothing to be compared to.
# Data selection is irrelevant when the defender never retrains, so only one
# level is run rather than three identical ones.
# ---------------------------------------------------------------------------
echo "=== [1/3] frozen-classifier baseline ==="
python -u experiments/run_multiseed.py \
  --config configs/ember_factorial.yaml \
  --cadences never --selections full_replay \
  --seeds 5 --out results/ember_baseline_frozen

# ---------------------------------------------------------------------------
# 2. Retrain-mode axis (scratch vs finetune), the third factorial axis.
# Compared against the completed scratch arm, so only finetune is run here.
# ---------------------------------------------------------------------------
echo "=== [2/3] retrain-mode axis (finetune) ==="
python -u experiments/run_multiseed.py \
  --config configs/ember_factorial.yaml \
  --modes finetune \
  --seeds 5 --out results/ember_mode_axis

# ---------------------------------------------------------------------------
# 3. Attacker axis (DQN, PPO) -- the roadmap's secondary axis.
# The question is whether the data-selection conclusion survives a different
# attacker, so cadence is pinned to every_round and only the data-selection
# levels vary. Both RL agents need per-round training, which is what makes this
# the expensive study; 3 seeds is enough to test whether the ordering holds.
# ---------------------------------------------------------------------------
echo "=== [3/3] attacker axis (dqn, ppo) ==="
python -u experiments/run_multiseed.py \
  --config configs/ember_factorial.yaml \
  --attackers dqn,ppo --cadences every_round \
  --seeds 3 --out results/ember_attacker_axis

echo "=== all three studies complete ==="
for d in ember_baseline_frozen ember_mode_axis ember_attacker_axis; do
  echo "--- $d ---"
  wc -l < "results/$d/multiseed_raw.csv" 2>/dev/null || echo "missing"
done
