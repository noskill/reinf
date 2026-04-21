#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wm_maze_joint_rnn_${TIMESTAMP}}"

CMD=(
  "${PYTHON_BIN}" wm_maze_train.py
  --n-episodes 3000
  --num-envs 64
  --max-steps 256
  --random-dim 10
  --random-extra-openings 0
  --seed 42
  --device cuda
  --save-interval 100
  --experiment-name "${EXPERIMENT_NAME}"
  --policy-lr 3e-4
  --wm-lr 3e-4
  --discount 0.99
  --entropy-coef 0.01
  --wm-weight-decay 0.01
  --wm-updates-per-policy 1
  --wm-replay-capacity 2048
  --wm-train-episodes 64
  --intrinsic-reward-scale 1.0
  --env-reward-scale 0.0
  --wm-model-type rnn
  --wm-sensor-mode categorical
  --wm-hidden-size 176
  --wm-layers 3
  --wm-heads 4
  --wm-head-dim 44
  --wm-intermediate 704
  --wm-rnn-state-norm none
  --wm-obs-latent-dim 64
  --wm-probe-hidden-dim 128
  --wm-probe-layers 2
  --wm-contrastive-dim 64
  --wm-contrastive-steps 3
  --wm-contrastive-temp 0.1
  --wm-contrastive-discount 0.75
  --wm-contrastive-negatives 128
  --wm-sensor-weight 1.0
  --wm-loc-weight 0.0
  --wm-head-weight 0.0
  --wm-turn-weight 0.0
  --wm-step-weight 0.0
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
