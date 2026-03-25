#!/usr/bin/env bash
set -euo pipefail

BASE_CKPT="${1:-outputs/wm_rnn_next_300e.pt}"
OUT_CKPT="${2:-outputs/wm_rnn_state_probes_mlp_30e.pt}"
OUT_LOG="${3:-wm_rnn_state_probes_mlp_30e.log}"

python3 wm_train.py --data data.txt \
    --model-type rnn \
    --obs-loss-mode soft --sensor-mode categorical \
    --load-path "$BASE_CKPT" \
    --train-state-probes-only --probe-hidden-dim 256 \
    --batch-size 8 --epochs 30 --lr 3e-4 \
    --hidden-size 128 --layers 2 --heads 4 --head-dim 32 --intermediate 512 \
    --sensor-weight 0.0 --loc-weight 1.0 --head-weight 1.0 \
    --turn-weight 0.0 --step-weight 0.0 \
    --save-path "$OUT_CKPT" | tee "$OUT_LOG"
