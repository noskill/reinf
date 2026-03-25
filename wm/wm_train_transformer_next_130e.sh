#!/usr/bin/env bash
set -euo pipefail

python3 wm_train.py --data data.txt \
    --model-type transformer \
    --obs-loss-mode soft --sensor-mode categorical \
    --batch-size 8 --epochs 300 --lr 2e-4 --weight-decay 0.02 \
    --attention-dropout 0.10 \
    --hidden-size 128 --layers 2 --heads 4 --head-dim 32 --intermediate 512 \
    --prior-rollout-weight 0.0 --prior-rollout-steps 0 \
    --z-only-weight 0.0 --h-only-weight 0.0 \
    --turn-weight 0.0 --step-weight 0.0 \
    --save-path outputs/wm_transformer_next_300e.pt | tee wm_transformer_next_300e.log
