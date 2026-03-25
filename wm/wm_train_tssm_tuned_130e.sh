#!/usr/bin/env bash
set -euo pipefail

python3 wm_train.py --data data.txt \
    --model-type tssm \
    --obs-loss-mode soft --sensor-mode categorical \
    --batch-size 8 --epochs 130 --lr 2e-4 \
    --hidden-size 128 --layers 2 --heads 4 --head-dim 32 --intermediate 512 \
    --stoch-size 8 --stoch-classes 8 --stoch-temp 0.8 \
    --kl-dyn-beta 1.5 --kl-rep-beta 0.05 --kl-free-nats 0.5 \
    --prior-rollout-weight 0.0 --prior-rollout-steps 0 --bptt-horizon 0 \
    --save-path outputs/wm_tssm_tuned_130e.pt | tee wm_tssm_tuned_130e.log
