#!/usr/bin/env bash
set -euo pipefail

python3 wm_train.py --data data.txt \
    --model-type tssm \
    --obs-loss-mode soft --sensor-mode categorical \
    --batch-size 8 --epochs 130 --lr 3e-4 \
    --hidden-size 128 --layers 2 --heads 4 --head-dim 32 --intermediate 512 \
    --stoch-size 8 --stoch-classes 8 --stoch-temp 1.0 \
    --kl-dyn-beta 2.0 --kl-rep-beta 0.1 --kl-free-nats 0.25 \
    --prior-rollout-weight 0.5 --bptt-horizon 0 \
    --save-path outputs/wm_tssm_both_130e.pt | tee wm_tssm_both_130e.log
