#!/usr/bin/env bash
set -euo pipefail

python3 wm_train.py --data data.txt \
    --model-type rssm-discrete \
    --obs-loss-mode soft --sensor-mode categorical \
    --batch-size 8 --epochs 600 --lr 2e-4 --weight-decay 0.01 \
    --hidden-size 128 --obs-latent-dim 64 \
    --rssm-transition residual --rssm-residual-scale 0.01 --rssm-state-norm layernorm \
    --stoch-size 8 --stoch-classes 8 --stoch-temp 0.8 \
    --kl-dyn-beta 1.5 --kl-rep-beta 0.05 --kl-free-nats 0.5 \
    --prior-rollout-weight 0.0 --prior-rollout-steps 0 --bptt-horizon 0 \
    --z-only-weight 0.0 --h-only-weight 0.0 \
    --turn-weight 0.0 --step-weight 0.0 \
    --save-path outputs/wm_rssm_residual_ln_next_600e.pt | tee wm_rssm_residual_ln_next_600e.log
