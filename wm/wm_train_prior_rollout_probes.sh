#!/usr/bin/env bash
set -euo pipefail

python3 wm_train.py --data data.txt \
    --model-type tssm \
    --obs-loss-mode soft --sensor-mode categorical \
    --batch-size 8 --epochs 30 --lr 3e-4 \
    --hidden-size 128 --layers 2 --heads 4 --head-dim 32 --intermediate 512 \
    --stoch-size 8 --stoch-classes 8 --stoch-temp 1.0 \
    --kl-dyn-beta 0.0 --kl-rep-beta 0.0 --kl-free-nats 0.25 \
    --prior-rollout-weight 0.0 --bptt-horizon 0 \
    --sensor-weight 1.0 --loc-weight 0.0 --head-weight 0.0 --turn-weight 0.0 --step-weight 0.0 \
    --z-only-weight 1.0 --h-only-weight 1.0 \
    --train-recon-heads-only \
    --load-path outputs/wm_tssm_prior_rollout_130e.pt \
    --save-path outputs/wm_tssm_prior_rollout_130e_probes.pt | tee wm_tssm_prior_rollout_130e_probes.log
