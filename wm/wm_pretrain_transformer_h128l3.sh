#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-${EPOCHS:-300}}"
SAVE_PATH="${2:-${SAVE_PATH:-outputs/wm_transformer_pretrain_h128l3.pt}}"
LOG_PATH="${3:-${LOG_PATH:-wm_transformer_pretrain_h128l3.log}}"
CPC_WEIGHT="${CPC_WEIGHT:-0.1}"
CPC_DIM="${CPC_DIM:-64}"
CPC_STEPS="${CPC_STEPS:-10}"
CPC_DISCOUNT="${CPC_DISCOUNT:-0.75}"
CPC_NEGATIVES="${CPC_NEGATIVES:-128}"

python3 wm_train.py --data data.txt \
    --model-type transformer \
    --obs-loss-mode soft --sensor-mode categorical \
    --sensor-sigma 0.0 \
    --batch-size 8 --epochs "${EPOCHS}" --lr 2e-4 --weight-decay 0.02 \
    --attention-dropout 0.10 \
    --hidden-size 128 --layers 3 --heads 4 --head-dim 32 --intermediate 512 \
    --prior-rollout-weight 0.0 --prior-rollout-steps 0 \
    --z-only-weight 0.0 --h-only-weight 0.0 \
    --contrastive-weight "${CPC_WEIGHT}" \
    --contrastive-dim "${CPC_DIM}" \
    --contrastive-steps "${CPC_STEPS}" \
    --contrastive-horizon-discount "${CPC_DISCOUNT}" \
    --contrastive-negatives "${CPC_NEGATIVES}" \
    --loc-weight 0.0 --head-weight 0.0 --turn-weight 0.0 --step-weight 0.0 \
    --save-path "${SAVE_PATH}" | tee "${LOG_PATH}"
