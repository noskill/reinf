#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-${EPOCHS:-500}}"
SAVE_PATH="${2:-${SAVE_PATH:-outputs/wm_rnn_pretrain_h176l3.pt}}"
LOG_PATH="${3:-${LOG_PATH:-wm_rnn_pretrain_h176l3.log}}"
RNN_NORM="${RNN_NORM:-none}"
CPC_WEIGHT="${CPC_WEIGHT:-0.1}"
CPC_DIM="${CPC_DIM:-64}"
CPC_STEPS="${CPC_STEPS:-10}"
CPC_DISCOUNT="${CPC_DISCOUNT:-0.75}"
CPC_NEGATIVES="${CPC_NEGATIVES:-128}"

python3 wm_train.py --data data.txt \
    --model-type rnn \
    --obs-loss-mode soft --sensor-mode categorical \
    --sensor-sigma 0.0 \
    --batch-size 8 --epochs "${EPOCHS}" --lr 2e-4 --weight-decay 0.02 \
    --attention-dropout 0.0 \
    --hidden-size 176 --layers 3 --heads 4 --head-dim 44 --intermediate 704 \
    --rnn-state-norm "${RNN_NORM}" \
    --probe-hidden-dim 256 --probe-layers 2 \
    --prior-rollout-weight 0.0 --prior-rollout-steps 0 \
    --z-only-weight 0.0 --h-only-weight 0.0 \
    --contrastive-weight "${CPC_WEIGHT}" \
    --contrastive-dim "${CPC_DIM}" \
    --contrastive-steps "${CPC_STEPS}" \
    --contrastive-horizon-discount "${CPC_DISCOUNT}" \
    --contrastive-negatives "${CPC_NEGATIVES}" \
    --loc-weight 0.0 --head-weight 0.0 --turn-weight 0.0 --step-weight 0.0 \
    --save-path "${SAVE_PATH}" | tee "${LOG_PATH}"
