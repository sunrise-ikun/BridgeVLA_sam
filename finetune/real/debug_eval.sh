#!/usr/bin/env bash
# Offline debug-eval launcher. Mirrors train.sh's env setup but always runs
# single-GPU, no DDP. All CLI args are forwarded to debug_eval.py.
#
# Usage:
#   bash finetune/real/debug_eval.sh
#   bash finetune/real/debug_eval.sh --checkpoint /path/to/model_X.pth --num_episodes 3
#
# Env-var toggles:
#   CUDA_VISIBLE_DEVICES — pin a specific GPU
#   SWANLAB_API_KEY      — unused here, but carried over for parity
#   REAL_EVAL_CHECKPOINT — default checkpoint if --checkpoint is not passed

BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DEFAULT_CKPT="${REAL_EVAL_CHECKPOINT:-${BRIDGEVLA_ROOT}/data/bridgevla_data/logs_real/train/model_last.pth}"
CKPT_ARGS=()
if [[ " $* " != *" --checkpoint "* ]]; then
    CKPT_ARGS+=(--checkpoint "${DEFAULT_CKPT}")
fi

set -e -x

cd "${FINETUNE_DIR}"

python real/debug_eval.py \
    "${CKPT_ARGS[@]}" \
    "$@"
