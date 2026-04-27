#!/usr/bin/env bash

source /home/zyz/miniconda3/envs/bridgevla_pretrain/bin/activate

BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
PRETRAIN_DIR="${BRIDGEVLA_ROOT}/pretrain"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
SAM3_LIB_DIR="${BRIDGEVLA_ROOT}/libs/sam3"

export PYTHONPATH="${FINETUNE_DIR}:${SAM3_LIB_DIR}:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export HF_HOME="/DATA/disk1/zyz/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ROBOPOINT_DATA_ROOT="${BRIDGEVLA_ROOT}/data/bridgevla_data/pretrain_data"
IMAGE_FOLDER="${ROBOPOINT_DATA_ROOT}"
JSON_DETECTION_PATH="${ROBOPOINT_DATA_ROOT}/detection_data.json"
CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_data/logs/pretrain/bridgevla_pretrain_coco/20260426_140113/pretrain_epoch_0.pth"

set -e -x

python "${PRETRAIN_DIR}/pretrain.py" \
    --branches 3 \
    --config_path "${PRETRAIN_DIR}/pretrain_config.yaml" \
    --image_folder "${IMAGE_FOLDER}" \
    --json_detection_path "${JSON_DETECTION_PATH}" \
    "$@"
