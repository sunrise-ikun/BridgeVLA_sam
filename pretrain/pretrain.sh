#!/usr/bin/env bash

source /robot/robot-research-exp-0/user/lpy/BridgeVLA_sam_env/bridgevla_sam/bin/activate
BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
PRETRAIN_DIR="${BRIDGEVLA_ROOT}/pretrain"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

# The pretrain script imports from bridgevla.*, so point PYTHONPATH at the
# finetune directory (which contains the `bridgevla` package) and the bundled
# sam3 library.
export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export HF_HOME="/robot/robot-rfm/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SWANLAB_API_KEY="pRP4aOFOIGQGP468x0O8f"

# RoboPoint pretrain data (adjust paths as needed).
export ROBOPOINT_DATA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/pretrain_data/coco"
IMAGE_FOLDER="${ROBOPOINT_DATA_ROOT}/images"
JSON_DETECTION_PATH="${ROBOPOINT_DATA_ROOT}/detection_data.json"

# Cluster env vars (same pattern as finetune/RLBench/train_h20.sh)
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export MLP_WORKER_GPU=${RESOURCE_GPU:-2}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29503}

set -e -x

cd "${PRETRAIN_DIR}"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    pretrain.py \
    --branches 2 \
    --config_path pretrain_config.yaml \
    --image_folder "${IMAGE_FOLDER}" \
    --json_detection_path "${JSON_DETECTION_PATH}" \
    "$@"
