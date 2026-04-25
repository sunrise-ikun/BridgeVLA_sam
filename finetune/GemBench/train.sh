#!/usr/bin/env bash

source /robot/robot-research-exp-0/user/lpy/BridgeVLA_sam_env/miniconda3/bin/activate bridgevla_sam_gembench
BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
export DATA_CACHE_ROOT="/robot/robot-rfm/.data_cache"
export HF_HOME="/robot/robot-rfm/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SWANLAB_API_KEY="pRP4aOFOIGQGP468x0O8f"
export GEMBENCH_DATA_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/GEMBench/train_dataset"

# Cluster env vars (same pattern as mibot/scripts/train.sh)
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export MLP_WORKER_GPU=${RESOURCE_GPU:-2}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29502}

set -e -x

cd "${FINETUNE_DIR}/GemBench"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    train.py \
    --exp_cfg_path configs/gembench_config.yaml \
    --data_folder "${GEMBENCH_DATA_FOLDER}" \
    $@