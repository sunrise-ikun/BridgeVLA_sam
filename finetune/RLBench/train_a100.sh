#!/usr/bin/env bash
#source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate bridgevla_sam

BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RLBENCH_DATA_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/RLBench"
export RLBENCH_REPLAY_STORAGE_DIR="${BRIDGEVLA_ROOT}/data/bridgevla_data/replay_train"

# Cluster env vars (same pattern as mibot/scripts/train.sh)
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}
export MLP_WORKER_GPU=${RESOURCE_GPU:-7}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29501}

set -e -x

cd "${FINETUNE_DIR}/RLBench"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    train.py \
    --exp_cfg_path configs/rlbench_config_a100.yaml \
    $@
