#!/usr/bin/env bash
# Real-robot offline trainer launcher. Mirrors finetune/RLBench/train_h20.sh.
#
# Usage:
#   bash finetune/real/train.sh [extra args forwarded to train.py]
#
# Env-var toggles (all optional):
#   DEBUG=true          — single-GPU debug path (requires WORLD_SIZE=1, RESOURCE_GPU=1)
#   VISUALIZE=1         — start-of-epoch viz on (default). VISUALIZE=0 turns it off.
#   REAL_PRETRAIN_PATH  — override default pretrain checkpoint path.
#   SWANLAB_API_KEY     — override the baked-in SwanLab key.
#
# Single-GPU debug:
#   DEBUG=true bash finetune/real/train.sh --debug --epochs 1 --max_iter 2
#
# Disable viz:
#   VISUALIZE=0 bash finetune/real/train.sh

BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Same key that finetune/RLBench/train_h20.sh exports. Override via env var
# externally if you want a different account.
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-pRP4aOFOIGQGP468x0O8f}"

# ---- default pretrain checkpoint ----
# Start real-robot finetuning from the BridgeVLA pretrain weights.
# Callers can override by passing --pretrain_path ... on the CLI or by setting
# REAL_PRETRAIN_PATH=... in the environment. Pass --no-pretrain to skip.
DEFAULT_PRETRAIN_PATH="${REAL_PRETRAIN_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/bridgevla_sam_pretrain/pretrain_epoch_1.pth}"
PRETRAIN_ARGS=()
if [[ " $* " != *" --pretrain_path "* && " $* " != *" --no-pretrain "* ]]; then
    PRETRAIN_ARGS+=(--load_pretrain --pretrain_path "${DEFAULT_PRETRAIN_PATH}")
fi
# Strip our private --no-pretrain sentinel before forwarding to train.py.
FORWARD_ARGS=()
for a in "$@"; do
    [[ "$a" == "--no-pretrain" ]] && continue
    FORWARD_ARGS+=("$a")
done

# ---- start-of-epoch visualization toggle ----
# VISUALIZE=1  → pass --visualize    (default, captures a pre-training baseline)
# VISUALIZE=0  → pass --no-visualize (skip viz every epoch)
# CLI --visualize / --no-visualize always wins over the env var.
VISUALIZE="${VISUALIZE:-1}"
VIZ_ARGS=()
if [[ " $* " != *" --visualize "* && " $* " != *" --no-visualize "* ]]; then
    if [[ "${VISUALIZE}" == "0" ]]; then
        VIZ_ARGS+=(--no-visualize)
    else
        VIZ_ARGS+=(--visualize)
    fi
fi

export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export MLP_WORKER_GPU=${RESOURCE_GPU:-4}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29503}

set -e -x

cd "${FINETUNE_DIR}"

if [[ "${DEBUG:-false}" == "true" && "${MLP_WORKER_NUM}" == "1" && "${MLP_WORKER_GPU}" == "1" ]]; then
    python real/train.py \
        --exp_cfg_path real/configs/real_config.yaml \
        --mvt_cfg_path real/configs/mvt_cfg.yaml \
        "${PRETRAIN_ARGS[@]}" \
        "${VIZ_ARGS[@]}" \
        "${FORWARD_ARGS[@]}"
else
    torchrun \
        --nnodes=${MLP_WORKER_NUM} \
        --node_rank=${MLP_ROLE_INDEX} \
        --nproc_per_node=${MLP_WORKER_GPU} \
        --master_addr=${MLP_WORKER_0_HOST} \
        --master_port=${MLP_WORKER_0_PORT} \
        real/train.py \
        --exp_cfg_path real/configs/real_config.yaml \
        --mvt_cfg_path real/configs/mvt_cfg.yaml \
        "${PRETRAIN_ARGS[@]}" \
        "${VIZ_ARGS[@]}" \
        "${FORWARD_ARGS[@]}"
fi
