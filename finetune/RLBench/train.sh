#!/usr/bin/env bash
set -e

# =======================================================================
# Paths (absolute; edit here if you move data)
# =======================================================================
BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

# PaliGemma base weights
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TOKENIZERS_PARALLELISM=false

# =======================================================================
# PYTHONPATH: finetune/ must be on the path so that
#   "import RLBench.utils.peract_utils_rlbench" resolves correctly.
# =======================================================================
export PYTHONPATH="${FINETUNE_DIR}:${PYTHONPATH:-}"

# =======================================================================
# CoppeliaSim / Qt (needed by import chain, not by simulation)
# =======================================================================
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}:/usr/lib/x86_64-linux-gnu/qt5/plugins"

# =======================================================================
# Xvfb: CoppeliaSim's Qt needs an X display even during training
#   (the import chain initializes Qt internally).
#   Auto-detect or start a virtual X display.
# =======================================================================
XVFB_DISPLAY=":99"

if [ -z "${DISPLAY:-}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    if xdpyinfo -display "${XVFB_DISPLAY}" >/dev/null 2>&1; then
        echo "[Info] Reusing existing Xvfb on ${XVFB_DISPLAY}"
    else
        echo "[Info] Starting Xvfb on ${XVFB_DISPLAY} ..."
        Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        sleep 2
        if ! kill -0 $! 2>/dev/null; then
            echo "[Error] Xvfb failed to start"; exit 1
        fi
    fi
    export DISPLAY="${XVFB_DISPLAY}"
fi

# =======================================================================
# Distributed training env vars
#
# These variables are set automatically by most cluster schedulers:
#   SLURM    : SLURM_NTASKS, SLURM_PROCID, SLURM_NODELIST, ...
#   MLP/PAI  : WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT, RESOURCE_GPU
#   torchrun : sets them internally
#
# For local single-machine use, defaults apply (1 node, all GPUs).
# =======================================================================
NNODES=${WORLD_SIZE:-1}                  # number of nodes
NODE_RANK=${RANK:-0}                     # rank of this node
GPUS_PER_NODE=${RESOURCE_GPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "========================================================"
echo "[Info] NNODES        = ${NNODES}"
echo "[Info] NODE_RANK     = ${NODE_RANK}"
echo "[Info] GPUS_PER_NODE = ${GPUS_PER_NODE}"
echo "[Info] MASTER_ADDR   = ${MASTER_ADDR}"
echo "[Info] MASTER_PORT   = ${MASTER_PORT}"
echo "[Info] DISPLAY       = ${DISPLAY}"
echo "========================================================"

cd "${FINETUNE_DIR}/RLBench"

# =======================================================================
# Launch training via torchrun
#   $@ passes through all extra arguments (config overrides, etc.)
# =======================================================================
torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    "$@"
