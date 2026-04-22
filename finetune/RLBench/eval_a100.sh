#!/usr/bin/env bash
set -e

# =======================================================================
# Paths (absolute; edit here if you move data)
# =======================================================================
BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"

# Model & data
MODEL_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/logs/train/2task_lr5e-5_transformer_04_21_20_18"
MODEL_NAME="model_26.pth"                # checkpoint file inside MODEL_FOLDER
EVAL_DATAFOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/RLBench"

# PaliGemma base weights (picked up by bridgevla/mvt/mvt_single.py via env var)
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"

# Keep HF offline to avoid accidental network fetches (optional, comment out if undesired)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# =======================================================================
# PYTHONPATH: finetune/ must be on the path so that
#   "import RLBench.utils.peract_utils_rlbench" resolves correctly.
# =======================================================================
export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"

# =======================================================================
# CoppeliaSim / Qt / display
# =======================================================================
cd "${FINETUNE_DIR}"
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

# Qt xcb needs GLX integration plugins from the system Qt5 package
# (libqt5gui5 must be installed: apt-get install -y libqt5gui5)
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}:/usr/lib/x86_64-linux-gnu/qt5/plugins"

# =======================================================================
# Xvfb: auto-detect or start a virtual X display for headless rendering.
#   - Reuses an existing Xvfb on :99 if already running.
#   - On SLURM clusters, each node should run its own Xvfb instance.
# =======================================================================
XVFB_DISPLAY=":99"

if [ -z "${DISPLAY:-}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    if xdpyinfo -display "${XVFB_DISPLAY}" >/dev/null 2>&1; then
        echo "[Info] Reusing existing Xvfb on ${XVFB_DISPLAY}"
    else
        echo "[Info] Starting Xvfb on ${XVFB_DISPLAY} ..."
        Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
        sleep 2
        if ! kill -0 ${XVFB_PID} 2>/dev/null; then
            echo "[Error] Xvfb failed to start"; exit 1
        fi
        echo "[Info] Xvfb started (PID=${XVFB_PID})"
    fi
    export DISPLAY="${XVFB_DISPLAY}"
else
    echo "[Info] Using existing DISPLAY=${DISPLAY}"
fi

cd "${FINETUNE_DIR}/RLBench"

# Sanity checks
[ -d "${MODEL_FOLDER}" ]        || { echo "[Error] MODEL_FOLDER missing: ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/${MODEL_NAME}" ] || { echo "[Error] checkpoint missing: ${MODEL_FOLDER}/${MODEL_NAME}"; exit 1; }
[ -f "${MODEL_FOLDER}/exp_cfg.yaml" ]  || { echo "[Error] exp_cfg.yaml missing in ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/mvt_cfg.yaml" ]  || { echo "[Error] mvt_cfg.yaml missing in ${MODEL_FOLDER}"; exit 1; }
[ -d "${EVAL_DATAFOLDER}" ]     || { echo "[Error] EVAL_DATAFOLDER missing: ${EVAL_DATAFOLDER}"; exit 1; }
[ -d "${PALIGEMMA_PATH}" ]      || { echo "[Error] PALIGEMMA_PATH missing: ${PALIGEMMA_PATH}"; exit 1; }

echo "[Info] MODEL_FOLDER     = ${MODEL_FOLDER}"
echo "[Info] MODEL_NAME       = ${MODEL_NAME}"
echo "[Info] EVAL_DATAFOLDER  = ${EVAL_DATAFOLDER}"
echo "[Info] PALIGEMMA_PATH   = ${PALIGEMMA_PATH}"
echo "[Info] COPPELIASIM_ROOT = ${COPPELIASIM_ROOT}"
echo "[Info] DISPLAY          = ${DISPLAY}"

# =======================================================================
# Run evaluation
# =======================================================================
python3 eval.py \
    --model-folder    "${MODEL_FOLDER}" \
    --eval-datafolder "${EVAL_DATAFOLDER}" \
    --model-name      "${MODEL_NAME}" \
    --tasks           "place_shape_in_shape_sorter" \
    --eval-episodes   25 \
    --episode-length  25 \
    --log-name        "eval_rlbench_sam_$(date +%Y%m%d_%H%M%S)" \
    --device          0 \
    --headless \
    --visualize \
    --save-video
