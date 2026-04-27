#!/usr/bin/env bash
set -euo pipefail

source /robot/robot-research-exp-0/user/lpy/BridgeVLA_sam_env/miniconda3/bin/activate bridgevla_sam_gembench

BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
MODEL_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/logs/train_gembench/gembench_31tasks_fixSAM_lr8e-5_04_26_03_09"
GEMBENCH_ROOT="${BRIDGEVLA_ROOT}/data/bridgevla_data/GEMBench"
SEED="${1:-300}"
MODEL_EPOCH="${2:-44}"
MICROSTEP_DATA_DIR="${3:-${GEMBENCH_ROOT}/test_dataset/microsteps/seed${SEED}}"
OUTPUT_ROOT="${4:-${MODEL_FOLDER}/eval/gembench}"
PORT="${PORT:-13044}"
IP="${IP:-localhost}"

export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export DATA_CACHE_ROOT="/robot/robot-rfm/.data_cache"
export HF_HOME="/robot/robot-rfm/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

[ -d "${MICROSTEP_DATA_DIR}" ] || { echo "[Error] MICROSTEP_DATA_DIR missing: ${MICROSTEP_DATA_DIR}"; echo "[Hint] Extract ${GEMBENCH_ROOT}/test_dataset/microsteps.tar.gz under ${GEMBENCH_ROOT}/test_dataset first."; exit 1; }

RECORD_VIDEO="${RECORD_VIDEO:-0}"
VIDEO_ROTATE_CAM="${VIDEO_ROTATE_CAM:-0}"
VIDEO_RES_W="${VIDEO_RES_W:-320}"
VIDEO_RES_H="${VIDEO_RES_H:-180}"
VISUALIZE="${VISUALIZE:-0}"
VISUALIZE_ROOT_DIR="${VISUALIZE_ROOT_DIR:-${OUTPUT_ROOT}/model_${MODEL_EPOCH}/seed${SEED}/visualize}"

EXTRA_ARGS=()
[ "${RECORD_VIDEO}" = "1" ] && EXTRA_ARGS+=(--record_video --video_resolution_width "${VIDEO_RES_W}" --video_resolution_height "${VIDEO_RES_H}")
[ "${VIDEO_ROTATE_CAM}" = "1" ] && EXTRA_ARGS+=(--video_rotate_cam)
[ "${VISUALIZE}" = "1" ] && EXTRA_ARGS+=(--visualize --visualize_root_dir "${VISUALIZE_ROOT_DIR}")

echo "[Info] SERVER       = http://${IP}:${PORT}"
echo "[Info] MODEL_EPOCH  = ${MODEL_EPOCH}"
echo "[Info] SEED         = ${SEED}"
echo "[Info] MICROSTEPS   = ${MICROSTEP_DATA_DIR}"
echo "[Info] OUTPUT_ROOT  = ${OUTPUT_ROOT}"
echo "[Info] RECORD_VIDEO = ${RECORD_VIDEO} (${VIDEO_RES_W}x${VIDEO_RES_H}, rotate=${VIDEO_ROTATE_CAM})"
echo "[Info] VISUALIZE    = ${VISUALIZE} (${VISUALIZE_ROOT_DIR})"

cd "${FINETUNE_DIR}/GemBench"

TASKS_DIR="${FINETUNE_DIR}/GemBench/assets"
declare -A TASK_JSONS
TASK_JSONS[train]="${TASKS_DIR}/taskvars_train.json"
TASK_JSONS[test_l2]="${TASKS_DIR}/taskvars_test_l2.json"
TASK_JSONS[test_l3]="${TASKS_DIR}/taskvars_test_l3.json"
TASK_JSONS[test_l4]="${TASKS_DIR}/taskvars_test_l4.json"

for split in train test_l2 test_l3 test_l4; do
    mapfile -t taskvars < <(python3 -c "import json,sys; [print(t) for t in json.load(open('${TASK_JSONS[$split]}'))]")
    for taskvar in "${taskvars[@]}"; do
        xvfb-run -a python3 client.py \
            --ip "${IP}" \
            --port "${PORT}" \
            --output_file "${OUTPUT_ROOT}/model_${MODEL_EPOCH}/seed${SEED}/${split}/result.json" \
            --microstep_data_dir "${MICROSTEP_DATA_DIR}" \
            --taskvar "${taskvar}" \
            "${EXTRA_ARGS[@]}"
    done
done

