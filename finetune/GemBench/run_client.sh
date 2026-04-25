#!/usr/bin/env bash
set -euo pipefail

source /robot/robot-research-exp-0/user/lpy/BridgeVLA_sam_env/miniconda3/bin/activate bridgevla_sam_gembench

BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
MODEL_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/logs/train_gembench/gembench_lr8e-5_04_24_21_43"
GEMBENCH_ROOT="${BRIDGEVLA_ROOT}/data/bridgevla_data/GEMBench"
SEED="${1:-300}"
MODEL_EPOCH="${2:-50}"
MICROSTEP_DATA_DIR="${3:-${GEMBENCH_ROOT}/test_dataset/microsteps/seed${SEED}}"
OUTPUT_ROOT="${4:-${MODEL_FOLDER}/eval/gembench}"
PORT="${PORT:-13003}"
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

echo "[Info] SERVER       = http://${IP}:${PORT}"
echo "[Info] MODEL_EPOCH  = ${MODEL_EPOCH}"
echo "[Info] SEED         = ${SEED}"
echo "[Info] MICROSTEPS   = ${MICROSTEP_DATA_DIR}"
echo "[Info] OUTPUT_ROOT  = ${OUTPUT_ROOT}"

cd "${FINETUNE_DIR}/GemBench"

declare -A TASKVARS
TASKVARS[train]="open_door+0 open_drawer+0 open_drawer+2 close_jar_peract+15 close_jar_peract+16"
TASKVARS[test_l2]="close_jar_peract+3 close_jar_peract+4"
TASKVARS[test_l3]="open_drawer_long+0 open_drawer_long+1 open_drawer_long+2 open_drawer_long+3 open_door2+0 open_drawer2+0 open_drawer3+0 open_drawer+1 close_drawer+0"
TASKVARS[test_l4]="put_items_in_drawer+0 put_items_in_drawer+2 put_items_in_drawer+4"

for split in train test_l2 test_l3 test_l4; do
    for taskvar in ${TASKVARS[$split]}; do
            xvfb-run -a python3 client.py \
                        --ip "${IP}" \
                                    --port "${PORT}" \
                                                --output_file "${OUTPUT_ROOT}/model_${MODEL_EPOCH}/seed${SEED}/${split}/result.json" \
                                                            --microstep_data_dir "${MICROSTEP_DATA_DIR}" \
                                                                        --taskvar "${taskvar}"
                                                                            done
                                                                            done

                                                                            