#!/usr/bin/env bash
set -euo pipefail

source /robot/robot-research-exp-0/user/lpy/BridgeVLA_sam_env/miniconda3/bin/activate bridgevla_sam_gembench

BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
MODEL_EPOCH="${1:-4}"
MODEL_FOLDER="${2:-${BRIDGEVLA_ROOT}/data/bridgevla_data/logs/train_gembench/gembench_31tasks_fixSAM_lr8e-5_04_26_03_09}"
PORT="${PORT:-13004}"

export PYTHONPATH="${FINETUNE_DIR}:${BRIDGEVLA_ROOT}/libs/sam3:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224"
export SAM3_CHECKPOINT_PATH="${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3"
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM=offscreen
export DATA_CACHE_ROOT="/robot/robot-rfm/.data_cache"
export HF_HOME="/robot/robot-rfm/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

[ -d "${MODEL_FOLDER}" ] || { echo "[Error] MODEL_FOLDER missing: ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/model_${MODEL_EPOCH}.pth" ] || { echo "[Error] checkpoint missing: ${MODEL_FOLDER}/model_${MODEL_EPOCH}.pth"; exit 1; }
[ -f "${MODEL_FOLDER}/exp_cfg.yaml" ] || { echo "[Error] exp_cfg.yaml missing in ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/mvt_cfg.yaml" ] || { echo "[Error] mvt_cfg.yaml missing in ${MODEL_FOLDER}"; exit 1; }
[ -d "${PALIGEMMA_PATH}" ] || { echo "[Error] PALIGEMMA_PATH missing: ${PALIGEMMA_PATH}"; exit 1; }

echo "[Info] MODEL_FOLDER = ${MODEL_FOLDER}"
echo "[Info] MODEL_EPOCH  = ${MODEL_EPOCH}"
echo "[Info] PORT         = ${PORT}"

cd "${FINETUNE_DIR}/GemBench"
xvfb-run -a python3 server.py --port "${PORT}" --model_epoch "${MODEL_EPOCH}" --base_path "${MODEL_FOLDER}"
