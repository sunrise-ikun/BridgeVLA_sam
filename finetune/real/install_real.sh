#!/usr/bin/env bash
set -e

# =======================================================================
# BridgeVLA-SAM  Real-Robot Environment Installer
# =======================================================================
# Sets up all dependencies needed for:
#   1. Real-robot training   (finetune/real/train.py)
#   2. Real-robot inference   (bridgevla_sam_real_eval)
#
# Prerequisites:
#   - A conda env with Python 3.10+ already activated (e.g. 3d_vla)
#   - CUDA 12.4 compatible GPU driver
#   - ZED SDK installed if using ZED camera  (provides pyzed via its own installer)
#   - Intel RealSense SDK if using RealSense (provides pyrealsense2)
#
# Usage:
#   conda activate 3d_vla   # or your target env
#   bash finetune/real/install_real.sh
#
# What this does NOT install (unlike install_rlbench.sh):
#   - CoppeliaSim / PyRep / RLBench  (simulation-only, not needed for real robot)
# =======================================================================

BRIDGEVLA_ROOT="/home/zk/Projects/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
LIBS_DIR="${FINETUNE_DIR}/bridgevla/libs"
SAM3_DIR="${BRIDGEVLA_ROOT}/libs/sam3"

echo "============================================================"
echo "  BridgeVLA-SAM  Real-Robot Environment Installer"
echo "============================================================"
echo "  BRIDGEVLA_ROOT : ${BRIDGEVLA_ROOT}"
echo "  FINETUNE_DIR   : ${FINETUNE_DIR}"
echo "  SAM3_DIR       : ${SAM3_DIR}"
echo "  Python         : $(python --version 2>&1)"
echo "  Conda env      : ${CONDA_DEFAULT_ENV:-<none>}"
echo "============================================================"

# --- Optional: GitHub mirror (leave empty for direct) ---
GITHUB_MIRROR=""
# GITHUB_MIRROR="https://ghfast.top/https://github.com/"
if [ -n "${GITHUB_MIRROR}" ]; then
    git config --global url."${GITHUB_MIRROR}".insteadOf "https://github.com/"
    echo "[Info] GitHub mirror: ${GITHUB_MIRROR}"
fi

# --- Optional: PyPI mirror (Tsinghua) ---
# Uncomment the following block if you are in mainland China.
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# --- Optional: PyTorch index (for mainland China) ---
# TORCH_INDEX_URL="https://mirror.sjtu.edu.cn/pytorch-wheels/cu124"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

# ======================================================================
# 1. Core build tools
# ======================================================================
echo ""
echo "[Step 1/7] Core build tools ..."
pip install --upgrade pip setuptools wheel ninja pyyaml

# ======================================================================
# 2. PyTorch ecosystem  (torch 2.6.0 + cu124)
# ======================================================================
echo ""
echo "[Step 2/7] PyTorch + xformers ..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url ${TORCH_INDEX_URL}
# xformers 0.0.29.post3 matches torch 2.6.0
pip install xformers==0.0.29.post3 --index-url ${TORCH_INDEX_URL}

# ======================================================================
# 3. bridgevla package (installs numpy, scipy, einops, transformers, etc.)
# ======================================================================
echo ""
echo "[Step 3/7] bridgevla package ..."
cd "${FINETUNE_DIR}"
pip install -e .
# Pin transformers version for PaliGemma compatibility
pip install 'transformers==4.51.3'
pip install 'accelerate>=0.26.0'

# ======================================================================
# 4. Local library dependencies
# ======================================================================
echo ""
echo "[Step 4/7] Local libs (point-renderer, YARR, peract_colab, SAM3) ..."

# point-renderer (required by MVT for rendering)
if [ -d "${LIBS_DIR}/point-renderer" ]; then
    pip install -e "${LIBS_DIR}/point-renderer"
else
    echo "[WARN] point-renderer not found at ${LIBS_DIR}/point-renderer — skipping"
fi

# YARR (agent interface, needed by load_agent fallback)
if [ -d "${LIBS_DIR}/YARR" ]; then
    pip install -e "${LIBS_DIR}/YARR"
else
    echo "[WARN] YARR not found at ${LIBS_DIR}/YARR — skipping"
fi

# peract_colab (utility functions)
if [ -d "${LIBS_DIR}/peract_colab" ]; then
    pip install -e "${LIBS_DIR}/peract_colab"
else
    echo "[WARN] peract_colab not found at ${LIBS_DIR}/peract_colab — skipping"
fi

# SAM3 (frozen vision-language encoder, new in 4_28_real branch)
if [ -d "${SAM3_DIR}" ]; then
    pip install -e "${SAM3_DIR}"
else
    echo "[WARN] sam3 not found at ${SAM3_DIR} — skipping"
fi

# ======================================================================
# 5. Additional Python packages
# ======================================================================
echo ""
echo "[Step 5/7] Additional Python packages ..."

# pytorch3d (needed by point-renderer / augmentation)
pip install "git+${GITHUB_MIRROR:-https://github.com/}facebookresearch/pytorch3d.git@stable"

# CLIP (needed by bridgevla setup.py but may fail due to git dep)
pip install "git+${GITHUB_MIRROR:-https://github.com/}openai/CLIP.git" || true

# Common training / eval deps
pip install yacs swanlab bitsandbytes h5py

# ======================================================================
# 6. Real-robot specific packages
# ======================================================================
echo ""
echo "[Step 6/7] Real-robot packages ..."

# Flask inference server
pip install flask

# 3D visualization
pip install open3d

# Robot control (Dobot TCP/Modbus)
pip install pymodbus

# Quaternion utilities
pip install pyquaternion

# OpenCV (prefer non-headless for real-robot GUI display)
pip install opencv-python

# Note: The following packages are hardware-specific and typically installed
# via their respective SDK installers rather than pip:
#
#   pyrealsense2  — Intel RealSense SDK
#       sudo apt install librealsense2-dkms librealsense2-utils
#       pip install pyrealsense2
#
#   pyzed         — Stereolabs ZED SDK
#       Download from https://www.stereolabs.com/developers/release
#       Run the ZED SDK installer, which installs pyzed into the active env.
#
# If these are NOT already installed, uncomment and adapt:
# pip install pyrealsense2
echo "[Info] Skipping pyrealsense2 / pyzed (install via their SDK installers)"

# ======================================================================
# 7. System packages (may need sudo)
# ======================================================================
echo ""
echo "[Step 7/7] System packages (optional, may need sudo) ..."

# OpenGL / display dependencies for Open3D and pyrender
if command -v apt-get &>/dev/null; then
    echo "[Info] Installing system display libs (requires sudo) ..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        libffi-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
        libxcb-cursor0 libxcb-xinerama0 libxcb-xinput0 \
        libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 \
        libxcb-shape0 libxcb-randr0 libxcb-sync1 libxcb-util1 \
        libxcb-glx0 libxcb-xkb1 libxkbcommon-x11-0 \
        ffmpeg
else
    echo "[Info] apt-get not available — skipping system packages"
fi


# ======================================================================
# Done
# ======================================================================
echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  Environment variables to set before running:"
echo ""
echo "    # PaliGemma local snapshot (avoid HuggingFace download)"
echo "    export PALIGEMMA_PATH=\"${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/paligemma-3b-pt-224\""
echo ""
echo "    # SAM3 checkpoint directory (contains sam3.pt)"
echo "    export SAM3_CHECKPOINT_PATH=\"${BRIDGEVLA_ROOT}/data/bridgevla_ckpt/sam3\""
echo ""
echo "    # PYTHONPATH for imports"
echo "    export PYTHONPATH=\"${FINETUNE_DIR}:${SAM3_DIR}:\${PYTHONPATH:-}\""
echo ""
echo "    # (Optional) Offline mode for HuggingFace"
echo "    export HF_HUB_OFFLINE=1"
echo "    export TRANSFORMERS_OFFLINE=1"
echo ""
echo "  Quick verify:"
echo "    python -c \"import bridgevla; import sam3; import point_renderer; print('All imports OK')\""
echo ""
echo "  To train:  bash finetune/real/train.sh"
echo "  To serve:  python bridgevla_sam_real_eval/rvt_our/eval_flask_app.py"
echo "============================================================"
