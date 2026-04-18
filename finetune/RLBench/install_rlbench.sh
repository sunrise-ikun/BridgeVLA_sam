#!/usr/bin/env bash
set -e

# =======================================================================
# Adapted for current workspace:
#   BRIDGEVLA_ROOT = /robot/robot-research-exp-0/user/lpy/BridgeVLA_sam
#   CoppeliaSim is already extracted under finetune/
#   PyRep / RLBench need to be cloned into finetune/bridgevla/libs
# =======================================================================

BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
LIBS_DIR="${FINETUNE_DIR}/bridgevla/libs"

# --- 配置清华 PyPI 镜像源 ---
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

pip install wheel ninja pyyaml

# --- Clone RLBench / PyRep into bridgevla/libs (skip if already present) ---
mkdir -p "${LIBS_DIR}"
cd "${LIBS_DIR}"
if [ ! -d "RLBench" ]; then
    git clone https://github.com/buttomnutstoast/RLBench.git
    cd RLBench
    git checkout 587a6a0e6dc8cd36612a208724eb275fe8cb4470
    cd ..
fi
if [ ! -d "PyRep" ]; then
    git clone https://github.com/stepjam/PyRep.git
    cd PyRep
    git checkout 231a1ac6b0a179cff53c1d403d379260b9f05f2f
    cd ..
fi

# --- CoppeliaSim (already extracted under finetune/) ---
cd "${FINETUNE_DIR}"
COPP_TAR="CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
if [ ! -d "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" ]; then
    if [ ! -f "${COPP_TAR}" ]; then
        echo "[Info] ${COPP_TAR} 未找到，开始下载..."
        wget https://www.coppeliarobotics.com/files/V4_1_0/${COPP_TAR}
    else
        echo "[Info] 检测到已下载的 ${COPP_TAR}，跳过下载"
    fi
    echo "[Info] 解压 ${COPP_TAR} ..."
    tar -xf "${COPP_TAR}"
else
    echo "[Info] CoppeliaSim 已解压，跳过下载与解压"
fi
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
pip3 install pip==25.0.1
pip3 install setuptools==76.1.0
# --ignore-installed blinker: 绕过系统 apt 装的 distutils 版 blinker 1.4 无法被 pip 卸载的问题
pip3 install --ignore-installed blinker open3d

cd "${FINETUNE_DIR}"
pip install -e .
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121 #
# 必须从 cu121 索引装 torchvision/torchaudio，默认 PyPI 是 cu124 构建，会与 torch 2.5.1+cu121 冲突
pip install --force-reinstall --no-deps \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install 'accelerate>=0.26.0'
pip3 install transformers==4.51.3
pip install git+https://github.com/openai/CLIP.git
sudo apt-get update
sudo apt-get install -y libffi-dev
sudo apt-get install -y xvfb
sudo apt-get install -y libfontconfig1
sudo apt install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
sudo apt install libxcb-cursor0
sudo apt install libxcb-xinerama0
pip install pyqt6
pip3 install yacs
pip3 install wandb
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

sudo apt-get update
sudo apt-get install -y  libxcb-xinput0  libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 libxcb-shape0 libxcb-randr0 libxcb-image0 libxcb-keysyms1 libxcb-icccm4 libxcb-sync1 libxcb-xinerama0 libxcb-util1
sudo apt-get install -y libxcb-glx0 libxcb-xkb1 libxkbcommon-x11-0
sudo apt install -y ffmpeg
pip install ffmpeg-python


pip uninstall -y opencv-python opencv-contrib-python
pip install  opencv-python-headless        
pip uninstall  -y opencv-python-headless      
pip install  opencv-python-headless   

pip install -e bridgevla/libs/PyRep 
pip install -e bridgevla/libs/RLBench 
pip install -e bridgevla/libs/YARR 
pip install -e bridgevla/libs/peract_colab
pip install -e bridgevla/libs/point-renderer    

