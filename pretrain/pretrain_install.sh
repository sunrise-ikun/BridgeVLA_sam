#!/usr/bin/env bash
set -e

BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"

# --- 配置 GitHub 代理镜像 ---
GITHUB_MIRROR="https://gh.llkk.cc/https://github.com/"
if [ -n "${GITHUB_MIRROR}" ]; then
    git config --global url."${GITHUB_MIRROR}".insteadOf "https://github.com/"
fi

# --- 配置 PyTorch 国内镜像 (上海交大) ---
TORCH_INDEX_URL="https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"

# --- 配置清华 PyPI 镜像源 ---
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 1. 安装 PyTorch
pip install --force-reinstall --no-deps \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url ${TORCH_INDEX_URL}

# 2. 安装 bridgevla 包
cd "${BRIDGEVLA_ROOT}/finetune"
pip install -e .

# 3. 安装其余依赖
pip install datasets peft bitsandbytes tf-keras swanlab
pip install numpy==1.26.4
pip install transformers==4.51.3
pip uninstall byted-wandb -y