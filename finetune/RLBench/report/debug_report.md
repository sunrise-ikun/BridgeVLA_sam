# BridgeVLA RLBench eval.sh Debug Report

- **Date**: 2026-04-18
- **Script**: `BridgeVLA_sam/finetune/RLBench/eval.sh`
- **Task**: `close_jar`, 3 episodes, episode length 25
- **Checkpoint**: `model_80.pth`
- **Final Result**: 3/3 episodes score 100.0

---

## Issue Summary

Running `eval.sh` directly会遇到 4 个串联的环境问题，依次解决后评估才能正常执行。

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `ModuleNotFoundError: No module named 'RLBench'` | `eval.py` import 的 `RLBench.utils.peract_utils_rlbench` 依赖 finetune 目录在 `PYTHONPATH` 中 | `export PYTHONPATH=".../finetune:$PYTHONPATH"` |
| 2 | `Could not find the Qt platform plugin "xcb"` | OpenCV (cv2) 捆绑了自己的 Qt 插件目录，覆盖了 CoppeliaSim 设置的 `QT_QPA_PLATFORM_PLUGIN_PATH` | 将 cv2 的 Qt 插件目录软链接到 CoppeliaSim 的插件 |
| 3 | `QXcbIntegration: Cannot create platform OpenGL context, neither GLX nor EGL are enabled` | Qt xcb 插件加载后找不到 GLX integration 插件（`libqxcb-glx-integration.so`） | 安装 `libqt5gui5`；设置 `QT_PLUGIN_PATH` 包含系统 Qt5 插件目录 |
| 4 | `signal 11 (SIGSEGV)` in LLVM symbols | TensorFlow 捆绑的 LLVM 与系统 libLLVM-15 符号冲突 | 卸载未使用的 TensorFlow |

---

## Issue 1: Python Module Import Error

### Symptom

```
ModuleNotFoundError: No module named 'RLBench'
```

### Analysis

`eval.py` 通过 `bridgevla_agent.py` (line 29) 间接 import：

```python
import RLBench.utils.peract_utils_rlbench as rlbench_utils
```

这里的 `RLBench` 指的是 `finetune/RLBench/` 目录（项目内模块），而非 pip 安装的 `rlbench` 包。`eval.sh` 通过 `cd` 切换到 `finetune/RLBench/` 但没有将 `finetune/` 加入 `PYTHONPATH`。

### Fix

在运行前设置：

```bash
export PYTHONPATH="/path/to/BridgeVLA_sam/finetune:${PYTHONPATH:-}"
```

---

## Issue 2: Qt Platform Plugin Not Found

### Symptom

```
qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in
  "/root/miniconda3/envs/bridgevla_sam/lib/python3.9/site-packages/cv2/qt/plugins"
```

### Analysis

`eval.sh` 设置了 `QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT`，期望 Qt 从 CoppeliaSim 目录加载 xcb 插件。但环境中安装的 `opencv-python`（非 headless 版本）捆绑了自己的 Qt 运行时，其内部硬编码了 `cv2/qt/plugins` 作为插件搜索路径，**覆盖了环境变量**。

cv2 的 Qt 插件目录只有一个 28KB 的 `libqxcb.so`，缺少 CoppeliaSim 需要的完整 Qt5 插件集。

### Fix

将 cv2 原始 Qt 目录备份，然后创建软链接指向 CoppeliaSim 的插件：

```bash
# 备份原始 cv2 Qt
mv .../site-packages/cv2/qt .../site-packages/cv2/qt_bak

# 创建新目录并软链接到 CoppeliaSim 插件
mkdir -p .../site-packages/cv2/qt/plugins
ln -sf $COPPELIASIM_ROOT/platforms .../site-packages/cv2/qt/plugins/platforms
ln -sf $COPPELIASIM_ROOT/xcbglintegrations .../site-packages/cv2/qt/plugins/xcbglintegrations
```

> **Note**: 更彻底的方案是按 `install_rlbench.sh` 的做法，用 `opencv-python-headless` 替换 `opencv-python`，彻底移除 cv2 的 Qt 捆绑。

---

## Issue 3: OpenGL Context Creation Failure

### Symptom

```
QXcbIntegration: Cannot create platform offscreen surface, neither GLX nor EGL are enabled
QXcbIntegration: Cannot create platform OpenGL context, neither GLX nor EGL are enabled
Error: signal 11 (SIGSEGV) in COffscreenGlContext constructor
```

### Analysis

Qt xcb 平台插件通过**子插件**（xcbglintegrations）提供 GLX/EGL 支持：

```
xcbglintegrations/
  libqxcb-glx-integration.so   ← GLX 渲染
  libqxcb-egl-integration.so   ← EGL 渲染
```

CoppeliaSim 自带了这些文件（在 `$COPPELIASIM_ROOT/xcbglintegrations/`），但 Qt 运行时只在标准插件路径中搜索，而不是 `QT_QPA_PLATFORM_PLUGIN_PATH`。系统上没有安装 `libqt5gui5`，导致系统标准路径下也没有这些插件。

此外，环境中没有运行的 X server（`$DISPLAY` 为空），xcb 插件需要一个 X display 才能工作。

### Fix

三步：

```bash
# 1. 安装系统 Qt5 GUI 包（提供 xcbglintegrations）
apt-get install -y libqt5gui5

# 2. 设置 QT_PLUGIN_PATH 包含 CoppeliaSim 和系统 Qt5 插件
export QT_PLUGIN_PATH="$COPPELIASIM_ROOT:/usr/lib/x86_64-linux-gnu/qt5/plugins"

# 3. 启动 Xvfb 虚拟显示
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99
```

---

## Issue 4: SIGSEGV due to TensorFlow/LLVM Conflict

### Symptom

```
Error: signal 11:
.../libtensorflow_framework.so.2(_ZN4llvm19raw_svector_ostream10write_implEPKcm+0x11)
/lib/x86_64-linux-gnu/libLLVM-15.so.1(_ZN4llvm11raw_ostream5writeEPKcm+0x183)
```

### Analysis

TensorFlow 2.20 捆绑了自己的 LLVM 实现（`libtensorflow_framework.so.2`），与系统安装的 `libLLVM-15.so.1` 存在 **ABI 符号冲突**。当 CoppeliaSim 创建 OpenGL 上下文时，动态链接器混合调用了两个 LLVM 版本的函数，导致 SIGSEGV。

TensorFlow 实际上**不被 eval 代码使用**——它是 `setup.py` 的 optional dependency，被 pip 安装后间接加载到进程中。

### Fix

```bash
pip uninstall -y tensorflow
```

确认 eval 代码中没有 `import tensorflow` 的调用链：

```bash
grep -r "tensorflow" finetune/RLBench/eval.py finetune/bridgevla/  # 无结果
```

---

## Final Working Command

所有修复已整合进 `eval.sh`，直接运行即可：

```bash
bash eval.sh
```

`eval.sh` 现在会自动完成：
- 设置 `PYTHONPATH`（Fix 1）
- 设置 `QT_PLUGIN_PATH` 包含系统 Qt5 xcbglintegrations（Fix 3）
- 自动检测/启动 Xvfb 虚拟 X display（Fix 3），已有则复用
- 不再依赖 `xvfb-run`，改为直接管理 Xvfb 进程

## Prerequisites (One-time Setup)

```bash
# Fix 2: 软链接 cv2 Qt 插件到 CoppeliaSim（如果仍在使用 opencv-python 非 headless 版）
SITE_PACKAGES="/root/miniconda3/envs/bridgevla_sam/lib/python3.9/site-packages"
mv ${SITE_PACKAGES}/cv2/qt ${SITE_PACKAGES}/cv2/qt_bak
mkdir -p ${SITE_PACKAGES}/cv2/qt/plugins
ln -sf ${COPPELIASIM_ROOT}/platforms ${SITE_PACKAGES}/cv2/qt/plugins/platforms
ln -sf ${COPPELIASIM_ROOT}/xcbglintegrations ${SITE_PACKAGES}/cv2/qt/plugins/xcbglintegrations

# Fix 3: 安装系统 Qt5
apt-get install -y libqt5gui5

# Fix 4: 卸载 TensorFlow
pip uninstall -y tensorflow
```

---

## Evaluation Output

```
Evaluating close_jar | Episode 0 | Score: 100.0 | Episode Length: 5 | Lang Goal: close the black jar
Evaluating close_jar | Episode 1 | Score: 100.0 | Episode Length: 5 | Lang Goal: close the olive jar
Evaluating close_jar | Episode 2 | Score: 100.0 | Episode Length: 5 | Lang Goal: close the cyan jar
[Evaluation] Finished close_jar | Final Score: 100.0
```

## Notes

- `--device` 参数只接受单个 GPU 编号（int），此脚本不支持多 GPU 并行评估
- 此次运行使用 GPU 0（`--device 0`）
- 如需更换 OpenCV 可运行 `pip install opencv-python-headless` 从根本上解决 Issue 2
