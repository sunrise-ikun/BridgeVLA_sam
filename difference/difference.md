# A100 机器适配变动记录

> 从 H20 迁移至 A100 时所做的修改。回到 H20 时按此文档还原。
>
> - **H20 路径前缀：** `/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam`
> - **A100 路径前缀：** `/DATA/disk1/zyz/projects/BridgeVLA_sam`

---

## 1. `finetune/RLBench/train_a100.sh`

**改动：** `BRIDGEVLA_ROOT` 路径 + 新增两个数据路径 env var。

还原时将此文件内容替换为 `train_h20.sh` 的内容（或做如下定向修改）：

```diff
-BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
+BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"

-export RLBENCH_DATA_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/RLBench"
-export RLBENCH_REPLAY_STORAGE_DIR="${BRIDGEVLA_ROOT}/data/bridgevla_data/replay_train"
+export RLBENCH_DATA_FOLDER="/robot/robot-research-exp-0/user/lpy/data/RLBench"
+export RLBENCH_REPLAY_STORAGE_DIR="${BRIDGEVLA_ROOT}/data/bridgevla_data/replay_train"
```

---

## 2. `finetune/RLBench/configs/rlbench_config_a100.yaml`

**改动：** `log_dir` 路径。

```diff
-log_dir: "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/logs"
+log_dir: "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_data/logs"
```

---

## 3. `finetune/RLBench/eval.sh`

**改动：** `BRIDGEVLA_ROOT` 路径。

```diff
-BRIDGEVLA_ROOT="/DATA/disk1/zyz/projects/BridgeVLA_sam"
+BRIDGEVLA_ROOT="/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam"
```

---

## 4. `finetune/RLBench/utils/peract_utils_rlbench.py`

**改动：** 新增 `import os`，`DATA_FOLDER` 和 `TRAIN_REPLAY_STORAGE_DIR` 改为从环境变量读取。

```diff
+import os
 from omegaconf import OmegaConf

-DATA_FOLDER="/robot/robot-research-exp-0/user/lpy/data/RLBench"
-TRAIN_REPLAY_STORAGE_DIR = "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_data/replay_train"
+DATA_FOLDER = os.environ.get("RLBENCH_DATA_FOLDER", "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/RLBench")
+TRAIN_REPLAY_STORAGE_DIR = os.environ.get("RLBENCH_REPLAY_STORAGE_DIR", "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/replay_train")
```

还原到 H20 时，可直接删除 `import os` 行并将两个变量改回原始硬编码值：

```python
DATA_FOLDER="/robot/robot-research-exp-0/user/lpy/data/RLBench"
TRAIN_REPLAY_STORAGE_DIR = "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_data/replay_train"
```

或者**保留** `os.environ.get()` 写法（推荐），只需确保 `train_h20.sh` 正确 export 对应路径（已同步更新）。

---

## 5. `finetune/bridgevla/config.py`

**改动：** 默认 `log_dir`（实际运行时会被 yaml 覆盖，影响较小）。

```diff
-_C.log_dir = "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/logs"
+_C.log_dir = "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_data/logs"
```

---

## 6. `finetune/RLBench/train_h20.sh`（仅新增，未修改原逻辑）

**改动：** 与 `train_a100.sh` 保持对称，新增了数据路径 env var 的显式 export。H20 还原时此两行可保留也可删除，不影响原逻辑。

```diff
+export RLBENCH_DATA_FOLDER="/robot/robot-research-exp-0/user/lpy/data/RLBench"
+export RLBENCH_REPLAY_STORAGE_DIR="${BRIDGEVLA_ROOT}/data/bridgevla_data/replay_train"
```

---

## 未修改的文件（H20 参考保留）

- `finetune/RLBench/train_h20.sh` — `BRIDGEVLA_ROOT` 仍为 H20 路径，无需改动
- `finetune/RLBench/configs/rlbench_config_h20.yaml` — `log_dir` 仍为 H20 路径，无需改动
- `finetune/RLBench/install_rlbench.sh` — 注释中提到旧路径，`BRIDGEVLA_ROOT` 变量本身已是 A100 路径（迁移前已修改），无需改动
