# Issue: SAM3 Encoder 输入范围不匹配 [0,1] vs [-1,1]

## 严重程度：高

## 问题描述

`SAM3EncoderWrapper` 接收到的图像范围是 **[0, 1]**，但 SAM3 的 ViT backbone 训练时使用的输入范围是 **[-1, 1]**（通过 `Normalize(mean=0.5, std=0.5)` 归一化）。这导致 SAM3 的特征提取工作在错误的输入分布上。

## 影响范围

- **RLBench 训练** (`bridgevla_agent.py` → `update()`)
- **GemBench 训练** (`bridgevla_agent.py` → `update_gembench()`)
- 两者共享同一个 `mvt_single.py`，所以都受影响。

## 根因分析

数据流：

1. `_preprocess_inputs` / `_preprocess_inputs_gembench` 中 `_norm_rgb()` 将 RGB 从 [0,255] 归一化到 **[-1, 1]**：
   ```python
   # peract_utils_rlbench.py / peract_utils_gembench.py
   def _norm_rgb(x):
       return (x.float() / 255.0) * 2.0 - 1.0   # → [-1, 1]
   ```

2. `rvt_utils.get_pc_img_feat()` 又把 img_feat 从 [-1,1] 转回 **[0, 1]**：
   ```python
   # rvt_utils.py, line 25
   img_feat = (img_feat + 1) / 2   # → [0, 1]
   ```

3. 渲染器保持 [0, 1] 不变，输出到 `mvt_single.py`。

4. `mvt_single.py` 中取 RGB 通道并直接传入 SAM3：
   ```python
   # mvt_single.py, line ~408
   img = img[:,:, 3:6, :, :]  # RGB channels, 值域 [0, 1]
   sam3_images = img.reshape(bs * self.num_img, 3, h, w)
   # 注释写的 "already in [-1,1]" 是错误的
   ```

5. `SAM3EncoderWrapper.forward()` 直接把图像传给 backbone，不做额外归一化。

## SAM3 期望 [-1,1] 的证据

### 证据 1：`sam3_image_processor.py` 官方预处理
```python
# libs/sam3/sam3/model/sam3_image_processor.py, line 21-27
self.transform = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(resolution, resolution)),
    v2.ToDtype(torch.float32, scale=True),          # → [0, 1]
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
])
```

### 证据 2：`io_utils.py` 图像加载
```python
# libs/sam3/sam3/model/io_utils.py, line 56-62
img_np = img_np / 255.0                  # → [0, 1]
img -= img_mean   # default (0.5,0.5,0.5) → [-0.5, 0.5]
img /= img_std    # default (0.5,0.5,0.5) → [-1, 1]
```

## 建议修复

在 `mvt_single.py` 中，将 SAM3 输入从 [0,1] 转到 [-1,1]：

```python
# mvt_single.py, SAM3 path
sam3_images = img.reshape(bs * self.num_img, 3, h, w) * 2.0 - 1.0  # [0,1] → [-1,1]
```

或者修正注释并确保上游提供正确范围。

## 相关文件

- `finetune/bridgevla/mvt/mvt_single.py` — SAM3 输入处（line ~408-412）
- `finetune/bridgevla/utils/rvt_utils.py` — `get_pc_img_feat()` line 25
- `finetune/bridgevla/mvt/sam3_utils.py` — `SAM3EncoderWrapper.forward()` 文档
- `libs/sam3/sam3/model/sam3_image_processor.py` — 官方预处理 transform
- `libs/sam3/sam3/model/io_utils.py` — 官方图像加载归一化
