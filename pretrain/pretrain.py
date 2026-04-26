'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn

Pretrain on the RoboPoint detection subset using the new BridgeVLA
architecture (PaliGemma + SAM3 + pali_proj / fusion_transformer / up0).
Manual DDP loop, two-stage freeze/unfreeze (matching finetune/RLBench/train.py).
'''
import os
import sys
import ast
import json
import time
import yaml
import argparse
import datetime
import subprocess
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from bridgevla.mvt.raft_utils import ConvexUpSample
from bridgevla.mvt.sam3_utils import SAM3EncoderWrapper
import bridgevla.mvt.utils as mvt_utils


USE_SWANLAB = False


# ---------------------------------------------------------------------------
# Utilities (kept from original file)
# ---------------------------------------------------------------------------
def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Independent softmax over non-zero regions per sample."""
    mask = (heatmap != 0).float()
    stable_input = heatmap * mask
    exp_vals = torch.exp(stable_input) * mask
    sum_exp = exp_vals.sum(dim=(1, 2), keepdim=True)
    return exp_vals / (sum_exp + eps)


def is_list_string(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']') and len(s) >= 2):
        return False
    try:
        parsed = ast.literal_eval(s)
        return isinstance(parsed, list)
    except (SyntaxError, ValueError):
        return False


def convert_xyxy_to_cxcywh(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def masked_mean(tensor):
    mask = (tensor != 0).float()
    count = mask.sum(dim=0, keepdim=True)
    count = torch.where(count == 0, torch.ones_like(count), count)
    return tensor.sum(dim=0, keepdim=True) / count


def visualize_bboxes_and_heatmap(image, bboxes_norm, heatmap_tensor, save_path,
                                 bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                                 bbox_width=2):
    resized_img = image.resize((224, 224))
    draw = ImageDraw.Draw(resized_img)
    color_cycle = cycle(bbox_colors)
    for bbox in bboxes_norm:
        cx, cy, w, h = bbox
        x0 = max(0, int((cx - w / 2) * 224))
        y0 = max(0, int((cy - h / 2) * 224))
        x1 = min(223, int((cx + w / 2) * 224))
        y1 = min(223, int((cy + h / 2) * 224))
        draw.rectangle([x0, y0, x1, y1], outline=next(color_cycle), width=bbox_width)

    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(resized_img); ax1.set_title(f'Image with {len(bboxes_norm)} BBoxes'); ax1.axis('off')
    ax2.imshow(heatmap, cmap='viridis', alpha=0.95)
    ax2.imshow(resized_img, alpha=0.05)
    ax2.set_title('Heatmap Overlay'); ax2.axis('off')
    plt.savefig(save_path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RoboPointDataset(Dataset):
    """Detection subset of the RoboPoint dataset."""

    def __init__(self, image_folder, json_detection_path, res):
        self.samples = []
        self.image_folder = image_folder
        self.res = res
        if json_detection_path is not None:
            self.list_data_detection = json.load(open(json_detection_path, "r"))
            self.samples.extend(self._parse_detection(self.list_data_detection))

    def __len__(self):
        return len(self.samples)

    def _parse_detection(self, list_data_dict):
        samples = []
        for source_data in tqdm(list_data_dict, desc="Parsing RoboPoint detection"):
            length_conversations = len(source_data["conversations"])
            assert length_conversations % 2 == 0
            for i in range(1, length_conversations, 2):
                if not is_list_string(source_data["conversations"][i]["value"]):
                    continue
                matched1 = "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: "
                matched1_ = "Please provide the bounding box coordinate of the region this sentence describes: "
                matched2 = " Format the result as a list of tuples, i.e. [(x1, y1, w1, h1), (x2, y2, w2, h2), ...], where x and y are the normalized pixel locations of the object centers, and w and h are the normalized object widths and heights. All values of x, y, w, and h should be between 0 and 1."
                text = source_data["conversations"][i - 1]["value"]
                if matched1 in text or matched1_ in text:
                    text = text.replace(matched1, "").replace(matched1_, "")
                    flag = "detection_1"
                elif matched2 in text:
                    text = text.replace(matched2, "")
                    flag = "detection_2"
                else:
                    assert False
                samples.append({
                    "text": text,
                    "image_path": os.path.join(self.image_folder, source_data["image"]),
                    "raw_label": source_data["conversations"][i]["value"],
                    "flag": flag,
                })
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "text": sample["text"].replace("<image>\n", ""),
            "image": Image.open(sample["image_path"]).convert("RGB"),
            "raw_label": sample["raw_label"],
            "flag": sample["flag"],
        }


@dataclass
class DataCollator(object):
    processor: AutoProcessor

    def __call__(self, data):
        texts = [ex["text"] for ex in data]
        images = [ex["image"] for ex in data]
        raw_label = [ex["raw_label"] for ex in data]
        flag = [ex["flag"] for ex in data]
        # Prepend one <image> token per image to each prompt to match
        # PaliGemmaProcessor's expected format (same convention as finetune
        # mvt_single.py) and silence its inference warning.
        prompts = ["<image>" + t for t in texts]
        tokens = self.processor(text=prompts, images=images,
                                return_tensors="pt", padding="longest")
        # Also retain raw texts (for SAM3 text encoder).
        tokens["raw_text"] = texts
        tokens["raw_label"] = raw_label
        tokens["flag"] = flag
        return tokens


def load_dataset(processor, image_folder, json_detection_path, res):
    return RoboPointDataset(image_folder, json_detection_path, res), DataCollator(processor=processor)


# ---------------------------------------------------------------------------
# Model — pretrain counterpart of the finetune mvt_single.py architecture
# ---------------------------------------------------------------------------
class BridgeVLAPretrainModel(nn.Module):
    """
    Pretrain-only model that mirrors finetune/bridgevla/mvt/mvt_single.py:
      PaliGemma -> pali_ln -> pali_proj  ─┐
                                          ├─ concat ─► fusion_transformer ─► up0
                          SAM3 -> sam3_ln ┘
    Only uses a single 2D image per sample (num_img = 1), so there is no
    renderer / point-cloud handling. Saved state-dict keys match the submodule
    paths under mvt_single.MVT (i.e. `model.*`, `pali_ln.*`, `pali_proj.*`,
    `sam3_ln.*`, `fusion_transformer.*`, `up0.*`), so finetune can load directly.
    """

    def __init__(self, model_id, sam3_ckpt_path,
                 img_size=224, img_patch_size=14):
        super().__init__()
        self.img_size = img_size
        self.img_patch_size = img_patch_size
        self.num_pat_img = img_size // img_patch_size  # 16
        self.num_img = 1

        # PaliGemma backbone (bf16 — same as mvt_single)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.vlm_dim = self.model.config.hidden_size  # 2048

        # Keep a reference to the processor but DO NOT register as submodule.
        self._processor = AutoProcessor.from_pretrained(model_id)

        # SAM3 encoder (frozen, weights loaded from checkpoint here)
        self.sam3_encoder = SAM3EncoderWrapper(
            checkpoint_path=sam3_ckpt_path, input_size=img_size
        )

        self.sam3_dim = 256
        self.pali_proj_dim = 768
        self.fused_dim = self.sam3_dim + self.pali_proj_dim  # 1024

        self.pali_ln = nn.LayerNorm(self.vlm_dim)
        self.pali_proj = nn.Conv2d(self.vlm_dim, self.pali_proj_dim, kernel_size=1)
        self.sam3_ln = nn.LayerNorm(self.sam3_dim)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.fused_dim, nhead=8,
            dim_feedforward=self.fused_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=4)

        self.up0 = ConvexUpSample(
            in_dim=self.fused_dim, out_dim=1, up_ratio=self.img_patch_size,
        )
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    # ------------------------------------------------------------------
    # Heat-map label construction (same logic as the original file)
    # ------------------------------------------------------------------
    def _build_gt_heatmap(self, raw_label, flag, h, w, device):
        action_trans = []
        for lab, f in zip(raw_label, flag):
            if f == "detection_1":
                ans = ast.literal_eval(lab)
                assert type(ans[0]) is float and len(ans) == 4
                bbox = convert_xyxy_to_cxcywh(ans)
                label = torch.tensor([[bbox[0], bbox[1]]])
                hm = mvt_utils.generate_hm_from_pt(
                    label.reshape(-1, 2) * h, (w, h), sigma=2, thres_sigma_times=3,
                )
            elif f == "detection_2":
                ans = ast.literal_eval(lab)
                assert type(ans[0]) is tuple and len(ans[0]) == 4
                labels = torch.tensor([[a[0], a[1]] for a in ans])
                hm_all = mvt_utils.generate_hm_from_pt(
                    labels.reshape(-1, 2) * h, (w, h), sigma=2, thres_sigma_times=3,
                )
                hm = masked_softmax(masked_mean(hm_all))
            else:
                assert False
            action_trans.append(hm)
        action_trans = torch.stack(action_trans)  # (bs, 1, h, w)
        return action_trans.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_ids, pixel_values, attention_mask,
                raw_label, flag, raw_text=None):
        """
        input_ids:      (bs, seq_len)
        pixel_values:   (bs, 3, 224, 224)  already normalised to [-1, 1]
        attention_mask: (bs, seq_len)
        raw_label/flag/raw_text: per-sample python lists (len == bs)
        """
        bs = input_ids.shape[0]
        h = w = self.img_size
        assert pixel_values.shape[-1] == h and pixel_values.shape[-2] == w

        # Cast pixel_values to PaliGemma dtype (bf16) — kept separately so we
        # can feed SAM3 with fp32 below.
        pali_pixel_values = pixel_values.to(self.model.dtype)

        # --- PaliGemma ---
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pali_pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        x = outputs.hidden_states[-1]  # (bs, seq, vlm_dim)

        # Extract the 256 image tokens (same strategy as mvt_single).
        image_tokens = []
        for i in range(bs):
            nz = torch.nonzero(attention_mask[i] != 0, as_tuple=True)[0]
            non_zero_output = x[i][nz]
            assert non_zero_output.shape[0] > 256 * self.num_img
            image_tokens.append(non_zero_output[: 256 * self.num_img])
        image_tokens = torch.stack(image_tokens)  # (bs, 256, vlm_dim)
        x_pali = rearrange(
            image_tokens,
            'b (c h1 h2) d -> (b c) d h1 h2',
            c=self.num_img, h1=self.num_pat_img, h2=self.num_pat_img,
        )  # (bs, vlm_dim, 16, 16)

        x_pali = x_pali.to(torch.float32)
        x_pali = x_pali.permute(0, 2, 3, 1)
        x_pali = self.pali_ln(x_pali)
        x_pali = x_pali.permute(0, 3, 1, 2)
        x_pali = self.pali_proj(x_pali)  # (bs, pali_proj_dim, 16, 16)

        # --- SAM3 (frozen) ---
        # SAM3 expects images in [-1, 1] which is the PaliGemma-processor range.
        sam3_prompts = list(raw_text) if raw_text is not None else ["" for _ in range(bs)]
        x_sam3 = self.sam3_encoder(pixel_values.to(torch.float32), sam3_prompts)
        x_sam3 = x_sam3.to(torch.float32)
        x_sam3 = x_sam3.permute(0, 2, 3, 1)
        x_sam3 = self.sam3_ln(x_sam3)
        x_sam3 = x_sam3.permute(0, 3, 1, 2)  # (bs, sam3_dim, 16, 16)

        # --- Fusion ---
        x_fused = torch.cat([x_pali, x_sam3], dim=1)  # (bs, fused_dim, 16, 16)
        x_fused_seq = x_fused.flatten(2).permute(0, 2, 1)  # (bs, 256, fused_dim)
        x_fused_seq = self.fusion_transformer(x_fused_seq)
        x_fused = x_fused_seq.permute(0, 2, 1).view(
            bs, self.fused_dim, self.num_pat_img, self.num_pat_img
        )

        # --- Upsample and loss ---
        trans = self.up0(x_fused)  # (bs, 1, 224, 224)
        q_trans = trans.view(bs, self.num_img, h * w).transpose(1, 2)  # (bs, h*w, 1)

        action_trans = self._build_gt_heatmap(raw_label, flag, h, w, q_trans.device)
        action_trans = action_trans.view(bs, self.num_img, h * w).transpose(1, 2).clone()

        loss = self._cross_entropy_loss(q_trans, action_trans).mean()
        return {"loss": loss, "q_trans": q_trans}


# ---------------------------------------------------------------------------
# Distributed setup (copied from finetune/RLBench/train.py)
# ---------------------------------------------------------------------------
def setup_distributed(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("Cannot find RANK and WORLD_SIZE — entering single-GPU debug mode")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "9001")
        os.environ.setdefault("LOCAL_RANK", "0")
    else:
        raise RuntimeError(
            "Distributed env vars not found. "
            "Launch with torchrun / srun, or set DEBUG=true for single-GPU mode."
        )

    dist.init_process_group(
        backend=backend,
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def freeze_params(model, freeze_paligemma):
    """Apply freeze policy.

    - Always frozen: SAM3 encoder, lm_head, embed_tokens
    - Stage 1 (freeze_paligemma=True): additionally freeze the whole PaliGemma
      backbone (module path `model.*`).
    - Stage 2 (freeze_paligemma=False): unfreeze PaliGemma (still keep
      lm_head / embed_tokens frozen).
    """
    always_freeze = ["lm_head", "embed_tokens"]
    for name, param in model.named_parameters():
        # SAM3 encoder — always frozen
        if name.startswith("sam3_encoder."):
            param.requires_grad = False
            continue
        if name.startswith("model."):
            if any(af in name for af in always_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = not freeze_paligemma
        else:
            # pali_ln, pali_proj, sam3_ln, fusion_transformer, up0 — always trainable.
            param.requires_grad = True


def build_optimizer(model, lr, weight_decay=0.0, betas=(0.9, 0.95)):
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params_to_optimize, lr=lr, weight_decay=weight_decay, betas=betas,
    )


def save_checkpoint(model, path, epoch, extra=None):
    """Save full state-dict excluding SAM3 encoder weights (always frozen and
    loaded independently from the SAM3 checkpoint)."""
    model_to_save = model.module if isinstance(model, DDP) else model
    sd = model_to_save.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("sam3_encoder.")}
    payload = {"epoch": epoch, "model_state": sd}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_epoch(ddp_model, loader, sampler, state, *,
              epoch, base_lr, warmup_steps, weight_decay,
              freeze_threshold_step, iters_per_epoch,
              device, rank, logging_steps):
    """Run one epoch. Checks every iteration for the stage-1→stage-2 transition
    (triggered when the cumulative step count reaches `freeze_threshold_step`).

    `state` is a mutable dict with keys:
        stage            (int, 1 or 2)
        warmup_step      (int, resets to 0 at the stage-2 transition)
        cumulative_step  (int, never resets across the run)
        optimizer        (torch.optim.Optimizer, replaced on transition)
    """
    ddp_model.train()
    # SAM3 must remain in eval mode regardless of the parent .train() call.
    inner = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    inner.sam3_encoder.eval()

    sampler.set_epoch(epoch)

    log_prefix = f"[S{state['stage']}] "
    iterator = tqdm(loader, disable=(rank != 0), desc=f"{log_prefix}epoch {epoch}")
    for it, batch in enumerate(iterator):
        # ---- Mid-epoch stage-2 transition ----
        if state["stage"] == 1 and state["cumulative_step"] >= freeze_threshold_step:
            freeze_params(inner, freeze_paligemma=False)
            state["optimizer"] = build_optimizer(
                ddp_model, lr=base_lr, weight_decay=weight_decay,
            )
            state["warmup_step"] = 0
            state["stage"] = 2
            log_prefix = f"[S{state['stage']}] "
            if rank == 0:
                n_train = sum(
                    p.numel() for p in ddp_model.parameters() if p.requires_grad
                ) / 1e9
                print(
                    f"[Stage 2] Unfroze PaliGemma at epoch={epoch} iter={it} "
                    f"(cumulative_step={state['cumulative_step']}). "
                    f"Trainable params: {n_train:.3f}B",
                    flush=True,
                )

        optimizer = state["optimizer"]

        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = ddp_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            raw_label=batch["raw_label"],
            flag=batch["flag"],
            raw_text=batch["raw_text"],
        )
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        state["warmup_step"] += 1
        lr_scale = (
            1.0 if warmup_steps <= 0
            else min(1.0, state["warmup_step"] / warmup_steps)
        )
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * lr_scale
        optimizer.step()

        state["cumulative_step"] += 1
        dist.barrier()

        if rank == 0 and (it % logging_steps == 0):
            cur_lr = optimizer.param_groups[0]["lr"]
            iterator.set_postfix_str(
                f"{log_prefix}epoch={epoch} iter={it} "
                f"loss={loss.item():.4f} lr={cur_lr:.3e}"
            )
            if USE_SWANLAB:
                import swanlab
                swanlab.log(
                    {"loss": loss.item(), "lr": cur_lr, "stage": state["stage"]},
                    step=state["cumulative_step"],
                )


# ---------------------------------------------------------------------------
# Checkpoint loader (used by finetune; also handy for validation here)
# ---------------------------------------------------------------------------
def split_pretrain_state(state_dict):
    """Partition a pretrain state_dict by submodule prefix. Returns a dict of
    {submodule_name: inner_state_dict} plus 'model' for PaliGemma."""
    buckets = {
        "model": {}, "pali_ln": {}, "pali_proj": {}, "sam3_ln": {},
        "fusion_transformer": {}, "up0": {},
    }
    for k, v in state_dict.items():
        for prefix in buckets:
            p = prefix + "."
            if k.startswith(p):
                buckets[prefix][k[len(p):]] = v
                break
    return buckets


def load_pretrain_checkpoint(path):
    """Load a pretrain checkpoint saved by this script (torch.save dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    return ckpt


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------
def experiment(cmd_args):
    with open(cmd_args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Paths / model IDs
    model_id = os.environ.get("PALIGEMMA_PATH", "google/paligemma-3b-pt-224")
    sam3_ckpt = os.environ.get(
        "SAM3_CHECKPOINT_PATH",
        "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_ckpt/sam3",
    )

    image_folder = cmd_args.image_folder or cfg.get("image_folder")
    json_detection_path = cmd_args.json_detection_path or cfg.get("json_detection_path")
    assert image_folder is not None and json_detection_path is not None, (
        "Need --image_folder / --json_detection_path (or set in config)."
    )

    # Config knobs (with sensible defaults)
    bs = int(cfg["bs"])
    lr = float(cfg["lr"])
    num_epochs = int(cfg["num_train_epochs"])
    # freeze_epochs can be fractional: 0.5 ⇒ transition half-way through epoch 0.
    freeze_epochs = float(cfg.get("freeze_epochs", 2))
    warmup_steps = int(cfg.get("warmup_steps", 400))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_total_limit = int(cfg.get("save_total_limit", 30))
    num_workers = int(cfg.get("dataloader_num_workers", 8))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    # Output dir (rank-0 builds, broadcast so all ranks agree)
    exp_name = cfg.get("exp_name", "pretrain")
    if rank == 0:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(cfg["output_dir"], exp_name, stamp)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.yaml"), "w") as fp:
            yaml.safe_dump(cfg, fp)
        out_list = [output_path]
    else:
        out_list = [None]
    dist.broadcast_object_list(out_list, src=0)
    output_path = out_list[0]

    if rank == 0:
        print(f"[Pretrain] world_size={world_size}, output_dir={output_path}")

    # ---- Model ----
    model = BridgeVLAPretrainModel(
        model_id=model_id, sam3_ckpt_path=sam3_ckpt,
        img_size=cfg.get("img_size", 224),
        img_patch_size=cfg.get("img_patch_size", 14),
    )
    processor = model._processor
    model.model.gradient_checkpointing_enable()
    model = model.to(device_id)

    # Stage 1 freezing BEFORE wrapping in DDP so DDP sees the correct requires_grad flags.
    freeze_params(model, freeze_paligemma=True)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
        print(f"[Stage 1] PaliGemma frozen. Trainable params: {n_train:.3f}B")

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ---- Dataset ----
    train_ds, collate_fn = load_dataset(processor, image_folder, json_detection_path,
                                        res=cfg.get("img_size", 224))
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=(num_workers > 0),
    )

    if rank == 0:
        print(f"[Pretrain] dataset size: {len(train_ds)}, iters/epoch: {len(loader)}")

    # ---- Optimizer ----
    optimizer = build_optimizer(ddp_model, lr=lr, weight_decay=weight_decay)

    # ---- SwanLab ----
    global USE_SWANLAB
    if rank == 0 and not cmd_args.debug:
        try:
            import swanlab
            api_key = os.environ.get("SWANLAB_API_KEY", "")
            if api_key:
                swanlab.login(api_key=api_key)
                swanlab.init(
                    project=cfg.get("swanlab_project", "bridgevla_pretrain"),
                    experiment_name=exp_name + "_" + os.path.basename(output_path),
                    config=cfg,
                )
                USE_SWANLAB = True
                print("[Info] SwanLab enabled.")
        except Exception as e:
            print(f"[Info] SwanLab init failed ({e}); continuing without SwanLab.")

    # ---- Training loop with two-stage freeze (fractional freeze_epochs) ----
    iters_per_epoch = len(loader)
    freeze_threshold_step = int(round(freeze_epochs * iters_per_epoch))
    if rank == 0:
        print(
            f"[Pretrain] freeze_epochs={freeze_epochs} ⇒ "
            f"transition at cumulative step {freeze_threshold_step} "
            f"(iters_per_epoch={iters_per_epoch})"
        )

    state = {
        "stage": 1,
        "warmup_step": 0,
        "cumulative_step": 0,
        "optimizer": optimizer,
    }
    saved_ckpts = []
    for epoch in range(num_epochs):
        if rank == 0:
            print(
                f"=== Stage {state['stage']} | epoch {epoch}/{num_epochs - 1} ===",
                flush=True,
            )
        run_epoch(
            ddp_model, loader, sampler, state,
            epoch=epoch, base_lr=lr, warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            freeze_threshold_step=freeze_threshold_step,
            iters_per_epoch=iters_per_epoch,
            device=device_id, rank=rank, logging_steps=logging_steps,
        )

        # Save at the end of every epoch.
        if rank == 0:
            ck_path = os.path.join(output_path, f"pretrain_epoch_{epoch}.pth")
            save_checkpoint(ddp_model, ck_path, epoch, extra={"stage": state["stage"]})
            saved_ckpts.append(ck_path)
            while save_total_limit > 0 and len(saved_ckpts) > save_total_limit:
                old = saved_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            last_path = os.path.join(output_path, "pretrain_last.pth")
            save_checkpoint(ddp_model, last_path, epoch, extra={"stage": state["stage"]})
            print(f"[Save] {ck_path}")

        dist.barrier()

    if rank == 0:
        print("[Pretrain] Finished.")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Stand-alone helpers kept from the original file
# ---------------------------------------------------------------------------
def visualise_dataset(image_folder, json_detection_path, save_path="./debug.png"):
    ds = RoboPointDataset(image_folder, json_detection_path, res=224)
    print("Total samples:", len(ds))
    for i in range(0, len(ds), 1000):
        data = ds[i]
        text, image, raw_label, flag = data["text"], data["image"], data["raw_label"], data["flag"]
        print("Image Size:", image.size, "Text:", text)
        if flag == "detection_1":
            ans = ast.literal_eval(raw_label)
            assert type(ans[0]) is float and len(ans) == 4
            bbox = convert_xyxy_to_cxcywh(ans)
            hm = mvt_utils.generate_hm_from_pt(
                torch.tensor([[bbox[0], bbox[1]]]).reshape(-1, 2) * 224,
                (224, 224), sigma=2, thres_sigma_times=3,
            )
            visualize_bboxes_and_heatmap(image, [bbox], hm, save_path)
        elif flag == "detection_2":
            ans = ast.literal_eval(raw_label)
            assert type(ans[0]) is tuple and len(ans[0]) == 4
            labels = torch.tensor([[a[0], a[1]] for a in ans])
            hm_all = mvt_utils.generate_hm_from_pt(
                labels.reshape(-1, 2) * 224, (224, 224), sigma=2, thres_sigma_times=3,
            )
            hm = masked_softmax(masked_mean(hm_all))
            visualize_bboxes_and_heatmap(image, ans, hm, save_path)
        else:
            assert False


def test_inference(cmd_args):
    """Load a pretrain checkpoint and run a few forward passes for sanity check."""
    with open(cmd_args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    model_id = os.environ.get("PALIGEMMA_PATH", "google/paligemma-3b-pt-224")
    sam3_ckpt = os.environ.get(
        "SAM3_CHECKPOINT_PATH",
        "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_ckpt/sam3",
    )
    model = BridgeVLAPretrainModel(model_id=model_id, sam3_ckpt_path=sam3_ckpt)
    sd = load_pretrain_checkpoint(cfg["checkpoint_dir"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Missing keys:", missing[:10])
    print("Unexpected keys:", unexpected[:10])
    model = model.to(device).eval()
    processor = model._processor

    ds = RoboPointDataset(cmd_args.image_folder, cmd_args.json_detection_path, res=224)
    save_path = cfg.get("test_save_path", "./pretrain_test.png")
    with torch.no_grad():
        for i in range(len(ds) - 1, 0, -300):
            sample = ds[i]
            tokens = processor(text=[sample["text"]], images=[sample["image"]],
                               return_tensors="pt", padding="longest")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(
                input_ids=tokens["input_ids"],
                pixel_values=tokens["pixel_values"],
                attention_mask=tokens["attention_mask"],
                raw_label=[sample["raw_label"]],
                flag=[sample["flag"]],
                raw_text=[sample["text"]],
            )
            print("loss=", out["loss"].item(), "text=", sample["text"])
            q_trans = out["q_trans"].view(224, 224, 1).detach()
            heatmap = F.softmax(q_trans.view(-1), dim=0).view(224, 224, 1)
            if sample["flag"] == "detection_1":
                ans = ast.literal_eval(sample["raw_label"])
                bbox = convert_xyxy_to_cxcywh(ans)
                visualize_bboxes_and_heatmap(sample["image"], [bbox], heatmap, save_path)
            else:
                ans = ast.literal_eval(sample["raw_label"])
                visualize_bboxes_and_heatmap(sample["image"], ans, heatmap, save_path)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=int, default=2,
                        help="1: visualise data; 2: pretrain; 3: test inference")
    parser.add_argument("--config_path", type=str, default="pretrain_config.yaml")
    parser.add_argument("--json_detection_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    cmd_args = parser.parse_args()

    if cmd_args.branches == 1:
        visualise_dataset(cmd_args.image_folder, cmd_args.json_detection_path)
    elif cmd_args.branches == 2:
        experiment(cmd_args)
    elif cmd_args.branches == 3:
        test_inference(cmd_args)
    else:
        raise ValueError(f"Unknown branch: {cmd_args.branches}")
