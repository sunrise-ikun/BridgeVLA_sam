"""
SAM3 Encoder Wrapper for BridgeVLA integration.

Wraps SAM3's ViT backbone, FPN neck, text encoder, transformer encoder,
and geometry encoder to extract 16x16 patch tokens from rendered views.
"""

import os
import pkg_resources

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import (
    _create_position_encoding,
    _create_vit_backbone,
    _create_vit_neck,
    _create_text_encoder,
    _create_transformer_encoder,
    _create_geometry_encoder,
)
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model.geometry_encoders import Prompt


class SAM3EncoderWrapper(nn.Module):
    """Wraps SAM3 encoder components for extracting patch tokens.

    Processes images (padded to 1008x1008) and text through SAM3's vision-language
    pipeline, then extracts the center 16x16 patch tokens (256-dim) corresponding
    to the non-padded region.
    """

    SAM3_IMG_SIZE = 1008
    SAM3_PATCH_SIZE = 14

    def __init__(self, checkpoint_path, input_size=224):
        super().__init__()
        self.input_size = input_size
        self.pad_total = self.SAM3_IMG_SIZE - input_size  # 784
        self.pad_each = self.pad_total // 2               # 392
        # Patch indices for the non-padded center region
        self.patch_start = self.pad_each // self.SAM3_PATCH_SIZE   # 28
        self.patch_end = self.patch_start + input_size // self.SAM3_PATCH_SIZE  # 44

        # --- Build SAM3 components ---
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )
        position_encoding = _create_position_encoding(
            precompute_resolution=self.SAM3_IMG_SIZE
        )
        vit_backbone = _create_vit_backbone()
        vit_neck = _create_vit_neck(
            position_encoding, vit_backbone, enable_inst_interactivity=False
        )
        text_encoder = _create_text_encoder(bpe_path)

        self.backbone = SAM3VLBackbone(
            visual=vit_neck, text=text_encoder, scalp=1
        )
        self.encoder = _create_transformer_encoder()
        self.geometry_encoder = _create_geometry_encoder()

        # --- Load pretrained weights ---
        self._load_weights(checkpoint_path)


        # --- Freeze all parameters ---
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Always stay in eval mode to save memory and avoid BN / dropout issues."""
        return super().train(False)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _load_weights(self, checkpoint_dir):
        ckpt_file = os.path.join(checkpoint_dir, "sam3.pt")
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        weight_dict = {}
        for k, v in ckpt.items():
            if k.startswith("detector.backbone."):
                weight_dict[k.replace("detector.", "")] = v
            elif k.startswith("detector.transformer.encoder."):
                weight_dict[k.replace("detector.transformer.", "")] = v
            elif k.startswith("detector.geometry_encoder."):
                weight_dict[k.replace("detector.", "")] = v

        missing, unexpected = self.load_state_dict(weight_dict, strict=False)
        _rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if _rank == 0:
            print(f"[SAM3EncoderWrapper] Loaded weights from {ckpt_file}")
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:10]}...")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, images, text_prompts):
        """
        Args:
            images:       (B, 3, H, W)  in [-1, 1], H=W=input_size (224)
            text_prompts: list of B strings
        Returns:
            tokens: (B, 256, 16, 16)  patch tokens from SAM3 encoder
        """
        B = images.shape[0]
        device = images.device
        # Determine autocast device type from input (handles both CPU and CUDA)
        amp_device = "cuda" if device.type == "cuda" else "cpu"

        # Run everything under autocast to handle SAM3's mixed-precision ops
        # (addmm_act internally converts to bfloat16; geometry encoder creates
        #  float32 tensors; autocast reconciles both automatically).
        return self._forward_impl(images, text_prompts, B, device, amp_device)

    @torch.no_grad()
    def _forward_impl(self, images, text_prompts, B, device, amp_device):
        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16):
            return self._forward_body(images, text_prompts, B, device)

    def _forward_body(self, images, text_prompts, B, device):

        # 1. Center-pad to 1008x1008 (pad_value=-1 = black in [-1,1] space)
        padded = F.pad(
            images,
            [self.pad_each, self.pad_each, self.pad_each, self.pad_each],
            value=-1.0,
        )

        # 2. Vision backbone  → FPN features + position encodings
        vis_output = self.backbone.forward_image(padded)

        # 3. Text backbone → language features
        text_output = self.backbone.forward_text(text_prompts, device=device)

        # 4. Prepare encoder inputs (use only scale-1.0 = last level)
        img_feat = vis_output["backbone_fpn"][-1]     # (B, 256, 72, 72)
        img_pos  = vis_output["vision_pos_enc"][-1]   # (B, 256, 72, 72)
        feat_size = (img_feat.shape[-2], img_feat.shape[-1])  # (72, 72)

        # Convert to seq-first  (HW, B, C)
        img_feats_seq = [img_feat.flatten(2).permute(2, 0, 1)]
        img_pos_seq   = [img_pos.flatten(2).permute(2, 0, 1)]

        # 5. Geometry prompt (text-only; empty geometry → cls token only)
        dummy_prompt = Prompt(
            box_embeddings=torch.zeros(0, B, 4, device=device),
            box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
        )
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=dummy_prompt,
            img_feats=img_feats_seq,
            img_sizes=[feat_size],
            img_pos_embeds=img_pos_seq,
        )
        # geo_feats: (1, B, 256)   geo_masks: (B, 1)

        # 6. Build composite prompt  =  [text | geometry | (empty visual)]
        txt_feats = text_output["language_features"]  # (seq, B, 256)
        txt_masks = text_output["language_mask"]       # (B, seq)  True=padding

        visual_prompt_embed = torch.zeros(
            (0, *geo_feats.shape[1:]), device=device
        )
        visual_prompt_mask = torch.zeros(
            (*geo_masks.shape[:-1], 0), device=device, dtype=geo_masks.dtype
        )

        prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)

        # 7. Run transformer encoder  (text–vision fusion)
        prompt_pos_embed = torch.zeros_like(prompt)
        encoder_out = self.encoder(
            src=[x.clone() for x in img_feats_seq],
            src_key_padding_mask=None,
            src_pos=[x.clone() for x in img_pos_seq],
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[feat_size],
        )

        # 8. Extract center 16×16 patch tokens
        memory = encoder_out["memory"]  # (HW, B, 256)  seq-first
        H, W = feat_size
        memory = memory.permute(1, 2, 0).view(B, 256, H, W)  # (B, 256, 72, 72)
        tokens = memory[
            :, :, self.patch_start : self.patch_end, self.patch_start : self.patch_end
        ]
        return tokens.contiguous()  # (B, 256, 16, 16)
