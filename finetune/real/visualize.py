"""Visualization helpers for the real-robot trainer & offline debug eval.

Two public entry points:

  - ``visualize_epoch(agent, dataset, ...)``   used by ``real/train.py`` at the
    start of every epoch (rank 0 only). Defaults to visualizing only the
    first MVT stage (``mvt1``), which is the full-scene top/front/right view.

  - ``visualize_samples(agent, samples, save_dir, stages=("mvt1","mvt2"), ...)``
    general helper used by ``real/debug_eval.py``. Runs a single forward pass
    over the given samples in eval mode and dumps pred + GT heatmap overlays
    for the requested stages.

For every sample/stage we save, per 3-view ``i in {0,1,2}``:

    {sample_dir}/{stage}/original_{i}.png       (rendered RGB, saved once)
    {sample_dir}/{stage}/overlay_{i}.png        (pred | gt side-by-side)
    {sample_dir}/{stage}/logits_{i}.png         (pred | gt side-by-side)
    {sample_dir}/meta.txt
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from PIL import Image

from bridgevla.models.bridgevla_agent import apply_channel_wise_softmax
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils

from real.utils.peract_utils import _preprocess_inputs_real


# PaliGemma bf16 attention kernel FPEs on H20 with bs < 4. Pad the forward
# batch up to this minimum by duplicating the first sample; duplicates are
# never written to disk.
MIN_FORWARD_BS = 4

_STAGE_INFO = {
    # stage -> (rendered-image key in `out`, q_trans channel slice offset)
    "mvt1": "mvt1_ori_img",
    "mvt2": "mvt2_ori_img",
}


# ---------------------------------------------------------------------------
# tensor/dict helpers
# ---------------------------------------------------------------------------

def _move_to_device(x, device):
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def _collate_samples(samples: List[dict]) -> dict:
    """Stack a list of Real_Dataset samples into a batch dict.

    Can't use torch default_collate directly because the per-camera entries
    return numpy arrays; we just stack them tensor-by-tensor here.
    """
    assert len(samples) > 0
    keys = samples[0].keys()
    out: dict = {}
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, dict):
            out[k] = {
                sub_k: torch.from_numpy(np.stack([s[k][sub_k] for s in samples], axis=0))
                for sub_k in v0
            }
        elif isinstance(v0, np.ndarray):
            out[k] = torch.from_numpy(np.stack([s[k] for s in samples], axis=0))
        elif isinstance(v0, (int, float, np.floating, np.integer)):
            out[k] = torch.tensor([float(s[k]) for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


# ---------------------------------------------------------------------------
# forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_viz_forward(
    agent,
    samples: List[dict],
    cameras: Sequence[str],
) -> Tuple[dict, torch.Tensor, torch.Tensor, int, int, int]:
    """Run a single eval-mode forward through ``agent._net_mod`` (bypassing
    DDP) and return everything the visualization layer needs.

    Returns
    -------
    out          : dict produced by MVT.forward (contains mvt1_ori_img and,
                   when stage_two is on, mvt2_ori_img + the mvt2 sub-dict)
    q_trans      : (bs, h*w, nc*stage_count) translation logits
    action_trans : (bs, h*w, nc*stage_count) GT Gaussian heatmaps
    bs, h, w     : batch size (after padding) and image dims
    """
    num_real = len(samples)
    assert num_real > 0, "need at least one sample"

    # Pad to MIN_FORWARD_BS with copies of the first sample. Never saved.
    forward_bs = max(num_real, MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - num_real)

    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)
    batch["lang_goal"] = [[[s["lang_goal"]]] for s in padded]
    batch["tasks"] = [s["tasks"] for s in padded]

    # --- mirrors RVTAgent.update_gembench, no backprop, no augmentation ---
    action_gripper_pose = batch["gripper_pose"]
    action_trans_con = action_gripper_pose[:, 0:3]

    obs, pcd = _preprocess_inputs_real(batch, list(cameras))
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
    pc, img_feat = rvt_utils.move_pc_in_bound(
        pc, img_feat, agent.scene_bounds, no_op=not agent.move_pc_in_bound,
    )
    wpt = [x[:3] for x in action_trans_con]

    wpt_local_list = []
    for _pc, _wpt in zip(pc, wpt):
        a, _ = mvt_utils.place_pc_in_cube(
            _pc, _wpt,
            with_mean_or_bounds=agent._place_with_mean,
            scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
        )
        if getattr(agent, "align_real_frame", False):
            a = a.clone()
            a[..., 0:2] = -a[..., 0:2]
        wpt_local_list.append(a.unsqueeze(0))
    wpt_local = torch.cat(wpt_local_list, dim=0)

    pc = [mvt_utils.place_pc_in_cube(
        _pc,
        with_mean_or_bounds=agent._place_with_mean,
        scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
    )[0] for _pc in pc]
    if getattr(agent, "align_real_frame", False):
        pc = [_pc.clone() for _pc in pc]
        for _pc in pc:
            _pc[..., 0:2] = -_pc[..., 0:2]

    bs = len(pc)
    nc = agent._net_mod.num_img
    h = w = agent._net_mod.img_size

    # In eval mode MVT.verify_inp does not require wpt_local/rot_x_y; stage_two
    # will rely on the *predicted* waypoint, matching real-robot inference.
    out = agent._net_mod(
        pc=pc,
        img_feat=img_feat,
        lang_emb=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        language_goal=batch["lang_goal"],
    )
    q_trans, _, _, _, _, pts = agent.get_q(out, dims=(bs, nc, h, w))
    action_trans = agent.get_action_trans(
        wpt_local, pts, out, None, dims=(bs, nc, h, w),
    )
    # Carry the batch tensor back out for downstream meta.txt writing.
    out["_batch_gripper_pose"] = batch["gripper_pose"]
    out["_batch_lang_goal"] = [s["lang_goal"] for s in padded]
    out["_batch_tasks"] = [s["tasks"] for s in padded]
    return out, q_trans, action_trans, bs, h, w


# ---------------------------------------------------------------------------
# per-sample save helpers
# ---------------------------------------------------------------------------

def _make_overlay(gray_vals: np.ndarray, orig_img: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Blend a single-channel heatmap onto an RGB image. Returns uint8 (H,W,3)."""
    import matplotlib.cm as cm
    hm = gray_vals.astype(np.float64)
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max - hm_min > 1e-8:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm)
    heatmap_rgb = (cm.jet(hm) * 255).astype(np.uint8)[..., :3]
    blended = (
        (1 - alpha) * orig_img.astype(np.float64)
        + alpha * heatmap_rgb.astype(np.float64)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _save_stage(
    out: dict,
    q_trans: torch.Tensor,
    action_trans: torch.Tensor,
    sample_idx: int,
    sample_dir: str,
    stage: str,
    nc: int,
    h: int,
    w: int,
) -> bool:
    """Save pred|gt side-by-side viz for one stage ("mvt1" or "mvt2").

    For each of the ``nc`` views saves:
      - ``original_{i}.png`` — rendered RGB (single image)
      - ``overlay_{i}.png``  — pred overlay | gt overlay (1×2)
      - ``logits_{i}.png``   — pred logits  | gt logits  (1×2, matplotlib)

    Returns True on success, False if the stage's rendered image is missing.
    """
    import matplotlib.pyplot as plt

    if stage not in _STAGE_INFO:
        raise ValueError(f"unknown stage {stage!r}")
    img_key = _STAGE_INFO[stage]
    if img_key not in out:
        print(f"[viz] {img_key} missing from model output; skip {stage}")
        return False

    # Channel layout: stage_two concat is [mvt1 (nc chans), mvt2 (nc chans)].
    if stage == "mvt1":
        sl = slice(0, nc)
    else:  # mvt2
        sl = slice(nc, 2 * nc)

    rgb = out[img_key][sample_idx, :, 3:6].float()  # (nc, 3, H, W) in [0,1]

    pred_raw = q_trans[sample_idx, :, sl].clone().view(h, w, nc).float()
    pred_hm = apply_channel_wise_softmax(pred_raw) * 100.0

    gt_raw = action_trans[sample_idx, :, sl].clone().view(h, w, nc).float()
    gt_hm = gt_raw * 100.0

    save_dir = os.path.join(sample_dir, stage)
    os.makedirs(save_dir, exist_ok=True)

    color_imgs = rgb.cpu().numpy().transpose(0, 2, 3, 1)   # (nc, H, W, 3)
    pred_gray = pred_hm.cpu().numpy().transpose(2, 0, 1)    # (nc, H, W)
    gt_gray = gt_hm.cpu().numpy().transpose(2, 0, 1)
    pred_logits = pred_raw.cpu().numpy().transpose(2, 0, 1)
    gt_logits = gt_raw.cpu().numpy().transpose(2, 0, 1)

    logits_vmin, logits_vmax = -10.0, 40.0

    for i in range(nc):
        orig = (np.clip(color_imgs[i], 0, 1) * 255).astype(np.uint8)

        # --- original (single) ---
        Image.fromarray(orig).save(os.path.join(save_dir, f"original_{i}.png"))

        # --- overlay: pred | gt ---
        pred_ov = _make_overlay(pred_gray[i], orig)
        gt_ov = _make_overlay(gt_gray[i], orig)
        combined_ov = np.concatenate([pred_ov, gt_ov], axis=1)
        Image.fromarray(combined_ov).save(
            os.path.join(save_dir, f"overlay_{i}.png"))

        # --- logits: pred | gt (matplotlib) ---
        p_l = pred_logits[i].astype(np.float64)
        g_l = gt_logits[i].astype(np.float64)
        fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(12, 5))
        ax_p.imshow(p_l, cmap="jet", vmin=logits_vmin, vmax=logits_vmax,
                    interpolation="nearest")
        ax_p.set_title(f"pred view {i}\n"
                       f"min={p_l.min():.3f}  max={p_l.max():.3f}")
        ax_p.set_axis_off()
        im_g = ax_g.imshow(g_l, cmap="jet", vmin=logits_vmin, vmax=logits_vmax,
                           interpolation="nearest")
        ax_g.set_title(f"gt view {i}\n"
                       f"min={g_l.min():.3f}  max={g_l.max():.3f}")
        ax_g.set_axis_off()
        fig.colorbar(im_g, ax=[ax_p, ax_g], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"logits_{i}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    return True


def _write_meta(sample_dir: str, meta: dict) -> None:
    with open(os.path.join(sample_dir, "meta.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_samples(
    agent,
    samples: List[dict],
    save_dirs: List[str],
    stages: Sequence[str] = ("mvt1",),
    cameras: Sequence[str] = ("3rd",),
    extra_meta: Optional[List[dict]] = None,
) -> None:
    """Run a single forward over ``samples`` and dump pred+GT viz for each.

    ``samples`` and ``save_dirs`` must be the same length; one save_dir per
    real sample (padding copies are not written). ``stages`` subset of
    ``("mvt1", "mvt2")``. If ``stage_two`` is False on the agent, ``mvt2``
    requests are silently skipped.
    """
    assert len(samples) == len(save_dirs), "samples and save_dirs length mismatch"
    num_real = len(samples)
    if num_real == 0:
        return

    was_training = agent._network.training
    agent.eval()
    try:
        out, q_trans, action_trans, _bs, h, w = _run_viz_forward(
            agent, samples, cameras,
        )
        nc = agent._net_mod.num_img

        for i in range(num_real):
            sample_dir = save_dirs[i]
            os.makedirs(sample_dir, exist_ok=True)
            for stage in stages:
                _save_stage(out, q_trans, action_trans, i, sample_dir, stage,
                            nc=nc, h=h, w=w)

            gp = out["_batch_gripper_pose"][i].cpu().tolist()
            meta = {
                "lang_goal":    out["_batch_lang_goal"][i],
                "task":         out["_batch_tasks"][i],
                "gt_xyz_m":     gp[0:3],
                "gt_quat_xyzw": gp[3:7],
                "gt_claw":      gp[7],
                "stages":       list(stages),
            }
            if extra_meta is not None and i < len(extra_meta):
                meta.update(extra_meta[i])
            _write_meta(sample_dir, meta)
    finally:
        if was_training:
            agent.train()


@torch.no_grad()
def visualize_epoch(
    agent,
    dataset,
    epoch: int,
    log_dir: str,
    num_samples: int = 2,
    cameras: Sequence[str] = ("3rd",),
    seed: Optional[int] = None,
    stages: Sequence[str] = ("mvt1",),
) -> None:
    """Sample ``num_samples`` random items from the training dataset and save
    pred+GT viz under ``{log_dir}/viz/epoch_{epoch:04d}/sample_{k}_idx{N}/``.

    Default ``stages=("mvt1",)`` keeps the training-time artifact light; the
    debug_eval script calls this same pipeline with ``stages=("mvt1","mvt2")``.
    Safe to call on rank 0 only — all network ops use ``agent._net_mod`` so no
    NCCL collective is triggered.
    """
    assert len(dataset) > 0, "empty dataset"
    num_samples = min(num_samples, len(dataset))

    rng = np.random.default_rng(seed if seed is not None else epoch)
    indices = rng.choice(len(dataset), size=num_samples, replace=False).tolist()
    samples = [dataset[int(i)] for i in indices]

    viz_root = os.path.join(log_dir, "viz", f"epoch_{epoch:04d}")
    save_dirs = [
        os.path.join(viz_root, f"sample_{k}_idx{int(indices[k])}")
        for k in range(num_samples)
    ]
    extra_meta = [{"dataset_idx": int(indices[k])} for k in range(num_samples)]

    visualize_samples(
        agent, samples, save_dirs,
        stages=stages, cameras=cameras, extra_meta=extra_meta,
    )
    print(f"[viz] epoch {epoch}: saved {num_samples} sample(s) under {viz_root}")
