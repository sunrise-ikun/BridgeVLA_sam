"""
BridgeVLA-SAM Agent for Real Robot Inference

Adapted from BridgeVLA_sam/finetune/bridgevla/models/bridgevla_agent.py.
Training-only dependencies (yarr, RLBench, GemBench) are removed.
Only inference-relevant code is kept, plus a new `act_real()` method for
real robot deployment.

Author: Peiyan Li (original), refactored for real-robot eval.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel

# ── BridgeVLA model imports ─────────────────────────────────────────────────
# Ensure BridgeVLA_sam/finetune is on sys.path *before* this module is loaded
# (done in the top-level eval scripts).
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import aug_utils


# ---------------------------------------------------------------------------
# Constants for real robot (override RLBench defaults)
# ---------------------------------------------------------------------------
REAL_SCENE_BOUNDS = [
    -1.1, -0.6, -0.2,
     0.2,  0.5,  0.6,
]
REAL_CAMERAS = ["3rd"]


def _align_real_frame_local(x):
    y = x.clone()
    y[..., 0:2] = -y[..., 0:2]
    return y


def _align_real_frame_rev_trans(rev_trans):
    def _rev_trans(x):
        return rev_trans(_align_real_frame_local(x))
    return _rev_trans


# ---------------------------------------------------------------------------
# Preprocessing helper (real robot observation → model input)
# ---------------------------------------------------------------------------
def _norm_rgb(x):
    """Normalise uint8 [0,255] → float [-1,1]."""
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_real(observation, cameras):
    """
    Convert a real-robot observation dict into the (obs, pcds) format
    expected by ``rvt_utils.get_pc_img_feat``.

    Args:
        observation: dict with camera keys, each containing 'rgb' and 'pcd'
                     tensors of shape (1, C, H, W).
        cameras: list of camera name strings, e.g. ["3rd"].

    Returns:
        obs:  list of [normed_rgb, pcd] per camera
        pcds: list of pcd tensors per camera
    """
    obs, pcds = [], []
    for cam in cameras:
        rgb = observation[cam]["rgb"]
        pcd = observation[cam]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds


# ---------------------------------------------------------------------------
# Visualization helpers (used by act_real when return_views=True)
# ---------------------------------------------------------------------------
def _apply_channel_wise_softmax(x):
    """Spatial softmax per channel.  x: (H, W, C) → same shape."""
    h, w, c = x.shape
    return torch.softmax(x.view(h * w, c), dim=0).view(h, w, c)


def _make_overlay(gray_vals: np.ndarray, orig_img: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Blend a single-channel heatmap onto an RGB image. Returns uint8 (H,W,3)."""
    import matplotlib
    matplotlib.use("Agg")
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


# ---------------------------------------------------------------------------
# RVTAgent – inference only
# ---------------------------------------------------------------------------
class RVTAgent:
    """BridgeVLA-SAM agent trimmed for real-robot inference.

    Constructor arguments mirror the training agent so that existing configs
    and ``load_agent`` still work.  Training-only paths are simply ignored.
    """

    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int = 72,
        stage_two: bool = False,
        move_pc_in_bound: bool = True,
        lr: float = 0.0001,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        weight_decay: float = 0.01,
        betas: list = [0.9, 0.95],
        warmup_steps: int = 2000,
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = None,
        cameras: list = None,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir: str = "",
    ):
        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self._weight_decay = weight_decay
        self._betas = tuple(betas)
        self._warmup_steps = warmup_steps
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds if scene_bounds is not None else REAL_SCENE_BOUNDS
        self.cameras = cameras if cameras is not None else REAL_CAMERAS
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

    # ------------------------------------------------------------------
    # Build / lifecycle
    # ------------------------------------------------------------------
    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Q-value extraction
    # ------------------------------------------------------------------
    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """Extract q-values from network output."""
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            if self.stage_two:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat(
                (out["feat_x"], out["feat_y"], out["feat_z"]), dim=-1
            ).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            raise NotImplementedError(f"rot_ver={self.rot_ver}")

        y_q = None
        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    # ------------------------------------------------------------------
    # Prediction decoding
    # ------------------------------------------------------------------
    def get_pred(self, out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info):
        """Decode network output into (wpt, rot_quat, grip, coll)."""
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(out, mvt1_or_mvt2, dyn_cam_info, y_q)

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
                ].argmax(1, keepdim=True),
                rot_q[
                    :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
                ].argmax(1, keepdim=True),
                rot_q[
                    :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    # ------------------------------------------------------------------
    # Real-robot inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act_real(self, observation: dict, cameras_view: list, return_views: bool = False):
        """
        Run a single inference step for a real-robot observation.

        Args:
            observation: dict with keys
                ``"language_goal"``  – nested list ``[[[text]]]``
                ``"<cam>"``          – dict with ``"rgb"`` and ``"pcd"``
                                       tensors of shape (1, 3, H, W)
            cameras_view: list of camera names, e.g. ``["3rd"]``
            return_views: if True, also return a dict of rendered view
                          images (original + overlay) for mvt1 and mvt2.

        Returns:
            target_pos     (np.ndarray, shape (3,))
            target_quat    (np.ndarray, shape (4,))
            target_gripper (np.ndarray, shape (1,))
            [views_info]   (dict) – only when return_views=True.
                keys: "mvt1" (and "mvt2" if stage_two), each mapping to
                {"originals": [img0,img1,img2], "overlays": [img0,img1,img2]}
                where each img is a uint8 (H, W, 3) numpy array.
        """
        language_goal = observation["language_goal"]

        # 1. preprocess
        obs, pcd = _preprocess_inputs_real(observation, cameras_view)
        pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # 2. place point cloud in unit cube
        pc_new, rev_trans = [], []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            a = _align_real_frame_local(a)
            b = _align_real_frame_rev_trans(b)
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        # 3. forward
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,
            language_goal=language_goal,
        )

        # 4. decode – also get q_trans when views are requested
        if return_views:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        # Un-rotate predicted quaternion from aligned frame back to real-robot frame
        _Rz180 = Rotation.from_euler('z', 180, degrees=True)
        for i in range(len(pred_rot_quat)):
            pred_rot_quat[i] = (_Rz180 * Rotation.from_quat(pred_rot_quat[i])).as_quat()

        result = (
            pred_wpt[0].cpu().numpy(),
            pred_rot_quat[0],
            pred_grip[0].cpu().numpy(),
        )

        if return_views:
            views_info = {}
            stages = [("mvt1", "mvt1_ori_img", slice(0, nc))]
            if self.stage_two:
                stages.append(("mvt2", "mvt2_ori_img", slice(nc, 2 * nc)))

            for stage_name, img_key, ch_slice in stages:
                if img_key not in out:
                    continue
                rgb = out[img_key][0, :, 3:6].float()  # (nc, 3, H, W) in [0,1]
                color_imgs = rgb.cpu().numpy().transpose(0, 2, 3, 1)  # (nc, H, W, 3)

                pred_raw = q_trans[0, :, ch_slice].clone().view(h, w, nc).float()
                pred_hm = _apply_channel_wise_softmax(pred_raw) * 100.0
                pred_gray = pred_hm.cpu().numpy().transpose(2, 0, 1)  # (nc, H, W)
                pred_logits = pred_raw.cpu().numpy().transpose(2, 0, 1)  # (nc, H, W)

                originals, overlays, logits = [], [], []
                for i in range(nc):
                    orig = (np.clip(color_imgs[i], 0, 1) * 255).astype(np.uint8)
                    originals.append(orig)
                    overlays.append(_make_overlay(pred_gray[i], orig))
                    logits.append(pred_logits[i].astype(np.float64))

                views_info[stage_name] = {
                    "originals": originals,
                    "overlays": overlays,
                    "logits": logits,
                }

            return result + (views_info,)

        return result

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"RVTAgent(\n"
            f"  rot_ver={self.rot_ver}, stage_two={self.stage_two},\n"
            f"  num_rotation_classes={self._num_rotation_classes},\n"
            f"  scene_bounds={self.scene_bounds},\n"
            f"  cameras={self.cameras},\n"
            f"  place_with_mean={self._place_with_mean},\n"
            f"  move_pc_in_bound={self.move_pc_in_bound},\n"
            f")"
        )
