"""Real_Dataset for the Dobot + ZED 20251011 layout.

Directory layout assumed:

    {data_root}/{episode_name}/{step_idx}/
        pose.pkl             # full-episode trajectory as a string
        instruction.pkl      # language goal string
        extrinsic_matrix.pkl # 4x4 camera->base
        zed_rgb/{i}.pkl      # uint8 (H,W,3) camera RGB, i = 0..num_frames-1
        zed_pcd/{i}.pkl      # float32 (H,W,3) camera-frame XYZ

For each step_idx we read the {step_idx}.pkl frame (matches the convention the
old BridgeVLA_Real trainer used), transform the point cloud to the Dobot base
frame, and build a sample compatible with `RVTAgent.update_gembench()`:

    sample = {
        "3rd":   {"rgb": (3,H,W) uint8, "pcd": (3,H,W) float32 in base frame},
        "gripper_pose": (8,) float32 = [x, y, z (m), qx, qy, qz, qw, claw],
        "ignore_collisions": float32 scalar = 1.0,
        "low_dim_state":    (2,) float32 = [claw, time],
        "lang_goal": str,
        "tasks":     str,
    }

The VLM/SAM3 language encoding wrapping ([[[text]]]) is applied in the training
loop, not here.
"""

import os
import pickle
import time
from typing import List, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def _parse_pose_string(data_str: str) -> List[dict]:
    """Parse a tab-/space-separated pose.pkl string.

    Each data line is: timestamp x_mm y_mm z_mm rx_deg ry_deg rz_deg claw [arm_flag]
    Returns a list of dicts (header skipped).
    """
    lines = data_str.strip().split("\n")
    entries = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        parts = line.strip().split()
        timestamp = parts[0]
        position = [float(x) for x in parts[1:4]]
        orientation = [float(x) for x in parts[4:7]]
        if len(parts) >= 8:
            claw_status = int(parts[7])
        else:
            # fall back to alternating open/close heuristic from the old loader
            claw_status = 1 if (i in (1, 2, 5)) else 0
        arm_flag = int(parts[8]) if len(parts) >= 9 else 0
        entries.append({
            "timestamp": timestamp,
            "position": position,
            "orientation": orientation,
            "claw_status": claw_status,
            "arm_flag": arm_flag,
        })
    return entries


def _load_pose_file(path: str) -> List[dict]:
    with open(path, "rb") as f:
        data_str = pickle.load(f)
    return _parse_pose_string(data_str)


def _convert_pcd_to_base(pcd_camera_hw3: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply 4x4 camera->base transform to an (H,W,3) camera-frame point cloud."""
    h, w, _ = pcd_camera_hw3.shape
    pc = pcd_camera_hw3.reshape(-1, 3)
    pc_h = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=1)
    pc_base = (transform_4x4 @ pc_h.T).T[:, :3]
    return pc_base.reshape(h, w, 3)


class Real_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Union[str, List[str]],
        cameras: List[str] = ("3rd",),
        ep_per_task: int = 10_000,
        verbose: bool = True,
        img_stride: int = 4,
    ):
        """
        :param img_stride: spatial stride applied to ZED RGB + PCD before they
            are returned. The raw 1080x1920 ZED frames contain ~2M points per
            sample, which is >30x the point count the sim trainer was tuned for;
            a default stride of 4 downsamples to ~130K points per sample while
            preserving the image aspect ratio. Set to 1 to disable.
        """
        if isinstance(data_path, str):
            self.data_paths = [data_path]
        else:
            self.data_paths = list(data_path)
        self.cameras = list(cameras)
        self.verbose = verbose
        self.img_stride = max(1, int(img_stride))

        # Per-sample index: list of (episode_dir, step_idx, num_steps, task_name).
        # We lazy-load RGB/PCD in __getitem__ so memory stays flat regardless of
        # dataset size.
        self.index: List[dict] = []
        self._build_index(ep_per_task)

    def _build_index(self, ep_per_task: int):
        t0 = time.time()
        n_episodes = 0
        for data_path in self.data_paths:
            if not os.path.isdir(data_path):
                raise FileNotFoundError(f"data_path not found: {data_path}")
            for ep_name in sorted(os.listdir(data_path)):
                ep_path = os.path.join(data_path, ep_name)
                if not os.path.isdir(ep_path):
                    continue
                step_dirs = sorted(
                    [d for d in os.listdir(ep_path)
                     if d.isdigit() and os.path.isdir(os.path.join(ep_path, d))],
                    key=int,
                )
                if len(step_dirs) < 2:
                    # need at least one current->next transition
                    continue
                n_episodes += 1
                if n_episodes > ep_per_task * max(1, self._num_tasks_cached(data_path)):
                    # ep_per_task cap is informational; we don't enforce a hard cap
                    # unless the user relied on the old loader's per-task semantics.
                    pass
                num_steps = len(step_dirs)
                # Each step_dir k has (current=pose_entries[k], target=pose_entries[k+1]).
                # The last step_dir has no "next" target, so we skip it.
                for k in range(num_steps - 1):
                    self.index.append({
                        "episode_path": ep_path,
                        "episode_name": ep_name,
                        "step_idx": int(step_dirs[k]),
                        "num_steps": num_steps,
                        "task_name": ep_name,
                    })
        if self.verbose:
            print(f"[Real_Dataset] indexed {len(self.index)} samples from "
                  f"{n_episodes} episodes across {len(self.data_paths)} root(s) "
                  f"in {time.time() - t0:.1f}s")

    def _num_tasks_cached(self, data_path: str) -> int:
        if not hasattr(self, "_task_counts"):
            self._task_counts = {}
        if data_path not in self._task_counts:
            self._task_counts[data_path] = sum(
                1 for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d))
            )
        return self._task_counts[data_path]

    @property
    def num_tasks(self) -> int:
        return sum(self._num_tasks_cached(p) for p in self.data_paths)

    @property
    def num_task_paths(self) -> int:
        return len({entry["episode_path"] for entry in self.index})

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        ep_path = entry["episode_path"]
        step = entry["step_idx"]
        num_steps = entry["num_steps"]
        step_dir = os.path.join(ep_path, str(step))

        # --- pose trajectory (shared across step_dirs, but we read the local one) ---
        gripper_pose = _load_pose_file(os.path.join(step_dir, "pose.pkl"))
        # GT target for this step is the next-frame pose; current is this step's.
        target_idx = step + 1
        if target_idx >= len(gripper_pose):
            target_idx = len(gripper_pose) - 1
        tgt = gripper_pose[target_idx]
        cur = gripper_pose[min(step, len(gripper_pose) - 1)]

        tgt_xyz_m = np.asarray(tgt["position"], dtype=np.float32) / 1000.0  # mm->m
        tgt_quat = R.from_euler(
            "xyz", tgt["orientation"], degrees=True
        ).as_quat().astype(np.float32)  # (qx, qy, qz, qw)
        gt_gripper_pose = np.concatenate(
            [tgt_xyz_m, tgt_quat, np.array([tgt["claw_status"]], dtype=np.float32)]
        )  # shape (8,)

        # --- low-dim state: [current claw, time embedding] ---
        time_embed = (1.0 - (step / float(max(num_steps - 1, 1)))) * 2.0 - 1.0
        low_dim_state = np.asarray(
            [cur["claw_status"], time_embed], dtype=np.float32
        )

        # --- camera extrinsic (4x4 camera->base) ---
        with open(os.path.join(step_dir, "extrinsic_matrix.pkl"), "rb") as f:
            extrinsic = np.asarray(pickle.load(f), dtype=np.float64)

        sample = {
            "gripper_pose": gt_gripper_pose,          # (8,) float32
            "low_dim_state": low_dim_state,           # (2,) float32
            "ignore_collisions": np.float32(1.0),
        }

        for cam in self.cameras:
            if cam != "3rd":
                # Only the ZED (logical "3rd") camera is present in the 20251011 layout.
                raise ValueError(f"Camera {cam!r} is not available in this dataset")
            rgb_path = os.path.join(step_dir, "zed_rgb", f"{step}.pkl")
            pcd_path = os.path.join(step_dir, "zed_pcd", f"{step}.pkl")
            with open(rgb_path, "rb") as f:
                rgb_hw3 = pickle.load(f)[:, :, :3]   # drop alpha if present
            if self.img_stride > 1:
                rgb_hw3 = rgb_hw3[::self.img_stride, ::self.img_stride]
            rgb_hw3 = np.ascontiguousarray(rgb_hw3)
            rgb_chw = np.transpose(rgb_hw3, (2, 0, 1)).astype(np.uint8)  # (3,H,W)

            with open(pcd_path, "rb") as f:
                pcd_hw3 = pickle.load(f)[:, :, :3].astype(np.float32)
            if self.img_stride > 1:
                pcd_hw3 = pcd_hw3[::self.img_stride, ::self.img_stride]
            pcd_hw3 = _convert_pcd_to_base(pcd_hw3, extrinsic)
            pcd_chw = np.transpose(pcd_hw3, (2, 0, 1)).astype(np.float32)  # (3,H,W)

            sample[cam] = {
                "rgb": rgb_chw,
                "pcd": pcd_chw,
            }

        # --- language instruction and task label ---
        with open(os.path.join(step_dir, "instruction.pkl"), "rb") as f:
            instruction = pickle.load(f)
        sample["lang_goal"] = str(instruction).strip()
        sample["tasks"] = entry["task_name"]

        return sample


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/data/bridgevla_data/Real/20251011"
    )
    ds = Real_Dataset(path, cameras=["3rd"])
    print(f"total samples: {len(ds)}")
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print(f"  {k}.{k2}: {getattr(v2, 'shape', None)} {getattr(v2, 'dtype', None)}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} -> {v!r}")
