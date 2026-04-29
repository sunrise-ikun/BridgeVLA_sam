"""Real_Dataset for the task-grouped Dobot + ZED layout.

Directory layout assumed:

    {data_root}/{task_group}/{episode_name}/
        pose.pkl             # full-episode trajectory as a string
        instruction.pkl      # language goal string
        extrinsic_matrix.pkl # 4x4 camera->base
        zed_rgb/{i}.pkl      # uint8 (H,W,3) camera RGB, i = 0..num_frames-1
        zed_pcd/{i}.pkl      # float32 (H,W,3) camera-frame XYZ
        cam_img/             # optional regular camera images (unused by loader)

Examples of task_group directories: plate_tasks, shelf_tasks, drawer_tasks.

For each frame i we produce one training sample (current=i, target=i+1).
The last frame in each episode has no next target so it is skipped.

The point cloud is transformed from camera frame to the Dobot base frame, and
the sample is built compatible with `RVTAgent.update_gembench()`:

    sample = {
        "3rd":   {"rgb": (3,H,W) uint8, "pcd": (3,H,W) float32 in base frame},
        "gripper_pose": (8,) float32 = [x, y, z (m), qx, qy, qz, qw, claw],
        "ignore_collisions": float32 scalar = 1.0,
        "low_dim_state":    (2,) float32 = [claw, time],
        "lang_goal": str,
        "tasks":     str,   # equals the task_group name
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
        tasks: Union[str, List[str]] = "all",
        ep_per_task: int = 10_000,
        verbose: bool = True,
        img_stride: int = 4,
    ):
        """
        :param data_path: Root directory (or list of roots) containing
            task_group subdirectories (e.g. plate_tasks, shelf_tasks).
        :param tasks: Which task_group directories to include. Pass ``"all"``
            (default) to include every task_group under data_path, or a
            comma-separated string / list of names to select specific groups.
        :param img_stride: Spatial stride applied to ZED RGB + PCD before they
            are returned. The raw 1080x1920 ZED frames contain ~2 M points per
            sample, which is >30x the point count the sim trainer was tuned for;
            a default stride of 4 downsamples to ~130 K points per sample while
            preserving the image aspect ratio. Set to 1 to disable.
        """
        if isinstance(data_path, str):
            self.data_paths = [data_path]
        else:
            self.data_paths = list(data_path)
        self.cameras = list(cameras)
        self.verbose = verbose
        self.img_stride = max(1, int(img_stride))

        # Resolve the set of task_group names to include.
        if isinstance(tasks, str):
            if tasks.strip().lower() == "all":
                self._task_filter: Union[None, set] = None  # None = accept all
            else:
                self._task_filter = {t.strip() for t in tasks.split(",") if t.strip()}
        else:
            self._task_filter = set(tasks) if tasks else None

        # Per-sample index: list of dicts with episode metadata.
        # RGB/PCD are lazy-loaded in __getitem__ so memory stays flat.
        self.index: List[dict] = []
        self._build_index(ep_per_task)

    def _build_index(self, ep_per_task: int):
        t0 = time.time()
        n_episodes = 0
        for data_path in self.data_paths:
            if not os.path.isdir(data_path):
                raise FileNotFoundError(f"data_path not found: {data_path}")
            for task_group in sorted(os.listdir(data_path)):
                task_group_path = os.path.join(data_path, task_group)
                if not os.path.isdir(task_group_path):
                    continue
                if self._task_filter is not None and task_group not in self._task_filter:
                    continue
                for ep_name in sorted(os.listdir(task_group_path)):
                    ep_path = os.path.join(task_group_path, ep_name)
                    if not os.path.isdir(ep_path):
                        continue
                    zed_rgb_dir = os.path.join(ep_path, "zed_rgb")
                    if not os.path.isdir(zed_rgb_dir):
                        continue
                    frame_files = sorted(
                        [f for f in os.listdir(zed_rgb_dir) if f.endswith(".pkl")],
                        key=lambda x: int(os.path.splitext(x)[0]),
                    )
                    num_frames = len(frame_files)
                    if num_frames < 2:
                        # need at least one current->next transition
                        continue
                    n_episodes += 1
                    # Frame k → current observation; frame k+1 → target action.
                    # The last frame has no next target so we skip it.
                    for k in range(num_frames - 1):
                        self.index.append({
                            "episode_path": ep_path,
                            "episode_name": ep_name,
                            "task_group": task_group,
                            "step_idx": k,
                            "num_steps": num_frames,
                            "task_name": task_group,
                        })
        if self.verbose:
            task_groups_found = {e["task_group"] for e in self.index}
            print(f"[Real_Dataset] indexed {len(self.index)} samples from "
                  f"{n_episodes} episodes, task_groups={sorted(task_groups_found)}, "
                  f"across {len(self.data_paths)} root(s) in {time.time() - t0:.1f}s")

    @property
    def num_tasks(self) -> int:
        return len({e["task_group"] for e in self.index})

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

        # --- pose trajectory (shared file at episode level) ---
        gripper_pose = _load_pose_file(os.path.join(ep_path, "pose.pkl"))
        # GT target for this step is the next-frame pose; current is this frame.
        target_idx = min(step + 1, len(gripper_pose) - 1)
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

        # --- camera extrinsic (4x4 camera->base, shared at episode level) ---
        with open(os.path.join(ep_path, "extrinsic_matrix.pkl"), "rb") as f:
            extrinsic = np.asarray(pickle.load(f), dtype=np.float64)

        sample = {
            "gripper_pose": gt_gripper_pose,          # (8,) float32
            "low_dim_state": low_dim_state,           # (2,) float32
            "ignore_collisions": np.float32(1.0),
        }

        for cam in self.cameras:
            if cam != "3rd":
                # Only the ZED (logical "3rd") camera is present in this layout.
                raise ValueError(f"Camera {cam!r} is not available in this dataset")
            rgb_path = os.path.join(ep_path, "zed_rgb", f"{step}.pkl")
            pcd_path = os.path.join(ep_path, "zed_pcd", f"{step}.pkl")
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
        with open(os.path.join(ep_path, "instruction.pkl"), "rb") as f:
            instruction = pickle.load(f)
        sample["lang_goal"] = str(instruction).strip()
        sample["tasks"] = entry["task_name"]

        return sample


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/Real"
    )
    tasks_arg = sys.argv[2] if len(sys.argv) > 2 else "all"
    ds = Real_Dataset(path, cameras=["3rd"], tasks=tasks_arg)
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
