#Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/utils/peract_utils.py
import os

from peract_colab.arm.utils import stack_on_channel


CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}
DATA_FOLDER = os.environ.get("RLBENCH_DATA_FOLDER", "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/RLBench")
TRAIN_REPLAY_STORAGE_DIR = os.environ.get("RLBENCH_REPLAY_STORAGE_DIR", "/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/replay_train")
EPISODE_FOLDER = "episode%d"
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration
DEMO_AUGMENTATION_EVERY_N = 10  # sample n-th frame in demo
ROTATION_RESOLUTION = 5  # degree increments per axis

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        rgb = stack_on_channel(replay_sample["%s_rgb" % n])
        pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds
