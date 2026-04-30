"""
Constants and preprocessing utilities for real-robot evaluation.

Scene bounds and camera names are configured for the Dobot real-robot setup.
"""

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# [x_min, y_min, z_min, x_max, y_max, z_max]
SCENE_BOUNDS = [
    -1.1, -0.6, -0.2,
     0.2,  0.5,  0.6,
]

CAMERAS = ["3rd"]
IMAGE_SIZE = 128


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _norm_rgb(x):
    """Normalise uint8 [0, 255] → float [-1, 1]."""
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_real(replay_sample, cameras):
    """
    Build (obs, pcds) lists from a real-robot observation dict.

    Args:
        replay_sample: dict with camera name keys, each holding
                       ``"rgb"`` and ``"pcd"`` tensors of shape (1, C, H, W).
        cameras: list of camera name strings, e.g. ``["3rd"]``.

    Returns:
        obs:  list of ``[normed_rgb, pcd]`` per camera
        pcds: list of pcd tensors per camera
    """
    obs, pcds = [], []
    for cam in cameras:
        rgb = replay_sample[cam]["rgb"]
        pcd = replay_sample[cam]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds

