"""Constants + preprocessing for the Dobot real-robot (ZED) pipeline.

Scene bounds and camera list are carried over from the old BridgeVLA_Real
setup that produced the 20251011 dataset. Image normalization matches the
GemBench/RLBench preprocessing (RGB in [-1, 1]).
"""

CAMERAS_REAL = ["3rd"]

SCENE_BOUNDS_REAL = [
    -1.1, -0.6, -0.2,
     0.2,  0.5,  0.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] in Dobot base frame (meters)

IMAGE_SIZE = 224


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_real(replay_sample, cameras):
    """Dict-style preprocessing. Mirrors gembench_utils._preprocess_inputs_gembench.

    Expects replay_sample[n]["rgb"] shape (b, 3, H, W) uint8 and
    replay_sample[n]["pcd"] shape (b, 3, H, W) float32 in base frame.
    """
    obs, pcds = [], []
    for n in cameras:
        rgb = replay_sample[n]["rgb"]
        pcd = replay_sample[n]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds
