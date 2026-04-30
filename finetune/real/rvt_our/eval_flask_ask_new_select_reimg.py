"""
BridgeVLA-SAM  Real-Robot Evaluation Client
=============================================
Captures camera images, sends them to the Flask inference server, and
controls the Dobot robot arm to execute predicted actions.

Usage::

    python eval_flask_ask_new_select_reimg.py

Make sure ``eval_flask_app.py`` is already running on the same machine
(or adjust ``SERVER_URL``).
"""

import os
import re
import sys
import copy
import json
import time
import base64

import cv2
import numpy as np
import torch
import requests
import open3d as o3d
from transforms3d.euler import euler2mat

from botarm import Server, DobotController, Point
from utils.real_camera_utils_new import Camera, get_cam_extrinsic


# =====================================================================
# Configuration  (edit these to match your setup)
# =====================================================================
SERVER_URL       = "http://127.0.0.1:5000/predict"
EPISODE_LENGTH   = 100
DEVICE           = "cuda:0"

# Language instruction for the current task
#INSTRUCTION = "put the wolf in the upper drawer"
INSTRUCTION = "put the Redbull in the top shelf"


# Safety clipping ranges (metres, base frame)
Y_RANGE = (-0.6, 0.3)
Z_RANGE = (-0.1, 0.5)

# Dobot arm network config (single-arm)
ARM_IP       = "192.168.201.1"
ARM_PORT     = 29999
LOCAL_IP     = "192.168.201.38"
LOCAL_PORT   = 12345


# =====================================================================
# Coordinate transform helpers
# =====================================================================
def convert_pcd_to_base(cam_type: str, pcd: np.ndarray) -> np.ndarray:
    """Transform point cloud from camera frame to robot base frame."""
    transform = get_cam_extrinsic(cam_type, "left")
    h, w = pcd.shape[:2]
    pts = pcd.reshape(-1, 3)
    pts_h = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    pts_base = (transform @ pts_h.T).T[:, :3]
    return pts_base.reshape(h, w, 3)


# =====================================================================
# Visualisation
# =====================================================================
VIS_SCENE_BOUNDS = [-1.3, -1.5, -0.1, 0.4, 0.7, 0.6]


def vis_pcd_with_end_pred(pcd, rgb, end_pose, pred_pose):
    """Visualise point cloud with current arm pose and predicted target."""
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    x_min, y_min, z_min, x_max, y_max, z_max = VIS_SCENE_BOUNDS
    mask = (
        (pcd_flat[:, 0] >= x_min) & (pcd_flat[:, 0] <= x_max) &
        (pcd_flat[:, 1] >= y_min) & (pcd_flat[:, 1] <= y_max) &
        (pcd_flat[:, 2] >= z_min) & (pcd_flat[:, 2] <= z_max)
    )
    pcd_flat, rgb_flat = pcd_flat[mask], rgb_flat[mask]

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    # current arm pose
    end_pose = [float(x) for x in end_pose]
    T_end = np.eye(4)
    T_end[:3, :3] = euler2mat(*np.deg2rad(end_pose[3:]), axes="sxyz")
    T_end[:3, 3] = np.array(end_pose[:3]) * 0.001
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_end.transform(T_end)

    # predicted pose
    if isinstance(pred_pose, str):
        pred_pose = [float(x) for x in pred_pose.strip("{}").split(",")]
    else:
        pred_pose = [float(x) for x in pred_pose]
    T_pred = np.eye(4)
    T_pred[:3, :3] = euler2mat(*np.deg2rad(pred_pose[3:]), axes="sxyz")
    T_pred[:3, 3] = np.array(pred_pose[:3]) * 0.001
    axis_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_pred.transform(T_pred)

    o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end, axis_pred])


# =====================================================================
# Serialisation / HTTP query
# =====================================================================
def _ndarray_to_base64(arr: np.ndarray) -> dict:
    return {
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "dtype": str(arr.dtype),
        "shape": arr.shape,
    }


def serialize_observation(language_goal, rgb: torch.Tensor, pcd: torch.Tensor) -> str:
    """Serialise observation tensors to a JSON string for the server."""
    return json.dumps({
        "language_goal": language_goal,
        "rgb":  _ndarray_to_base64(rgb.detach().cpu().numpy()),
        "pcd":  _ndarray_to_base64(pcd.detach().cpu().numpy()),
    })


def query_policy(json_str: str, url: str = SERVER_URL):
    """Send observation to inference server and return (pos, quat, gripper)."""
    try:
        resp = requests.post(
            url, data=json_str,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"Server returned {resp.status_code}")
            return None, None, None

        result = resp.json()
        print("Server result:", result)
        if "error" in result:
            print(f"Server error: {result['error']}")
            return None, None, None

        return (
            np.array(result["target_pos"],        dtype=np.float32),
            np.array(result["target_quat"],        dtype=np.float32),
            np.array(result["target_gripper"][0],  dtype=np.float32),
        )
    except Exception as e:
        print(f"query_policy failed: {e}")
        return None, None, None


# =====================================================================
# User prompts
# =====================================================================
def prompt_before_action() -> str:
    """Returns 'retry', 'execute', or 'quit'."""
    while True:
        print("\n>>> Pre-action: [r]etry | [n] execute | [q]uit <<<")
        try:
            cmd = input().strip().lower()
        except KeyboardInterrupt:
            return "quit"
        if cmd in ("r", "n", "q"):
            return {"r": "retry", "n": "execute", "q": "quit"}[cmd]
        print(f"Invalid input '{cmd}'.")


def prompt_after_action() -> str:
    """Returns 'continue' or 'quit'."""
    print("\n>>> Action done. Press Enter for next step, or 'q' to quit <<<")
    try:
        cmd = input().strip().lower()
    except KeyboardInterrupt:
        return "quit"
    return "quit" if cmd == "q" else "continue"


# =====================================================================
# Observation preprocessing (camera → tensor)
# =====================================================================
def preprocess_observation(camera_info: dict, device: str = DEVICE) -> tuple:
    """
    Convert raw camera data into tensors for the server.

    Returns:
        observation_raw : dict with numpy arrays (for visualisation)
        rgb_tensor      : (1, 3, H, W)  float tensor
        pcd_tensor      : (1, 3, H, W)  float tensor
    """
    pcd_base = convert_pcd_to_base("3rd", camera_info["3rd"]["pcd"])
    rgb_bgr  = cv2.cvtColor(camera_info["3rd"]["rgb"], cv2.COLOR_RGB2BGR)

    obs_raw = {"pcd": pcd_base.copy(), "rgb": rgb_bgr.copy()}

    rgb_t = torch.from_numpy(
        np.transpose(rgb_bgr, [2, 0, 1])
    ).to(device).unsqueeze(0).float().contiguous()

    pcd_t = torch.from_numpy(
        np.transpose(pcd_base, [2, 0, 1])
    ).to(device).unsqueeze(0).float().contiguous()

    return obs_raw, rgb_t, pcd_t


# =====================================================================
# Gripper state reading
# =====================================================================
def read_gripper_state(bot, server) -> str:
    """Read the gripper open/close state (1=closed, 0=open)."""
    raw = bot.claws_read_command(server.modbusRTU, 258, 1)
    match = re.search(r"\{(.*?)\}", str(raw))
    return match.group(1)


# =====================================================================
# Main evaluation loop
# =====================================================================
def _eval():
    instructions = [[[INSTRUCTION]]]
    cameras = Camera(camera_type="3rd")

    # ---- initialise robot arm ----
    server = Server(ARM_IP, ARM_PORT, LOCAL_IP, LOCAL_PORT)
    server.sock.connect((server.ip, server.port))
    bot = DobotController(server.sock)
    server.init_com(bot)
    bot._initialize(server)

    try:
        for step in range(EPISODE_LENGTH - 1):
            print(f"\n{'='*50}\n=== STEP {step} ===\n{'='*50}")

            # ── retry loop ─────────────────────────────────
            while True:
                # capture (flush buffer with 10 grabs)
                for _ in range(10):
                    time.sleep(0.1)
                    camera_info = cameras.capture()
                    time.sleep(0.1)

                # ZedCam.capture() returns numpy *views* into sl.Mat buffers
                # that are freed when capture() returns.  Force-copy NOW
                # before the backing memory is recycled.
                camera_info["3rd"] = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in camera_info["3rd"].items()
                }

                obs_raw, rgb_t, pcd_t = preprocess_observation(camera_info)

                # read arm state (safe to do after camera data is copied)
                current_gripper = read_gripper_state(bot, server)
                print(f"Gripper state (1=closed, 0=open): {current_gripper}")
                end_pose = bot.get_pose()

                # query server
                t0 = time.time()
                json_str = serialize_observation(
                    instructions[0][0][0], rgb_t, pcd_t,
                )
                target_pos, target_quat, target_gripper = query_policy(json_str)
                print(f"Inference time: {time.time() - t0:.3f}s")
                assert target_pos is not None, "Server returned None"
                _gripper_state_str = "CLOSE" if int(target_gripper) == 1 else "OPEN"
                _gripper_cmd_str   = "OPEN"  if int(target_gripper) == 1 else "CLOSE"
                print(f"Predicted next gripper: {_gripper_state_str} "
                      f"(training={int(target_gripper)}) → robot cmd: {_gripper_cmd_str}")

                # quaternion sign convention
                if target_quat[0] < 0:
                    target_quat = -target_quat

                # safety clipping
                target_pos[1] = np.clip(target_pos[1], *Y_RANGE)
                target_pos[2] = np.clip(target_pos[2], *Z_RANGE)

                # convert metres → millimetres
                target_pos_mm = target_pos * 1000.0

                # gripper command (training: 0=open,1=close → robot: 1=open,0=close)
                target_gripper_cmd = 1 - int(target_gripper)

                target_point = Point(target_pos_mm, target_quat, target_gripper_cmd)

                print(f"Pos(mm): {target_pos_mm}  Quat: {target_quat}  "
                      f"Gripper: {target_gripper} → cmd {target_gripper_cmd}")

                vis_pcd_with_end_pred(
                    obs_raw["pcd"], obs_raw["rgb"],
                    end_pose, target_point.position_quaternion_claw,
                )

                decision = prompt_before_action()
                if decision == "retry":
                    print("Retrying current step ...\n")
                    continue
                elif decision == "execute":
                    print("Executing action.\n")
                    break
                elif decision == "quit":
                    raise StopIteration

            # ── execute action ─────────────────────────────
            pos_response = bot.point_control(target_point)
            print(f"pos_response: {pos_response}")
            if pos_response == "Success":
                bot.claws_control(target_gripper_cmd, server.modbusRTU)
            else:
                print("Position control failed — skipping gripper.")

            # ── post-action ────────────────────────────────
            if prompt_after_action() == "quit":
                print("User quit.")
                break

    except StopIteration:
        print("\nTask terminated by user.")
    finally:
        print("Shutting down ...")
        try:
            cameras.stop()
            del cameras
            print("Camera closed.")
        except Exception as e:
            print(f"Camera close error: {e}")
        bot.interrupt_close()
        server.sock.close()
        server.app.close()


# =====================================================================
if __name__ == "__main__":
    _eval()