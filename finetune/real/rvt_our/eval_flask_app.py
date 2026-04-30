"""
BridgeVLA-SAM  Real-Robot Inference Server
===========================================
Flask server that:
  1. Loads the BridgeVLA-SAM model (PaliGemma + SAM3 backbone + MVT head).
  2. Exposes a ``/predict`` HTTP endpoint for real-robot action inference.

Environment variables (optional)::

    PALIGEMMA_PATH         local HuggingFace snapshot of PaliGemma-3b-pt-224
    SAM3_CHECKPOINT_PATH   directory containing ``sam3.pt``

Usage::

    python eval_flask_app.py          # starts on 0.0.0.0:5000
"""

import os
import sys
import json
import base64
import datetime
import traceback

import numpy as np
import torch
from flask import Flask, request, jsonify, Response
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Environment silencers ────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# ── Derive project root from file location ──────────────────────────────────
# <repo>/finetune/real/rvt_our/eval_flask_app.py  →  <repo>
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGEVLA_SAM_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))

# ── Pretrained model paths (PaliGemma & SAM3) ───────────────────────────────
os.environ.setdefault(
    "PALIGEMMA_PATH",
    os.path.join(BRIDGEVLA_SAM_ROOT, "data", "bridgevla_ckpt", "paligemma-3b-pt-224"),
)
os.environ.setdefault(
    "SAM3_CHECKPOINT_PATH",
    os.path.join(BRIDGEVLA_SAM_ROOT, "data", "bridgevla_ckpt", "sam3"),
)

# ── Path setup ───────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `rvt_our` is importable as a package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# BridgeVLA_sam model code (MVT, renderer, augmentation, etc.)
BRIDGEVLA_SAM_FINETUNE = os.path.join(BRIDGEVLA_SAM_ROOT, "finetune")
POINT_RENDERER_DIR = os.path.join(
    BRIDGEVLA_SAM_FINETUNE, "bridgevla", "libs", "point-renderer"
)
SAM3_LIB_DIR = os.path.join(BRIDGEVLA_SAM_ROOT, "libs", "sam3")
YARR_LIB_DIR = os.path.join(
    BRIDGEVLA_SAM_FINETUNE, "bridgevla", "libs", "YARR"
)
for p in [BRIDGEVLA_SAM_FINETUNE, POINT_RENDERER_DIR, SAM3_LIB_DIR, YARR_LIB_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── BridgeVLA imports ────────────────────────────────────────────────────────
import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.config as default_exp_cfg
from bridgevla.mvt.mvt import MVT
from bridgevla.utils.rvt_utils import load_agent as load_agent_state

# Local inference-only agent (no training / yarr dependency)
from rvt_our.models.bridgevla_agent import RVTAgent
from rvt_our.utils.peract_utils import SCENE_BOUNDS, IMAGE_SIZE


# =====================================================================
# 1. Model loading
# =====================================================================
def load_agent(
    model_path: str,
    exp_cfg_path: str = None,
    mvt_cfg_path: str = None,
    device: int = 0,
    load_pretrain: bool = False,
    pretrain_path: str = None,
) -> RVTAgent:
    """
    Instantiate MVT + RVTAgent, load checkpoint, set eval mode.

    Args:
        model_path:    Path to ``model_*.pth`` checkpoint.
        exp_cfg_path:  (optional) experiment config YAML override.
        mvt_cfg_path:  (optional) MVT config YAML override.
        device:        CUDA device ordinal.
        load_pretrain: Whether MVT should load a pre-trained VLM checkpoint.
        pretrain_path: Directory with ``model.safetensors.*`` shards.

    Returns:
        Ready-to-use :class:`RVTAgent` in eval mode.
    """
    device_str = f"cuda:{device}"
    model_folder = os.path.dirname(model_path)

    # ---- experiment config ----
    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path is not None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    old_place_with_mean = exp_cfg.rvt.place_with_mean
    exp_cfg.rvt.place_with_mean = True
    exp_cfg.freeze()

    # ---- MVT config ----
    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path is not None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
    mvt_cfg.freeze()

    # For stage-two, restore the original place_with_mean
    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    # ---- build network ----
    rvt = MVT(
        renderer_device=device_str,
        load_pretrain=load_pretrain,
        pretrain_path=pretrain_path,
        **mvt_cfg,
    )

    # ---- build agent ----
    agent = RVTAgent(
        network=rvt.to(device_str),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=["3rd"],
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=device_str)
    load_agent_state(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent


# =====================================================================
# 2. Request deserialisation
# =====================================================================
def deserialize_data(data: dict, device: str = "cuda:0") -> dict:
    """
    Convert JSON payload → observation dict for ``agent.act_real()``.

    Expected keys in *data*:
        language_goal  (str)
        rgb            (base64-encoded ndarray, shape (1, 3, H, W))
        pcd            (base64-encoded ndarray, shape (1, 3, H, W))

    ``low_dim_state`` is accepted but **ignored** — the new BridgeVLA-SAM
    model does not use proprioception.
    """

    def _b64_to_ndarray(b64_dict):
        raw = base64.b64decode(b64_dict["data"])
        return np.frombuffer(
            raw, dtype=np.dtype(b64_dict["dtype"])
        ).reshape(b64_dict["shape"])

    language_goal = data["language_goal"]
    rgb = torch.from_numpy(_b64_to_ndarray(data["rgb"]).copy()).to(device)
    pcd = torch.from_numpy(_b64_to_ndarray(data["pcd"]).copy()).to(device)

    observation = {
        "language_goal": [[[language_goal]]],
        "3rd": {"rgb": rgb, "pcd": pcd},
    }
    return observation


# =====================================================================
# 3. Flask application
# =====================================================================
app = Flask(__name__)
model: RVTAgent = None
cameras_view = ["3rd"]

# ---- Rendered-view logging ------------------------------------------------
VIEW_LOG_DIR = os.path.join(_THIS_DIR, os.pardir, "logs_real")
os.makedirs(VIEW_LOG_DIR, exist_ok=True)
_session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_predict_step_counter = 0

# ---- Eager model loading (runs at import / startup) ----
print("Starting BridgeVLA-SAM inference server ...")
print("Loading model, please wait ...")

# >>>>>>>  Checkpoint configuration (edit these for your setup)  <<<<<<<<
BASE_PATH     = os.path.join(BRIDGEVLA_SAM_ROOT, "data", "bridgevla_ckpt", "bridgevla_sam", "real_zed_dobot_bs4_lr5e-5_20251011_04_29_20_34")
MODEL_NAME    = "model_80.pth"
DEVICE        = 0
# Set these two if you want to initialise MVT from a pre-trained VLM snapshot
LOAD_PRETRAIN = False
PRETRAIN_PATH = None
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

model_path   = os.path.join(BASE_PATH, MODEL_NAME)
exp_cfg_path = os.path.join(BASE_PATH, "exp_cfg.yaml")
mvt_cfg_path = os.path.join(BASE_PATH, "mvt_cfg.yaml")

try:
    model = load_agent(
        model_path=model_path,
        exp_cfg_path=exp_cfg_path,
        mvt_cfg_path=mvt_cfg_path,
        device=DEVICE,
        load_pretrain=LOAD_PRETRAIN,
        pretrain_path=PRETRAIN_PATH,
    )
    print(f"Model loaded on cuda:{DEVICE} — server ready.")
    print(f"Listening at http://0.0.0.0:5000/predict")
except Exception as e:
    print(f"Model loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# =====================================================================
# 4. Inference endpoint
# =====================================================================
@app.route("/predict", methods=["POST"])
def predict():
    global model
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        observation = deserialize_data(data, device=f"cuda:{DEVICE}")

        with torch.no_grad():
            target_pos, target_quat, target_gripper, views_info = model.act_real(
                observation, cameras_view, return_views=True
            )

        # ---- save rendered three-views to disk ----
        global _predict_step_counter
        if views_info:
            step_dir = os.path.join(
                VIEW_LOG_DIR, _session_timestamp,
                f"step_{_predict_step_counter:04d}",
            )
            for stage_name, stage_data in views_info.items():
                stage_dir = os.path.join(step_dir, stage_name)
                os.makedirs(stage_dir, exist_ok=True)
                for i, orig in enumerate(stage_data["originals"]):
                    Image.fromarray(orig).save(
                        os.path.join(stage_dir, f"original_{i}.png"))
                for i, ov in enumerate(stage_data["overlays"]):
                    Image.fromarray(ov).save(
                        os.path.join(stage_dir, f"overlay_{i}.png"))
                for i, p_l in enumerate(stage_data.get("logits", [])):
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    im = ax.imshow(p_l, cmap="jet", vmin=-10.0, vmax=40.0,
                                   interpolation="nearest")
                    ax.set_title(f"pred view {i}\n"
                                 f"min={p_l.min():.3f}  max={p_l.max():.3f}")
                    ax.set_axis_off()
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(os.path.join(stage_dir, f"logits_{i}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
            meta_lines = [
                f"step: {_predict_step_counter}",
                f"instruction: {observation['language_goal'][0][0][0]}",
                f"target_pos: {target_pos.tolist()}",
                f"target_quat: {target_quat.tolist()}",
                f"target_gripper (training 0=open,1=close): {target_gripper.tolist()[0]}",
                f"robot_cmd (1=open,0=close): {1 - int(target_gripper.tolist()[0])}",
            ]
            with open(os.path.join(step_dir, "meta.txt"), "w") as f:
                f.write("\n".join(meta_lines) + "\n")
            print(f"[view_log] saved views to {step_dir}")
        _predict_step_counter += 1

        result = {
            "target_pos":     target_pos.tolist(),
            "target_quat":    target_quat.tolist(),
            "target_gripper": target_gripper.tolist(),
        }
        print("result:", result)
        return Response(json.dumps(result), mimetype="application/json")

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "details": traceback.format_exc(),
        }), 500


# =====================================================================
# 5. Entry point
# =====================================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )