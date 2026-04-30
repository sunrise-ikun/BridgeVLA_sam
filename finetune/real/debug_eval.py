"""Offline debug evaluation for a trained BridgeVLA-SAM real-robot checkpoint.

Each run of this script:

  1. Loads the configs next to the checkpoint (exp_cfg.yaml + mvt_cfg.yaml),
     falls back to ``real/configs/*.yaml`` if they are missing.
  2. Builds MVT + RVTAgent on a single GPU (no DDP) and loads the checkpoint
     state dict.
  3. Randomly picks ``--num_episodes`` episodes from the dataset.
  4. For every step in each picked episode, runs the eval-mode forward and
     saves pred + GT heatmap visualizations for **both** MVT stages
     (``mvt1_pred / mvt1_gt / mvt2_pred / mvt2_gt``).

Output layout::

    {output_dir}/
        ep_00_{episode_name}/
            step_00_idx{N}/
                mvt1_pred/{original,gray,overlay,logits}_{0,1,2}.png
                mvt1_gt/...
                mvt2_pred/...
                mvt2_gt/...
                meta.txt
            step_01_idx{N}/ ...
        ep_01_{episode_name}/ ...
        run_meta.txt

By default the output dir is ``{ckpt_dir}/debug_eval/{ckpt_name}_{MM_DD_HH_MM}``.

Launch::

    bash finetune/real/debug_eval.sh
    # or directly:
    bash finetune/real/debug_eval.sh \
        --checkpoint /path/to/model_X.pth \
        --num_episodes 3
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import defaultdict
from typing import List

import numpy as np
import torch

# Make `finetune/` importable (this file lives in finetune/real/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FINETUNE_DIR = os.path.dirname(_THIS_DIR)
if _FINETUNE_DIR not in sys.path:
    sys.path.insert(0, _FINETUNE_DIR)

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import bridgevla.config as exp_cfg_mod                      # noqa: E402
import bridgevla.models.bridgevla_agent as bridgevla_agent  # noqa: E402
import bridgevla.mvt.config as mvt_cfg_mod                  # noqa: E402
from bridgevla.mvt.mvt import MVT                           # noqa: E402
from bridgevla.utils.rvt_utils import get_num_feat          # noqa: E402

from real.real_dataset import Real_Dataset                  # noqa: E402
from real.utils.peract_utils import (                       # noqa: E402
    CAMERAS_REAL,
    IMAGE_SIZE,
    SCENE_BOUNDS_REAL,
)
from real.visualize import visualize_samples                # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_cfg_path(cli_path: str, ckpt_dir: str, fallback_name: str) -> str:
    """Prefer ``{ckpt_dir}/{fallback_name}`` (produced by train.py), fall back
    to ``real/configs/{fallback_name}`` shipped with the repo. CLI override
    always wins."""
    if cli_path:
        return cli_path
    beside_ckpt = os.path.join(ckpt_dir, fallback_name)
    if os.path.isfile(beside_ckpt):
        return beside_ckpt
    return os.path.join(_THIS_DIR, "configs", fallback_name)


def _load_ckpt_into(backbone: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    # Strip DDP's "module." prefix if present.
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"[debug_eval] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    if missing:
        print(f"  missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"  unexpected[:5]: {unexpected[:5]}")


def build_agent(ckpt_path: str, exp_cfg_path: str, mvt_cfg_path: str,
                device: str):
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    exp_cfg.merge_from_file(exp_cfg_path)
    exp_cfg.freeze()

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    mvt_cfg.merge_from_file(mvt_cfg_path)
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    # load_pretrain=False: we'll overwrite with the trained checkpoint below,
    # which already contains the PaliGemma/SAM3 weights we need.
    backbone = MVT(
        renderer_device=device,
        load_pretrain=False,
        pretrain_path=None,
        **mvt_cfg,
    ).to(device)
    _load_ckpt_into(backbone, ckpt_path)

    # NB: no DDP wrapping — this is single-process inference.
    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS_REAL,
        cameras=CAMERAS_REAL,
        log_dir="/tmp/debug_eval_logs",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 300)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )
    agent.build(training=False, device=device)
    agent.eval()
    return agent, exp_cfg, mvt_cfg


def group_samples_by_episode(dataset: Real_Dataset) -> dict:
    """Group dataset indices by their episode path, preserving step order."""
    by_ep: defaultdict = defaultdict(list)
    for idx, entry in enumerate(dataset.index):
        by_ep[entry["episode_path"]].append((idx, entry["step_idx"]))
    for ep_path in by_ep:
        by_ep[ep_path].sort(key=lambda t: t[1])  # sort by step index
    return dict(by_ep)


def group_episodes_by_task(by_ep: dict) -> dict:
    """Further group episode paths by task_group (parent-directory name)."""
    by_task: defaultdict = defaultdict(list)
    for ep_path in by_ep:
        task_group = os.path.basename(os.path.dirname(ep_path))
        by_task[task_group].append(ep_path)
    return dict(by_task)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default="/DATA/disk1/zyz/projects/BridgeVLA_sam/data/bridgevla_data/logs_real/train/"
                "real_zed_dobot_bs4_lr5e-5_20251011_04_29_19_14/model_25.pth",
        help="Path to a model_X.pth produced by real/train.py.",
    )
    parser.add_argument(
        "--data_folder", type=str,
        default="/DATA/disk1/zyz/projects/BridgeVLA_sam/"
                "data/bridgevla_data/Real",
    )
    parser.add_argument("--num_episodes", type=int, default=2,
                        help="Number of episodes to sample **per task**. "
                             "Total episodes = num_tasks × num_episodes.")
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Where to write episode directories. Defaults to "
             "{ckpt_dir}/debug_eval/{ckpt_name}_{MM_DD_HH_MM}.",
    )
    parser.add_argument(
        "--exp_cfg_path", type=str, default="",
        help="Optional override. Otherwise looks for {ckpt_dir}/exp_cfg.yaml, "
             "then falls back to real/configs/real_config.yaml.",
    )
    parser.add_argument(
        "--mvt_cfg_path", type=str, default="",
        help="Optional override. Otherwise looks for {ckpt_dir}/mvt_cfg.yaml, "
             "then falls back to real/configs/mvt_cfg.yaml.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for episode picking (None = nondeterministic).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--stages", type=str, nargs="+", default=["mvt1", "mvt2"],
        choices=["mvt1", "mvt2"],
        help="Which MVT stages to visualize. Default is both.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Forward-pass batch size. PaliGemma bf16 FPEs on H20 when bs<4; "
             "don't set this below 4.",
    )
    args = parser.parse_args()

    # --- resolve paths ---
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    if not args.output_dir:
        stamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        args.output_dir = os.path.join(ckpt_dir, "debug_eval",
                                        f"{ckpt_name}_{stamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    exp_cfg_path = _resolve_cfg_path(args.exp_cfg_path, ckpt_dir, "exp_cfg.yaml")
    mvt_cfg_path = _resolve_cfg_path(args.mvt_cfg_path, ckpt_dir, "mvt_cfg.yaml")

    print(f"[debug_eval] checkpoint:  {ckpt_path}")
    print(f"[debug_eval] exp_cfg:     {exp_cfg_path}")
    print(f"[debug_eval] mvt_cfg:     {mvt_cfg_path}")
    print(f"[debug_eval] output_dir:  {args.output_dir}")
    print(f"[debug_eval] data_folder: {args.data_folder}")
    print(f"[debug_eval] num_episodes={args.num_episodes} per task, stages={args.stages}, seed={args.seed}")

    torch.cuda.set_device(args.device)

    # --- agent + dataset ---
    agent, _exp_cfg, _mvt_cfg = build_agent(
        ckpt_path, exp_cfg_path, mvt_cfg_path, args.device,
    )

    tasks_filter = getattr(_exp_cfg, "tasks", "all")
    dataset = Real_Dataset(args.data_folder, cameras=CAMERAS_REAL,
                           tasks=tasks_filter, verbose=True)

    # --- pick episodes: num_episodes per task ---
    by_ep = group_samples_by_episode(dataset)
    by_task = group_episodes_by_task(by_ep)
    assert len(by_task) > 0, "dataset has no episodes"
    rng = np.random.default_rng(args.seed)

    # chosen: list of (task_group, ep_path)
    chosen = []
    for task_group in sorted(by_task.keys()):
        eps = sorted(by_task[task_group])
        n_pick_task = min(args.num_episodes, len(eps))
        idxs = rng.choice(len(eps), size=n_pick_task, replace=False).tolist()
        for i in idxs:
            chosen.append((task_group, eps[i]))
    n_pick = len(chosen)

    run_meta_path = os.path.join(args.output_dir, "run_meta.txt")
    with open(run_meta_path, "w") as f:
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"data_folder: {args.data_folder}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_episodes_per_task: {args.num_episodes}\n")
        f.write(f"num_tasks: {len(by_task)}\n")
        f.write(f"num_episodes_total: {n_pick}\n")
        f.write(f"stages: {args.stages}\n")
        f.write("episodes:\n")
        for tg, ep in chosen:
            f.write(f"  - [{tg}] {ep}\n")

    # --- per-episode visualization ---
    for ep_i, (task_group, ep_path) in enumerate(chosen):
        ep_name = os.path.basename(ep_path)
        ep_dir = os.path.join(args.output_dir, f"ep_{ep_i:02d}_{task_group}_{ep_name}")
        os.makedirs(ep_dir, exist_ok=True)

        step_pairs: List = by_ep[ep_path]  # list of (dataset_idx, step_idx)
        print(f"[debug_eval] episode {ep_i+1}/{n_pick}: {ep_name} "
              f"({len(step_pairs)} step(s))", flush=True)

        # Run all steps of this episode in mini-batches of args.batch_size so
        # we amortize the PaliGemma forward cost and stay above MIN_FORWARD_BS.
        bs = max(1, int(args.batch_size))
        for start in range(0, len(step_pairs), bs):
            chunk = step_pairs[start:start + bs]
            samples = [dataset[int(dsi)] for (dsi, _) in chunk]
            save_dirs = [
                os.path.join(ep_dir, f"step_{chunk[k][1]:02d}_idx{int(chunk[k][0])}")
                for k in range(len(chunk))
            ]
            extra_meta = [
                {"dataset_idx": int(chunk[k][0]),
                 "episode": ep_name,
                 "step_idx_in_episode": int(chunk[k][1])}
                for k in range(len(chunk))
            ]
            visualize_samples(
                agent, samples, save_dirs,
                stages=args.stages,
                cameras=CAMERAS_REAL,
                extra_meta=extra_meta,
            )

    print(f"[debug_eval] done. Output under: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
