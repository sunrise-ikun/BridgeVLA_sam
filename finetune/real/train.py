"""BridgeVLA-SAM real-robot offline trainer (Dobot + ZED, 20251011 dataset).

Loosely modeled on finetune/RLBench/train.py, but:
  * no replay buffer — we stream samples from `Real_Dataset` via DataLoader
  * uses `RVTAgent.update_gembench()` (dict-style replay_sample format)
  * passes `cameras=["3rd"]` to the agent and `SCENE_BOUNDS_REAL` at construction
  * the real dataset is small, so epochs/bs/lr are tuned accordingly in the yaml

Launch (single GPU debug):
    DEBUG=true python finetune/real/train.py \
        --exp_cfg_path finetune/real/configs/real_config.yaml \
        --mvt_cfg_path finetune/real/configs/mvt_cfg.yaml

Launch (multi-GPU):
    torchrun --nproc_per_node=4 finetune/real/train.py ...
"""

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.distributed as dist
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from real.visualize import visualize_epoch                  # noqa: E402

try:
    import swanlab  # noqa: F401
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

USE_SWANLAB = False


# ------------------------------------------------------------------
# distributed helpers
# ------------------------------------------------------------------

def setup_distributed(backend: str = "nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % max(num_gpus, 1))
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        pass  # set by torchrun
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("[real/train] No distributed env vars — entering single-GPU debug mode.")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # random free port to avoid clashing with concurrent debug runs
        import random as _rnd
        os.environ.setdefault("MASTER_PORT", str(_rnd.randint(29600, 29999)))
        os.environ.setdefault("LOCAL_RANK", "0")
    else:
        raise RuntimeError(
            "Distributed env vars not found. Launch with torchrun / srun, "
            "or set DEBUG=true for single-GPU mode."
        )
    dist.init_process_group(
        backend=backend,
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )


def reduce_mean(value) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return float(value)
    t = torch.tensor(float(value), device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


# ------------------------------------------------------------------
# misc helpers
# ------------------------------------------------------------------

def get_time_folder() -> str:
    import datetime
    now = datetime.datetime.now()
    return f"{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"


def get_logdir(cmd_args, exp_cfg) -> str:
    root = cmd_args.log_dir or exp_cfg.log_dir
    run_name = f"{exp_cfg.swanlab_run}_{get_time_folder()}"
    log_dir = os.path.join(root, "train", run_name)
    if cmd_args.debug:
        log_dir = os.path.join(log_dir, "debug")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    # stash the run name as-is so swanlab uses it even when debug appends a sub-dir
    setattr(get_logdir, "last_run_name", run_name)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir: str):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as f:
        with redirect_stdout(f):
            print(exp_cfg.dump())
    with open(f"{log_dir}/mvt_cfg.yaml", "w") as f:
        with redirect_stdout(f):
            print(mvt_cfg.dump())
    with open(f"{log_dir}/args.yaml", "w") as f:
        yaml.dump(cmd_args.__dict__, f)


def save_agent(agent, path: str, epoch: int):
    model = agent._network
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    torch.save({"epoch": epoch, "model_state": model_state}, path)


# ------------------------------------------------------------------
# data loading
# ------------------------------------------------------------------

def build_dataloader(dataset, rank: int, world_size: int,
                     batch_size: int, num_workers: int):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
    )
    return loader, sampler


# ------------------------------------------------------------------
# training
# ------------------------------------------------------------------

def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def train_one_epoch(agent, loader, cameras, rank, max_iter=None,
                    epoch_idx=0, global_step_base=0):
    agent.train()
    epoch_losses = defaultdict(list)

    pbar = tqdm.tqdm(loader, disable=(rank != 0), position=0, leave=True)
    for step_idx, raw_batch in enumerate(pbar):
        if max_iter is not None and step_idx >= max_iter:
            break
        batch = move_batch_to_device(raw_batch, agent._device)
        # Wrap language strings as [[[text]]] so mvt_single's `text[0][0]` works.
        batch["lang_goal"] = [[[item]] for item in raw_batch["lang_goal"]]
        # Keep task names as plain list.
        batch["tasks"] = list(raw_batch["tasks"])

        out = agent.update_gembench(
            replay_sample=batch,
            backprop=True,
            reset_log=(step_idx == 0),
            cameras=cameras,
        )
        for k, v in out.items():
            epoch_losses[k].append(reduce_mean(v))

        if rank == 0:
            if USE_SWANLAB:
                swanlab.log(out, step=global_step_base + step_idx)
            if step_idx % 10 == 0:
                pbar.set_postfix(
                    loss=f"{out.get('total_loss', 0.0):.3f}",
                    lr=f"{out.get('lr', 0.0):.2e}",
                )

    return {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}


# ------------------------------------------------------------------
# experiment
# ------------------------------------------------------------------

def experiment(cmd_args):
    setup_distributed()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

    # ---- configs ----
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path:
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts:
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    epochs = cmd_args.epochs if cmd_args.epochs is not None else exp_cfg.epochs

    if cmd_args.exp_cfg_opts:
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts:
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    exp_cfg.freeze()

    BATCH_SIZE = exp_cfg.bs
    if rank == 0:
        print(f"[real/train] world_size={world_size}, bs_per_gpu={BATCH_SIZE}, epochs={epochs}")

    # ---- log dir ----
    if rank == 0:
        log_dir = get_logdir(cmd_args, exp_cfg)
        log_dir_list = [log_dir]
    else:
        log_dir_list = [None]
    dist.broadcast_object_list(log_dir_list, src=0)
    log_dir = log_dir_list[0]

    # ---- dataset ----
    t0 = time.time()
    dataset = Real_Dataset(
        data_path=cmd_args.data_folder,
        cameras=CAMERAS_REAL,
        verbose=(rank == 0),
    )
    if rank == 0:
        print(f"[real/train] dataset: {len(dataset)} samples, "
              f"{dataset.num_task_paths} episodes")
    loader, sampler = build_dataloader(
        dataset, rank, world_size, BATCH_SIZE, exp_cfg.num_workers
    )
    if rank == 0:
        print(f"[real/train] dataloader built in {time.time() - t0:.1f}s")

    # ---- mvt + agent ----
    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path:
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts:
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    ).to(device_id)
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS_REAL,
        cameras=CAMERAS_REAL,
        log_dir=f"{log_dir}/test_run/",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 300)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    # ---- Stage 1 freeze: PaliGemma frozen (SAM3 frozen in __init__) ----
    always_freeze = ["lm_head", "embed_tokens"]
    FREEZE_EPOCHS = int(getattr(exp_cfg, "freeze_epochs", 4))
    for name, param in agent._network.named_parameters():
        if "mvt1.model" in name:
            param.requires_grad = False
    if rank == 0:
        print(f"[real/train] Stage 1: PaliGemma + SAM3 frozen for {FREEZE_EPOCHS} epochs "
              "— training fusion/up0/heads.")
    trainable = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[real/train] trainable params: {trainable / 1e9:.3f} B")

    agent.build(training=True, device=device_id)

    # ---- dump cfgs (after construction so mvt_cfg is finalized) ----
    if rank == 0:
        exp_cfg.defrost()
        t_lr = exp_cfg.peract.lr
        t_id = exp_cfg.exp_id
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = t_lr
        exp_cfg.exp_id = t_id
        exp_cfg.freeze()

    # ---- swanlab (mirrors RLBench/train.py; debug uses mode="disabled") ----
    global USE_SWANLAB
    swanlab_run_name = getattr(get_logdir, "last_run_name", os.path.basename(log_dir))
    if rank == 0 and HAS_SWANLAB:
        try:
            swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY", ""))
            swanlab.init(
                project=exp_cfg.swanlab_project,
                experiment_name=swanlab_run_name,
                mode="disabled" if cmd_args.debug else "cloud",
            )
            USE_SWANLAB = True
            mode = "disabled" if cmd_args.debug else "cloud"
            print(f"[real/train] SwanLab init OK, project={exp_cfg.swanlab_project}, "
                  f"run={swanlab_run_name}, mode={mode}")
        except Exception as e:
            print(f"[real/train] SwanLab init failed ({e}); continuing without.")

    # ---- training loop ----
    if rank == 0:
        print(f"[real/train] begin training ({epochs} epochs)", flush=True)

    save_every = int(getattr(exp_cfg, "save_every_n_epochs", 5))
    iters_per_epoch = max(1, len(loader))  # for computing the global step base

    def run_viz(epoch_idx: int) -> None:
        """End-of-epoch / pre-epoch visualization on rank 0 only.

        Bypasses DDP (uses agent._net_mod under the hood) so non-zero ranks
        don't need to participate. We follow it with a dist.barrier() in the
        caller to re-sync before the next training step.
        """
        if rank != 0 or not cmd_args.visualize or cmd_args.viz_per_epoch <= 0:
            return
        try:
            visualize_epoch(
                agent, dataset, epoch=epoch_idx, log_dir=log_dir,
                num_samples=cmd_args.viz_per_epoch,
                cameras=CAMERAS_REAL,
                seed=epoch_idx,
            )
        except Exception as e:
            print(f"[real/train] visualize_epoch failed at epoch {epoch_idx}: {e}",
                  flush=True)

    for epoch in range(epochs):
        # Pre-epoch visualization: runs at the *start* of every epoch, so the
        # very first viz (epoch=0) shows the model state BEFORE any training
        # steps have run — a useful baseline to track how the heatmap tightens
        # over time.
        run_viz(epoch)
        dist.barrier()

        # Stage 2 transition: unfreeze PaliGemma.
        if epoch == FREEZE_EPOCHS:
            for name, param in agent._network.named_parameters():
                if "mvt1.model" in name:
                    param.requires_grad = not any(af in name for af in always_freeze)
            agent.rebuild_optimizer()
            trainable = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
            if rank == 0:
                print(f"[real/train] Stage 2 @ epoch {epoch}: PaliGemma unfrozen. "
                      f"Trainable params: {trainable/1e9:.3f} B")

        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"[real/train] epoch {epoch}/{epochs}", flush=True)
        losses = train_one_epoch(
            agent, loader, cameras=CAMERAS_REAL, rank=rank,
            max_iter=cmd_args.max_iter, epoch_idx=epoch,
            global_step_base=epoch * iters_per_epoch,
        )

        if rank == 0:
            loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
            print(f"[real/train] epoch {epoch} done — {loss_str}", flush=True)
            if USE_SWANLAB:
                swanlab.log(
                    {f"epoch/{k}": v for k, v in losses.items()},
                    step=(epoch + 1) * iters_per_epoch - 1,
                )

        is_periodic = save_every > 0 and epoch > 0 and (epoch % save_every == 0)
        is_final = epoch == epochs - 1
        if rank == 0 and (is_periodic or is_final):
            save_agent(agent, f"{log_dir}/model_{epoch}.pth", epoch)
            save_agent(agent, f"{log_dir}/model_last.pth", epoch)
            print(f"[real/train] saved checkpoint at epoch {epoch}", flush=True)

        dist.barrier()

    if rank == 0:
        print("[real/train] done.", flush=True)
    dist.destroy_process_group()


# ------------------------------------------------------------------
# entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_cfg_path", type=str,
                        default="real/configs/real_config.yaml")
    parser.add_argument("--mvt_cfg_path", type=str,
                        default="real/configs/mvt_cfg.yaml")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--data_folder", type=str, nargs="+",
                        default=["/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/"
                                 "data/bridgevla_data/Real/20251011"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Cap iterations per epoch (for smoke-testing).")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true",
                        help="Load BridgeVLA pretrain weights into MVT.")
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="Path to pretrain checkpoint (.pth or .safetensors dir).")
    # Visualization master switch. Default ON so a fresh `bash train.sh` also
    # dumps a pre-training baseline at epoch 0. `--no-visualize` disables it.
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument("--visualize", dest="visualize",
                           action="store_true", default=True,
                           help="Enable start-of-epoch visualization (default).")
    viz_group.add_argument("--no-visualize", dest="visualize",
                           action="store_false",
                           help="Disable start-of-epoch visualization.")
    parser.add_argument("--viz_per_epoch", type=int, default=2,
                        help="Number of random training samples to visualize at "
                             "the START of each epoch (rank 0 only). Epoch 0's "
                             "viz captures the pre-training baseline. Ignored "
                             "when --no-visualize is set.")
    cmd_args = parser.parse_args()
    experiment(cmd_args)
