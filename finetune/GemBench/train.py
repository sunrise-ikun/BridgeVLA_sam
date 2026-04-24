'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/train.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import subprocess
import time
import tqdm
import yaml
import argparse
from contextlib import redirect_stdout
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import swanlab
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from bridgevla.models.bridgevla_agent import print_loss_log
from bridgevla.utils.rvt_utils import (
    get_num_feat,
)
from utils.peract_utils_gembench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from gembench_dataset import Gembench_Dataset

USE_SWANLAB = False


def create_dataloader(dataset, rank, world_size, batch_size, num_workers):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=True
    )
    return dataloader


def train(agent, data_loader, epoch, cameras=["front", "left_shoulder", "right_shoulder", "wrist"], rank=0, debug_dir=None):
    agent.train()

    def move_tensors_to_device(d, device):
        if isinstance(d, dict):
            return {k: move_tensors_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()}
        return d

    iteration = 0
    epoch_losses = {}
    for raw_batch in tqdm.tqdm(data_loader, disable=(rank != 0), position=0, leave=True):
        iteration += 1
        batch = move_tensors_to_device(raw_batch, agent._device)
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = [[[item]] for item in raw_batch["lang_goal"]]
        update_args = {
            "cameras": cameras,
            "replay_sample": batch,
            "backprop": True,
            "reset_log": (iteration == 0),
        }
        if debug_dir is not None and iteration <= 100:
            update_args["debug_dir"] = debug_dir
            update_args["debug_step"] = epoch * len(data_loader) + iteration
        out = agent.update_gembench(**update_args)

        if epoch_losses == {}:
            epoch_losses = {key: [] for key in out.keys()}
        for key in epoch_losses:
            epoch_losses[key].append(out[key])

        step = epoch * len(data_loader) + iteration
        if rank == 0 and USE_SWANLAB and step % 10 == 0:
            swanlab.log(out, step=step)

    if rank == 0:
        print_loss_log(agent)

    avg_losses = {key: sum(values) / len(values) for key, values in epoch_losses.items()}
    return avg_losses


def save_agent(agent, path, epoch):
    model = agent._network

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
        },
        path,
    )


def get_time():
    import datetime
    now = datetime.datetime.now()
    return f"{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"


def build_run_name(exp_cfg, world_size):
    return f"{exp_cfg.swanlab_run}_{get_time()}"


def get_logdir(cmd_args, exp_cfg, dist, run_name):
    root = cmd_args.log_dir if cmd_args.log_dir else exp_cfg.log_dir
    log_dir = os.path.join(root, "train_gembench", run_name)
    if cmd_args.debug:
        log_dir = os.path.join(log_dir, "debug")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.

    Supports three launch modes (auto-detected in order):
      1. SLURM  — srun sets SLURM_JOB_ID / SLURM_PROCID / SLURM_NTASKS
      2. torchrun / MLP / PAI — sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR
      3. DEBUG  — single-GPU fallback when DEBUG=true
    """
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
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("Cannot find RANK and WORLD_SIZE — entering single-GPU debug mode")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "9001")
        os.environ.setdefault("LOCAL_RANK", "0")
        rank = 0
        world_size = 1
    else:
        raise RuntimeError(
            "Distributed env vars not found. "
            "Launch with torchrun / srun, or set DEBUG=true for single-GPU mode."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


def experiment(cmd_args):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    ddp = int(os.environ['WORLD_SIZE']) > 1
    if dist.get_rank() == 0:
        print(f"Total devices: {dist.get_world_size()}")
        if ddp:
            print(f"Running DDP (world_size={dist.get_world_size()}).")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"

    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    if local_rank == 0:
        print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    EPOCHS = exp_cfg.epochs
    FREEZE_EPOCHS = int(getattr(exp_cfg, "freeze_epochs", 2))

    # Unified run name: built on rank 0 and broadcast so all ranks use the same directory.
    if dist.get_rank() == 0:
        run_name = build_run_name(exp_cfg, dist.get_world_size())
        run_name_list = [run_name]
    else:
        run_name_list = [None]
    dist.broadcast_object_list(run_name_list, src=0)
    run_name = run_name_list[0]

    log_dir = get_logdir(cmd_args, exp_cfg, dist, run_name)

    t_start = time.time()
    train_dataset = Gembench_Dataset(
        cmd_args.data_folder,
        device=device_id,
        cameras=cmd_args.cameras,
        ep_per_task=cmd_args.ep_per_task,
        tasks=cmd_args.tasks,
    )
    if local_rank == 0:
        print("Total tasks:", train_dataset.num_tasks)
        print("Total trajectories:", train_dataset.num_task_paths)
        print("Dataset Length:", len(train_dataset))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_dataloader = create_dataloader(train_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers)

    t_end = time.time()
    if local_rank == 0:
        print("Created Dataset. Time Cost: {:.1f} minutes".format((t_end - t_start) / 60.0))

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    )
    backbone = backbone.to(local_rank)
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 1000)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    # ---- Stage 1 freeze: PaliGemma frozen, SAM3 frozen (in its __init__) ----
    always_freeze = ["lm_head", "embed_tokens"]
    if cmd_args.freeze_vision_tower:
        always_freeze.append("vision_tower")
        if dist.get_rank() == 0:
            print("Freeze vision tower")

    for name, param in agent._network.named_parameters():
        if "mvt1.model" in name:
            param.requires_grad = False
    if dist.get_rank() == 0:
        print(f"[Stage 1] Frozen PaliGemma + SAM3. Training fusion/up0/heads for {FREEZE_EPOCHS} epochs.")

    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    if dist.get_rank() == 0:
        print(f'Total trainable parameters: {total_params / 1e9:.2f} billion')

    agent.build(training=True, device=device_id)

    start_epoch = 0
    end_epoch = EPOCHS

    if dist.get_rank() == 0:
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()

    # Initialize Logging =>> SwanLab
    global USE_SWANLAB
    if dist.get_rank() == 0:
        swanlab_project = exp_cfg.swanlab_project
        try:
            swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY", ""))
            swanlab.init(
                project=swanlab_project,
                experiment_name=run_name,
                mode="disabled" if cmd_args.debug else "cloud",
            )
            USE_SWANLAB = True
            print(f"[Info] SwanLab enabled, project={swanlab_project}, run={run_name}")
        except Exception as e:
            print(f"[Info] SwanLab init failed ({e}), training continues without SwanLab")

    if dist.get_rank() == 0:
        print("Start training ...", flush=True)

    debug_vis_dir = None
    if cmd_args.debug and dist.get_rank() == 0:
        debug_vis_dir = os.path.join(log_dir, "vis")

    i = start_epoch
    while True:
        if i == end_epoch:
            break

        # ---- Stage 2 transition: unfreeze PaliGemma backbone ----
        if i == FREEZE_EPOCHS:
            for name, param in agent._network.named_parameters():
                if "mvt1.model" in name:
                    if any(af in name for af in always_freeze):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            agent.rebuild_optimizer()
            total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
            if dist.get_rank() == 0:
                print(f"[Stage 2] Unfroze PaliGemma. Trainable params: {total_params / 1e9:.2f}B")

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")

        train_dataloader.sampler.set_epoch(i)
        _epoch_debug_dir = None
        if debug_vis_dir is not None:
            _epoch_debug_dir = os.path.join(debug_vis_dir, f"epoch_{i}")
        out = train(agent, train_dataloader, epoch=i, rank=dist.get_rank(), cameras=cmd_args.cameras, debug_dir=_epoch_debug_dir)

        if dist.get_rank() == 0 and USE_SWANLAB:
            swanlab.log({f"epoch_{k}": v for k, v in out.items()}, step=i)

        save_every = int(getattr(exp_cfg, "save_every_n_epochs", 20))
        is_periodic_save = save_every > 0 and i > 0 and (i % save_every == 0)
        is_final_save = save_every > 0 and i == end_epoch - 1
        if dist.get_rank() == 0 and (is_periodic_save or is_final_save):
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)

        i += 1
        dist.barrier()

    dist.barrier()
    if dist.get_rank() == 0:
        print("[Finish]")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/gembench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--data_folder", type=str, default="/PATH_TO_TRAIN_DATA/train_dataset")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ep_per_task", type=int, default=10000)
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of task names to train on. If None, use all tasks."
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["left_shoulder", "right_shoulder", "wrist", "front"],
        help="List of camera names"
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)
