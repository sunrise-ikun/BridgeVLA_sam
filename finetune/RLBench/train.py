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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/train.py
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
import time
from collections import defaultdict
from contextlib import redirect_stdout
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from utils.get_dataset import get_dataset
from bridgevla.utils.rvt_utils import (
    get_num_feat,
    RLBENCH_TASKS,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    TRAIN_REPLAY_STORAGE_DIR,
)

USE_WANDB = False

def train(agent, dataset, training_iterations,epoch,rank=0):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)
        dist.barrier()
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args={
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
            }
        out=agent.update(**update_args)
        dist.barrier()
        if rank == 0:
            step=epoch*training_iterations+iteration
            if USE_WANDB:
                wandb.log(out, step=step)
            if step % 100 == 0:
                print(f"  step {step}, loss keys: {list(out.keys())}", flush=True)
    return log

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



def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks



def get_time():
    import datetime
    now = datetime.datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    #  'MM-DD-HH-MM'
    folder_name = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return folder_name


def build_run_name(exp_cfg, world_size):
    """Final run/folder name: f"{wandb_run}_{MM_DD_HH_MM}"."""
    return f"{exp_cfg.wandb_run}_{get_time()}"


def get_logdir(cmd_args, exp_cfg, dist, run_name):
    # cmd_args.log_dir overrides cfg if explicitly provided
    root = cmd_args.log_dir if cmd_args.log_dir else exp_cfg.log_dir
    log_dir = os.path.join(root, "train", run_name)
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
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
        print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    NUM_TRAIN = 100
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * dist.get_world_size()))

    if exp_cfg.epochs!=cmd_args.epochs and dist.get_rank() == 0:
        print(f"cmd args epochs != exp cfg epochs You are using {cmd_args.epochs}")
    EPOCHS = cmd_args.epochs

    data_folder=DATA_FOLDER
    # Unified run name: shared by the log folder and the W&B run.
    # Built on rank 0 and broadcast so all ranks land in the same directory.
    if dist.get_rank() == 0:
        run_name = build_run_name(exp_cfg, dist.get_world_size())
        run_name_list = [run_name]
    else:
        run_name_list = [None]
    dist.broadcast_object_list(run_name_list, src=0)
    run_name = run_name_list[0]
    log_dir = get_logdir(cmd_args, exp_cfg, dist, run_name)
    tasks = get_tasks(exp_cfg)
    if dist.get_rank() == 0:
        print("Training on {} tasks: {}".format(len(tasks), tasks))
    t_start = time.time()

    # Replay buffer build: rank 0 builds (with optional refresh), others poll a
    # flag file until rank 0 finishes.  This avoids NCCL barrier timeouts when
    # building takes a long time.
    os.makedirs(TRAIN_REPLAY_STORAGE_DIR, exist_ok=True)
    replay_flag = os.path.join(TRAIN_REPLAY_STORAGE_DIR, ".replay_ready")
    if dist.get_rank() == 0:
        if os.path.exists(replay_flag):
            os.remove(replay_flag)
        get_dataset(
            tasks, BATCH_SIZE_TRAIN, None, TRAIN_REPLAY_STORAGE_DIR, None,
            data_folder, NUM_TRAIN, None,
            cmd_args.refresh_replay, device_id,
            num_workers=exp_cfg.num_workers, only_train=True,
            sample_distribution_mode=exp_cfg.sample_distribution_mode,
        )
        with open(replay_flag, "w") as f:
            f.write("ready")
        print("Rank 0: replay buffer ready.", flush=True)
    else:
        print(f"Rank {dist.get_rank()}: waiting for rank 0 to build replay ...", flush=True)
        while not os.path.exists(replay_flag):
            time.sleep(5)
        print(f"Rank {dist.get_rank()}: replay flag detected, loading ...", flush=True)

    # All non-zero ranks load from the completed replay on disk.
    if dist.get_rank() != 0:
        train_dataset, _ = get_dataset(
            tasks, BATCH_SIZE_TRAIN, None, TRAIN_REPLAY_STORAGE_DIR, None,
            data_folder, NUM_TRAIN, None,
            False, device_id,
            num_workers=exp_cfg.num_workers, only_train=True,
            sample_distribution_mode=exp_cfg.sample_distribution_mode,
        )
    else:
        # Rank 0 already has train_dataset from the build call above.
        # Re-load to get a fresh iterator (the build call consumed it).
        train_dataset, _ = get_dataset(
            tasks, BATCH_SIZE_TRAIN, None, TRAIN_REPLAY_STORAGE_DIR, None,
            data_folder, NUM_TRAIN, None,
            False, device_id,
            num_workers=exp_cfg.num_workers, only_train=True,
            sample_distribution_mode=exp_cfg.sample_distribution_mode,
        )

    dist.barrier()
    t_end = time.time()
    if local_rank== 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    # for maintaining backward compatibility
    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    )
    backbone=backbone.to(local_rank)
    # if ddp:
    backbone = DDP(backbone, device_ids=[local_rank],find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    freeze_names=["lm_head","embed_tokens"]
    if cmd_args.freeze_vision_tower:
        freeze_names.append("vision_tower")
        if dist.get_rank() == 0:
            print("Freeze vision tower")

    for name, module in agent._network.named_modules():
        for freeze_name in freeze_names:
            if freeze_name in name:
                for param in module.parameters():
                    param.requires_grad = False
                break
    
    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    total_params_billion = total_params / 1e9  
    if dist.get_rank() == 0:
        print(f'Total trainable parameters: {total_params_billion:.2f} billion')


    agent.build(training=True, device=device_id)
    start_epoch = 0
    end_epoch = EPOCHS

    if dist.get_rank() == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
    # Initialize Logging =>> W&B
    global USE_WANDB
    if dist.get_rank() == 0:
        wandb_project = exp_cfg.wandb_project
        # Reuse the unified run name (same as log_dir basename) for W&B
        try:
            wandb.login()
            wandb.init(
                project=wandb_project,
                name=run_name,
                mode="offline" if cmd_args.debug else "online",
            )
            USE_WANDB = True
            print(f"[Info] W&B enabled, project={wandb_project}, run={run_name}")
        except Exception as e:
            print(f"[Info] W&B init failed ({e}), training continues without W&B")

    if dist.get_rank() == 0:
        print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")

        out = train(agent, train_dataset, TRAINING_ITERATIONS,epoch=i,rank=dist.get_rank())

        save_every = int(getattr(exp_cfg, "save_every_n_epochs", 10))
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
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/rlbench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    cmd_args = parser.parse_args()
    experiment(cmd_args)
