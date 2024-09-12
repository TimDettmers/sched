import numpy as np
import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
import math
import easyapi
from itertools import product
from torch.optim.lr_scheduler import OneCycleLR

from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--uuid', type=str, default='')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--server_ip', type=str, default='hyak')
parser.add_argument('--server_port', type=int, default=5000)
parser.add_argument('--hf_token', type=str, default='')
parser.add_argument('--model', type=str, default='Qwen/Qwen2-7B-Instruct')
parser.add_argument('--sgl_args_string', type=str, default='')
parser.add_argument('--priority', type=str, default='')
parser.add_argument('--constraint', type=str, default='[a40|a100|l40|l40s]')

args = parser.parse_args()


cpus_per_task = 4
mem = (48*(8 if args.gpus > 8 else args.gpus))+20
seed_offset = 0
time_hours = 5
time_minutes = 0

begin = None
partition = 'ckpt-all'
account = 'zlab'

change_dir = 'sched/'
repo = 'sched'
exclude = 'g3091,g3104'

s = gpuscheduler.HyakScheduler(account=account, partition=partition, use_gres=False)

rdm_port = np.random.randint(12200, 12999, 1)[0]
logfolder = 'easyapi/sglang/'
cores_per_job = 4
mem = 32

home_path = '/data/input/timd'
base_path = join(home_path, 'git/sched')

pre_cmds = ["nvidia-smi", "export HOST=$(hostname -I | awk '{print $2}')", f"export MODEL={args.model}"]
pre_cmds = pre_cmds + [f'echo $HOST', 'echo $MODEL', 'echo "test"']
pre_cmds = pre_cmds + [f'export HF_TOKEN={args.hf_token}']
pre_cmds = pre_cmds + [f'curl -X POST http://{args.server_ip}:{args.server_port}/register_model -H \"Content-Type: application/json\" -d \
        \'{{ "ip": \"\'\"$SLURMD_NODENAME\"\'\", "port": {rdm_port}, "model": \"\'\"$MODEL\"\'\", \
        "uuid": "{args.uuid}", "gpus": {args.gpus}, \
        "job_id": \"\'\"$SLURM_JOB_ID\"\'\" \
          }}\'']

cmd = f"python -m sglang.launch_server --model-path $MODEL --port {rdm_port} --host $SLURMD_NODENAME {args.sgl_args_string}"

post_cmds = []


cmds = pre_cmds + [cmd] + post_cmds
s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cpus_per_task, mem=mem, constraint=args.constraint, exclude=exclude, time_minutes=time_minutes, gpus=args.gpus)
s.run_jobs(gpus_per_node=args.gpus, requeue=False, as_array=False, single_process=True, log_id=args.model.replace('/', '-'))
