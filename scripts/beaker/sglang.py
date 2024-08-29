import numpy as np
import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
import math
from itertools import product
from torch.optim.lr_scheduler import OneCycleLR

from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--uuid', type=str, default='')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--server_ip', type=str, default='jupiter-cs-aus-199.reviz.ai2.in')
parser.add_argument('--server_port', type=int, default=5000)
parser.add_argument('--hf_token', type=str, default='')
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B')

args = parser.parse_args()

s = gpuscheduler.GantryScheduler('./config/austin.cfg', cluster='ai2/jupiter-cirrascale-2', budget='ai2/allennlp', workspace='ai2/timd', weka='oe-training-default:/data/input')

rdm_port = np.random.randint(12200, 12999, 1)[0]
logfolder = 'sglang/llama8b'
cores_per_job = 4
mem = 32

home_path = '/data/input/timd'
base_path = join(home_path, 'git/sched')

pre_cmds = [f'cd {base_path}', 'printenv', "export HOST=$(hostname -I | awk '{print $1}')", f"export MODEL={args.model}"]
pre_cmds = pre_cmds + [f'export HF_TOKEN={args.hf_token}']
pre_cmds = pre_cmds + [f'curl -X POST {args.server_ip}:{args.server_port}/register_model -H \"Content-Type: application/json\" -d \
        \'{{ "ip": \"\'\"$HOST\"\'\", "port": {rdm_port}, "model": \"\'\"$MODEL\"\'\", \
        "uuid": "{args.uuid}", "gpus": {args.gpus}, \
        "job_id": \"\'\"$BEAKER_JOB_ID\"\'\" \
          }}\'']

cmd = f"python -m sglang.launch_server --model-path $MODEL --port {rdm_port} --host $HOST"

post_cmds = []


cmds = pre_cmds + [cmd] + post_cmds
s.add_job(logfolder, cmds, cores=cores_per_job, mem=mem, gpus=args.gpus)
s.run_jobs(gpus_per_node=args.gpus, requeue=False, as_array=True, single_process=True)
