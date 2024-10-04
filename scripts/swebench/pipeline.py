import numpy as np
import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
import math
import time
from itertools import product

from os.path import join
import uuid
import base64
import easyapi

easy_url = 'http://saturn-cs-aus-253.reviz.ai2.in:5000'
#easy_url = 'http://jupiter-cs-aus-227.reviz.ai2.in:5000'
api = easyapi.Api(server_url=easy_url)

def short_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes)[:8].decode('ascii')

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--launch', action='store_true')
parser.add_argument('--scheduler', type=str, default='slurm')
parser.add_argument('--cluster', type=str, default='saturn')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()


begin = None

constraint = ''
change_dir = 'Agentless_private/'
repo = 'Agentless_private'
exclude = ''
#partition = 'ckpt-all'
partition = 'gpu-rtx6k'
#partition = 'gpu-a40'
#account = 'stf'
account = 'zlab'
#account = 'ark'
#account = 'efml'


if args.scheduler == 'slurm':
    s = gpuscheduler.HyakScheduler(account=account, partition=partition, use_gres=False)
else:
    s = gpuscheduler.GantryScheduler('./config/austin.cfg', cluster=f'ai2/{args.cluster}-cirrascale*', budget='ai2/allennlp', workspace='ai2/timd', weka='oe-training-default:/data/input')

limit = 64
job_gpus = 0
gpus_per_node = job_gpus
#memory, constraint = 48, '"[a100|a40]"'
port = np.random.randint(12200, 12999, 1)
cpus_per_task = cores_per_job = num_threads = limit+32
memory = 128
num_seeds = 1
seed_offset = 2
time_hours = 5
time_minutes = 0

home_path = '/data/input/timd'
base_path = join(home_path, 'git/Agentless')
swebench_path = join(home_path, 'git/SWE-bench')
checkpoint_base_dir = join(home_path, 'checkpoints')

launch_gpus = 2
launch_tp = 2
models = []

#models.append(('Qwen/Qwen2.5-0.5B-Instruct-AWQ', launch_gpus, launch_tp, 1, short_uuid())) #1
#models.append(('Qwen/Qwen2.5-1.5B-Instruct-AWQ', launch_gpus, launch_tp, 1, short_uuid())) #2
models.append(('Qwen/Qwen2.5-14B-Instruct-AWQ', launch_gpus, launch_tp, 7, short_uuid())) # 2

#models.append(('Qwen/Qwen2.5-7B-Instruct-AWQ', launch_gpus, launch_tp, 4, short_uuid())) # 4
#models.append(('Qwen/Qwen2.5-Coder-7B-Instruct-AWQ', launch_gpus, launch_tp, 4, short_uuid())) # 4

#models.append(('Qwen/Qwen2.5-32B-Instruct-AWQ', launch_gpus, launch_tp, 16, short_uuid())) # 8
#models.append(('Qwen/Qwen2.5-72B-Instruct-AWQ', launch_gpus, launch_tp, 36, short_uuid())) # 8
#models.append(('Qwen/Qwen2.5-3B-Instruct-AWQ', launch_gpus, launch_tp, 2, short_uuid())) # 8
#models.append(('hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', launch_gpus, launch_tp, 36, short_uuid())) # 8
#models.append(('hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4', launch_gpus, launch_tp, 36, short_uuid())) # 8
#models.append(('TechxGenus/Mistral-Large-Instruct-2407-AWQ', launch_gpus, launch_tp, 36, short_uuid())) # 8
#models.append(('hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4', launch_gpus, launch_tp, 36, short_uuid())) # 8

num_launches = 1
pre_cmds = []
#sgl_args = f'--schedule-conservativeness 0.1 --max-prefill-tokens {8*32*1024} --max-num-reqs 64'
sgl_args = f'--schedule-conservativeness 0.1 --max-num-reqs 64 --mem-fraction-static 0.8'
if args.launch:
    print('launching models ...')
    for model, gpus, tp, mem, uuid in models:
        if api.has_model(model) and not args.launch: break

        mem = round(mem/tp)
        for i in range(num_launches):
            api.launch_model(model, gpus=gpus, hf_token='', sgl_args_string=f'--tp {tp} {sgl_args}', priority='high', constraint="[l40|l40s|a40]")

if not args.dry:
    for model, gpus, tp, mem, uuid in models:
        while not api.has_model(model):
            print(f'waiting for {model} ...')
            time.sleep(5)

n = 4
top_n = 2
max_samples = 42
jobs = []
name = logfolder = f'replication6'
parallel = ' --parallel'
#parallel = ''
for model, gpus, tp, mem, uuid in models:
    model_stripped = model.replace('/', '_')
    #uuid = 'lsvGQZmb2'
    cmds = []
    cmds.append(f'echo "experiment {model_stripped}_{uuid}: single" 2 samples top_n=2')
    cmds.append(f'export PATH=$PATH:/usr/local/bin/')
    cmds.append(f'export EASY_URL={easy_url}')
    cmds.append(f'export PYTHONPATH={base_path}:{swebench_path}')
    cmds.append(f'cp -r {base_path}/repo_structures /tmp/')
    cmds.append(f'cd {base_path}')
    cmds.append('export PROJECT_FILE_LOC=/tmp/repo_structures')
    cmds.append(f'python agentless/fl/localize.py --file_level --related_level --fine_grain_line_level --output_folder {name}/{model_stripped}_{uuid}/location --top_n {top_n} --compress --context_window=20 --temperature 0.8 --num_samples {n} --backend easyapi --model {model} --num_threads {num_threads} --match_partial_paths --limit {limit} --max_context_length {32*1024}')
    cmds.append(f'python agentless/fl/localize.py --merge --output_folder {name}/{model_stripped}_{uuid}/location_merged --start_file {name}/{model_stripped}_{uuid}/location/loc_outputs.jsonl --num_samples {n} --limit {limit} --max_context_length {32*1024}')
    cmds.append(f'python agentless/repair/repair.py --loc_file {name}/{model_stripped}_{uuid}/location_merged/loc_merged_0-1_outputs.jsonl --output_folder {name}/{model_stripped}_{uuid}/repair_run_1 --loc_interval --top_n={top_n} --context_window=20 --max_samples {max_samples}  --cot --diff_format --gen_and_process --backend easyapi --model {model} --num_threads {num_threads} {parallel}')
    cmds.append(f'python agentless/repair/repair.py --loc_file {name}/{model_stripped}_{uuid}/location_merged/loc_merged_2-3_outputs.jsonl --output_folder {name}/{model_stripped}_{uuid}/repair_run_2 --loc_interval --top_n={top_n} --context_window=20 --max_samples {max_samples}  --cot --diff_format --gen_and_process --backend easyapi --model {model} --num_threads {num_threads} {parallel}')
    cmds.append(f'python agentless/repair/rerank.py --patch_folder {name}/{model_stripped}_{uuid}/repair_run_1,{name}/{model_stripped}_{uuid}/repair_run_2 --num_samples {max_samples} --deduplicate --plausible --output_file {name}/{model_stripped}_{uuid}/all_preds.jsonl --num_threads {num_threads}')
    #cmds.append(f'python agentless/repair/rerank.py --patch_folder {name}/{model_stripped}_{uuid}/repair_run_1 --num_samples {max_samples} --deduplicate --plausible --output_file {name}/{model_stripped}_{uuid}/all_preds.jsonl')
    cmds.append(f'echo "python -m swebench.harness.run_evaluation     --dataset_name princeton-nlp/SWE-bench_Lite     --predictions_path ../Agentless/{name}/{model_stripped}_{uuid}/all_preds.jsonl     --max_workers 32     --run_id {name}_{model_stripped}_{uuid}"')
    #cmds.append(f'python agentless/repair/rerank.py --patch_folder {name}/{model_stripped}_{uuid}/repair_run_1 --num_samples 42 --deduplicate --plausible --output_file {name}/{model_stripped}_{uuid}/all_preds.jsonl')

    if args.scheduler == 'slurm':
        s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cpus_per_task, mem=memory, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
    else:
        s.add_job(logfolder+f'/{model_stripped}', cmds, 0, False, cores=cpus_per_task, mem=memory, constraint='', exclude='', time_minutes=0, gpus=gpus)
    jobs.append((cmds))
post_cmds = []

args2 = {}
args3 = {}
args4 = []
args5 = {}
args6 = {}


rdm = np.random.RandomState(5345)

for key, value in args2.items():
    if value == True:
        cmd = cmd + ' --{0}'.format(key)
    else:
        cmd = cmd + ' --{0} {1}'.format(key, value)

args_prod = []
for key, values in args3.items():
    if isinstance(key, tuple):
        keyvalues = []
        for tups in values:
            arg = ''
            for i, v in enumerate(tups):
                if v is True: v = ''
                if v is False: continue
                if len(key[i]) == 0:
                    arg += '{0} '.format(v)
                else:
                    arg += '--{0} {1} '.format(key[i], v)
            keyvalues.append(arg)
    elif isinstance(key, str):
        keyvalues = []
        for v in values:
            if v is True: v = ''
            if v is False:
                keyvalues.append('')
            else:
                keyvalues.append(' --{0} {1}'.format(key, v))
    args_prod.append(keyvalues)

if len(args_prod) >= 2:
    args_prod = list(product(*args_prod))
else:
    new_args = []
    if len(args_prod) > 0:
        for arg in args_prod[0]:
            new_args.append([arg])
        args_prod = new_args

if args.dry:
    for i, job in enumerate(jobs):
        print(i)
        for cmd in job:
            print(cmd)
    print('')
    print('Total jobs', len(jobs))
    print('GPUs: {0}'.format(gpus_per_node))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True)
