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

hf_token = os.environ['HF_TOKEN'] if 'HF_TOKEN' in os.environ else ''

api = easyapi.Api()

def short_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes)[:8].decode('ascii')

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--launch', action='store_true')
parser.add_argument('--openai', action='store_true')
parser.add_argument('--scheduler', type=str, default='beaker')
parser.add_argument('--cluster', type=str, default='ai2/saturn-cirrascale')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()

print(args)


begin = None

constraint = ''
change_dir = 'ScholarQABench/'
repo = 'ScholarQABench'
exclude = ''
#partition = 'ckpt-all'
partition = 'gpu-rtx6k'
#partition = 'gpu-a40'
#account = 'stf'
account = 'zlab'
#account = 'ark'
#account = 'efml'


# cluster = ['ai2/jupiter-cirrascale-2', 'ai2/neptune-cirrascale', 'ai2/saturn-cirrascale']
if args.scheduler == 'slurm':
    s = gpuscheduler.HyakScheduler(account=account, partition=partition, use_gres=False)
else:
    s = gpuscheduler.GantryScheduler('/weka/oe-adapt-default/saurabhs/repos/sched/config/austin.cfg', cluster=args.cluster, budget='ai2/allennlp', workspace='ai2/saurabhs', weka='oe-adapt-default:/weka/oe-adapt-default')

job_gpus = 0
gpus_per_node = job_gpus
#memory, constraint = 48, '"[a100|a40]"'
port = np.random.randint(12200, 12999, 1)
memory = 128
num_seeds = 1
seed_offset = 5
time_hours = 5
time_minutes = 0

home_path = '/weka/oe-adapt-default/saurabhs'
base_path = join(home_path, 'repos/nora_adapt')

models = []

#models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct-AWQ', 1, 4)) #1
#models.append(('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct-AWQ', 2, 4)) #2

#models.append(('Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct-AWQ', 4, 4)) # 4
#models.append(('Qwen2.5-7B-Coder', 'Qwen/Qwen2.5-Coder-7B-Instruct-AWQ', 4)) # 4
#models.append(('gpt-4o-2024-08-06', 'gpt-4o-2024-08-06', 4, 16)) # 4
#models.append(('gpt-4o-2024-05-13', 'gpt-4o-2024-05-13', 4, 16)) # 4
#models.append(('gpt-4o-mini', 'gpt-4o-mini', 4, 16)) # 4

#models.append(('Qwen2.5-Coder-7B', 'Qwen/Qwen2.5-Coder-7B-Instruct', 4, 16)) # 4
# models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct', 1, 16)) # 2
# models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 1, 16)) # 2
# models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-3B-Instruct', 1, 16)) # 2
# models.append(('Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct', 2, 16)) # 2
# models.append(('Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct', 2, 16)) # 2
# models.append(('Qwen2.5-32B', 'Qwen/Qwen2.5-32B-Instruct', 4, 16)) # 8
# models.append(('Qwen2.5-72B', 'Qwen/Qwen2.5-72B-Instruct', 8, 16)) # 8

models.append(('DeepSeek-R1-Distill-Qwen-7B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 2, 16)) # 2
#models.append(('prometheus-7b', 'prometheus-eval/prometheus-7b-v2.0', 8, 16)) # 8
#models.append(('DeepSeek-R1-AWQ', 'cognitivecomputations/DeepSeek-R1-AWQ', 8, 4)) # 8

#models.append(('Nemotron-70B', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', 8, 16))
#models.append(('Mistral_Large-123B', 'SillyTilly/Mistral-Large-Instruct-2407', 8, 16))
#models.append(('DeepSeek-Coder-236B', 'deepseek-ai/DeepSeek-Coder-V2-Instruct-0724', 8, 16))

#models.append(('Llama-8B', 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 8, 4)) # 8
#models.append(('Llama-70B', 'meta-llama/Llama-3.1-70B', 8, 16)) # 8
#models.append(('Llama-405B', 'hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4', 8, 4)) # 8


if not args.openai:
    num_gpus = 2
    num_launches = 1
    if num_launches is not None:
        set_gpus = True
    #sgl_args = f'--schedule-conservativeness 0.1 --max-prefill-tokens {8*32*1024} --max-num-reqs 64'
    if args.launch:
        print('launching models ...')
        for name, model, tp, bits in models:
            print(f'launching {name} ...')
            if set_gpus: num_gpus = tp
            if api.has_model(model) and not args.launch: break
            num_req = 512 if not 'awq' in model.lower() else 32
            sgl_args = f'--allow-auto-truncate --grammar-backend xgrammar'

            #num_launches = num_gpus//tp
            for i in range(num_launches):
                print(f'launching {i+1}/{num_launches} ...')
                quant = ' --quantization fp8' if bits == 8 else ''
                api.launch_model(model, gpus=tp, cluster=cluster, hf_token=hf_token, sgl_args_string=f'--tp {tp} {quant} --trust-remote-code {sgl_args}', priority='high', constraint="[l40|l40s|a40]")

    if not args.dry:
        for name, model, tp, bits in models:
            while not api.has_model(model):
                print(f'waiting for {model} ...')
                time.sleep(5)


args2 = {}
args3 = {}

cpus_per_task = cores_per_job = num_threads = 0


base = 'Qwen/Qwen2.5-{params}B-Instruct'
p = [0.5, 1.5, 3, 7, 14, 32, 72]
name = logfolder = f'scholarqa_grid11'
#args3['model'] = [base.format(params=params) for params in p[:-1]]
#args3['model'] = [base.format(params=params) for params in p[3:-2]]
#args3['model'] = [base.format(params=params) for params in p[4:]]
args3['model'] = [] # [base.format(params=params) for params in p[:1]]
args3['model'].append('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
args3['proc'] = [250]
args3['n'] = [100]


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


jobs = []
pre_cmds = []
pre_cmds.append('sleep $(((RANDOM % 10)+1))')
pre_cmds.append(f'export PATH=$PATH:/usr/local/bin/')
#pre_cmds.append(f'export EASY_URL={easy_url}')
#pre_cmds.append(f'export PYTHONPATH={base_path}:{swebench_path}')
pre_cmds.append(f'source {home_path}/.bashrc')
pre_cmds.append(f'eval "$(conda shell.bash hook)"')
#pre_cmds.append(f'mkdir -p /data/tmp')
#pre_cmds.append(f'cp -r {base_path}/repo_structures /data/tmp/')
pre_cmds.append(f'ls -la')
pre_cmds.append(f'cd {base_path}')
pre_cmds.append(f'export BEAKER_TOKEN=+B+Vhnbwacnx5t/z')
#pre_cmds.append('export PROJECT_FILE_LOC=/data/tmp/repo_structures')
cmd = 'python synthetic_rubric_tuning.py     --qa-dir data/scholarqa_cs/src_answers     --test-config data/scholarqa_cs/test_configs_snippets.json     --rubrics --snippets'
for seed in range(seed_offset, seed_offset+num_seeds):
    for i, values in enumerate(args_prod):
        idval = short_uuid()
        cmds = pre_cmds + [f'redis-server --port {6379+i} &', 'sleep 5'] + [cmd + ''.join(values) + f' --store {logfolder}_{i} --redis_db {i} --redis_port {6379+i}']
        print(i, cmds[-1])

        if args.scheduler == 'slurm':
            s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cpus_per_task, mem=memory, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=0)
        else:
            s.add_job(logfolder, cmds, 0, False, cores=cpus_per_task, mem=memory, constraint='', exclude='', time_minutes=0, gpus=0)
        jobs.append((cmds))
post_cmds = []


if args.dry:
    print('')
    print('Total jobs', len(jobs))
    print('GPUs: {0}'.format(gpus_per_node))
    print('CPU cores: {0}'.format(cpus_per_task))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True, priority='normal')
