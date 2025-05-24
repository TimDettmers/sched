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

easy_url = 'http://neptune-cs-aus-267.reviz.ai2.in:5000'
#easy_url = 'http://saturn-cs-aus-253.reviz.ai2.in:5000'
#easy_url = 'http://jupiter-cs-aus-227.reviz.ai2.in:5000'
api = easyapi.Api(server_url=easy_url)

def short_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes)[:8].decode('ascii')

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--launch', action='store_true')
parser.add_argument('--openai', action='store_true')
parser.add_argument('--scheduler', type=str, default='slurm')
parser.add_argument('--cluster', type=str, default='saturn')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()

print(args)


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


cluster = ['ai2/jupiter-cirrascale-2', 'ai2/neptune-cirrascale', 'ai2/saturn-cirrascale']
if args.scheduler == 'slurm':
    s = gpuscheduler.HyakScheduler(account=account, partition=partition, use_gres=False)
else:
    s = gpuscheduler.GantryScheduler('/data/input/timd/git/sched/config/austin.cfg', cluster=cluster, budget='ai2/allennlp', workspace='ai2/timd', weka='oe-training-default:/data/input')

job_gpus = 0
gpus_per_node = job_gpus
#memory, constraint = 48, '"[a100|a40]"'
port = np.random.randint(12200, 12999, 1)
memory = 128
num_seeds = 1
seed_offset = 5
time_hours = 5
time_minutes = 0

home_path = '/data/input/timd'
base_path = join(home_path, 'git/Agentless')
swebench_path = join(home_path, 'git/SWE-bench')
checkpoint_base_dir = join(home_path, 'checkpoints')

models = []

#models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct-AWQ', 1, 4)) #1
#models.append(('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct-AWQ', 2, 4)) #2

#models.append(('Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct-AWQ', 4, 4)) # 4
#models.append(('Qwen2.5-7B-Coder', 'Qwen/Qwen2.5-Coder-7B-Instruct-AWQ', 4)) # 4
#models.append(('gpt-4o-2024-08-06', 'gpt-4o-2024-08-06', 4, 16)) # 4
#models.append(('gpt-4o-2024-05-13', 'gpt-4o-2024-05-13', 4, 16)) # 4
#models.append(('gpt-4o-mini', 'gpt-4o-mini', 4, 16)) # 4

#models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct', 1, 16)) # 2
#models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 1, 16)) # 2
#models.append(('Qwen2.5-0.5B', 'Qwen/Qwen2.5-3B-Instruct', 1, 16)) # 2
#models.append(('Qwen2.5-Coder-7B', 'Qwen/Qwen2.5-Coder-7B-Instruct', 4, 16)) # 4
#models.append(('Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct', 1, 16)) # 2
#models.append(('Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct', 1, 16)) # 2
#models.append(('Qwen2.5-32B', 'Qwen/Qwen2.5-32B-Instruct', 2, 16)) # 8
#models.append(('Qwen2.5-32B', 'Qwen/Qwen2.5-32B-Instruct', 4, 16)) # 8
models.append(('Qwen2.5-72B', 'Qwen/Qwen2.5-72B-Instruct', 8, 16)) # 8
#models.append(('prometheus-7b', 'prometheus-eval/prometheus-7b-v2.0', 8, 16)) # 8
#models.append(('DeepSeek-R1-AWQ', 'cognitivecomputations/DeepSeek-R1-AWQ', 8, 4)) # 8

#models.append(('Nemotron-70B', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', 8, 16))
#models.append(('Mistral_Large-123B', 'SillyTilly/Mistral-Large-Instruct-2407', 8, 16))
#models.append(('DeepSeek-Coder-236B', 'deepseek-ai/DeepSeek-Coder-V2-Instruct-0724', 8, 16))

#models.append(('Llama-8B', 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', 8, 4)) # 8
#models.append(('Llama-70B', 'meta-llama/Llama-3.1-70B', 8, 16)) # 8
#models.append(('Llama-405B', 'hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4', 8, 4)) # 8


if not args.openai:
    num_gpus = 4
    #sgl_args = f'--schedule-conservativeness 0.1 --max-prefill-tokens {8*32*1024} --max-num-reqs 64'
    if args.launch:
        print('launching models ...')
        for name, model, tp, bits in models:
            if api.has_model(model) and not args.launch: break
            num_req = 512 if not 'awq' in model.lower() else 32
            sgl_args = f'--schedule-conservativeness 0.1 --max-num-reqs {num_req} --mem-fraction-static 0.85 --allow-auto-truncate'

            num_launches = 1
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

cpus_per_task = cores_per_job = num_threads = 30

name = logfolder = f'test_12'
args3['max_samples'] = [20]
args3['top_n'] = [2]
args3['num_samples'] = [4]
args3['limit'] = [128]
args3['temperature'] = [0.8]
args3['context_window'] = [20]
args3['backend'] = ['easyapi']
#args3['backend'] = ['openai']


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
pre_cmds.append(f'export EASY_URL={easy_url}')
pre_cmds.append(f'export PYTHONPATH={base_path}:{swebench_path}')
pre_cmds.append(f'source /data/input/timd/.bashrc')
pre_cmds.append(f'eval "$(conda shell.bash hook)"')
pre_cmds.append(f'mkdir -p /data/tmp')
pre_cmds.append(f'cp -r {base_path}/repo_structures /data/tmp/')
pre_cmds.append(f'cd {base_path}')
pre_cmds.append(f'export BEAKER_TOKEN=4jEJObA+dQIm7Qfn')
pre_cmds.append('export PROJECT_FILE_LOC=/data/tmp/repo_structures')
for seed in range(seed_offset, seed_offset+num_seeds):
    for i, values in enumerate(args_prod):
        idval = short_uuid()
        for model_name, model, tp, bits in models:
            addtional_args_str = ' '
            for val in values:
                addtional_args_str += '{0}' .format(val)
            addtional_args_str += f' --seed {seed}'
            addtional_args_str += f' --bits {bits} --model_name {model_name}'
            model_stripped = model_name.replace('/', '_')
            model_stripped += f'_{bits}bits'
            cmds = [] + pre_cmds
            cmds.append(f'python agentless/fl/localize.py --file_level --related_level --fine_grain_line_level --output_folder {name}/{model_stripped}_{idval}/location --compress --model {model} --num_threads {num_threads} --match_partial_paths --max_context_length {128*1024} {addtional_args_str}')
            cmds.append(f'python agentless/fl/localize.py --merge --output_folder {name}/{model_stripped}_{idval}/location_merged --start_file {name}/{model_stripped}_{idval}/location/loc_outputs.jsonl --max_context_length {128*1024} {addtional_args_str}')
            cmds.append(f'python agentless/repair/repair.py --loc_file {name}/{model_stripped}_{idval}/location_merged/loc_merged_0-1_outputs.jsonl --output_folder {name}/{model_stripped}_{idval}/repair_run_1 --loc_interval --cot --diff_format --gen_and_process --model {model} --num_threads {num_threads} --playground_path /data/tmp/playground {addtional_args_str}')
            cmds.append(f'python agentless/repair/repair.py --loc_file {name}/{model_stripped}_{idval}/location_merged/loc_merged_2-3_outputs.jsonl --output_folder {name}/{model_stripped}_{idval}/repair_run_2 --loc_interval --cot --diff_format --gen_and_process --model {model} --num_threads {num_threads} --playground_path /data/tmp/playground {addtional_args_str}')
            cmds.append(f'python agentless/repair/rerank.py --patch_folder {name}/{model_stripped}_{idval}/repair_run_1,{name}/{model_stripped}_{idval}/repair_run_2 --deduplicate --plausible --output_file {name}/{model_stripped}_{idval}/all_preds.jsonl --num_threads {num_threads} {addtional_args_str}')
            cmds.append(f'echo "python -m swebench.harness.run_evaluation     --dataset_name princeton-nlp/SWE-bench_Lite     --predictions_path ../Agentless/{name}/{model_stripped}_{idval}/all_preds.jsonl     --max_workers 120     --run_id {name}_{model_stripped}_{idval} --cache_level instance"')
            cmds.append(f'python log_experiment.py {addtional_args_str} --folder {logfolder} --command \"python -m swebench.harness.run_evaluation     --dataset_name princeton-nlp/SWE-bench_Lite     --predictions_path ../Agentless/{name}/{model_stripped}_{idval}/all_preds.jsonl     --max_workers 120     --run_id {name}_{model_stripped}_{idval} --cache_level instance\"')

            if args.scheduler == 'slurm':
                s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cpus_per_task, mem=memory, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=0)
            else:
                s.add_job(logfolder+f'/{model_stripped}', cmds, 0, False, cores=cpus_per_task, mem=memory, constraint='', exclude='', time_minutes=0, gpus=0)
            jobs.append((cmds))
post_cmds = []


if args.dry:
    for i, job in enumerate(jobs):
        print(i)
        for cmd in job:
            print(cmd)
    print('')
    print('Total jobs', len(jobs))
    print('GPUs: {0}'.format(gpus_per_node))
    print('CPU cores: {0}'.format(cpus_per_task))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True, priority='high')
