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

api = easyapi.Api()

def short_uuid():
    return base64.urlsafe_b64encode(uuid.uuid4().bytes)[:8].decode('ascii')

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--launch', action='store_true')
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


s = gpuscheduler.HyakScheduler(account=account, partition=partition, use_gres=False)
#s = gpuscheduler.GantryScheduler('./config/cirrascale.cfg', cluster='ai2/*cirrascale', budget='ai2/allennlp', workspace='ai2/timd')
#s = gpuscheduler.GantryScheduler('./config/austin.cfg', cluster='ai2/jupiter-cirrascale-2', budget='ai2/allennlp', workspace='ai2/timd', weka='oe-training-default:/data/input')

gpus = 0
gpus_per_node = gpus
#memory, constraint = 48, '"[a100|a40]"'
port = np.random.randint(12200, 12999, 1)
logfolder = f'agentless/qwen_2.5'
ckp_name = logfolder
cpus_per_task = cores_per_job = num_threads = 20
memory = 128
num_seeds = 1
seed_offset = 2
time_hours = 5
time_minutes = 0

#home_path = '/net/nfs.cirrascale/allennlp/timd'
home_path = '/data/input/timd'
base_path = join(home_path, 'git/sched')
checkpoint_base_dir = join(home_path, 'checkpoints')

models = []
#models.append(('Qwen/Qwen2.5-0.5B-Instruct-AWQ', 1, 1, 1, short_uuid()))
#models.append(('Qwen/Qwen2.5-1.5B-Instruct-AWQ', 1, 1, 1, short_uuid()))
models.append(('Qwen/Qwen2.5-3B-Instruct-AWQ', 1, 1, 2, short_uuid()))
models.append(('Qwen/Qwen2.5-7B-Instruct-AWQ', 1, 1, 4, short_uuid()))
models.append(('Qwen/Qwen2.5-14B-Instruct-AWQ', 1, 1, 7, short_uuid()))
models.append(('Qwen/Qwen2.5-32B-Instruct-AWQ', 1, 1, 16, short_uuid()))
#models.append(('Qwen/Qwen2.5-72B-Instruct-AWQ', 2, 2, 36, short_uuid()))

num_launches = 1
pre_cmds = []
if not args.dry or args.launch:
    print('launching models ...')
    for model, gpus, tp, mem, uuid in models:
        if api.has_model(model) and not args.launch: break

        mem = round(mem/tp)
        for i in range(num_launches):
            api.launch_model(model, gpus=gpus, hf_token='', sgl_args_string=f'--tp {tp}', priority='normal', constraint="[l40|l40s|a40]")

    for model, gpus, tp, mem, uuid in models:
        while not api.has_model(model): time.sleep(5)

jobs = []
for model, gpus, tp, mem, uuid in models:
    model_stripped = model.replace('/', '_')
    cmds = []
    cmds.append('cp -r repo_structures /tmp/')
    cmds.append('export PROJECT_FILE_LOC=/tmp/repo_structures')
    cmds.append(f'python agentless/fl/localize.py --file_level --related_level --fine_grain_line_level --output_folder {model_stripped}_{uuid}/location --top_n 3 --compress --context_window=20 --temperature 0.8 --num_samples 4 --backend easyapi --model {model} --num_threads {num_threads} --match_partial_paths')
    cmds.append(f'python agentless/fl/localize.py --merge --output_folder {model_stripped}_{uuid}/location_merged --start_file {model_stripped}_{uuid}/location/loc_outputs.jsonl --num_samples 4')
    cmds.append(f'python agentless/repair/repair.py --loc_file {model_stripped}_{uuid}/location_merged/loc_merged_0-1_outputs.jsonl --output_folder {model_stripped}_{uuid}/repair_run_1 --loc_interval --top_n=4 --context_window=20 --max_samples 42  --cot --diff_format --gen_and_process --backend easyapi --model {model} --num_threads {num_threads}')
    cmds.append(f'python agentless/repair/repair.py --loc_file {model_stripped}_{uuid}/location_merged/loc_merged_2-3_outputs.jsonl --output_folder {model_stripped}_{uuid}/repair_run_2 --loc_interval --top_n=4 --context_window=20 --max_samples 42  --cot --diff_format --gen_and_process --backend easyapi --model {model} --num_threads {num_threads}')
    cmds.append(f'python agentless/repair/rerank.py --patch_folder {model_stripped}_{uuid}/repair_run_1,{model_stripped}_{uuid}/repair_run_2 --num_samples 42 --deduplicate --plausible --output_file {model}_{uuid}/all_preds.jsonl')

    s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cpus_per_task, mem=memory, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
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

#port = port[0]
#if len(args4) == 0: args4.append('')
#for seed in range(num_seeds):
#    seed = seed + seed_offset
#    for arg4 in args4:
#        if len(args_prod) == 0: args_prod.append(('', ''))
#        for i, values in enumerate(args_prod):
#            port += 1
#            #job_cmd = cmd + arg4
#            #for val in values:
#                #job_cmd += ' {0}' .format(val)
#            #checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, checkpoint_base_dir)
#            #save_dir = ' --checkpoint_dir {0}'.format(checkpoint_dir)
#            #job_cmd += save_dir
#            #job_cmd = job_cmd.replace('MASTER_PORT', str(port))
#            #job_cmd = job_cmd + ' --seed {0}'.format(seed)
#            cmds = pre_cmds + cmds + post_cmds
#            jobs = cmds
#            s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cpus_per_task, mem=memory, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('GPUs: {0}'.format(gpus))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True)
