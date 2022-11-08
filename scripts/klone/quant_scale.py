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
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()


gpus = 4
#memory, constraint = 21, '"[rtx6k|a40]"'
#memory, constraint = 21, '"[rtx6k|a40]"'
#memory, constraint = 21, '"[rtx6k]"'
#memory, constraint = 70, '"[a100]"'
memory, constraint = 38, '"[a40]"'
#memory, constraint = 10, '"[2080ti]"'

cmd = f'python ~/git/forked/lm-evaluation-harness/main.py --model gpt2 --use_accelerate --max_memory_per_gpu {memory}GB --no_cache --skip_tokenizer'

name = 'quant1'

logfolder = 'quant_scale/opt/{0}'.format(name)
ckp_name = logfolder
cpus_per_task = 2
mem = ((1*memory)*(8 if gpus > 8 else gpus))+20
num_seeds = 1
seed_offset = 4
time_hours = 2
time_minutes = 0

begin = None
partition = 'ckpt'
account = 'stf'
#account = 'zlab'

#begin = 'now+8hours'
#begin = '19:00'
#begin = '03:00'
#partition = 'scavenge'

change_dir = 'forked/lm-evaluation-harness/'
repo = 'forked/lm-evaluation-harness'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=partition, use_gres=False)

checkpoint_base_dir = '/gscratch/scrubbed/timdettmers'

fp16 = True


args2 = {}
args2['limit'] = 400

args3 = {}

args3['tasks'] = ['pile_pile-cc,winogrande,piqa,hellaswag,lambada']

model_str = 'pretrained=facebook/opt-{}'
#args3['model_args'] = [model_str.format('125m'), model_str.format('350m'), model_str.format('1.3b'), model_str.format('2.7b')] # 1 2080ti
#args3['model_args'] = [model_str.format('6.7b')] # 1 rtx 6k
#args3['model_args'] = [model_str.format('13b')] # 2 rtx6k
#args3['model_args'] = [model_str.format('30b')] # 2 a40 or 4 rtx6k
args3['model_args'] = [model_str.format('66b')] # 4 a40 or 7 rtx6k
#bits = [8, 7, 6, 5, 4, 3, 2]
bits = [8, 6, 4, 2]

# blockwise methods
args3[('total_bits', 'ebits')] = list(zip(bits, [3, 3, 3, 1]))
args3['blocksize'] = [4096, 2048, 1024, 512]
args3['method'] = ['blockwise_linear', 'blockwise_dynamic', 'blockwise_quantile', 'blockwise_fp8']


# non-blockwise and variable methods
#key = ('total_bits', 'ffn_bits', 'attn_bits')
#args3[key] = []
#for b in bits:
#    args3[key].append((b, b-1, b-1))
#    args3[key].append((b, b, b))
#args3['offset'] = ['mean', 'median']
##args3['offset'] = [False, 'mean', 'median']
#args3[('method', 'metric')] = [('linear', 'std'), ('error', 'relerr')]
#args3['ensure_total_bits'] = [True]


# old

#args3['ffn_bits'] = [8]
#args3['attn_bits'] = [8]

#key = ('method', 'ebits', 'total_bits')
#args3[key] = []
#for b in bits:
#    for i in range(1, b):
#        args3[key].append(('blockwise_fp8', i, b))

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
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            if i < args.skip: continue
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            if not fp16: job_cmd = job_cmd.replace('--fp16 ', ' ')
            #job_cmd = job_cmd + ' --seed {0}'.format(seed)
            checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, checkpoint_base_dir)
            save_dir = ' --save-dir {0}'.format(checkpoint_dir)
            #job_cmd = job_cmd + save_dir
            #cmds = ['source /private/home/timdettmers/.bashrc', 'source activate base2', job_cmd]
            cmds = [job_cmd]
            if rdm.rand(1) <= args.p:
                jobs.append(job_cmd)
                s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cpus_per_task, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('Memory: {0}'.format(mem))
    print('begin: {0}'.format(begin))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))
    print('Jobs will be run on: {1} {0}'.format(partition, account))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs(begin=begin, single_process=True)

