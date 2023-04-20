import numpy as np
import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
import math
import random
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
#memory, constraint = 10, '"[2080ti]"
#memory, constraint = 21, '"[rtx6k|a40]"'
#memory, constraint = 21, '"[rtx6k]"'
#memory, constraint = 70, '"[a100]"'
memory, constraint = 38, '"[a40]"'
#memory, constraint = 38, '"[a40|a100]"'
#memory, constraint = 21, '"[rtx6k|a40|a100]"'
#memory, constraint = 10, '"[2080ti|rtx6k|titan]"'
#memory, constraint = 10, '"[2080ti|rtx6k|titan|a40]"'
#memory, constraint = 10, '"[2080ti|rtx6k|titan|a40|a100]"'

cmd = f'python ~/git/forked/lm-evaluation-harness/main.py --model hf --use_accelerate --max_memory_per_gpu {memory}GB --no_cache --skip_tokenizer'

name = 'fewshot1'
offset = 0

logfolder = 'evo/normal/{0}'.format(name)
ckp_name = logfolder
cpus_per_task = 2
mem = ((1*memory)*(8 if gpus > 8 else gpus))+20
num_seeds = 5
seed_offset = 0
time_hours = 4
time_minutes = 0

begin = None
#partition = 'ckpt'
#partition = 'gpu-rtx6k'
partition = 'gpu-a40'
#account = 'stf'
account = 'zlab'
#account = 'ark'
#account = 'efml'

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
#args2['limit'] = 10

args3 = {}
#args3['limit'] = [400, 500, 600, 700, 800]
#args2['total_bits'] = 16


#args3['tasks'] = ['winogrande,piqa,hellaswag,lambada']
#args3['tasks'] = ['winogrande,piqa,hellaswag,lambada,pile_pile-cc,wikitext']
#args3['tasks'] = ['winogrande,piqa,hellaswag,lambada,pile_pile-cc']
#args3['tasks'] = ['winogrande,piqa,hellaswag,pile_pile-cc']
#args3['tasks'] = ['winogrande']
args3['tasks'] = ['winogrande,piqa,hellaswag,pile_pile-cc,arc_easy,arc_challenge']
#args3['tasks'] = ['pile_pile-cc,wikitext']

#ms = 'pretrained=gpt2{}'
#args3['model_args'] = [ms.format(''), ms.format('-medium'), ms.format('-large'), ms.format('-xl')] # 1 rtx6k or 2080 ti

#ms = 'pretrained=bigscience/bloom-{}'
#args2['use_fast_tok'] = True
##args3['model_args'] = [ms.format('560m'), ms.format('1b1'), ms.format('1b7'), ms.format('3b')] # 1 2080 Ti
#args3['model_args'] = [ms.format('7b1')] # 1 rtx6k
#args3['model_args'] = ['pretrained=bigscience/bloom'] # 6 A100
#args3['model_args'] = [ms.format('560m'), ms.format('1b1'), ms.format('1b7'), ms.format('3b'), ms.format('7b1')] # 1 rtx6k

#ms = 'pretrained=bigscience/bloomz-{}'
#args2['use_fast_tok'] = True
##args3['model_args'] = [ms.format('560m'), ms.format('1b1'), ms.format('1b7'), ms.format('3b')] # 1 2080 Ti
##args3['model_args'] = [ms.format('7b1')] # 1 rtx6k
#args3['model_args'] = [ms.format('560m'), ms.format('1b1'), ms.format('1b7'), ms.format('3b'), ms.format('7b1')] # 1 rtx6k


#ms = 'pretrained=EleutherAI/pythia-{}-deduped'
#args2['use_fast_tok'] = True
#args3['model_args'] = [ms.format('19m'), ms.format('125m'), ms.format('350m'), ms.format('1.3b')] # 1 2080 Ti
#args3['model_args'] = [ms.format('6.7b')] # 1 rtx6k
#args3['model_args'] = [ms.format('19m'), ms.format('125m'), ms.format('350m'), ms.format('1.3b'), ms.format('6.7b')] # 1 rtx6k
#args3['model_args'] = [ms.format('13b')]
#args3['model_args'] = [ms.format('13b'), 'pretrained=EleutherAI/gpt-neox-20b'] # 2 a40
#args3['model_args'] = [ms.format('6.7b'), ms.format('13b'), 'pretrained=EleutherAI/gpt-neox-20b'] # 2 a40


model_str = 'pretrained=/gscratch/zlab/llama/'
args3['model_args'] = []
#args3['model_args'].append(model_str + '7B') # 1x A40
#args3['model_args'].append(model_str + '13B') # 1x A40 or 2 rtx6k
#args3['model_args'].append(model_str + '30B') # 2x A40
args3['model_args'].append(model_str + '65B') # 4x A40 or 7 rtx6b

#model_str = 'pretrained=/gscratch/scrubbed/timdettmers/models/hf_checkpoint/'
#args3['model_args'] = []
#model_str = 'pretrained=facebook/opt-{}'
#args3['model_args'].append(model_str.format('125m'))
#args3['model_args'].append(model_str.format('350m'))
#args3['model_args'].append(model_str.format('1.3b'))
#args3['model_args'].append(model_str.format('2.7b')) # 1x rtx 2080 ti

#args3['model_args'] = [model_str.format('6.7b')] # 1 rtx 6k
#args3['model_args'] = [model_str.format('13b')] # 2 rtx6k or 1 a40
#args3['model_args'] = [model_str.format('30b')] # 2 a40 or 4 rtx6k or 1 A100
#args3['model_args'] = [model_str.format('66b')] # 4 a40 or 7 rtx6k or 2 A100

args3['dtype'] = ['float16']
args3['total_bits'] = [4]
args3['blocksize'] = [64]

#args3['ebits'] = [2, 3]
#args3['method'] = ['blockwise_fp8']
#args3['method'] = ['blockwise_linear']

args3['custom_scale'] = [0.9677083]
#args3['use_extra_value'] = [True, False]
args3[('use_extra_value', 'method', 'ebits', 'nested')] = []
args3[('use_extra_value', 'method', 'ebits', 'nested')].append((True, 'blockwise_normal', 3, True))
args3[('use_extra_value', 'method', 'ebits', 'nested')].append((True, 'blockwise_normal', 3, False))
args3[('use_extra_value', 'method', 'ebits', 'nested')].append((False, 'blockwise_normal', 3, False))
args3[('use_extra_value', 'method', 'ebits', 'nested')].append((False, 'blockwise_fp8', 3, False))
args3[('use_extra_value', 'method', 'ebits', 'nested')].append((False, 'blockwise_fp8', 2, False))
args3[('use_extra_value', 'method', 'ebits', 'nested')].append((False, 'blockwise_linear', 3, False))


#bits = [8, 7, 6, 5, 4, 3]
#args3['total_bits'] = bits

#key = ('ffn_bits', 'attn_bits')
#args3[key] = []
#args3[key].append((4, 5))
#args3[key].append((4, 6))
#args3[key].append((5, 4))
#args3[key].append((5, 6))
#args3[key].append((6, 4))
#args3[key].append((6, 5))

#args3[('total_bits', 'ebits')] = list(zip(bits, [2, 3]))
# blockwise methods
#args3[('total_bits', 'ebits')] = list(zip(bits, [3]))
#args3['blocksize'] = [2048, 1024, 512]

# BASE
#bits = [8, 7, 6, 5, 4, 3]
#args3[('total_bits', 'ebits')] = list(zip(bits, [3, 3, 3, 3, 2, 1]))
#args3['blocksize'] = [64, 128]
#args3['method'] = ['blockwise_linear', 'blockwise_dynamic', 'blockwise_quantile', 'blockwise_fp8']

# 175b
#bits = [8, 4, 3]
#args3[('total_bits', 'ebits')] = list(zip(bits, [3, 3, 2]))
#bits = [7, 6, 5]
#args3[('total_bits', 'ebits')] = list(zip(bits, [3, 3, 3]))

#bits = [8, 7, 6, 5, 4, 3]
#args3[('total_bits', 'ebits')] = list(zip(bits, [3, 3, 3, 3, 3, 2]))
#args3['blocksize'] = [64]
#args3['method'] = ['blockwise_linear', 'blockwise_dynamic', 'blockwise_quantile', 'blockwise_fp8']

# BASE 3,4
#bits = [4]
#args3[('total_bits', 'ebits')] = list(zip(bits, [3]))
#args3['blocksize'] = [64]
#args3['method'] = ['blockwise_linear', 'blockwise_dynamic', 'blockwise_quantile', 'blockwise_fp8']
##args3['method'] = ['blockwise_quantile']
#args3['outliers'] = [True, False]

#bits = [3]
#args3[('total_bits', 'ebits')] = list(zip(bits, [2]))
#args3['blocksize'] = [64]
#args3['method'] = ['blockwise_fp8']
#args3['outliers'] = [True]

#args3['method'] = ['blockwise_linear', 'blockwise_dynamic', 'blockwise_quantile', 'blockwise_fp8']
#args3['offset'] = ['mean', 'median', False]
#args3['offset'] = ['mean']

# non-blockwise and variable methods
#key = ('total_bits', 'ffn_bits', 'attn_bits')
#args3[key] = []
#for b in bits:
#    args3[key].append((b, b-1, b-1))
#    #args3[key].append((b, b, b))
#args3['offset'] = [False, 'mean', 'median']
#
#args3['method'] = ['iterative', 'iterative_block']
#args3['blocksize'] = [1024]

#args3[('method', 'metric')] = [('linear', 'std'), ('error', 'relerr')]
#args3['ensure_total_bits'] = [True]


# old

#args3['ffn_bits'] = [8]
#args3['attn_bits'] = [8]
#args3['blocksize'] = [64]
#bits = [8, 7, 6, 5, 4, 3]
#args3['method'] = ['blockwise_fp8']
#key = ('ebits', 'total_bits')
#args3[key] = []
#for b in bits:
#    for i in range(1, b):
#        args3[key].append((i, b))




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
            job_cmd = job_cmd + ' --seed {0}'.format(seed)
            checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, checkpoint_base_dir)
            save_dir = ' --save-dir {0}'.format(checkpoint_dir)
            #job_cmd = job_cmd + save_dir
            #cmds = ['source /private/home/timdettmers/.bashrc', 'source activate base2', job_cmd]
            #cmds = [f'sleep {random.randrange(15, 300)}', job_cmd]
            cmds = [job_cmd]
            if rdm.rand(1) <= args.p:
                jobs.append(job_cmd)
                s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cpus_per_task, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i+args.skip, job)
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

