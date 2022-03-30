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
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()


gpus = 128
cmd = 'python train.py             --task cf-natural --k 16384 --test_k 16 --split dev --seed 100 --train_seed 1             --do_train --use_demonstrations --individual             --batch_size 8 --lr 1e-05 --fairscale --out_dir /checkpoint/timdettmers/GPT --clip-norm 0.1 --warmup_steps 3000 --distributed_world_size {0}'.format(gpus)


name = 'test3'
constraint = 'volta32gb'

logfolder = 'metatraining/gpt-large/{0}'.format(name)
ckp_name = logfolder
cores_per_job = 10
mem = 64*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 0
time_hours = 1
time_minutes = 0

begin = None
#partition = 'learnlab,learnfair,scavenge'
partition = 'learnlab,learnfair'
#partition = 'devlab'

#begin = 'now+8hours'
#begin = '19:00'
#begin = '03:00'
#partition = 'scavenge'

change_dir = 'FewShotLearning/'
repo = 'FewShotLearning'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)

args2 = {}

#args2['gpt2'] = 'gpt2-large'
args2['gpt2'] = 'gpt-j-6B'

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

jobs = []
if len(args4) == 0: args4.append('')
assert num_seeds == 1
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            #job_cmd = job_cmd + ' --seed {0}'.format(seed)
            checkpoint_dir = '/checkpoint/timdettmers/GPT/{1}_{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            save_dir = ' --out_dir {0}'.format(checkpoint_dir)
            job_cmd = job_cmd + save_dir
            cmds = [job_cmd]
            jobs.append(job_cmd)
            s.add_job(logfolder, repo, change_dir, cmds, time_hours, True, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('begin: {0}'.format(begin))
    print('Jobs will be written to: {0}'.format(join('/private/home/timdettmers/logs/', logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs(begin=begin)

