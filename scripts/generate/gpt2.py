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


gpus = 1
cmd = 'python generate.py'

args2 = {}

name = 'gen3_dense'
constraint = 'volta'

logfolder = 'generations/gpt2/{0}'.format(name)
ckp_name = logfolder
cores_per_job = 5
mem = 48*(8 if gpus > 8 else gpus)
num_seeds = 100
seed_offset = 0
time_hours = 3
time_minutes = 0

begin = None
#partition = 'learnlab,learnfair,scavenge'
partition = 'learnlab,learnfair'
#partition = 'devlab'

#begin = 'now+8hours'
#begin = '19:00'
#begin = '03:00'
#partition = 'scavenge'

change_dir = 'fairseq_private'
repo = 'fairseq_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)


fp16 = True
args3 = {}
args4 = []
args5 = {}
args6 = {}

#args3['size'] = ['130b']
args3['checkpoint-folder'] = ['/checkpoint/timdettmers/sparse_attention/gpt2_3/050521537cabf90c1d7133dcb7eeb7cf/'] # wave
#args3['checkpoint-folder'] = ['/checkpoint/timdettmers/sparse_attention/gpt2_3/bd9a6bf2f18a6fe69b8600e3eedd8f9f'] # checker
#args3['checkpoint-folder'] = ['/checkpoint/timdettmers/sparse_attention/gpt2_3/26a3dadf1a4361bd04134c91256edf94'] # dense
args3['ckpt-name'] = ['checkpoint_last.pt']
args2['num-samples'] = 1000
args2['seqlen'] = 1024
#args2['half'] = ''
args3['samples-per-batch'] = [10]

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
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            #job_cmd += ' --checkpoint /checkpoint/timdettmers/{1}/{0}/model.pt'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            if not fp16: job_cmd = job_cmd.replace('--fp16 ', ' ')
            job_cmd = job_cmd + ' --seed {0}'.format(seed)
            out_dir = '/checkpoint/timdettmers/{1}/{0}/gen '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            save_dir = ' --out {0}'.format(out_dir)
            job_cmd = job_cmd + save_dir
            #cmds = ['source /private/home/timdettmers/.bashrc', 'source activate internal', job_cmd]
            cmds = [job_cmd]
            jobs.append(job_cmd)
            s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

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

