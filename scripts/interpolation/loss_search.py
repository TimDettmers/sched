import itertools
import argparse
import os
import uuid
import hashlib

import numpy as np
import gpuscheduler

from itertools import product, combinations
from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--scale', action='store_true')
args = parser.parse_args()

cmd = 'python wrangle_losses.py'

args2 = {}
args2['f'] = 'clean_losses'
args2['eval'] = 50
args2['train'] = ''
args2['data'] = 'all_1.npy'
args2['norm'] = ''


name = 'all5'
logfolder = 'interpolation/{0}/'.format(name)
time_hours = 0
time_minutes = 15
cores_per_job = 4
mem = 24
num_seeds = 3
seed_offset = 0
constraint = 'volta16gb'
ckp_name = name
fp16 = False

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
partition = 'learnfair'
#partition = 'uninterrupted'
#partition = 'dev'
change_dir = 'fairseq_private/'
repo = 'fairseq_private'
exclude = 'learnfair0285,learnfair0405'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
#args3['lbl'] = ['103', '25,50,75,103', '10,15,25,50,75,103']
args3['lbl'] = ['50']
args3['epochs'] = [50, 125]
args3['num-picks'] = [10, 25, 50]
args3['lr'] = [0.0006]
args3['hidden-size'] = [1024]
#args3['wt'] = ['1,2']
#args3['wt'] = ['1,2', '2,3', '1,3', '1,2,3']
#args3['wt'] = ['2,3', '3,4', '2,4', '2,3,4']
#args3['wt'] = ['1,2,3']
#args3['wt'] = ['1,2,3', '2,3,4', '3,4,5', '4,5,7', '5,7,10', '7,10,15', '10,15,25', '15,25,50', '25,50,75']
args3['layers'] = [1]
args3['input-drop'] = [0.0, 0.05, 0.1]
args3['dropout'] = [0.1, 0.2]
args3['batch-size'] = [128]
#args3['max-variation'] = ['dropout','decoder_embed_dim', 'decoder_ffn_embed_dim','decoder_layers', 'attention_dropout']
#args3['exclude-variation'] = ['dropout','decoder_embed_dim', 'decoder_ffn_embed_dim','decoder_layers', 'attention_dropout']

#args3['data-format'] = ['seq-len', 'mean', 'exact-seq']
args3['data-format'] = ['exact-seq']
#args3['data-format'] = ['seq-len']
#args3['data-format'] = ['flat']
args3['seq-len'] = [192]
args3[''] = ['linear-feats --cfg-feats', 'linear-feats', 'cfg-feats', '']
args4 = []

args_prod = []
for key, values in args3.items():
    if len(key) == 0:
        keyvalues = [' --{0}'.format(v) if len(v) > 0 else '{0}'.format(v) for v in values]
    else:
        keyvalues = [' --{0} {1}'.format(key, v) for v in values]
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
            job_cmd = cmd + ' --seed {0} '.format(seed) + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            if not fp16: job_cmd = job_cmd.replace('--fp16', '')
            jobs.append(job_cmd)
            s.add_job(logfolder, repo, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(partition))

if not args.dry:
    s.run_jobs()

