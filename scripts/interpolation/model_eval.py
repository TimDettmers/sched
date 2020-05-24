import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
from itertools import product

from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

cmd = 'python eval_models.py'

args2 = {}
args2['folder'] = 'lightconv_loss_cpy'
args2['folder_output'] = 'lightconv_loss_evaled'
#args2['folder'] = 'clean_loss_extended'
#args2['folder_output'] = 'clean_loss_full_eval'

name = 'model_eval'
logfolder = 'interpolation/{0}/'.format(name)
#time_hours = 24*2
cores_per_job = 4
mem = 24
num_seeds = 1
seed_offset = 0
constraint = ''
ckp_name = name
num_jobs = 96

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

fp16 = True
args3 = {}
args4 = []
time_hours = 0
time_minutes = 15
path = join('/private/home/timdettmers/git/', change_dir, args2['folder'], '*')

files = list(glob.iglob(path))
n = len(files)
print('Total files: {0}'.format(n))
print('Write to: {0}'.format(args2['folder_output']))

for start in list(range(0,n,n//num_jobs)):
    end = start + n//num_jobs
    args4.append(' --start-idx {0} --end-idx {1} '.format(start, end))


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
            job_cmd = cmd + arg4
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

