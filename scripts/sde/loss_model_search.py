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


cmd = 'python fit_loss_data.py --sde-verbose'


name = 'loss_grid24'
ckp_name = name
logfolder = 'sde/{0}/'.format(name)
cores_per_job = 5
mem = 128
num_seeds = 3
seed_offset = 0
constraint = ''
gpus = 1
time_hours = 0
time_minutes = 240

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
partition = 'learnfair'
#partition = 'uninterrupted'
#partition = 'dev'
change_dir = 'sparse_learning/mnist_cifar/'
repo = 'sparse_learning'
exclude = ''
fp16 = False

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

# basic arguments
args2 = {}
#args2['sde-num-picks'] = 5
args2['sde-folder'] = 'hook_data/cifar10-200-full-grid_full7/'
#args2['sde-folder'] = 'hook_data/cifar10-12-loss-grid_full1/'
args2['folder-to'] = 'hook_data/cifar10-200-full-grid_full7/'
args2['sde-restrict'] = 'model=models-mobilev2'
#args2['sde-use-config-feats'] = ''
args2['sde-remove-outliers'] = ''
#args2['sde-normalizer'] =  'std'
#args2['sde-use-encoder'] = ''
#args2['sde-use-model-feats'] = ''
#args2['sde-model-feat-type'] = 'one-hot-int'

# grid search arguments
args3 = {}
args3['sde-batch-size'] = [128]
args3['sde-epochs'] = [75]
args3['sde-lr'] = [0.0006]
args3['sde-dropout'] = [0.2]
args3['sde-layers'] = [1]
args3['sde-hidden-size'] = [1024]
args3['sde-input-drop'] = [0.0]
args3['sde-min-lr'] = [1e-06]
args3['subset-from'] = [0.1]
args3['sde-num-picks'] = [200]
args3['sde-use-config-feats'] = []
#args3['sde-remove-outliers'] = []
args3['sde-normalizer'] =  ['std']
args3['sde-use-encoder'] = []
args3['sde-use-model-feats'] = []
args3['sde-limit'] = [3000, -1]
args3['sde-k-shot-eval'] = [1]
args3['sde-use-emb'] = []
args3['sde-use-config-feats'] = ''
args3['sde-input-transform'] = ['logsoftmax', 'softmax', 'none']
args3['sde-output-transform'] = ['logsoftmax', 'softmax', 'crossentropy']


# conditional grid search arguments
args4 = []

# string matching conditional grid search arguments
args5 = {}
args5['sde-use-model-feats'] = {'sde-model-feat-type': ['one-hot-int', 'continuous']}


for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args_prod = []
for key, values in args3.items():
    if len(key) == 0:
        keyvalues = [' --{0}'.format(v) if len(v) > 0 else '{0}'.format(v) for v in values]
    elif len(values) == 0:
        keyvalues = [' --{0}'.format(key), ' ']
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
            job_cmd = cmd + ' --sde-seed {0} '.format(seed) + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            if not fp16: job_cmd = job_cmd.replace('--fp16', '')
            if any([k in job_cmd for k in args5.keys()]):
                for substr, pdict in args5.items():
                    if substr in job_cmd:
                        for key, values in pdict.items():
                            for v in values:
                                job_cmd5 = job_cmd + ' --{0} {1}'.format(key, v)
                                jobs.append(job_cmd5)
                                s.add_job(logfolder, repo, change_dir, job_cmd5, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                jobs.append(job_cmd)
                s.add_job(logfolder, repo, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(partition))

if not args.dry:
    s.run_jobs()

