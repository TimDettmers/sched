import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
from itertools import product
from torch.optim.lr_scheduler import OneCycleLR

from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


cmd = 'MKL_THREADING_LAYER=GNU python main.py --data cifar --sde-argconfig ~/git/sde/config/args_config.txt'

args2 = {}

name = 'grid_full7'
ckp_name = name
logfolder = 'sde/{0}/'.format(name)
#time_hours = 24*2
cores_per_job = 5
mem = 32
num_seeds = 10
seed_offset = 0
constraint = ''
gpus = 1

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

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

folder = './hook_data/cifar10-{0}-{1}-{2}'

fp16 = False
args3 = {}
args3['sde-subset-size'] = [0.1, 1.0]
args3['model'] = ['wrn-16-8', 'wrn-22-8', 'alexnet-s', 'alexnet-b','models-dense','models-efficient','models-google','models-mobile','models-regnext-200','models-regnext-400','models-preact-18 --lr 0.01','models-preact-50 --lr 0.01','models-resnet-18','models-resnet-50 --lr 0.01','models-resnext-2','models-shufflev2','models-dpn-26 --lr 0.01', 'models-mobilev2']

args4 = []
for epoch, metric in product([25, 200], ['full']):
    folder_name = folder.format(epoch, metric, name)
    args4.append(' --epochs {0} --metric {1} --sde-folder {2} '.format(epoch, metric, folder_name))

args5 = {}
args5['wrn'] = {'dropout' : ['0.3 --fp16']}
args5['models-'] = {'fp16' : ['']}
time_hours = 12
time_minutes = 0

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
            #job_cmd += ' --checkpoint /checkpoint/timdettmers/{1}/{0}/model.pt'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            if not fp16: job_cmd = job_cmd.replace('--fp16', '')
            if any([k in job_cmd for k in args5.keys()]):
                for substr, pdict in args5.items():
                    if substr in job_cmd:
                        for key, values in pdict.items():
                            for v in values:
                                job_cmd5 = job_cmd + ' --{0} {1}'.format(key, v)
                                job_cmd5 = job_cmd5 + ' --seed {0}'.format(seed)
                                jobs.append(job_cmd5)
                                s.add_job(logfolder, repo, change_dir, job_cmd5, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
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

