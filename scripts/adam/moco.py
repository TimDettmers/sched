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


gpus = 8

cmd = ' python main_moco.py  -a resnet50  --lr 0.03  --batch-size 256  --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 /datasets01/imagenet_full_size/061417/ --mlp --moco-t 0.2 --aug-plus --cos'
cmd2 = 'python main_lincls.py  -a resnet50  --lr 30.0  --batch-size 256  --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0  /datasets01/imagenet_full_size/061417/'
args2 = {}

name = 'blockwise1'
constraint = 'volta'

logfolder = 'adam/moco_v2/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 5
mem = 48*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 20
time_hours = 72
time_minutes = 0

partition = 'prioritylab,learnlab,learnfair'
change_dir = '/private/home/timdettmers/git/moco/'
repo = 'moco'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)

fp16 = False
args3 = {}
#args3[('adam-bits', 'percentile-clipping', 'adam8bits-method')] = [(32, 100, 'quantile'), (8, 100, 'quantile'), (8, 100, 'dynamic_tree')]
#args3[('adam-bits', 'percentile-clipping', 'adam8bits-method')] = [(32, 100, 'quantile')]
#args3[('adam-bits', 'percentile-clipping', 'adam8bits-method')] = [(8, 100, 'quantile'), (8, 100, 'dynamic_tree'), (32, 100, 'quantile')]
#args3[('adam-bits', 'percentile-clipping', 'adam8bits-method')] = [(8, 100, 'quantile')]
#args3[('adam-bits', 'percentile-clipping', 'adam8bits-method')] = [(8, 100, 'dynamic_tree')]
#args3[('adam-bits', 'percentile-clipping', 'adam8bits-method')] = [(32, 100, 'quantile'), (8, 100, 'quantile'), (8, 100, 'dynamic_tree')]

args3['adam-bits'] = [8, 32]



args4 = []

args5 = {}

args6 = {}

rdm = np.random.RandomState(5345)

for key, value in args2.items():
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
            job_cmd2 = cmd2 + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
                job_cmd2 += ' {0}' .format(val)
            #job_cmd += ' --checkpoint /checkpoint/timdettmers/{1}/{0}/model.pt'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            if not fp16: job_cmd = job_cmd.replace('--fp16 ', ' ')
            if not fp16: job_cmd2 = job_cmd2.replace('--fp16 ', ' ')
            job_cmd = job_cmd + ' --seed {0}'.format(seed)
            job_cmd2 = job_cmd2 + ' --seed {0}'.format(seed)
            checkpoint_dir = '/checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            checkpoint_dir2 = '/checkpoint/timdettmers/{1}/finetune_{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            save_dir = ' --save-dir {0}'.format(checkpoint_dir)
            save_dir2 = ' --save-dir {0}'.format(checkpoint_dir2)
            pretrained = ' --pretrained {0}/checkpoint_0199.pth.tar '.format(checkpoint_dir.strip())
            #raport = ' --raport-file {0}/raport_file.json '.format(checkpoint_dir.strip())
            os.makedirs(checkpoint_dir.strip(), exist_ok=True)
            job_cmd = job_cmd + save_dir
            job_cmd2 = job_cmd2 + save_dir2 + pretrained
            #cmds = [job_cmd, job_cmd2]
            cmds = [job_cmd]
            #cmds = [job_cmd2]
            if rdm.rand(1) <= args.p:
                jobs.append(cmds[0])
                s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('Jobs will be written to: {0}'.format(join('/private/home/timdettmers/logs/', logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs(single_process=True, comment='"ICLR internal review deadline 2019-09-28"')#, log_id='2139b6c61747dbd7cc9512638f90ef97')

