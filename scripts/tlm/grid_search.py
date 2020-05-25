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


#cmd = 'bash experiments/text8.sh {0}'.format(gpus)
#cmd = 'bash experiments/text8_apeximproved.sh 8'
gpus = 8
cmd = 'python -m torch.distributed.launch --nproc_per_node={0} --master_port 9919 main.py --data data/text8 --adapt-span --adapt-span-cache --distributed'.format(gpus)

args2 = {}
#args2['word-nlayers'] = 6
args2['word-hid-sz'] = 512
args2['word-inner-hid-sz'] = 2048
args2['word-nheads'] = 8
args2['word-attn-span'] = 8192
args2['word-block-sz'] = 512
#args2['char-nlayers'] = 1
args2['char-hid-sz'] = 512
args2['char-inner-hid-sz'] = 2048
args2['char-nheads'] = 8
args2['char-attn-span'] = 8192
args2['char-block-sz'] = 512
args2['char-vocab-size'] = 33
args2['lr'] = 0.07
args2['momentum'] =  0
args2['dropout'] = 0.3
args2['optim'] = 'adagrad'
args2['lr-warmup'] = 32000
args2['grad-clip'] = 0.03
args2['niter'] = 900
args2['adapt-span-loss'] = 0.0000005
args2['word-max-len'] = 15
args2['batch-sz'] = 64
args2['nbatches'] = 1000
args2['batch-split'] = 2
args2['fused-layernorm'] = ''
args2['init'] = 'xavier_uniform'

name = 'grid2'
ckp_name = name
logfolder = 'tlm/{0}/'.format(name)
#time_hours = 24*2
cores_per_job = gpus*4
mem = 128
num_seeds = 1
seed_offset = 0
constraint = 'volta32gb'

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
partition = 'learnfair'
#partition = 'uninterrupted'
#partition = 'dev'
change_dir = 'tlm/'
repo = 'tlm'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

fp16 = True
args3 = {}
#args3['init'] = ['xavier_uniform', 'sparse --init-sparsity 0.1']
#args3[''] = ['norm-comb-embs', '']
#args3[''] = ['fused-layernorm', '', 'char-tie-embs', 'fused-layernorm  --char-tie-embs']
args3['word-nlayers'] = [6,8,10,12]
args3['char-nlayers'] = [2,4,6]

args4 = []
time_hours = 72
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
            job_cmd += ' --checkpoint /checkpoint/timdettmers/{1}/{0}/model.pt'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            if not fp16: job_cmd = job_cmd.replace('--fp16', '')
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

