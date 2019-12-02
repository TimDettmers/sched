import itertools
import gpuscheduler
import argparse
import os
import uuid
from itertools import product

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
log_base = '/gscratch/cse/dettmers/logs'

cmd = 'OMP_NUM_THREADS=1 python control-tasks/run_experiment.py'

random_weight_list = [\
#'./data/raw.train.kaiming-uniform.elmo-layers.hdf5',
#'./data/raw.train.N00.0001.elmo-layers.hdf5',
#'./data/raw.train.N00.001.elmo-layers.hdf5',
'./data/raw.train.N00.01.elmo-layers.hdf5',
'./data/raw.train.xavier-normal.elmo-layers.hdf5',
'./data/raw.train.xavier-uniform.elmo-layers.hdf5',
#'./data/raw.train.xavier-uniform_N00.001.elmo-layers.hdf5',
'./data/raw.train.xavier-uniform_N00.01.elmo-layers.hdf5',
#'./data/raw.train.xavier-uniform_N00.1.elmo-layers.hdf5'
]

args2 = {}
#args2['task'] = 'pos'
#args2['task'] = 'edge'
#args2['task'] = 'corrupted-edge'
#args2['rank'] = 1000
#args2['epochs'] = 40
#args2['l2'] = 0.00
#args2['momentum'] = 0.9
args2['optim'] = 'adam'
args2['scheduler'] = 'cosine'
#args2['scheduler'] = 'none'
#args2['emb-train-path'] = './data/raw.train.elmo-layers.hdf5'
args2['emb-train-path'] = './data/raw.train.xavier-normal.elmo-layers.hdf5'
#args2['emb-train-path'] = './data/raw.train.xavier-uniform.elmo-layers.hdf5'
#args2['layer'] = 2
args2['lr'] = 0.001
args2['patience'] = 50
#args2['patience'] = 4
args2['layernorm'] = ''
args2['test'] = ''
args2['random'] = ''
args2['temp-path'] = '/gscratch/scrubbed/dettmers/'
#args2['type'] = 'bilinear'
#args2['type'] = 'linear'

time_hours = 2
cores_per_job = 5
num_seeds = 3
seed_offset = 0
mem_GB = 32

#account = 'stf'
account = 'cse'

s = gpuscheduler.HyakScheduler('/gscratch/cse/dettmers/git/sched/config/', verbose=args.verbose, account=account, partition=account + '-gpu')

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
#args3['epochs'] = [15, 40]
#args3['epochs'] = [15, 40, 100]
#args3['lr'] = [0.0006, 0.003, 0.001]
#args3['l2'] = [0.0, 0.01]
args3['layer'] = [1,2]
#args3['task'] = ['pos', 'corrupted-pos']
#args3['task'] = ['pos', 'corrupted-pos', 'edge', 'corrupted-edge']
args3['topk'] = [1024, 786, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0]
#args3['topk'] = [8, 768, 0, 2, 1, 32, 64]
#args3['topk'] = [768, 1024]
#args3['topk'] = [1024, 768, 256, 128, 2]
#args3['topk'] = [1024, 512, 128, 32, 16, 4, 2, 0]
#args3['topk'] = [512]
#args3['patience'] = [4, 50]
#args3['emb-train-path'] = random_weight_list
#args3['scheduler'] = ['cosine', 'plateau', 'none']

logfolder = 'probe_random/{0}/'.format('double_rand')
args4 = []
#args4.append('--task pos --type linear --epochs 40 --direct')
#args4.append('--task corrupted-pos --type linear --epochs 40 --direct')
# baseline
args4.append('--task edge --type bilinear --epochs 15 --rank 1000')
#args4.append('--task corrupted-edge --type bilinear --epochs 15 --rank 1000')
#args4.append('--task pos --type linear --epochs 40 --rank 1000')
#args4.append('--task corrupted-pos --type linear --epochs 40 --rank 1000')

#args4.append('--task edge --type bilinear --rank 1000')
#args4.append('--task corrupted-edge --type bilinear --rank 1000 --scheduler plateau --patience 4 --epochs 40 --lr 0.001')
#args4.append('--task pos --type linear --rank 1000')
#args4.append('--task corrupted-pos --type linear --rank 1000 --scheduler plateau --patience 4 --epochs 40 --lr 0.003')

# control
#args4.append('--task edge --type bilinear --epochs 15 --rank 1000 --l2 0.01')
#args4.append('--task corrupted-edge --type bilinear --epochs 15 --rank 1000 --l2 0.01')
#args4.append('--task pos --type linear --epochs 40 --rank 10')
#args4.append('--task corrupted-pos --type linear --epochs 40 --rank 10')

#args4.append('--task corrupted-edge --type bilinear --rank 1000 --scheduler plateau --patience 4 --epochs 40 --lr 0.001 --l2 0.01')
#args4.append('--task corrupted-pos --type linear --rank 10 --scheduler plateau --patience 4 --epochs 40 --lr 0.003')

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

seed_offset = 0

jobs = []

if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            fp16 = False
            job_cmd = cmd + ' --seed {0} --results-dir /tmp/probe/{1} '.format(seed, uuid.uuid4()) + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            job_cmd += ' base_config.yaml'
            jobs.append(job_cmd)
            s.add_job(logfolder, 'probe/', job_cmd, time_hours, fp16, cores=cores_per_job, mem=mem_GB)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(account))

if not args.dry:
    s.run_jobs(log_base, add_fp16=True)

