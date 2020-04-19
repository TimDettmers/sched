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

cmd = 'OMP_NUM_THREADS=1 python main.py'

args2 = {}
args2['batch-size'] = 128
args2['data'] = 'cifar'
#args2['epochs'] = 250
#args2['model'] = 'vgg-d'
args2['wave'] = ''
args2['no-batchnorm'] = ''

# grid6 dropout/4
# grid7 dropout/2
logfolder = 'feature_collapse/{0}/'.format('grid7')
time_hours = 4
cores_per_job = 4
mem = 12
num_seeds = 1
seed_offset = 0

account = 'cse-ckpt'
#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = account = '-gpu'
partition = 'ckpt-gpu'
change_dir = 'sparse_learning/mnist_cifar/'
repo = 'sparse_learning'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=partition)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3['lowlr'] = [0.03, 0.01]
args3['midlr'] = [0.01]
args3['highlr'] = [0.1]
#args3['decay_frequency'] = [15000, 30000]
args3['epochs'] = [250]
args3['jump-freq'] = [3, 5, 7, 10]
args3['dropout'] = [0.1, 0.3, 0.5]
args3['model'] = ['vgg-e']

args4 = []
#args4.append(' --lr-schedule multiplicative --decay-rate 0.97 ')
#args4.append(' --lr-schedule multiplicative --decay-rate 0.985 ')
args4.append(' --lr-schedule cosine ')
#args4.append(' --lr-schedule step --decay-rate 0.1 --decay_frequency 30000 ')

# good 30k 0.1 + dropout 0.2 >  30k 0.1 + dropout 0.0 + lowr 0.03
# okay 0.2 / 15k
# bad 0.33

# cosine + 0.03
# multiplicative + 0.98

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
fp16 = False
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + ' --seed {0} '.format(seed) + arg4# + ' ' + dataset
            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, repo, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job, mem=mem)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(account))

if not args.dry:
    s.run_jobs()

