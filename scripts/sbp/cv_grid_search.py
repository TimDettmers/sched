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

cmd = 'OMP_NUM_THREADS=1 python main.py --sbp --dense'

args2 = {}
args2['decay_frequency'] = 30000
args2['batch-size'] = 128
args2['data'] = 'cifar'
#args2['model'] = 'lenet5'
args2['epochs'] = 250
args2['epsilon'] = 0.0
#args2['beta'] = 2.0
#args2['method'] = 'simple'

# grid3 = deterministic
# grid4 = random
logfolder = 'sbp/{0}/'.format('cv_grid4')
time_hours = 4
cores_per_job = 4
mem = 16
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
args3['model'] = ['wrn-16-8']
#args3['method'] = ['KLmin']
args3['method'] = ['simple']
#args3['method'] = ['KLtopk']
#args3['method'] = ['KLmin', 'KLtopk']
args3['beta'] = [2.0, 4.0, 6.0]
#args3['history-size'] = [1000, 2000, 5000]
args3['history-size'] = [500, 2000, 5000]
args3['metric-history-size'] = [500]

args4 = []
#args4.append(' --del-history --del-metric-history')
args4.append(' --del-history ')
#args4.append(' --del-metric-history')
args4.append('')


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

