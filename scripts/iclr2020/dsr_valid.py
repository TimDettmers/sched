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
log_base = '/usr/lusers/dettmers/data/logs/'

cmd = 'OMP_NUM_THREADS=1 python main.py --model cifar10_WideResNet --schedule-file ./learning_schedules/wrnet_schedule.yaml --initial-sparsity-fc 0.00 --widen-factor 2 --rewire --sub-kernel-granularity 4 --epochs 250 --weight-decay 5.0e-4 --threshold-prune --n-prune-params 20000 --rewire-scaling --stop-rewire-epoch 240 --validate-set'

args2 = {}
#args2['data'] = 'cifar'
#args2['decay_frequency'] = 30000
#args2['epochs'] = 250
##args2['data'] = 'mnist'
##args2['epochs'] = 100
##args2['decay_frequency'] = 40
#args2['prune-rate'] = 0.2
#args2['fp16'] = ''
#args2['valid_split'] = 0.1
#args2['max-threads'] = 6
#args2['verbose'] = ''
#args2['growth'] = 'momentum'
#args2['redistribution'] = 'momentum'


logfolder = 'iclr2020/{0}/'.format('dsr')
time_hours = 8
cores_per_job = 3
num_seeds = 4
seed_offset = 1

#account = 'cse'
account = 'stf'
#account = 'ckpt'
change_dir = 'dynamic-reparameterization/'

s = gpuscheduler.HyakScheduler('/gscratch/cse/dettmers/git/sched/config/', verbose=args.verbose, account=account, partition=account + '-gpu')

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
#args3['lr'] = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
args3['initial-sparsity-conv'] = [0.5, 0.6, 0.7, 0.8, 0.9]

args4 = []
#args4.append('--model vgg-d --density 0.05')
#args4.append('--model alexnet-b --density 0.10')
#args4.append('--model wrn-16-8 --density 0.05')
#args4.append('--model lenet300-100 --density 0.05')


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
fp16 = True
job_idx = seed_offset*len(args_prod)
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + ' --seed {0} --job-idx {1}'.format(seed, job_idx) + arg4
            job_idx += 1
            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(account))

if not args.dry:
    s.run_jobs(log_base, add_fp16=True)

