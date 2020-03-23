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

#account = 'stf'
account = 'cse'

log_base = '/gscratch/cse/dettmers/logs'
s = gpuscheduler.HyakScheduler('/gscratch/cse/dettmers/git/sched/config/', verbose=args.verbose, account=account, partition=account + '-gpu')

#log_base = '/home/tim/logs/'
#s = gpuscheduler.SshScheduler('/home/tim/data/git/sched/config/', verbose=args.verbose)
#s.update_host_config('office', mem_threshold=1700, util_threshold=25)


cmd = 'OMP_NUM_THREADS=1 python main.py'

args2 = {}
args2['input-drop'] = 0.2

logfolder = 'conve/{0}/'.format('fixed_label_smoothing3')
time_hours = 4
cores_per_job = 5
num_seeds = 2
seed_offset = 0

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3['epochs'] = [60, 100, 150]
args3['label-smoothing'] = [0.1, 0.2, 0.3, 0.4, 0.5]

args4 = []
args4.append('--data FB15k-237 --hidden-drop 0.5 --lr 0.003')
args4.append('--data FB15k-237 --hidden-drop 0.5 --lr 0.001')
args4.append('--data WN18RR --hidden-drop 0.3 --lr 0.003')

args_prod = []
for key, values in args3.items():
    if len(key) == 0:
        keyvalues = [' --{0}'.format(v) if len(v) > 0 else '{0}'.format(v) for v in values]
    else:
        keyvalues = [' --{0} {1}'.format(key, v) for v in values]
    args_prod.append(keyvalues)

if len(args_prod) >= 2:
    args_prod = list(product(*args_prod))
elif len(args_prod) == 1:
    new_args = []
    for arg in args_prod[0]:
        new_args.append([arg])
    args_prod = new_args


jobs = []
if len(args4) == 0: args4.append('')
if len(args_prod) == 0: args_prod.append(('', ''))
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        for i, values in enumerate(args_prod):
            fp16 = False
            job_cmd = cmd + ' --seed {0} '.format(seed) + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, 'ConvE/', job_cmd, time_hours, fp16, cores=cores_per_job)

host2cmd = {}
cmds = []
remap = {}

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))

if not args.dry:
    s.run_jobs(log_base, cmds=cmds, add_fp16=True, host2cmd_adds=host2cmd, remap=remap)

