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
log_base = '/usr/lusers/dettmers/logs/'

cmd = 'OMP_NUM_THREADS=1 python main.py'
cmd = 'fairseq-train --task language_modeling data/wikitext-103 --arch transformer_lm_wiki103 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 --clip-norm 0.1 --criterion adaptive_loss --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16 --num-workers 2 --no-progress-bar --log-interval 10'


args2 = {}
args2['max-update'] = 286000
#args2['batch-size'] = 24
args2['update-freq'] = 1
args2['optimizer'] = 'nag'
args2['max-tokens'] = 2048
# seq length
args2['tokens-per-sample'] = 2048
args2['keep-last-epochs'] = 1

args2['lr-period-updates'] = 270000
args2['max-lr'] = 1.0
args2['min-lr'] = 1e-09
args2['warmup-init-lr'] = 1e-07
args2['lr'] = 0.0001
args2['warmup-updates'] = 16000


logfolder = 'multifilter/{0}/'.format('baseline_wiki103')
time_hours = 48
cores_per_job = 40
seed_offset = 1
num_seeds = 1
num_GPUs = 8

account = 'cse'
#account = 'stf'
change_dir = 'multifilter/'

s = gpuscheduler.HyakScheduler('/gscratch/cse/dettmers/git/sched/config/', verbose=args.verbose, account=account, partition=account + '-gpu')

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
#args3['lr'] = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
#args3['batch-size'] = [64]
#args3['batch-size'] = [6, 12]
#args3['lr'] = [0.0004, 0.0003]
#args3['max-lr'] = [0.0008, 0.0005]
#args2['min-lr'] = [1e-08, 1e-07]

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

seed_offset = 0

jobs = []
fp16 = True
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + ' --seed {0} '.format(seed) + arg4 + '--save-dir /tmp/multifilter/{0}'.format(str((uuid.uuid4())))

            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job, gpus=num_GPUs)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(account))

if not args.dry:
    s.run_jobs(log_base, add_fp16=True)

