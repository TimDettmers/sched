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

cmd = 'OMP_NUM_THREADS=1 python train.py --cuda --data ../data/wikitext-2/ --dataset wt103 --adaptive --gpu0_bsz 1 --fp16 --dynamic-loss-scale --log-interval 50 --eval-interval 400'

args2 = {}
#args2['n_layer'] = 16
args2['d_model'] = 400
args2['n_head'] = 10
args2['d_head'] = 40
#args2['d_inner'] = 2000
#args2['dropout'] = 0.0
#args2['dropatt'] = 0.0
#args2['lr'] = 0.0007
args2['warmup_step'] = 3000
#args2['max_step'] = 25000
args2['tgt_len'] = 150
args2['mem_len'] = 150
args2['eval_tgt_len'] = 150
args2['batch_size'] = 32
#args2['dropouti'] = 0.0
#args2['dropouto'] = 0.0
#args2['dropoute'] = 0.0



#12 with output drop at final layer; 11 without
# 15 with wdecay, a bit longer training
# 16 with wdecay, a bit longer training; and standard pre-softmax scaling
# 17 with wdecay, a bit longer training; and standard pre-softmax scaling; remove dropout after final hidden layer
# 18 with wdecay, a bit longer training; and standard pre-softmax scaling
# 19 with xavier
logfolder = 'replication/{0}/'.format('grid18')
#time_hours = 24*14
time_hours = 24
cores_per_job = 4
num_seeds = 1
seed_offset = 0

account = 'cse'
#account = 'stf'
#account = 'ark'
change_dir = 'transformer-xl/pytorch'
repo = 'transformer-xl'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=account + '-gpu')
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3['lr'] = [0.00035]
args3['max_step'] = [125000]
args3['dropouti'] = [0.6]
args3['dropouto'] = [0.5]
#args3['dropouti'] = [0.6, 0.5, 0.7]
#args3['dropouto'] = [0.5, 0.6, 0.4]
args3['dropoute'] = [0.2]
#args3['dropoute'] = [0.15, 0.2, 0.25]
#args3['dropout'] = [0.2, 0.15, 0.25]
#args3['dropatt'] = [0.2, 0.15, 0.25]
args3['dropout'] = [0.2]
args3['dropatt'] = [0.2]
args3['d_inner'] = [900]
args3['n_layer'] = [16]

args4 = []

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
            job_cmd = cmd + ' --work-dir /gscratch/scrubbed/dettmers/{0} '.format(str(uuid.uuid4())) + ' --seed {0} '.format(seed) + arg4
            #job_cmd = cmd + ' --seed {0} '.format(seed) + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, repo, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(account))

if not args.dry:
    s.run_jobs()

