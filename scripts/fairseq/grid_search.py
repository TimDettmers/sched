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

cmd = 'fairseq-train --task language_modeling data/wikitext-2 --arch transformer_lm --share-decoder-input-output-embed \
  --sample-break-mode none \
  --fp16 --ddp-backend=no_c10d'


args2 = {}
#args2['max-update'] = 5000
args2['warmup-updates'] = 400
#args2['batch-size'] = 24
args2['update-freq'] = 1
args2['optimizer'] = 'adam'
args2['adam-betas'] = "'(0.9, 0.98)'"

args2['max-tokens'] = 2048
#args2['weight-decay'] = 0.00
#args2['clip-norm'] = 0.0
#args2['dropout'] = 0.0
# seq length
#args2['tokens-per-sample'] = 512
args2['keep-last-epochs'] = 1
args2['lr-scheduler'] = 'inverse_sqrt'

#args2['lr-period-updates'] = 270000
#args2['max-lr'] = 1.0
args2['min-lr'] = 1e-09
args2['warmup-init-lr'] = 1e-07
#args2['lr'] = 0.0005
#args2['decoder-embed-dim'] = 512
#args2['decoder-ffn-embed-dim'] = 2048
#args2['decoder-layers'] = 6
#args2['decoder-attention-heads'] = 8
#gelu
#[--decoder-embed-dim N]
#                     [--decoder-output-dim N] [--decoder-input-dim N]
# [--decoder-learned-pos]



logfolder = 'fairseq/{0}/'.format('base11')
time_hours = 2
cores_per_job = 5
seed_offset = 1
num_seeds = 1
num_GPUs = 1

#account = 'ckpt'
account = 'cse'
#account = 'stf'
change_dir = 'fairseq_private/'

s = gpuscheduler.HyakScheduler('/gscratch/cse/dettmers/git/sched/config/', verbose=args.verbose, account=account, partition=account + '-gpu')

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
#args3['lr'] = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]
#args3['batch-size'] = [64]
#args3['batch-size'] = [6, 12]
args3['lr'] = [0.0007]
args3['batch-size'] = [16 ]
args3['max-update'] = [25000 ]
args3['decoder-embed-dim'] = [400]
args3['decoder-ffn-embed-dim'] = [2000]
args3['decoder-layers'] = [16]
args3['tokens-per-sample'] = [128]
args3['decoder-attention-heads'] = [10]
args3['weight-decay'] = [1e-07]
args3['clip-norm'] = [0.0003]
args3['dropout'] = [0.35]
args3['attention-dropout'] = [0.2]
args3['activation-dropout'] = [0.0]
#args3['adaptive-softmax-cutoff'] = ['5000,10000', '10000,20000']
#args3['criterion'] = ['cross_entropy', 'adaptive_loss']
# adaptive_softmax_cutoff None
# adaptive_softmax_factor 4
# decoder-learned-pos False
# activation-fn relu

#args3['max-lr'] = [0.0008, 0.0005]
#args2['min-lr'] = [1e-08, 1e-07]

args4 = []
#args4.append('--decoder-embed-dim 768 --decoder-ffn-embed-dim 768 ')
#args4.append('--decoder-embed-dim 768 --decoder-ffn-embed-dim 1536 ')
#args4.append('--decoder-embed-dim 768 --decoder-ffn-embed-dim 2304 ')


# 1. figure out how much updates I need for convergence (analyze training curves)
# 2. tune learning rates
# 3. search for overfitting threshold with a simple model
# 4. Tune tokens per sample, dropout and other parameters


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
            job_cmd = cmd + ' --seed {0} '.format(seed) + arg4 + '--save-dir /gscratch/scrubbed/dettmers/tmp/{0}'.format(str((uuid.uuid4())))

            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, change_dir, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job, gpus=num_GPUs)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(account))

if not args.dry:
    s.run_jobs(log_base, add_fp16=True)

