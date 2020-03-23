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
cmd = 'OMP_NUM_THREADS=1 fairseq-train --task language_modeling --arch transformer_lm --share-decoder-input-output-embed   --sample-break-mode none --ddp-backend=no_c10d --batch-size 16  --log-format simple --no-save --log-interval 50 --fp16 --save-dir /gscratch/scrubbed/dettmers/ '

args2 = {}
args2['warmup-updates'] = 400
args2['update-freq'] = 1
args2['optimizer'] = 'adam'
args2['adam-betas'] = "'(0.9, 0.98)'"

args2['max-tokens'] = 2048
args2['clip-norm'] = 0.0003
#args2['weight-decay'] = 1e-07
#args2['dropout'] = 0.35
#args2['attention-dropout'] = 0.2
#args2['activation-dropout'] = 0.0
# seq length
args2['tokens-per-sample'] = 128
args2['keep-last-epochs'] = 1
args2['lr-scheduler'] = 'inverse_sqrt'

#args2['lr-period-updates'] = 270000
#args2['max-lr'] = 1.0
#args2['max-update'] = 25000
args2['min-lr'] = 1e-09
args2['warmup-init-lr'] = 1e-07
args2['lr'] = 0.0007
#args2['decoder-embed-dim'] = 400
#args2['decoder-ffn-embed-dim'] = 2000
#args2['decoder-layers'] = 16
args2['decoder-attention-heads'] = 10
args2['decay'] = 1.0


logfolder = 'interpolation/{0}/'.format('grid1')
time_hours = 2
cores_per_job = 4
num_seeds = 1
seed_offset = 0

account = 'cse'
#account = 'stf'
#account = 'ark'
change_dir = 'fairseq_private/'
repo = 'fairseq_private'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=account + '-gpu')
#s = gpuscheduler.SshScheduler(verbose=args.verbose)

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
#args3['max-update'] = [25000]
args3['weight-decay'] = [1e-08]
args3['dropout'] = [0.3]
args3['attention-dropout'] = [0.20]
#args3['activation-dropout'] = [0.0, 0.05, 0.1]
args3['activation-dropout'] = [0.0]
args3['decoder-embed-dim'] = [200, 400]
args3['decoder-ffn-embed-dim'] = [2000, 4000]
args3['decoder-layers'] = [8, 16]
#args3['beta'] = [0.4]
#args3['epsilon'] = [0.0]
#args3['method'] = ['inverse_rescaled_sum', 'inverse_rescaled_max', 'percentile', 'topk_sum']
#args3['method'] = ['max']
#args3['epsilon'] = [0.1, 0.2, 0.5, 0.7]
#args3['decay'] = [0.995, 0.99, 0.98, 0.95]
args4 = []
args4.append(' data/wikitext-2 --max-update 25000')
args4.append(' data/wikitext-5 --max-update 62500')
args4.append(' data/wikitext-7 --max-update 87500')
args4.append(' data/wikitext-10 --max-update 125000')
args4.append(' data/wikitext-15 --max-update 187500')


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
            job_cmd = cmd + ' --seed {0} '.format(seed) + arg4
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

