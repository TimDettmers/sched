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


cmd = 'OMP_NUM_THREADS=1 python train.py --cuda --data ../data/wikitext-2/ --dataset wt103 --adaptive --n_layer 12 --optim adam --tgt_len 150 --mem_len 150 --eval_tgt_len 150 --fp16 --dynamic-loss-scale --eval-interval 100 --log-interval 10'

args2 = {}
#args2['conv'] = ''
#args2['dim2'] = ''
#args2['shape2'] = 2
#args2['kernel-size'] = 3
#args2['downsample-identity'] = ''
args2['d_emb'] = 400
#args2['d_model'] = 400
args2['n_head'] = 1
#args2['d_head'] = 40
#args2['d_inner'] = 2900
#args2['dropout'] = 0.0
args2['batch_chunk'] = 1
args2['batch_size'] = 32
#args2['dropatt'] = 0.0
#args2['use-batchnorm'] = ''
#args2['lr'] = 0.0006
#args2['max_step'] = 3000
#args2['warmup_step'] = 100
args2['multifilter'] = ''
#args2['index-std'] = 5.0




logfolder = 'fixed_multifilter/{0}/'.format('analysis')
time_hours = 4
cores_per_job = 5
num_seeds = 3

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3['d_model'] = [400]
#args3['n_head'] = [5]
#args3['d_head'] = [40]
args3['lr'] = [0.0006]
args3['max_step'] = [6000]
args3['warmup_step'] = [200]
#args3['d_model'] = [400]
#args3['d_head'] = [20]
#args3['n_head'] = [1]
#args3['dropout'] = [0.0]#, 0.15, 0.2]
args3['dropatt'] = [0.0]
args3['dropout'] = [0.1]
args3['batch_size'] = [32]
args3['filter-std'] = [5]
args3['index-std'] = [1]
args3['num-filters'] = [25, 50, 75, 100]
args3['filter-size'] = [5, 10, 25]

args4 = []

# 2.5 inner = 1 head
heads = [30]
inners = []
for head in heads:
    inners.append(int(2900-((head-20)*2.5)))

for head, inner in zip(heads, inners):
    args4.append('--d_head {0} --d_inner {1} '.format(head, inner))

#args4.append('--index-std 3 --filter-std 5 ')
#args4.append('--index-std 5 --filter-std 15 ')
#args4.append('--index-std 5 --filter-std 25 ')

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
    for arg in args_prod[0]:
        new_args.append([arg])
    args_prod = new_args

seed_offset = 0

jobs = []
if len(args4) == 0: args4.append('')
if len(args_prod) == 0: args_prod.append(('', ''))
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        for i, values in enumerate(args_prod):
            fp16 = False
            job_cmd = cmd + ' --seed {0} --work_dir=/gscratch/scrubbed/dettmers/{1} '.format(seed, uuid.uuid4()) + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            jobs.append(job_cmd)
            s.add_job(logfolder, 'convtransformer/pytorch/', job_cmd, time_hours, fp16, cores=cores_per_job)

host2cmd = {}
host2cmd['ofir3'] = ' --max-threads 4'

cmds = []
#cmds = ['git stash', 'git checkout 9bf460346ae133d5632066c4364e7d70437a1559'] # O1
#cmds = ['git stash', 'git checkout 85e6f84d7f5c2e92752f87994e1a71ffca4973d9'] # O2
#cmds = ['git stash', 'git checkout 24f59a80352c512106d0f3134fcf71b49ed6065e'] # O2 no float loss
#cmds = ['git stash', 'git checkout master', 'git pull']
#cmds = ['cd $GIT_HOME', 'git clone git@github.com:TimDettmers/convtransformer.git', 'cd convtransformer', 'git checkout max_out_sim', 'bash getdata.sh', 'cd pytorch']
cmds = cmds + ['git stash', 'git checkout master', 'git pull', 'git checkout conv_replication', 'git pull']

remap = {}
remap[('ofir4', 0)] = 1
remap[('ofir4', 1)] = 0
remap[('ofir1', 0)] = 1
remap[('ofir2', 1)] = 0
remap[('ofir2', 0)] = 1
remap[('ofir1', 1)] = 0
remap[('shoob', 2)] = 0
remap[('shoob', 0)] = 2

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))

if not args.dry:
    s.run_jobs(log_base, cmds=cmds, add_fp16=True, host2cmd_adds=host2cmd, remap=remap)

