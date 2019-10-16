import itertools
import gpuscheduler
import argparse
import os
from itertools import product

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

#s = gpuscheduler.Scheduler('/home/tim/data/git/sched/config/')
s = gpuscheduler.HyakScheduler('/gscratch/scrubbed/dettmers/git/sched/config/', verbose=args.verbose)

s.update_host_config('home', mem_threshold=1700, util_threshold=30)
s.update_host_config('office', mem_threshold=1700, util_threshold=25)
#s.update_host_config('ari', mem_threshold=2500, util_threshold=25)

cmd_raw = 'OMP_NUM_THREADS=1 python train.py --cuda --data ../data/wikitext-2/ --dataset wt103 --adaptive --n_layer 12 --dropatt 0.0 --optim adam --tgt_len 150 --mem_len 150 --eval_tgt_len 150 --batch_size 32 --batch_chunk 1 --fp16 --dynamic-loss-scale --eval-interval 100 --work_dir=LM-TFM-wt103/ITER/ --log-interval 10'

emb = 400
model = 400
heads = 10
d_head = 40
inner = 2000
dropout = 0.1
lr = 0.0006

cmd = cmd_raw.format(emb, model, heads, d_head, inner, dropout, lr)


args2 = {}
args2['conv'] = ''
args2['dim2'] = ''
args2['shape2'] = 1
args2['kernel-size'] = 1
#args2['downsample-identity'] = ''
args2['d_emb'] = 400
args2['d_model'] = 400
args2['n_head'] = 10
args2['d_head'] = 40
args2['d_inner'] = 2000
args2['dropout'] = 0.1
args2['lr'] = 0.0008
args2['max_step'] = 2000
args2['warmup_step'] = 100

log_base = '/usr/lusers/dettmers/logs/'
logfolder = 'convtransformers/{0}/'.format('base2d')
time_hours = 1

cores_per_job = 2

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3[''] = ['downsample-identity', '']

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
    args_prod = [[args_prod[0][0]], [args_prod[0][1]]]

num_seeds = 2
seed_offset = 0

jobs = []
for seed in range(num_seeds):
    if len(args_prod) == 0: args_prod.append(('', ''))
    for i, values in enumerate(args_prod):
        fp16 = False
        job_cmd = cmd.replace('ITER', str(i)) + ' --seed {0}'.format(seed)
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

