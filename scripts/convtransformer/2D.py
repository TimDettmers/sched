import itertools
import gpuscheduler
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
args = parser.parse_args()

s = gpuscheduler.Scheduler('/home/tim/data/git/sched/config/')

s.update_host_config('home', mem_threshold=1700, util_threshold=30)
s.update_host_config('office', mem_threshold=1700, util_threshold=25)
#s.update_host_config('ari', mem_threshold=2500, util_threshold=25)

cmd_raw = 'OMP_NUM_THREADS=1 python train.py --cuda --data ../data/wikitext-2/ --dataset wt103 --adaptive --n_layer 12 --dropatt 0.0 --optim adam --warmup_step 800 --tgt_len 150 --mem_len 150 --eval_tgt_len 150 --batch_size 32 --batch_chunk 1 --fp16 --dynamic-loss-scale --eval-interval 500 --work_dir=LM-TFM-wt103/gpu3/ --log-interval 10'

emb = 400
model = 400
heads = 10
d_head = 40
inner = 2000
dropout = 0.1
lr = 0.0006

cmd = cmd_raw.format(emb, model, heads, d_head, inner, dropout, lr)


args2 = {}
#args2['conv'] = ''
#args2['dim2'] = ''
#args2['shape2'] = 1
#args2['kernel-size'] = 1
#args2['downsample-identity'] = ''
args2['d_emb'] = 400
args2['d_model'] = 400
args2['n_head'] = 10
args2['d_head'] = 40
args2['d_inner'] = 2000
args2['dropout'] = 0.1
#args2['lr'] = 0.0006
#args2['max_step'] = 4000
logfolder = 'base'

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3['max_step'] = [4000, 6000]
args3['lr'] = [0.0003, 0.0006]

args_prod = []
for key, values in args3.items():
    keyvalues = [' --{0} {1}'.format(key, v) for v in values]
    args_prod.append(keyvalues)

args_prod = list(product(*args_prod))


num_seeds = 4
seed_offset = 0

jobs = []
for seed in range(num_seeds):
    for key, value in args_prod:
        fp16 = False
        jobs.append(['convtransformers/{0}/'.format(logfolder), 'convtransformer/pytorch/', cmd + ' {1} {2} --seed {0}'.format(seed, key, value), fp16])

print(jobs[0])
print(len(jobs))

for job in jobs:
    s.add_job(*job)

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
        print(job[2])
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(jobs[0][0]))

if not args.dry:
    s.run_jobs('/home/tim/logs/', cmds=cmds, add_fp16=True, host2cmd_adds=host2cmd, remap=remap)

