import itertools
import gpuscheduler
import argparse

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
args = parser.parse_args()

s = gpuscheduler.Scheduler('/home/tim/git/sched/config/')

s.update_host_config('home', mem_threshold=1700, util_threshold=30)
s.update_host_config('office', mem_threshold=1700, util_threshold=25)
#s.update_host_config('ari', mem_threshold=2500, util_threshold=25)

cmd_mnist = 'OMP_NUM_THREADS=1 python main.py --model {4} --density {5} --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose{6}'
cmd_cifar = 'OMP_NUM_THREADS=1 python main.py --model {4} --decay_frequency 30000 --batch-size 128 --data cifar --epochs 1 --density {5} --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose --prune-rate 0.2 {6} --bench'

cmd = '\
OMP_NUM_THREADS=1 python train.py \
--cuda \
--data ../data/wikitext-2/ \
--dataset wt103 \
--adaptive \
--n_layer 16 \
--d_model 400 \
--n_head 10 \
--d_head 40 \
--d_inner 2400 \
--dropout 0.2 \
--dropatt 0.0 \
--optim adam \
--lr 0.00025 \
--warmup_step 0 \
--max_step 7500 \
--tgt_len 150 \
--mem_len 150 \
--eval_tgt_len 150 \
--batch_size 32 \
--fp16 \
--dynamic-loss-scale \
--log-interval 50 \
--eval-interval 100 \
--maxout-level -1'


num_seeds = 10
seed_offset = 0

jobs = []
for seed in range(num_seeds):
    fp16 = True
    jobs.append(['convtransformers/maxout/', 'convtransformer/pytorch/', cmd, fp16])

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
cmds = ['cd $GIT_HOME', 'git clone git@github.com:TimDettmers/convtransformer.git', 'cd convtransformer', 'git checkout max_out_sim', 'bash getdata.sh', 'cd pytorch']
cmds = cmds + ['git stash', 'git checkout master', 'git pull', 'git checkout max_out_sim', 'git pull']

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

