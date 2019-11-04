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
#log_base = '/home/tim/logs/'
s = gpuscheduler.HyakScheduler('/gscratch/scrubbed/dettmers/git/sched/config/', verbose=args.verbose)
log_base = '/usr/lusers/dettmers/logs/'

s.update_host_config('home', mem_threshold=1700, util_threshold=30)
s.update_host_config('office', mem_threshold=1700, util_threshold=25)
#s.update_host_config('ari', mem_threshold=2500, util_threshold=25)

cmd = 'OMP_NUM_THREADS=1 python control-tasks/run_experiment.py --results-dir ./probe/name/ corrupt_config.yaml'

args2 = {}
args2['task'] = 'corrupted-pos'
args2['rank'] = 1024
args2['epochs'] = 40
args2['l2'] = 0.0
args2['momentum'] = 0.9
args2['optim'] = 'adam'
args2['scheduler'] = 'cosine'
args2['emb-train-path'] = './data/raw.train.elmo-layers.hdf5'
args2['layer'] = 2

#args2['lr'] = 0.001

logfolder = 'probe/{0}/'.format('pos')
time_hours = 1
cores_per_job = 5
num_seeds = 4

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)

args3 = {}
args3['lr'] = [0.001, 0.0001]

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
for seed in range(num_seeds):
    if len(args_prod) == 0: args_prod.append(('', ''))
    for i, values in enumerate(args_prod):
        fp16 = False
        job_cmd = cmd.replace('ITER', str(i)) + ' --seed {0}'.format(seed)
        for val in values:
            job_cmd += ' {0}' .format(val)
        jobs.append(job_cmd)
        s.add_job(logfolder, 'probe/', job_cmd, time_hours, fp16, cores=cores_per_job)

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

