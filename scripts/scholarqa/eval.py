import numpy as np
import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
import math
from itertools import product
from torch.optim.lr_scheduler import OneCycleLR

from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()

#s = gpuscheduler.GantryScheduler('./config/cirrascale.cfg', cluster='ai2/*cirrascale', budget='ai2/allennlp', workspace='ai2/timd')
s = gpuscheduler.GantryScheduler('./config/austin.cfg', cluster='ai2/jupiter-cirrascale-2', budget='ai2/allennlp', workspace='ai2/timd', weka='oe-training-default:/data/input')

gpus = 1
gpus_per_node = gpus
memory, constraint = 48, '"[a100|a40]"'
port = np.random.randint(12200, 12999, 1)
logfolder = 'gantry-beaker-sched/test_chdir1'
ckp_name = logfolder
cores_per_job = 4
mem = 32
num_seeds = 1
seed_offset = 2

#home_path = '/net/nfs.cirrascale/allennlp/timd'
home_path = '/data/input/timd'
base_path = join(home_path, 'git/sched')
checkpoint_base_dir = join(home_path, 'checkpoints')

pre_cmds = [f'cd {base_path}']
cmd = f'python tests/beaker_test.py'
post_cmds = []

args2 = {}
args3 = {}

n = 1
args3['comment'] = [f'"this is test number {i}"' for i in range(n)]
args3['lr'] = [0.001, 0.003, 0.006, 0.009]


args4 = []
args5 = {}
args6 = {}

rdm = np.random.RandomState(5345)

for key, value in args2.items():
    if value == True:
        cmd = cmd + ' --{0}'.format(key)
    else:
        cmd = cmd + ' --{0} {1}'.format(key, value)

args_prod = []
for key, values in args3.items():
    if isinstance(key, tuple):
        keyvalues = []
        for tups in values:
            arg = ''
            for i, v in enumerate(tups):
                if v is True: v = ''
                if v is False: continue
                if len(key[i]) == 0:
                    arg += '{0} '.format(v)
                else:
                    arg += '--{0} {1} '.format(key[i], v)
            keyvalues.append(arg)
    elif isinstance(key, str):
        keyvalues = []
        for v in values:
            if v is True: v = ''
            if v is False:
                keyvalues.append('')
            else:
                keyvalues.append(' --{0} {1}'.format(key, v))
    args_prod.append(keyvalues)

if len(args_prod) >= 2:
    args_prod = list(product(*args_prod))
else:
    new_args = []
    if len(args_prod) > 0:
        for arg in args_prod[0]:
            new_args.append([arg])
        args_prod = new_args

jobs = []
port = port[0]
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            port += 1
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, checkpoint_base_dir)
            save_dir = ' --checkpoint_dir {0}'.format(checkpoint_dir)
            job_cmd += save_dir
            job_cmd = job_cmd.replace('MASTER_PORT', str(port))
            job_cmd = job_cmd + ' --seed {0}'.format(seed)
            cmds = pre_cmds + [job_cmd] + post_cmds
            if rdm.rand(1) <= args.p:
                jobs.append(job_cmd)
                s.add_job(logfolder, cmds, cores=cores_per_job, mem=mem, exclude=exclude, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('GPUs: {0}'.format(gpus))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True)
