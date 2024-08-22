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


gpus = 1
gpus_per_node = gpus
port = np.random.randint(12200, 12999, 1)
base_path = '/net/nfs.cirrascale/allennlp/timd/git/sched'
cmd = f'python {join(base_path, "tests/beaker_test.py")}'

checkpoint_base_dir = '/net/nfs.cirrascale/allennlp/timd/checkpoints'

args2 = {}

#name = 'cola1'
#name = 'nf4_7_zlab'
#constraint = '"[rtx6k]"'
#memory, constraint = 24, '"[rtx6k|a40|a100]"'
memory, constraint = 48, '"[a100|a40]"'
#constraint = '"[a100]"'
#constraint = '"[rtx6k|a40]"'

logfolder = 'test1/test2/beaker_test3'
ckp_name = logfolder
cores_per_job = 4
mem = 32
num_seeds = 2
seed_offset = 2
requeue = False

begin = None
#partition = 'ckpt'
#partition = 'gpu-rtx6k'
partition = 'gpu-a40'

#partition = 'scavenge'

exclude = ''
#account = 'efml'
account = 'zlab'

s = gpuscheduler.GantryScheduler('./config/cirrascale.cfg', cluster='ai2/*cirrascale', budget='ai2/allennlp', workspace='ai2/timd')

args2 = {}
args3 = {}

n = 3
args3['comment'] = [f'"this is test number {i}"' for i in range(n)]
args3['lr'] = [0.001, 0.003]


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
            #job_cmd = job_cmd + f' --experiment_id {hashlib.md5(str(job_cmd).encode("utf-8")).hexdigest()}'
            #cmds = ['mkdir -p /tmp/huggingface/datasets', 'cp -r ~/.cache/huggingface/datasets/glue /tmp/huggingface/datasets/', 'cp -r ~/.cache/huggingface/hub/models--roberta-large /tmp/huggingface/']
            #cmds = cmds + [job_cmd]
            cmds = [job_cmd]
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
    print('Jobs will be run on: {0}'.format(partition))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True)
