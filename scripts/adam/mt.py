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


gpus = 8

cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train data/wmt16_en_de_bpe32k/wmt16_en_de_bpe32k/ --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --fp16  --save-dir /checkpoint/timdettmers/mt_test --fp16-no-flatten-grads --log-format simple --log-interval 50 --distributed-port 12597 --distributed-world-size {0} --keep-best-checkpoints 10 --keep-last-epochs 10 --keep-interval-updates 1 --ddp-backend=no_c10d'.format(gpus)


args2 = {}

args2['validate-interval-updates'] = 1000
args2['save-interval-updates'] = 1000

name = 'grid2'
constraint = 'volta32gb'

logfolder = 'adam/wmt16_en_de/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 5
mem = 48*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 0
time_hours = 48
time_minutes = 0

partition = 'learnfair'
change_dir = 'fairseq_private/'
repo = 'fairseq_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)

fp16 = True
args3 = {}

key = ('lr', 'max-lr', 'min-lr', 'warmup-init-lr')
args3[key] = []
#for params in [2e8]:
#for params in [1e1,  1e3, 1e2, 5e3]:
    #lr = 0.003239 + (-0.0001395*math.log(params))
for lr in [1e-3, 3e-3]:
    args3[key].append((lr, lr+1e-8, lr*0.1, lr*0.1 + 1e-8))
    #args3[key].append((lr, lr+1e-8, lr*0.1, lr*1.0 + 1e-8))

args2['optimizer'] = 'adam'
args2['lr-scheduler'] = 'cosine'
args3[('max-update', 'warmup-updates')] = [(9600, 3000)]
args3[('max-tokens', 'update-freq')] = [(3072, 128//gpus)]
args3[('fused', 'adam-bits', 'memory-efficient-fp16', 'adam8bits-method')] = [(True, 32, False, 'quantile'), (False, 32, True, 'quantile'), (False, 8, True, 'quantile'), (False, 8, True, 'dynamic_tree')]
args3['adam8bits-offset'] = [1/512]
args3['prob-quant'] = [True]
#args3['adam-betas'] = ["'(0.9, 0.995)'", "'(0.9, 0.99)'", "'(0.9, 0.98)'"]
args3['adam-betas'] = ["'(0.9, 0.98)'"]
#args3['adam-eps'] = [1e-6]
args3['adam8bits-qfreq'] = [1]
#args3['adam8bits-method'] = ['quantile', 'dynamic_tree']
args3['percentile-clipping'] = [5]
args3['dist-scale'] = [1.1]
args3['use-emb-norm'] = [True]
#args3[('memory-efficient-fp16', 'adam-bits')] = [(True, 8)]
#args3['clip-norm'] = [0.4, 0.8]



args4 = []

args5 = {}

args6 = {}

rdm = np.random.RandomState(5345)

for key, value in args2.items():
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
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            #job_cmd += ' --checkpoint /checkpoint/timdettmers/{1}/{0}/model.pt'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            if not fp16: job_cmd = job_cmd.replace('--fp16 ', ' ')
            if any([k in job_cmd for k in args5.keys()]):
                for substr, pdict in args5.items():
                    if substr in job_cmd:
                        for key, values in pdict.items():
                            for v in values:
                                job_cmd5 = job_cmd + ' --{0} {1}'.format(key, v)
                                job_cmd5 = job_cmd5 + ' --seed {0}'.format(seed)
                                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd5).encode('utf-8')).hexdigest(), ckp_name)
                                save_dir = ' --save-dir {0}'.format(checkpoint_dir)
                                job_cmd5 = job_cmd5 + save_dir
                                cmds = [job_cmd5]
                                if rdm.rand(1) <= args.p:
                                    jobs.append(job_cmd5)
                                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
                save_dir = ' --save-dir {0}'.format(checkpoint_dir)
                job_cmd = job_cmd + save_dir
                cmds = [job_cmd]
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('Jobs will be written to: {0}'.format(join('/private/home/timdettmers/logs/', logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs()

