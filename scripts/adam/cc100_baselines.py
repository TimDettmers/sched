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


gpus = 128

cmd = 'fairseq-train /private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin   --distributed-world-size {0} --distributed-port 54187  --fp16  --memory-efficient-fp16   --num-workers 2   --criterion cross_entropy   --task language_modeling   --sample-break-mode none --log-interval 25   --tokens-per-sample 1024 --arch transformer_lm_big   --share-decoder-input-output-embed            --decoder-layers 28 --decoder-attention-heads 16 --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 --activation-fn relu    --no-epoch-checkpoints --keep-best-checkpoints 0  --keep-interval-updates 1 --keep-last-epochs 0 --save-interval-updates 1000 --log-format simple --fp16-no-flatten-grads --ignore-unused-valid-subsets'.format(gpus)
args2 = {}

name = 'blockwise2'
constraint = 'volta32gb'


# 1024 tokens * 8 update_freq * 56250 steps = 0.4608e9 tokens -> optimal batch size 3460
# model sizes: 1.92bn, 2.43bn, 1.41bn


logfolder = 'adam/cc100/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 5
mem = 56*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 3
time_hours = 72
time_minutes = 0
#partition = 'dev'
#partition = 'learnlab,learnfair,scavenge'
partition = 'learnlab,learnfair'
change_dir = 'fairseq_private'
repo = 'fairseq_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)

fp16 = True
args3 = {}
args2['lr-scheduler'] =  'polynomial_decay'
args2['warmup-updates'] = 2000
args2['max-update'] = 56250
args2['total-num-update'] = 56250

#args2['lr-scheduler'] =  'cosine'
#args2['warmup-updates'] = 3000
#args2['max-update'] = 56250*4

args2['fp16-scale-window'] = 250
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(True, 32, 'quantile', 1), (False, 8, 'quantile', 1), (False, 8, 'dynamic_tree', 1), (False, 8, 'quantile', 25)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(True, 32, 'quantile', 1)]#, (False, 8, 'quantile', 1), (False, 8, 'dynamic_tree', 1), (False, 8, 'quantile', 25)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(True, 32, 'quantile', 1)]
#args3['adam8bits-offset'] = [1/512]
#args3['prob-quant'] = [False]

#args3['dist-scale'] = [1.0]
#args3[('percentile-clipping', 'clip-norm')] = [(100, 0.1)]
#args3['decoder-embed-dim'] = [2048+256]
#args3['decoder-ffn-embed-dim'] = [8192+2048]
#args3['max-tokens'] = [3072]
#args3['update-freq'] = [2]

key = ('max-tokens', 'decoder-embed-dim', 'decoder-ffn-embed-dim', 'update-freq', 'lr')
#key = ('max-tokens', 'decoder-embed-dim', 'decoder-ffn-embed-dim', 'update-freq')
args3[key] = []

#lrkey = ('lr', 'warmup-init-lr')
#args3[lrkey] = []

# 32-bit baseline
#args3['optimizer'] = ['adam']
#args3[('percentile-clipping', 'clip-norm')] = [(100, 0.1)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(True, 32, 'quantile', 1)]
##args3[key].append((2048,2048,8192,8, 0.00075))
#args3[key].append((2048,2048,8192,2))
#
#lr = 0.003239 + (-0.0001395*math.log(1.41e9))
#args3[lrkey].append((lr, lr+1e-8, lr*0.1, lr*0.1 + 1e-8))

# adafactor
#args3[('percentile-clipping', 'clip-norm')] = [(100, 0.1)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(False, 32, 'quantile', 1)]
#args2['optimizer'] = 'adafactor'
#args2['beta1'] = 0.9
#args2['decay-rate'] = 0.999
##args3[key].append((2048,2048,8192,8, 0.00075))
#args3[key].append((2048,2048+256,8192+2048,2))
##args3[key].append((2048,2688,10752,2))
#
#lr = 0.003239 + (-0.0001395*math.log(1.92e9))
#args3[lrkey].append((lr, lr+1e-8, lr*0.1, lr*0.1 + 1e-8))

# 8-bit
args3[('percentile-clipping', 'clip-norm')] = [(5, 0.0)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(False, 8, 'quantile', 1)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(False, 8, 'dynamic_tree', 1)]
#args3[('fused', 'adam-bits', 'adam8bits-method', 'adam8bits-qfreq')] = [(False, 8, 'dynamic_tree', 1), (False, 8, 'quantile', 1)]
args3['optimizer'] = ['adam']
args2['no-scale-embedding'] = ''
args2['use-bnb'] = ''
args2['stable-emb'] = ''

args3[('optim-bits', 'use-blockwise')] = [(8, True), (8, False)]

args3[key].append((2048,2048,8192,8, 0.00075))
#args3[key].append((2048,2048,8192,8, 0.00045))
#args3[key].append((2048,2688,10752,2))
#args3['use-emb-norm'] = [True]

#lr = 0.003239 + (-0.0001395*math.log(2.43e9))
#args3[lrkey].append((lr, 0.0))

#args2['train-subset'] = 'train11'

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
    s.run_jobs(comment='"NeurIPS deadline 2021-05-28"')

