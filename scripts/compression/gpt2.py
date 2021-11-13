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
parser.add_argument('--skip', nargs='+', type=int, default=None, help='Sets the [min, max] range of the metric value (two space separated values).')
args = parser.parse_args()


gpus = 8
cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train --task language_modeling --share-decoder-input-output-embed --sample-break-mode none --ddp-backend=no_c10d --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints --keep-interval-updates 1 --distributed-port 12597 --distributed-world-size {0} --valid-subset valid'.format(gpus)

args2 = {}

name = 'gpt2_8bit'
#name = 'gpt2_compression9'
constraint = ''

logfolder = 'compression/{0}'.format(name)
ckp_name = logfolder
cores_per_job = 4
mem = 16*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 4
time_hours = 36
time_minutes = 0

begin = None
exclude = ''

#account = 'ark'
#account = 'cse'
#account = 'stf'
account = 'zlab'


#partition = 'gpu-2080ti'
partition = 'gpu-rtx6k'
#partition = 'ckpt'

#begin = 'now+3hours'
#begin = '19:00'
#begin = '03:00'

change_dir = 'fairseq_private/'
repo = 'fairseq_private'
#exclude = 'g3007'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=partition, use_gres=False)

fp16 = True
args3 = {}

key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'decoder-attention-heads', 'decoder-input-dim', 'decoder-output-dim')
args3[key] = []
for model_dim in [1024]:
    heads = 8*(model_dim//512)
    for ff_dim in [4*model_dim]:
        args3[key].append((model_dim, ff_dim, heads, model_dim, model_dim))

args2['arch'] = 'transformer_lm'
args2['weight-decay'] = 0.00
args2['validate-interval-updates'] = 1000
args2['save-interval-updates'] = 1000
args2['lr-scheduler'] = 'cosine'
args2['fp16-no-flatten-grads'] = ''
args2['min-loss-scale'] = 1e-10
args2['fp16-scale-window'] = 250
args2['valid-subset'] = 'valid_wiki,valid,valid_1b,valid_lambada,valid_wiki2,valid_ptb'
#args2['no-scale-embedding'] = ''
#args2['stable-emb'] = ''
#args2['comm8bit'] = ''

#args2['use-bnb'] = ''
#args2['optim-bits'] = 8
#args2['memory-efficient-fp16'] = ''

args3[('ff-block','maxout', 'scale-factor')] = []
args3[('ff-block','maxout', 'scale-factor')].append(('bottleneck', 1, 2))
args3[('ff-block','maxout', 'scale-factor')].append(('bottleneck', 1, 4))
#
#args3[('ff-block','maxout', 'scale-factor')].append(('bottleneck', 2, 1))
#args3[('ff-block','maxout', 'scale-factor')].append(('bottleneck', 4, 1))

# done or will be done
# m4s4, m4s2,
# todo

# queued: baseline (823), m2s2 (817)
# experimental: m2s4 250 (60), m2s4 250 + lr (61)

# m2s2 (51)
# s2s4 (53)
# s4s2 (90)

# done: baseline, m4s2, s4s4
# running:  s2s2, s2s4, s4s2
# required: s4s4 rerun (optional), baseline + 8bit, m2s2, m2s4
# queued: m2s2 (zlab), m2s4 (zlab)


#args3['num-stages'] = [2, 4]
#args3['num-stages'] = [4]
args3['num-stages'] = [2]

args3['clip-norm'] = [0.1]
args2['optimizer'] = 'adam'
args3['adam-betas'] = ["'(0.9, 0.995)'"] # baseline params
args3['adam-eps'] = [1e-7] # baseline params
args3['decoder-layers'] = [16]
args3[('dropout', 'attention-dropout', 'relu-dropout')] = [(0.0, 0.0, 0.0)]

args3[('max-tokens', 'update-freq', 'tokens-per-sample', 'max-update')] = []
args3[('max-tokens', 'update-freq', 'tokens-per-sample', 'max-update')].append((2048, 128//gpus, 512, int(16944*2*1.0666)))
#args3[('max-tokens', 'update-freq', 'tokens-per-sample', 'max-update')].append((2048, 64//gpus, 1024, 16944*4))
args3[('warmup-updates', '')] = [(5000, ' /mmfs1/home/dettmers/data/openwebtext')]

args3['weight-decay'] = [0.00]

key = ('lr', 'warmup-init-lr')
args3[key] = []
for params in [1e4]:
    lr = 0.003239 + (-0.0001395*math.log(params))
    #args3[key].append((lr, lr*0.1))
    args3[key].append((lr, 0.0))
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
jobid = 0
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
            checkpoint_dir = '/gscratch/scrubbed/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
            save_dir = ' --save-dir {0}'.format(checkpoint_dir)
            job_cmd = job_cmd + save_dir
            #job_cmd += ' --attention-8bit 32bit '
            #job_cmd = job_cmd.replace('lr 0.0019', 'lr 0.0010')
            #job_cmd = job_cmd.replace('seed 3', 'seed 787878')
            #job_cmd = job_cmd.replace('keep-interval-updates 1', 'keep-interval-updates 2')
            #job_cmd = job_cmd.replace("'(0.9, 0.995)'", "'(0.9, 0.99)'")
            #job_cmd = job_cmd.replace('clip-norm 0.1', 'clip-norm 0.05')
            cmds = [job_cmd]
            if args.skip is not None and i in range(*args.skip):
                jobid += 1
                continue
            jobid += 1
            job_cmd = job_cmd.replace(str(12597), str(12597+jobid))
            if rdm.rand(1) <= args.p:
                jobs.append(job_cmd)
                s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(job)
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('begin: {0}'.format(begin))
    print('Jobs will be written to: {0}'.format(join('/mmfs1/home/dettmers/data/logs', logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs(begin=begin)

