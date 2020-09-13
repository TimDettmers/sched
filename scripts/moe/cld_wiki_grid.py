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
parser.add_argument('--baseline', action='store_true', help='Run baseline transformer')
args = parser.parse_args()


cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train --task language_modeling --share-decoder-input-output-embed --sample-break-mode none --ddp-backend=no_c10d --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints'
cmd2 = 'MKL_THREADING_LAYER=GNU fairseq-eval-lm --context-window 0 --task language_modeling --max-tokens 2048 --tokens-per-sample 128 --gen-subset {2} --skip-invalid-size-inputs-valid-test --path {1}/checkpoint_best.pt {0}'

args2 = {}
#baseline
gpus = 8


if args.baseline:
    #args2['optimizer'] = 'adam'
    #args2['adam-betas'] = "'(0.9, 0.98)'"
    #args2['lr'] = 0.0005


    args2['weight-decay'] = 0.01
    args2['lr-scheduler'] = 'inverse_sqrt'
    args2['min-lr'] = 1e-09
    args2['warmup-init-lr'] = 1e-07
    args2['max-tokens'] = 2048
    args2['arch'] = 'transformer_lm'

# moe
else:
    #args2['optimizer'] = 'lamb'
    #args2['lamb-betas'] = "'(0.9, 0.999)'"
    #args2['fp16-no-flatten-grads'] = ''

    #args2['warmup-updates'] = 400
    #args2['optimizer'] = 'adam'
    #args2['adam-betas'] = "'(0.9, 0.98)'"
    #args2['lr'] = 0.0005


    args2['arch'] = 'moe_lm'
    args2['lr-scheduler'] = 'inverse_sqrt'
    args2['max-update'] = 5000 # -> might need less
    args2['min-lr'] = 1e-09
    args2['warmup-init-lr'] = 1e-07
    args2['max-tokens'] = 2048
    args2['min-loss-scale'] = 1e-10


name = 'grid2'
logfolder = 'moe/cld/wiki/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 4*gpus
mem = 32*gpus
num_seeds = 1
seed_offset = 0
constraint = 'volta'
time_hours = 2
time_minutes = 0

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
#partition = 'learnfair'
#partition = 'uninterrupted'
partition = 'dev'
change_dir = 'fairseq_private/'
repo = 'fairseq_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)


#args2['dropout'] = 0.1
#args2['no-save'] = ''
args2['tokens-per-sample'] = 128
args2['weight-decay'] = 0.00
args2['update-freq'] = 8//gpus
#args2['update-freq'] = 1
args2['optimizer'] = 'lamb'
args2['lamb-betas'] = "'(0.9, 0.999)'"
args2['fp16-no-flatten-grads'] = ''


fp16 = True
args3 = {}

min_emb_dim = 32
min_dim = 32
increment_factor = 2

if not args.baseline:
    key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'moe-ff-dim', 'decoder-attention-heads', 'dummy', 'decoder-input-dim', 'decoder-output-dim')
    args3[key] = []
    for scale in range(5, 7, increment_factor):
        emb_dim = min_emb_dim + (32*scale//4)
        if scale  == 0: scale = 1
        heads = (min_dim*scale)//32
        heads = 1 if heads == 0 else heads
        ff_dim = int(min_dim*scale*4//2)
        ff_dim -= ff_dim % 32
        args3[key].append((min_dim*scale, min_dim*scale*4, ff_dim, heads, scale, emb_dim, emb_dim))

    args3['num-experts'] = [8]
    args3['experts-per-seq'] = [7]
    args3['moe-freq'] = [2]
    args3['moe-start-layer'] = [1]
    #args3['iloss-weight'] = [100000.0, 1000000.0]
    #args3['bloss-type'] = ['mean-prob', 'mean-prob-seg']
    args3['bloss-type'] = ['mean-prob']
    args3['criterion'] = ['moe_cross_entropy']
    #args3['lr'] = [0.0005, 0.0003, 0.0007]
    args3[('sample', 'sample-type')] = [(0, 'gumbel'), (1, 'proportional')]
    args3[('epsilon', 'epsilon-length')] = [(0.0, 1/4), (0.2, 1/4)]
    args3['epsilon-min'] = [0.0]
    args3['counter-reset-period'] = [1]
    args3['overflow-fraction'] = [0.0]
    args3['weight-decay'] = [0.0]
    args3['use-ff-norm'] = [True]
    #args3['gate-type'] = ['word-level']
    args3[('gate-type', 'iloss-weight')] = [('word-level', 100000.0), ('segments', 1000.0)]
else:
    key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'decoder-attention-heads', 'dummy', 'decoder-input-dim', 'decoder-output-dim')
    args3[key] = []
    for scale in range(5, 10, increment_factor):
        emb_dim = min_emb_dim + (32*scale//4)
        if scale  == 0: scale = 1
        heads = (min_dim*scale)//32
        heads = 1 if heads == 0 else heads
        args3[key].append((min_dim*scale, min_dim*scale*4, heads, scale, emb_dim, emb_dim))

args3['dropout'] = [0.0]
args3['attention-dropout'] = [0.0]#0.1, 0.2]
args3['clip-norm'] = [0.1]
#args3[('max-update', 'warmup-updates', '')] = [(30000, 3000, ' data/wikitext-25')]#, (3250, 400, ' data/wikitext-5')]
args3[('max-update', 'warmup-updates', '')] = [(1250, 125, ' data/cld_wiki_0.25')]
#args3[('max-update', 'warmup-updates', '')] = [(4000, 1000, ' data/wikitext-2')]#,(25000, 2000, ' data/wikitext-50')]

data_path = 'data/cld_wiki_0.25'
valid_subsets = ['valid', 'valid1', 'valid2', 'valid3']

args3['decoder-layers'] = [3]
args3['lr'] = [0.006]



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
                                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0}'.format(hashlib.md5(str(job_cmd5).encode('utf-8')).hexdigest(), ckp_name)
                                save_dir = ' --save-dir {0} '.format(checkpoint_dir)
                                job_cmd5 = job_cmd5 + save_dir
                                cmds = [job_cmd5]
                                for subset in valid_subsets:
                                    cmds.append(cmd2.format(data_path, checkpoint_dir, subset))
                                if rdm.rand(1) <= args.p:
                                    jobs.append(job_cmd5)
                                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0}'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
                save_dir = ' --save-dir {0} '.format(checkpoint_dir)
                job_cmd = job_cmd + save_dir
                cmds = [job_cmd]
                for subset in valid_subsets:
                    cmds.append(cmd2.format(data_path, checkpoint_dir, subset))
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs()

