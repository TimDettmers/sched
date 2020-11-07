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


cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train --task language_modeling --share-decoder-input-output-embed --sample-break-mode none --ddp-backend=no_c10d --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints --keep-interval-updates 5'
cmd2 = 'fairseq-eval-lm --context-window 0 --task language_modeling --max-tokens 2048 --tokens-per-sample 128 --gen-subset {2} --skip-invalid-size-inputs-valid-test --path {1}/checkpoint_best.pt {0}'

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
    args2['min-lr'] = 1e-09
    args2['warmup-init-lr'] = 1e-07
    args2['min-loss-scale'] = 1e-10
    args2['moe-start-layer'] = 0


if args.baseline:
    name = 'baselines20'
    constraint = 'volta32gb'
else:
    name = 'moe20'
    constraint = 'volta'

logfolder = 'moe/scaling/cc_news/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 4*gpus
mem = 48*gpus
num_seeds = 1
seed_offset = 0
time_hours = 16
time_minutes = 0

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
partition = 'learnfair'
#partition = 'uninterrupted'
#partition = 'dev'
change_dir = 'fairseq_private/'
repo = 'fairseq_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)


#args2['dropout'] = 0.1
#args2['no-save'] = ''
args2['weight-decay'] = 0.00
#args2['max-tokens'] = 1024
args2['special-eval'] = ''
args2['max-tokens'] = 2048


fp16 = True
args3 = {}

min_emb_dim = 32
min_dim = 32
increment_factor = 10

base_heads = 1
model_dim = 64
doublings = 5

if not args.baseline:
    key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'moe-ff-dim', 'decoder-attention-heads', 'dummy', 'decoder-input-dim', 'decoder-output-dim', 'num-experts')
    args3[key] = []
    #for num_experts in [16, 32]:
    for num_experts in [16]:
        #for ff_factor in [4, 8, 16, 32, 64]:
        for ff_factor in [16, 32, 64]:
            for i in range(doublings):
                if i < 4: continue
                factor = 2**i
                heads = base_heads*factor
                emb_dim = model_dim*factor
                ff_dim = model_dim*factor*ff_factor
                args3[key].append((emb_dim, ff_dim, ff_dim//num_experts, heads, i, emb_dim, emb_dim, num_experts))
                #args3[key].append((emb_dim, ff_dim, ff_dim//2, heads, i, emb_dim, emb_dim))

    args3['epsilon'] = [0.2]
    args3['moe-freq'] = [2]
    args3['sample-type'] = ['argmax']
    args3['criterion'] = ['moe_cross_entropy']
    args3['use-ff-norm'] = [False]
    args3[('gate-type', 'experts-per-seq', 'iloss-weight')] = []
    #args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('segments', 7, 0.3))
    #args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('segments', 7, 0.1))
    args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('segments', 7, 0.05))
    #args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('segments', 31, 0.01))
    args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('segments', 31, 0.05))
    #args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('segments', 31, 0.3))
    args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('word-level', 511, 0.01))
    #args3[('gate-type', 'experts-per-seq', 'iloss-weight')].append(('word-level', 255, 0.01))
    args3['agg-type'] = ['mean']
    #args3['iloss-weight'] = [0.01]
    #args3[('num-experts', 'iloss-weight')] = [(16, 0.01), (16, 0.05), (16, 0.1)]
    #args3[('num-experts', 'iloss-weight')] = [(8, 0.02)]
    #args3['num-experts'] = [16]
    #args3[('gate-type', 'experts-per-seq')] = [('segments', 255), ('segments', 127), ('word-level', 255)]
    #args3['iloss-weight'] = [0.01]
else:
    key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'decoder-attention-heads', 'dummy', 'decoder-input-dim', 'decoder-output-dim')
    args3[key] = []
    for ff_factor in [4, 8, 16, 32, 64]:
        for i in range(doublings):
            if i < 4: continue
            factor = 2**i
            heads = base_heads*factor
            emb_dim = model_dim*factor
            ff_dim = model_dim*factor*ff_factor
            args3[key].append((emb_dim, ff_dim, heads, i, emb_dim, emb_dim))

args2['validate-interval-updates'] = 1000
#args3['decoder-layers'] = [4, 8]
#args3[('dropout', 'attention-dropout', 'relu-dropout')] = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1)]
args3['decoder-layers'] = [4]
args3[('dropout', 'attention-dropout', 'relu-dropout')] = [(0.1, 0.1, 0.1)]
args3['clip-norm'] = [0.0]

# WT
#args3[('max-update', 'warmup-updates', '')] = [(80000, 24000, ' data/wikitext-103')]
#args3[('max-update', 'warmup-updates', '')] = [(31250, 10000, ' data/wikitext-25'), (50000, 15000, ' data/wikitext-50'), (100000, 30000, ' data/wikitext-103')]
#args3['tokens-per-sample'] = [256]
#args3['update-freq'] = [8//gpus]

# CC-News
args3[('max-update', 'warmup-updates', '')] = [(250000, 3000, ' data/cc_news')]
args2['save-interval-updates'] = 1000
args2['tokens-per-sample'] = 512
seqs_per_mini_batch = 512 # OpenAI scaling laws mini-batch size
args3['update-freq'] = [seqs_per_mini_batch//(args2['max-tokens']//args2['tokens-per-sample'])//gpus]


data_path = 'data/wikitext-10'
valid_subsets = []

#args3['decoder-layers'] = [3]
args3['weight-decay'] = [0.00]

#args2['optimizer'] = 'lamb'
#args2['lamb-betas'] = "'(0.9, 0.999)'"
#args2['fp16-no-flatten-grads'] = ''
#args3['lr'] = [0.001]

args2['warmup-init-lr'] = 1e-03
args2['lr'] = 1e-03
args2['optimizer'] = 'adafactor'
args2['fp16-no-flatten-grads'] = ''

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
                                for subset in valid_subsets:
                                    cmds.append(cmd2.format(data_path, checkpoint_dir, subset))
                                if rdm.rand(1) <= args.p:
                                    jobs.append(job_cmd5)
                                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
                save_dir = ' --save-dir {0}'.format(checkpoint_dir)
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
    print('')
    print('Total jobs', len(jobs))
    print('Time hours: {0}'.format(time_hours))
    print('GPUs: {0}'.format(gpus))
    print('Jobs will be written to: {0}'.format(join('/private/home/timdettmers/logs/', logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs()

