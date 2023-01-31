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
gpus_per_node = 2
port = np.random.randint(12200, 12999, 1)
cmd = 'srun fairseq-train --task language_modeling --share-decoder-input-output-embed --sample-break-mode none --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints --keep-interval-updates 1 --distributed-port {1} --distributed-world-size {0} --valid-subset valid --num-workers 2'.format(gpus, port[0])

args2 = {}

name = 'test3'
#constraint = '"[rtx6k|a40|2080ti|a100]"'
constraint = '"[rtx6k|a40]"'

logfolder = 'ccsb/{0}'.format(name)
ckp_name = logfolder
cores_per_job = 2
mem = 12*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 4
time_minutes = 0

begin = None
partition = 'ckpt'
#partition = 'gpu-rtx6k'

#begin = 'now+8hours'
#begin = '19:00'
#begin = '03:00'
#partition = 'scavenge'

change_dir = 'fairseq_private/'
repo = 'fairseq_private'
exclude = ''
account = 'zlab'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=partition, use_gres=False)

checkpoint_base_dir = '/gscratch/scrubbed/timdettmers'

l = 10
ffn = 8*1024
model = 1024

seq_len = 512
batch_size = 8*1024

update_freq = 24//gpus
remainder = 24/(update_freq*gpus)
max_updates = int(16000*remainder)

total_tokens = batch_size*update_freq*max_updates*gpus

params = 0
params += l*ffn*model*2 # ffn
params += l*model*model # attn output
params += 3*l*model*model # attn proj

FLOPS = params*6*total_tokens

# RTX 2080 Ti has 94 TFLOPS
# assume 50% utilization for large models
# assume 25%% utilization for large models
FLOPS_S = 94e12*0.25*gpus
tokens_per_sec = FLOPS_S/(params*6)

seconds = FLOPS/FLOPS_S


print('='*80)
print(f'total_time: {seconds/60/60:.2f}h')
print(f'Parameters: {params/1e9:.3f}b')
print(f'Tokens: {total_tokens/1e9:.3f}b')
print(f'Expected tokens per second: {tokens_per_sec:.0f}')
print('='*80)

time_hours = int(math.ceil(seconds/60/60*1.2))-2

fp16 = True
args3 = {}

key = ('decoder-embed-dim', 'decoder-ffn-embed-dim', 'decoder-attention-heads', 'decoder-input-dim', 'decoder-output-dim')
args3[key] = []
args3[key] = []
for model_dim in [model]:
    for heads in [model_dim//64]:
        for ff_dim in [ffn]:
            args3[key].append((model_dim, ff_dim, heads, model_dim, model_dim))

args2['ddp-backend'] = 'fully_sharded'
#args2['ddp-backend'] = 'no_c10d'
args2['arch'] = 'transformer_lm_big'
args2['weight-decay'] = 0.00
args2['validate-interval-updates'] = 1000
args2['save-interval-updates'] = 1000
args2['fp16-no-flatten-grads'] = ''
args2['min-loss-scale'] = 1e-10
args2['fp16-scale-window'] = 250
args2['clip-norm'] = 0.6
args3['decoder-layers'] = [l]
args2['bashcmd'] = '"cp -r /gscratch/scrubbed/timdettmers/data/data-bin/c4_small /tmp/"'
#args2['no-scale-embedding'] = ''
#args2['no-save'] = ''
#args2['adambits'] = 8
#args2['stable-emb'] = ''


#args3['ff-block'] = ['subsampled', 'ff']
#args3['ff-block'] = ['subsampled', 'ff']
#args3['ff-block'] = ['subsampled']
args3['ff-block'] = ['ff']
args2['lr-scheduler'] = 'cosine'
args2['optimizer'] = 'adam'
args3['adam-betas'] = ["'(0.9, 0.995)'"] # baseline params
args3['adam-eps'] = [1e-7] # baseline params

args3[('max-tokens', 'update-freq', 'tokens-per-sample')] = []
#args3[('max-tokens', 'update-freq', 'tokens-per-sample')].append((2048, 128//gpus, 512))
args3[('max-tokens', 'update-freq', 'tokens-per-sample')].append((batch_size, update_freq, seq_len))
#args3[('max-update', 'warmup-updates', '')] = [(16000, 3000, ' /gscratch/cse/data/cc_small')]
args3[('max-update', 'warmup-updates', '')] = [(max_updates, 3000, ' /tmp/c4_small')]

args3[('dropout', 'attention-dropout', 'relu-dropout')] = [(0.0, 0.0, 0.0)]


key = ('lr', 'warmup-init-lr')
args3[key] = []
#args3[key].append((0.00163*0.75, 0.0))
args3[key].append((0.00163, 0.0))
#args3[key].append((0.00163*1.25, 0.0))
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
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            if not fp16: job_cmd = job_cmd.replace('--fp16 ', ' ')
            job_cmd = job_cmd + ' --seed {0}'.format(seed)
            checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, checkpoint_base_dir)
            save_dir = ' --save-dir {0}'.format(checkpoint_dir)
            job_cmd = job_cmd + save_dir
            #cmds = ['source /private/home/timdettmers/.bashrc', 'source activate base2', job_cmd]

            ssd_copy_lines = []

            #cmds = ['cp -r /gscratch/scrubbed/timdettmers/data/data-bin/c4_small /tmp/c4_small', job_cmd]
            #cmds = ['cp -r /gscratch/cse/data/cc_small /tmp/c4_small', job_cmd]
            cmds = ssd_copy_lines + [job_cmd]
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
    print('begin: {0}'.format(begin))
    print('Jobs will be written to: {0}'.format(join(s.config['LOG_HOME'], logfolder)))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=False)

