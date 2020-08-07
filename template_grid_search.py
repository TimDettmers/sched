import itertools
import gpuscheduler
import argparse
import os
import uuid
import hashlib
import glob
from itertools import product
from torch.optim.lr_scheduler import OneCycleLR

from os.path import join

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()


cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train --task language_modeling --share-decoder-input-output-embed   --sample-break-mode none --ddp-backend=no_c10d --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints'

#cmd += ' data/wikitext-2'

args2 = {}
args2['no-save'] = ''
args2['warmup-updates'] = 400
args2['optimizer'] = 'lamb'
#args2['optimizer'] = 'adam'
#args2['adam-betas'] = "'(0.9, 0.98)'"

args2['max-tokens'] = 2048
args2['update-freq'] = 1

args2['clip-norm'] = 0.0003
#args2['weight-decay'] = 1e-07
#args2['dropout'] = 0.35
#args2['attention-dropout'] = 0.2
#args2['activation-dropout'] = 0.0
# seq length
args2['tokens-per-sample'] = 128
args2['lr-scheduler'] = 'inverse_sqrt'

#args2['lr-period-updates'] = 270000
#args2['max-lr'] = 1.0
#args2['max-update'] = 25000//gpus
args2['min-lr'] = 1e-09
args2['warmup-init-lr'] = 1e-07
#args2['lr'] = 0.0007
args2['decoder-ffn-embed-dim'] = 512
args2['decoder-input-dim'] = 512
args2['decoder-output-dim'] = args2['decoder-input-dim']
args2['decoder-layers'] = 4
args2['arch'] = 'moe_lm'
#args2['decoder-embed-dim'] = 128
#args2['decoder-attention-heads'] = 4
args2['criterion'] = 'moe_cross_entropy'
args2['fp16-no-flatten-grads'] = ''
#args2['adaptive-input-cutoff'] = '20000,60000'
#args2['adaptive-softmax-cutoff'] '20000,60000'
#args2['share-decoder-input-output-embed'] = ''
#args2['criterion'] = 'adaptive_loss'
#args2['decoder-input-dim'] = 1200
#args2['decoder-output-dim'] = 1200


name = 'grid10'
ckp_name = name
logfolder = 'moe/{0}/'.format(name)
#time_hours = 24*2
gpus = 2
cores_per_job = 5*gpus
mem = 32*gpus
num_seeds = 1
seed_offset = 0
constraint = 'volta'
time_hours = 4
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

for key, value in args2.items():
    cmd = cmd + ' --{0} {1}'.format(key, value)


fp16 = True
args3 = {}
args3['num-experts'] = [8, 16, 32]
args3['experts-per-seq'] = [1, 3, 7, 15, 31, 63, 127]
args3['moe-freq'] = [2]
args3['bloss-weight'] = [6.0]
args3['moe-noise-type'] = ['seq-additive']
args3['bloss-type'] = ['cv']
args3['lr'] = [0.006]
args3['weight-decay'] = [1e-06, 1e-08]
args3['dropout'] = [0.0]
args3['attention-dropout'] = [0.0]
args3['activation-dropout'] = [0.0]
args3['expert-compute-type'] = ['iter']
args3['lamb-betas'] = ["'(0.9, 0.999)'"]
args3['moe-noise-init'] = ['xavier']
#args3['warmup-updates'] = [400, 2000]
#args3['max-update'] = [15000//gpus]
#args3['dataset'] = ['data/wikitext-2', 'data/wikitext-5']
#args3['decoder-layers'] = [8, 16]
#args3['decoder-attention-heads'] = [8, 16]
args3[('decoder-embed-dim','decoder-attention-heads')] = [(128, 4), (256, 8)]
args3[('max-update', '', 'warmup-updates')] = [(25000//gpus, 'data/wikitext-2', 500), (50000//gpus, 'data/wikitext-5', 2000), (100000//gpus, 'data/wikitext-10', 5000)]




args4 = []
#args4.append(' --max-update {0} data/wikitext-2 --warmup-updates 500 '.format(25000//gpus))
#args4.append(' --max-update {0} data/wikitext-5 --warmup-updates 2000 '.format(50000//gpus))
#args4.append(' --max-update {0} data/wikitext-10 --warmup-updates 5000 '.format(100000//gpus))

args5 = {}

args6 = {}

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
        if len(key) == 0:
            keyvalues = [' --{0}'.format(v) if len(v) > 0 else '{0}'.format(v) for v in values]
        else:
            keyvalues = [' --{0} {1}'.format(key, v) for v in values]
    print(keyvalues)
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
            if not fp16: job_cmd = job_cmd.replace('--fp16', '')
            if any([k in job_cmd for k in args5.keys()]):
                for substr, pdict in args5.items():
                    if substr in job_cmd:
                        for key, values in pdict.items():
                            for v in values:
                                job_cmd5 = job_cmd + ' --{0} {1}'.format(key, v)
                                job_cmd5 = job_cmd5 + ' --seed {0}'.format(seed)
                                save_path = ' --save-dir /checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd5).encode('utf-8')).hexdigest(), ckp_name)
                                job_cmd5 = job_cmd5 + save_path
                                jobs.append(job_cmd5)
                                s.add_job(logfolder, repo, change_dir, job_cmd5, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
                save_path = ' --save-dir /checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
                job_cmd = job_cmd + save_path
                jobs.append(job_cmd)
                s.add_job(logfolder, repo, change_dir, job_cmd, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for job in jobs:
        print(job)
    print('total jobs', len(jobs))
    print('Jobs will be written to: {0}'.format(logfolder))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs()

