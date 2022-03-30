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


gpus = 32

cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train /private/home/timdettmers/data/wmt16_en_de_bpe32k/ --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --fp16  --fp16-no-flatten-grads --log-format simple --log-interval 50 --distributed-port 12597 --distributed-world-size {0} --keep-best-checkpoints 2 --keep-last-epochs 20 --keep-interval-updates 1 --ddp-backend=no_c10d'.format(gpus)


args2 = {}

name = 'snorm1'

constraint = 'volta32gb'

logfolder = '8bit_training/wmt/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 5
mem = 48*(8 if gpus > 8 else gpus)
num_seeds = 1
seed_offset = 0
time_hours = 12
time_minutes = 0


partition = 'learnlab'
#partition = 'scavenge'
change_dir = 'fairseq_private/'
repo = 'fairseq_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)

fp16 = True
args3 = {}

#key = ('lr', 'max-lr', 'min-lr', 'warmup-init-lr')
#args2['lr-scheduler'] = 'cosine'
#args3[key] = []
#for lr in [0.00075, 0.0005]:
#    args3[key].append((lr, lr+1e-8, lr*0.1, lr*0.1 + 1e-8))

args3['lr'] = [0.00075]
##args3['lr'] = [0.001]
args2['warmup-init-lr'] = 1e-07
args2['lr-scheduler'] = 'inverse_sqrt'
args2['fp16-scale-window'] = 250

args3['ff-block'] = [ '8bit']
args3['attention-8bit'] = ['linear']
args3['snorm'] = ['qk', 'v', 'qkv']
#args3['sparse-decomp'] = [True]
#args3['sparse-perc'] = [10, 5, 2]
#args3[('attention-8bit', 'attention-norm')] = [('off', True), ('off', False)]
#args3[('stable-emb', 'no-scale-embedding')] = [(False, False), (True, True)]

# adam
args2['optimizer'] = 'adam'
#args2['required-seq-len-multiple'] = 4
#args2['required-batch-size-multiple'] = 8
args3['adam-betas'] = ["'(0.9, 0.98)'"]

args3[('max-update', 'warmup-updates')] = [(24000, 4000)]
args3[('max-tokens', 'update-freq')] = [(3584, 128//gpus)]
args3[('clip-norm')] = [(0.6)]


#blockwise = ['/checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/4db316c9e499936f4d30762ea962ce2d',
#      ' /checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/3bc06fecfc416084d040ba90192b8911', ' /checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/8f481737785310cc2c4c293044ae5248']
#adafactor = ['/checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/d85280c5be946063fe6ce7d9b2982e8b', '/checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/0cf2c3e0bae987cdf47d1ff1355c0f8d', '/checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/c11ca192df29edb21ce51a0e2168cb55']
#baseline = ['/checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/e6051b9cfa9cbf5d12760b9acd48b8f5', ' /checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/dc5e671615c32d24ea7fb7fdd25546ba', '/checkpoint/timdettmers/adam/wmt16_en_de/blockwise_vs_adafactor/a27d300ab118ef93bbbddd04c653b2c7']




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
checkpoints = []
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
                                checkpoints.append(checkpoints.strip())
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
                checkpoints.append(checkpoint_dir.strip())
                job_cmd = job_cmd + save_dir
                cmds = [job_cmd]
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

extra_cmds = []
extra_cmds.append('python scripts/average_checkpoints.py --inputs {0} --num-epoch-checkpoints 5 --output {0}/checkpoint.avg10b.pt')
extra_cmds.append('rm {0}/gen*')
extra_cmds.append('fairseq-generate ~/data/wmt16_en_de_bpe32k/ --path {0}/checkpoint.avg10b.pt --beam 4 --lenpen 0.6 --remove-bpe > {0}/gen10b.out')
#extra_cmds.append('bash scripts/sacrebleu.sh wmt14/full en de {0}/gen10b.out')
extra_cmds.append('bash scripts/compound_split_bleu.sh {0}/gen10b.out')

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

for checkpoint in checkpoints:
    for cmd in extra_cmds:
        print(cmd.format(checkpoint))
    print('')

if not args.dry:
    s.run_jobs(comment='"ICLR review deadline 2021-09-27"')
    #s.run_jobs(skip_cmds=0)
    #s.run_jobs(skip_cmds=1)

