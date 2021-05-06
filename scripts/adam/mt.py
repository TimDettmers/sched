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

cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train data/wmt16_en_de_bpe32k/wmt16_en_de_bpe32k/ --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --fp16  --fp16-no-flatten-grads --log-format simple --log-interval 50 --distributed-port 12597 --distributed-world-size {0} --keep-best-checkpoints 2 --keep-last-epochs 20 --keep-interval-updates 1 --ddp-backend=no_c10d'.format(gpus)


args2 = {}

name = 'inf_grid4'
constraint = 'volta32gb'

logfolder = 'adam/wmt16_en_de/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2
cores_per_job = 5
mem = 48*(8 if gpus > 8 else gpus)
num_seeds = 2
seed_offset = 0
time_hours = 12
time_minutes = 0

#partition = 'learnfair'
partition = 'dev'
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

# adam
args2['optimizer'] = 'adam'
#args3['adam-betas'] = ["'(0.9, 0.98)'", "'(0.9, 0.995)'"]
args3['adam-betas'] = ["'(0.9, 0.98)'"]

# adafactor
#args2['optimizer'] = 'adafactor'
#args2['beta1'] = 0.9
#args2['decay-rate'] = 0.98

args3[('max-update', 'warmup-updates')] = [(24000, 4000)]
args3[('max-tokens', 'update-freq')] = [(3584, 128//gpus)]
args3[('fused', 'adam-bits', 'memory-efficient-fp16', 'adam8bits-method')] = [(True, 32, False, 'quantile'), (False, 32, True, 'quantile')]
#args3[('fused', 'adam-bits', 'memory-efficient-fp16', 'adam8bits-method')] = [(False, 32, True, 'quantile')]
#args3[('fused', 'adam-bits', 'memory-efficient-fp16', 'adam8bits-method')] = [(False, 8, True, 'quantile'), (False, 8, True, 'dynamic_tree')]
#args3[('fused', 'adam-bits', 'memory-efficient-fp16', 'adam8bits-method')] = [(False, 8, True, 'dynamic_tree')]
#args3[('fused', 'adam-bits', 'memory-efficient-fp16', 'adam8bits-method')] = [(False, 8, True, 'quantile')]
args3['adam8bits-offset'] = [1/512]
args3['prob-quant'] = [False]
args3['adam8bits-qfreq'] = [1]
args3['dist-scale'] = [1.0]

args3['percentile-clipping'] = [100]
args3['use-emb-norm'] = [False]
#log_id = '88ce0ff01b5b72a142a329cb5ffe7275'

extra_cmds = []
extra_cmds.append('python scripts/average_checkpoints.py --inputs {0} --num-epoch-checkpoints 20 --output {0}/checkpoint.avg10.pt')
extra_cmds.append('rm {0}/gen*')
extra_cmds.append('fairseq-generate data/wmt16_en_de_bpe32k/wmt16_en_de_bpe32k/ --path {0}/checkpoint.avg10.pt --beam 4 --lenpen 0.6 --remove-bpe > {0}/gen10.out')
extra_cmds.append('bash scripts/sacrebleu.sh wmt14/full en de {0}/gen10.out')
extra_cmds.append('bash scripts/compound_split_bleu.sh {0}/gen10.out')

#checkpoint = '/checkpoint/timdettmers/adam/wmt16_en_de/inf_steps5/d329d52e876972ede8e5383b5890f2df'
#for cmd in extra_cmds:
    #print(cmd.format(checkpoint))



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
            else:
                job_cmd = job_cmd + ' --seed {0}'.format(seed)
                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
                save_dir = ' --save-dir {0} '.format(checkpoint_dir.strip())

                job_cmd = job_cmd + save_dir
                cmds = [job_cmd]
                for c in extra_cmds:
                    cmds.append(c.format(checkpoint_dir.strip()))
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    for c in cmds[1:]:
                        print(c)
                    #s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
                    #s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=1)

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
    s.run_jobs(skip_cmds=0)
    #s.run_jobs(skip_cmds=1)

