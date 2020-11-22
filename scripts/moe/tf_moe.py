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
parser.add_argument('--tf', action='store_true', help='Run tensorflow baseline transformer')
parser.add_argument('--moe', action='store_true', help='Run mixture of expert baseline transformer')
parser.add_argument('--eval', action='store_true', help='Run with perplexity eval mode in TF.')
parser.add_argument('--inverse_sqrt', action='store_true', help='Use inverse_sqrt scheduler')
args = parser.parse_args()


#cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train --task language_modeling --share-decoder-input-output-embed --sample-break-mode none --ddp-backend=no_c10d --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints'

#cmd2 = 'MKL_THREADING_LAYER=GNU fairseq-eval-lm --context-window 0 --task language_modeling --max-tokens 2048 --tokens-per-sample 128 --gen-subset {2} --skip-invalid-size-inputs-valid-test --path {1}/checkpoint_best.pt {0}'

gpus = 8
args2 = {}

if args.tf:
    #cmd = '''python mesh_tensorflow/transformer/main.py  --gin_file=mesh_tensorflow/transformer/gin/defaults.gin --gin_file=mesh_tensorflow/transformer/gin/layouts/8dp_gpu_minxu.gin  --gin_file=mesh_tensorflow/transformer/gin/problems/t2t_lm1b_minxu.gin  --gin_param "run.mode='train'"'''
    cmd = '''python mesh_tensorflow/transformer/main.py  --gin_file=mesh_tensorflow/transformer/gin/defaults.gin --gin_file=mesh_tensorflow/transformer/gin/problems/t2t_lm1b_minxu.gin  --gin_param "run.mode='train'"'''
    args2['train_steps'] = 34400
else:
    cmd = 'MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 fairseq-train --task language_modeling --share-decoder-input-output-embed --sample-break-mode complete --ddp-backend=no_c10d --log-format simple --log-interval 50 --fp16 --keep-best-checkpoints 1 --no-epoch-checkpoints  --skip-invalid-size-inputs-valid-test --validate-interval-updates 1000 --save-interval-updates 1000 --keep-interval-updates 1 --distributed-port 12597 --distributed-world-size {0}'.format(gpus)

    if args.moe:
        args2['arch'] = 'moe_lm'
    else:
        args2['arch'] = 'transformer_lm'
        args2['criterion'] = 'cross_entropy'

    if args.inverse_sqrt:
        args2['lr-scheduler'] = 'inverse_sqrt'
        args2['warmup-init-lr'] = 1e-03
        args2['lr'] = 1e-03

    args2['max-tokens'] = 4096
    #args2['max-tokens'] = 2048
    args2['min-loss-scale'] = 1e-10
    args2['tokens-per-sample'] = 256
    #args2['update-freq'] = 1
    args2['optimizer'] = 'adafactor'
    #args2['lamb-betas'] = "'(0.9, 0.999)'"
    args2['fp16-no-flatten-grads'] = ''
    args2['weight-decay'] = 0.00
    #args2['decoder-ffn-embed-dim'] = 8192
    #args2['decoder-ffn-embed-dim'] = 4096
    args2['decoder-attention-heads'] = 4
    args2['decoder-layers'] = 4
    args2['decoder-embed-dim'] = 512

#baseline
if args.moe:
    name = 'attention1'
else:
    name = 'baseline_lr13'

#name = 'base_drop_batch1'
if args.tf:
    args2['dummy'] = name
    logfolder = 'moe/tf/baseline/{0}'.format(name)
else:
    logfolder = 'moe/tf/pytorch/{0}'.format(name)
ckp_name = logfolder
#time_hours = 24*2

cores_per_job = 1
mem = 32*gpus
num_seeds = 1
seed_offset = 0
constraint = 'volta'
#time_hours = int(48/gpus)
time_hours = 6
time_minutes = 0

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
#partition = 'dev'
#partition = 'uninterrupted'
#partition = 'dev'
partition = 'learnfair'
if args.tf:
    change_dir = 'mesh/'
    repo = 'mesh'
    env_name = 'tf'
else:
    change_dir = 'fairseq_private/'
    repo = 'fairseq_private'
    #change_dir = 'fairseq/'
    #repo = 'fairseq'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)


#args2['dropout'] = 0.1
#args2['no-save'] = ''
#args2['tokens-per-sample'] = 128
#args2['update-freq'] = 1
#args2['optimizer'] = 'lamb'
#args2['lamb-betas'] = "'(0.9, 0.999)'"
#args2['fp16-no-flatten-grads'] = ''
#args2['valid-subset'] = 'valid'

fp16 = True
args3 = {}
if args.tf:
    args3['gin_file'] = ['mesh_tensorflow/transformer/gin/models/lm_base.gin', 'mesh_tensorflow/transformer/gin/models/lm_moe_minxu.gin']
    #args3['gin_file'] = ['mesh_tensorflow/transformer/gin/models/lm_moe_minxu.gin']
    #args3['gin_file'] = ['mesh_tensorflow/transformer/gin/models/lm_base.gin']
else:
    if args.moe:
        args2['special-eval'] = ''
        args3['num-experts'] = [16]
        #args3['experts-per-seq'] = [7]
        args3['moe-freq'] = [2]
        #args3['bloss-type'] = ['mean-prob-seg']
        #args3['bloss-type'] = ['mean-prob']
        args3['criterion'] = ['moe_cross_entropy']
        args3[('sample', 'sample-type')] = [(0, 'argmax')]#, (1, 'proportional')]
        args3[('epsilon', 'epsilon-length')] = [(0.0, 1/4)]
        args3['epsilon-min'] = [0.0]
        args3['counter-reset-period'] = [1]
        #args3['conv-context-size'] = [32, 128]
        args3['overflow-fraction'] = [0.0]
        args3['use-ff-norm'] = [False]
        args3['no-expert-dropout'] = [True]
        #args3['agg-type'] = ['mean', 'conv']
        args3['agg-type'] = ['attention']
        args3['loss-type'] = ['mean']
        args3['gate-sharing'] = ['none']
        #args3['gate-type'] = ['word-level']
        args3[('iloss-weight', 'sample-type')] = [(0.01, 'argmax')]
        #args3[('gate-type', 'experts-per-seq')] = [('segments', 31), ('word-level', 255)]
        args3[('gate-type', 'experts-per-seq')] = [('segments', 7)]
        args3['moe-start-layer'] = [0]
        #args3[('moe-ff-dim', 'decoder-ffn-embed-dim')] = [(256, 4096), (512, 8192), (4096, 65536)]
        #args3[('moe-ff-dim', 'decoder-ffn-embed-dim')] = [(128, 4096), (256, 8192), (2048, 65536)] # 32 experts
        #args3[('moe-ff-dim', 'decoder-ffn-embed-dim')] = [(512, 4096), (1024, 8192), (8192, 65536)] # 8 experts
        #args3[('moe-ff-dim', 'decoder-ffn-embed-dim')] = [(256, 4096), (512, 8192)]
        #args3[('moe-ff-dim', 'decoder-ffn-embed-dim')] = [(4096, 65536)]

        # OpenAI scaling laws
        excess_expert_params = [args2['decoder-layers']/args3['moe-freq'][0]*(args3['num-experts'][0]-2)*p*args2['decoder-embed-dim']*2/args3['num-experts'][0] for p in [4096,8192,65546]]
        lr = lambda params: (0.003239 + (-0.0001395*math.log(params)))
        params = [2.1538e7, 3.885e7, 2.8946e8] # moe-freq = 2
        base_lrs = [lr(params[0]), lr(params[1]), lr(params[2])]
        adjusted_params = [p-e for e, p in zip(excess_expert_params, params)]
        params = adjusted_params
        lrs = [lr(params[0]), lr(params[1]), lr(params[2])]

        #args2['lr'] = [lr(
        if not args.inverse_sqrt:
            lrs = [lr*2.0 for lr in lrs]
            args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')] = []
            args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((256, 4096, lrs[0], lrs[0]+1e-8, lrs[0]*0.1, lrs[0]*0.1+1e-8))
            args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((512, 8192, lrs[1], lrs[1]+1e-8, lrs[1]*0.1, lrs[1]*0.1+1e-8))
            args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((4096, 65536, lrs[1], lrs[1]+1e-8, lrs[1]*0.1, lrs[1]*0.1+1e-8))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((128, 4096, lrs[0], lrs[0]+1e-8, lrs[0]*0.1, lrs[0]*0.1+1e-8))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((256, 8192, lrs[1], lrs[1]+1e-8, lrs[1]*0.1, lrs[1]*0.1+1e-8))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((2048, 65536, lrs[1], lrs[1]+1e-8, lrs[1]*0.1, lrs[1]*0.1+1e-8))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')] = [(128, 4096, lrs[0], lrs[0]+1e-8, lrs[0]*0.1, lrs[0]*0.1+1e-8),
            #                                                                                              (256, 8192, lrs[1], lrs[1]+1e-8, lrs[1]*0.1, lrs[1]*0.1+1e-8)]
            args2['lr-scheduler'] = 'cosine'
        else:
            lrs = [args2['lr']/blr*lr for blr, lr in zip(base_lrs, lrs)]
            args2.pop('lr')
            args2.pop('warmup-init-lr')
            args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')] = []
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((256, 4096, lrs[0], lrs[0]))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((512, 8192, lrs[1], lrs[1]))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((256, 4096, 3e-3, 3e-3))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((512, 8192, 3e-3, 3e-3))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((512, 8192, 3e-3, 3e-3))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((128, 4096, 3e-3, 3e-3))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((256, 8192, 3e-3, 3e-3))
            #args3[('moe-ff-dim', 'decoder-ffn-embed-dim', 'lr', 'warmup-init-lr')].append((2048, 8192, 3e-3, 3e-3))
    else:
        args2['lr-scheduler'] = 'cosine'
        args3[('decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')] = []
        for factor in [2.0]:
            lr = lambda params: (0.003239 + (-0.0001395*math.log(params)))*factor
            params = [2.1007e7, 3.789e7, 2.7295e8]
            lrs = [lr(params[0]), lr(params[1]), lr(params[2])]

            args3[('decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((4096, lrs[0], lrs[0]+1e-8, lrs[0]*0.1, lrs[0]*0.1+1e-8))
            args3[('decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((8192, lrs[1], lrs[1]+1e-8, lrs[1]*0.1, lrs[1]*0.1+1e-8))
            args3[('decoder-ffn-embed-dim', 'lr', 'max-lr', 'min-lr', 'warmup-init-lr')].append((65536, lrs[2], lrs[2]+1e-8, lrs[2]*0.1, lrs[2]*0.1+1e-8))

    args3['dropout'] = [0.1]
    args3['attention-dropout'] = [0.1]
    args3['relu-dropout'] = [0.1]
        #args3[('max-update', 'warmup-updates', '')] = [(30000, 3000, ' data/wikitext-25')]#, (3250, 400, ' data/wikitext-5')]
    args3[('max-update', 'warmup-updates', '', 'update-freq')] = []
    if args.inverse_sqrt:
        args3[('max-update', 'warmup-updates', '', 'update-freq')].append((34400, 10000, ' /private/home/timdettmers/data/t2t_data/data-bin', 8//gpus))
    else:
        args3[('max-update', 'warmup-updates', '', 'update-freq')].append((34400, 3000, ' /private/home/timdettmers/data/t2t_data/data-bin', 8//gpus))
    args3['clip-norm'] = [0.0]


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
        print(values)
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
                                #job_cmd5 = job_cmd5 + ' --seed {0}'.format(seed)
                                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0}'.format(hashlib.md5(str(job_cmd5).encode('utf-8')).hexdigest(), ckp_name)
                                if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
                                save_dir = ' --{1}dir {0} '.format(checkpoint_dir, 'save-' if not args.tf else 'model_')
                                job_cmd5 = job_cmd5 + save_dir
                                if args.eval: job_cmd = job_cmd.replace("'train'", "'perplexity_eval'")
                                cmds = ['export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/private/home/timdettmers/local/lib64', 'source /private/home/timdettmers/.bashrc', 'source activate '+env_name, job_cmd5]
                                if rdm.rand(1) <= args.p:
                                    jobs.append(job_cmd5)
                                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)
            else:
                #job_cmd = job_cmd + ' --seed {0}'.format(seed)
                checkpoint_dir = '/checkpoint/timdettmers/{1}/{0}'.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name)
                save_dir = ' --{1}dir {0} '.format(checkpoint_dir, 'save-' if not args.tf else 'model_')
                job_cmd = job_cmd + save_dir
                if args.eval: job_cmd = job_cmd.replace("'train'", "'perplexity_eval'")
                if args.tf:
                    cmds = ['export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/private/home/timdettmers/local/lib64', 'source /private/home/timdettmers/.bashrc', 'source activate '+env_name, job_cmd]
                else:
                    cmds = [job_cmd]
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    s.add_job(logfolder, repo, change_dir, cmds, time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    for i, job in enumerate(jobs):
        print(i, job)
    print('GPUs: {0}'.format(gpus))
    print('Time hours: {0}'.format(time_hours))
    print('total jobs', len(jobs))
    print('Jobs will be written to: ~/logs/{0}{1}'.format(logfolder, ' (does not exist)' if not os.path.exists(join('~/logs',logfolder)) else ''))
    print('Jobs will be run on: {0}'.format(partition))
    print('Run in folder: {0}'.format(change_dir))

if not args.dry:
    s.run_jobs()

