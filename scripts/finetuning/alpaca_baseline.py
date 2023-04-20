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
gpus_per_node = gpus
port = np.random.randint(12200, 12999, 1)
#cmd = f'python -m torch.distributed.launch --master_port MASTER_PORT --nproc_per_node={gpus} examples/text-classification/run_glue.py --model_name_or_path roberta-large --task_name mrpc --do_train --do_eval --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps 1000 --logging_dir ~/data/logs/adamix/roberta_large_rte --evaluation_strategy epoch --save_strategy epoch --warmup_ratio 0.1 --apply_expert_soup --adapter_size 16 --num_experts 4 --seed 0 --inference_level 3 --sharing_up 1 --sharing_down 0 --use_consistency_loss 1'
#cmd = f'python -m torch.distributed.launch --master_port MASTER_PORT --nproc_per_node={gpus} examples/text-classification/run_glue.py --model_name_or_path roberta-large --do_train --do_eval --per_device_eval_batch_size 8 --overwrite_output_dir --logging_steps 1000 --logging_dir ~/data/logs/adamix/roberta_large_rte --evaluation_strategy epoch --save_strategy epoch --warmup_ratio 0.1'
#cmd = 'python examples/pytorch/text-classification/run_glue.py --model_name_or_path roberta-large --do_train --do_eval --per_device_eval_batch_size 2 --logging_steps 1000 --logging_dir ~/data/logs/adamix/roberta_large_rte --evaluation_strategy epoch'
cmd = 'python train.py'
cmd = 'torchrun --nproc_per_node=8 --master_port=12212 train.py     --model_name_or_path /gscratch/zlab/llama/7B     --data_path ./alpaca_data.json     --bf16 True      --num_train_epochs 3     --per_device_train_batch_size 1     --per_device_eval_batch_size 4     --gradient_accumulation_steps 16     --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 100     --save_total_limit 20     --learning_rate 2e-5     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type "cosine"     --logging_steps 1     --fsdp "full_shard auto_wrap"     --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"     --tf32 True'

checkpoint_base_dir = '/gscratch/scrubbed/timdettmers/checkpoints/'

args2 = {}

#name = 'cola1'
name = 'alpaca_deepspeed_baseline1'
#name = 'nf4_7_zlab'
#constraint = '"[rtx6k]"'
#memory, constraint = 24, '"[rtx6k|a40|a100]"'
memory, constraint = 48, '"[a100|a40]"'
#constraint = '"[a100]"'
#constraint = '"[rtx6k|a40]"'

logfolder = 'finetuning/llama/{0}'.format(name)
ckp_name = logfolder
cores_per_job = 4
mem = 396
num_seeds = 1
seed_offset = 1
time_minutes = 0
requeue = False
requeue_length_hours = 4

begin = None
#partition = 'ckpt'
#partition = 'gpu-rtx6k'
partition = 'gpu-a40'

#begin = 'now+8hours'
#begin = '19:00'
#begin = '03:00'
#partition = 'scavenge'

#repo = change_dir = 'alpaca-lora/'
repo = change_dir = 'stanford_alpaca-private/'
exclude = ''
#account = 'efml'
account = 'zlab'

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account=account, partition=partition, use_gres=False)


args2 = {}
args3 = {}

time_hours = 72
if requeue:
    time_hours = 0
else:
    pass


def get_max_memory(size, num_gpus, memory, bits):
    if num_gpus == 1: return (memory*1024) - 6*1024
    model_GB = size *(bits/8)
    # round up a GB, so we do not round out of memory if something cannot be split well
    max_memory_MB = ((model_GB/gpus)-1023)//1024*1024
    return int(max_memory_MB)




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
port = port[0]
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            port += 1
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
            checkpoint_dir = '{2}/{1}/{0} '.format(hashlib.md5(str(job_cmd).encode('utf-8')).hexdigest(), ckp_name, checkpoint_base_dir)
            save_dir = ' --output_dir {0}'.format(checkpoint_dir)
            job_cmd += save_dir
            job_cmd = job_cmd.replace('MASTER_PORT', str(port))
            job_cmd = job_cmd + ' --seed {0}'.format(seed)
            #job_cmd = job_cmd + f' --experiment_id {hashlib.md5(str(job_cmd).encode("utf-8")).hexdigest()}'
            #cmds = ['mkdir -p /tmp/huggingface/datasets', 'cp -r ~/.cache/huggingface/datasets/glue /tmp/huggingface/datasets/', 'cp -r ~/.cache/huggingface/hub/models--roberta-large /tmp/huggingface/']
            #cmds = cmds + [job_cmd]
            cmds = [job_cmd]
            if rdm.rand(1) <= args.p:
                jobs.append(job_cmd)
                s.add_job(logfolder, repo, change_dir, cmds, time_hours, False, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

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
    if not requeue:
        s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=False, as_array=True, single_process=True)
    else:
        s.run_jobs(begin=begin, gpus_per_node=gpus_per_node, requeue=True, as_array=False, single_process=True, requeue_length_hours=requeue_length_hours)

