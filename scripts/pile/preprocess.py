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


gpus = 1
cmd = 'python the_pile/pile.py --force-write '
#cmd = 'MKL_THREADING_LAYER=GNU fairseq-preprocess --task language_modeling --bpe hf_byte_bpe --nwordssrc 50176 --workers 20 --only-source --srcdict /private/home/timdettmers/data/The.Pile.hf.byte.bpe.50k/dict.txt'
cmd = 'MKL_THREADING_LAYER=GNU fairseq-preprocess --task language_modeling --bpe gpt2 --nwordssrc 50176 --workers 20 --only-source --srcdict /private/home/timdettmers/data/The.Pile.gpt2.byte.bpe.50k/dict.txt'


args2 = {}
name = 'preprocess_sub_datasets'
constraint = ''
logfolder = 'pile/{0}'.format(name)
cores_per_job = 10
mem = 64
num_seeds = 1
seed_offset = 0
time_hours = 24
time_minutes = 0

#account = 'cse'
#account = 'stf'
#account = 'ark'
#partition = 'scavenge'
#partition = 'scavenge,learnfair'
partition = 'learnfair'
#partition = 'uninterrupted'
#partition = 'dev'
change_dir = 'pile_private/'
repo = 'pile_private'
exclude = ''

s = gpuscheduler.HyakScheduler(verbose=args.verbose, account='', partition=partition, use_gres=False)
#s = gpuscheduler.SshScheduler(verbose=args.verbose)


fp16 = True
args3 = {}
#args3[('destdir', 'trainpref', 'validpref', 'testpref')] = [('~/data/The.Pile.gpt2.byte.bpe.50k/{0}'.format(i), 'data/The_Pile/segments/train{0}.txt'.format(i), 'data/The_Pile/valid.txt', 'data/The_Pile/test.txt') for i in range(0, 30)]
#names = ['ArXiv', 'NIH.ExPorter', '"Wikipedia.(en)"', 'Bibliotik', 'OpenSubtitles', 'YoutubeSubtitles', 'BookCorpus', 'OpenWebText2', 'DM.Mathematics', 'PhilPapers', 'Enron.Emails', 'PubMed.Abstracts', 'EuroParl', 'PubMed.Central','FreeLaw', 'StackExchange','Github', '"Gutenberg.(PG-19)"', 'USPTO','HackerNews', 'Ubuntu.IRC']
names = ['Wikipedia.en', 'Gutenberg.PG19']
args3[('destdir', 'trainpref', 'validpref', 'testpref')] = []
for name in names:
    args3[('destdir', 'trainpref', 'validpref', 'testpref')].append(('~/data/{0}'.format(name), 'data/{0}/train.txt'.format(name), 'data/{0}/valid.txt'.format(name), 'data/{0}/test.txt'.format(name)))
#args3[('destdir', 'trainpref')] = [('~/data/pile_hf_bpe/hf_bpe-{0}'.format(i), 'data/The_Pile/segments/train{0}.txt'.format(i)) for i in range(1)]
#args3[('destdir', 'validpref', 'testpref')] = [('~/data/pile_hf_bpe/hf_bpe-validtest', 'data/The_Pile/valid.txt', 'data/The_Pile/test.txt')]
#grgs3[('destdir', 'validpref', 'testpref')] = [('~/data/pile_gpt2_raw/hf_bpe-validtest', 'data/The_Pile/valid.txt', 'data/The_Pile/test.txt')]
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
cmds = ['mv ~/git/data/pile_bin/gpt2-{0}/train.txt ~/data/gpt2/train{0}.txt']
if len(args4) == 0: args4.append('')
for seed in range(num_seeds):
    seed = seed + seed_offset
    for arg4 in args4:
        if len(args_prod) == 0: args_prod.append(('', ''))
        for i, values in enumerate(args_prod):
            job_cmd = cmd + arg4
            for val in values:
                job_cmd += ' {0}' .format(val)
                if rdm.rand(1) <= args.p:
                    jobs.append(job_cmd)
                    s.add_job(logfolder, repo, change_dir, [job_cmd], time_hours, fp16, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

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

