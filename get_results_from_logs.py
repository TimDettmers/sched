import os
import glob
import numpy as np
import argparse
import re
import difflib
import copy
from os.path import join
import pandas as pd
import operator

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def show_diff(seqm):
    """Unify operations between two compared strings
seqm is a difflib.SequenceMatcher instance whose a & b are strings"""
    output= []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            output.append(seqm.a[a0:a1])
        elif opcode == 'insert':
            output.append(bcolors.OKGREEN + seqm.b[b0:b1] + bcolors.ENDC)
        elif opcode == 'delete':
            output.append(bcolors.FAIL + seqm.a[a0:a1] + bcolors.ENDC)
        elif opcode == 'replace':
            output.append(bcolors.OKBLUE + seqm.b[b0:b1] + bcolors.ENDC)
        else:
            raise RuntimeError("unexpected opcode")
    return ''.join(output)

parser = argparse.ArgumentParser(description='Log file evaluator.')
parser.add_argument('-f', '--folder-path', type=str, default=None, help='The folder to evaluate if running in folder mode.')
parser.add_argument('-r', '--recursive', action='store_true', help='Apply folder-path mode to all sub-directories')
parser.add_argument('--contains', type=str, default='', help='The line of the test metric must contain this string.')
parser.add_argument('--start', type=str, default='', help='String after which the test score appears.')
parser.add_argument('--end', type=str, default='\n', help='String before which the test score appears.')
parser.add_argument('--groupby', type=str, default='', help='Argument(s) which should be grouped by (without --). Multiple arguments separated with a comma.')
parser.add_argument('--orderby', type=str, default='', help='Argument(s) which should be ordered by (without --). Multiple arguments separated with a comma.')
parser.add_argument('--filter', type=str, default='', help='Argument(s) which should be kept by value (arg=value). Multiple arguments separated with a comma.')
parser.add_argument('--metric-lt', type=float, default=float("inf"), help='Only display aggregate results with a metric less than this value.')
parser.add_argument('--metric-gt', type=float, default=-float("inf"), help='Only display aggregate results with a metric greater than this value.')
parser.add_argument('--all', action='store_true', help='Prints all individual scores.')
parser.add_argument('--name', action='store_true', help='Prints all scores and associated log file names')
parser.add_argument('--partial', type=str, default='', help='Prints only configuration which at least have a partial match with the keywords (comma separates)')
parser.add_argument('--namespaces', action='store_true', help='Prints all argparse arguments.')
parser.add_argument('--diff', action='store_true', help='Prints all argparse arguments with differences.')
parser.add_argument('--csv', type=str, default='', help='Prints all argparse arguments with differences.')
parser.add_argument('--lower-is-better', action='store_true', help='Whether a lower metric is better.')
parser.add_argument('--always-last', action='store_true', help='Whether to always overwrite the metric with the last value encountered.')
parser.add_argument('--vim', action='store_true', help='Prints a vim command to open the files for the presented results')
parser.add_argument('--vim-mode', type=str, default='one', help='Select vim print options: {one, all}. Use "one" for one file per seed or all for all of them.')
parser.add_argument('--no-agg', action='store_true', help='No aggregation is done. This is useful for building seaborn confidence intervals.')
parser.add_argument('--exclude', type=float, default=None, help='Exclude individual results above or below this value')
parser.add_argument('--exclude-inverse', action='store_true', help='Exclude the results better than the exclude metric.')
parser.add_argument('--mean-metrics', action='store_true', help='Takes the mean of all metrics gathered in a single log file.')

args = parser.parse_args()

if args.diff: args.namespaces = True

if args.recursive:
    folders = [x[0] for x in os.walk(args.folder_path)]

regex = re.compile(r'(?<={0}).*(?={1})'.format(args.start, args.end))


groupby = set(args.groupby.split(','))
partial = set([s.strip() for s in args.partial.split(',')])

data = []
names = []
fdata = {}
groups = {}
cfg2logs = {}
namespaces = set()
for folder in folders:
    fdata[folder] = []
    for log_name in glob.iglob(join(folder, '*.log')):
        has_error = False
        if os.stat(log_name.replace('.log','.err')).st_size > 0: has_error = True

        with open(log_name, 'r') as f:
            multimatch = False
            config = None
            for line in f:
                if 'Namespace(' in line:
                    if not line.startswith('Namespace('):
                        line = line[line.find('Namespace('):]
                    if args.namespaces:
                        hsh = line
                        if 'seed' in line:
                            idx = line.index('seed')
                            hsh = line[:idx] + line[idx+6:] 
                        if line not in namespaces:
                            print(bcolors.OKGREEN + hsh + bcolors.ENDC)
                            namespaces.add(hsh)
                    matches = re.findall(r'(?!^\()([^=,]+)=([^\0]+?)(?=,[^,]+=|\)$)', line[len('Namespace('):])
                    config = []
                    if 'has_error' in groupby:
                        config.append(('has_error', str(has_error)))
                    for m in matches:
                        if m[0].strip() in groupby:
                            config.append((m[0].strip(), m[1].strip().replace(')','').replace("'", '')))
                if args.contains == '' or args.contains in line:
                    matches = re.findall(regex, line)
                else:
                    matches = []

                if len(matches) > 0:
                    if config is None:
                        print('Config for {0} not found. Test metric: {1}'.format(log_name, matches[0]))
                        continue
                    if tuple(config) not in cfg2logs: cfg2logs[tuple(config)] = []
                    cfg2logs[tuple(config)].append(log_name)

                    metric = float(matches[0])
                    if multimatch:
                        if args.mean_metrics:
                            if not isinstance(groups[tuple(config)][-1], list):
                                groups[tuple(config)][-1] = [groups[tuple(config)][-1]]
                            groups[tuple(config)][-1].append(metric)
                        elif args.always_last:
                            groups[tuple(config)][-1] = metric
                        elif args.lower_is_better and metric < groups[tuple(config)][-1]:
                            #print(metric, groups[config][-1])
                            groups[tuple(config)][-1] = metric
                        elif not args.lower_is_better and metric > groups[tuple(config)][-1]:
                            groups[tuple(config)][-1] = metric
                        if args.name:
                            names[-1] = (names[-1][0], groups[config][-1])
                    else:
                        config = tuple(config)
                        if config not in groups:
                            groups[config] = []

                        groups[config].append(float(matches[0]))
                        if args.name:
                            names.append((join(folder, log_name), groups[config][-1]))
                        multimatch = True

        if config is None: continue
        if tuple(config) in groups and isinstance(groups[tuple(config)][-1], list):
            groups[tuple(config)][-1] = np.mean(groups[tuple(config)][-1])
        if args.exclude is not None:
            if tuple(config) in groups:
                metrics = groups[tuple(config)]
                pop = False
                i = 0
                while i < len(metrics):
                    metric = metrics[i]
                    if metric > args.exclude and args.lower_is_better and not args.exclude_inverse: pop = True
                    if metric < args.exclude and not args.lower_is_better and not args.exclude_inverse: pop = True
                    if metric < args.exclude and args.lower_is_better and args.exclude_inverse: pop = True
                    if metric > args.exclude and not args.lower_is_better and args.exclude_inverse: pop = True

                    if pop:
                        groups[tuple(config)].pop(i)
                    else:
                        i += 1
if args.name:
    sorted_x = sorted(names, key=operator.itemgetter(1), reverse=True)
    for path, score in sorted_x:
        print(path, score)

    

if args.diff:
    for n1 in namespaces:
        for n2 in namespaces:
            sm = difflib.SequenceMatcher(None, n1, n2)
            print(show_diff(sm))

if args.namespaces: exit()

keys = list(groups.keys())

sort_keys = args.orderby.split(',')
filter_keys = [] if args.filter == '' else args.filter.replace('\,', '___').split(',')
filter_keys = [key.replace('___', ',') for key in filter_keys]

order = []
filters = {}
if len(keys) > 0:
    for skey in sort_keys:
        for i, (key, value) in enumerate(keys[0]):
            if key == skey.strip():
                order.append(i)

    for fkey in filter_keys:
        for i, (key, value) in enumerate(keys[0]):
            k,v = fkey.split('=')
            if key == k.strip():
                filters[i] = v.strip()

for idx in order:
    keys = sorted(keys, key=lambda x: x[idx])

logs = []
metrics = []
for config in keys:
    data = groups[config]
    if len(data) > 0:
        m = np.mean(data)
    else:
        m = 0.0
    metrics.append(m)
if args.orderby == '':
    idx = np.argsort(metrics)
    if args.lower_is_better:
        idx = idx[::-1]
else:
    idx = range(len(metrics))

pandas_pairs = []
pandas_columns = set()
for i in idx:
    config = keys[i]
    if any([v!=config[idx][1] for idx, v in filters.items() if idx < len(config)]): continue
    if args.partial != '':
        vals = [v for k,v in config]
        skip = True
        for v in vals:
            for p in partial:
                if p in v:
                    skip = False
                    break
            if not skip: break
        if skip:
            continue


    data = groups[config]
    if len(data) > 0:
        m = np.mean(data)
        if m > args.metric_lt: continue
        if m < args.metric_gt: continue

        if args.vim_mode == 'one':
            logs = cfg2logs[config][0]
        elif args.vim_mode == 'all':
            logs += cfg2logs[config]
        elif args.vim_mode == 'config':
            logs = set(cfg2logs[config])
        else:
            print(args.vim_mode)
            raise NotImplementedError('Vim print mode not implemented')

        if len(data) == 1: se = 0.0
        else: se = np.std(data, ddof=1)/np.sqrt(len(data))
        conf95 = 1.96*se
        print('='*80)
        print('Summary for config {0}:'.format(config))
        for key, value in config:
            pandas_columns.add(key)

        if args.no_agg:
            pandas_columns.add('Value')
            for d in data:
                cfg = dict(copy.copy(config))
                cfg['Value'] = d
                pandas_pairs.append(cfg)
        else:
            pandas_columns.update(['Mean', 'SE', 'Median', 'CI Lower', 'CI Upper', 'n'])
            cfg = dict(copy.copy(config))
            cfg['Mean'] = m
            cfg['SE'] = se
            cfg['Median'] = np.median(data)
            cfg['CI Lower'] = m-conf95
            cfg['CI Upper'] = m+conf95
            cfg['n'] = len(data)
            pandas_pairs.append(cfg)

        if len(data) == 1:
            print('Metric mean value (SE): {0:.3f} ({4:.4f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}'.format(m, m-float('NaN'), m+float('NaN'), len(data), float('NaN')))
        else:
            print('Metric mean value (SE): {0:.3f} ({4:.4f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}'.format(m, m-conf95, m+conf95, len(data), se))
        if args.vim and args.vim_mode == 'one':
            print('vim {0}'.format(logs))
        if args.vim and args.vim_mode == 'config':
            print('vim {0}'.format(' '.join(logs)))
        if args.all:
            for d in data:
                print(d)
print('='*80)

if args.vim and args.vim_mode == 'all':
    print('vim {0}'.format(' '.join(logs)))

if args.csv != '':
    if not os.path.exists(os.path.dirname(args.csv)):
        os.makedirs(os.path.dirname(args.csv))

    columns = list(pandas_columns)
    data = []

    for cfg in pandas_pairs:
        row = []
        for column in columns:
            if column in cfg:
                row.append(cfg[column])
            else:
                row.append('')

        data.append(row)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(args.csv)

