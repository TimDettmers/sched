import os
import glob
import numpy as np
import argparse
import re
import difflib
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
namespaces = set()
for folder in folders:
    fdata[folder] = []
    for log_name in glob.iglob(join(folder, '*.log')):
        with open(log_name, 'r') as f:
            multimatch = False
            config = None
            for line in f:
                if 'Namespace(' in line:
                    if not line.startswith('Namespace('):
                        line = line[line.find('Namespace('):]
                    if args.namespaces:
                        idx = line.index('seed')
                        hsh = line[:idx] + line[idx+6:] 
                        if line not in namespaces:
                            print(bcolors.OKGREEN + hsh + bcolors.ENDC)
                            namespaces.add(hsh)
                    matches = re.findall(r'([^,]+)=([^,]+)', line[len('Namespace('):])
                    config = []
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

                    if multimatch:
                        metric = float(matches[0])
                        if metric < groups[config][-1]:
                            groups[config][-1] = float(matches[0])
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
filter_keys = [] if args.filter == '' else args.filter.split(',')

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

pandas_data = []
for group in keys:
    if any([v!=group[idx][1] for idx, v in filters.items()]): continue
    if args.partial != '':
        vals = [v for k,v in group]
        skip = True
        for v in vals:
            for p in partial:
                if p in v:
                    skip = False
                    break
            if not skip: break
        if skip:
            continue

    
    data = groups[group]
    if len(data) > 0:
        m = np.mean(data)
        if m > args.metric_lt: continue
        if m < args.metric_gt: continue
        if len(data) == 1: se = 0.0
        else: se = np.std(data, ddof=1)/np.sqrt(len(data))
        conf95 = 1.96*se
        print('='*80)
        print('Summary for config {0}:'.format(group))
        row = []
        for key, value in group:
            row.append(value)
        row.append(m)
        row.append(se)
        row.append(np.median(data))
        row.append(m-conf95)
        row.append(m+conf95)
        row.append(len(data))
        pandas_data.append(row)

        if len(data) == 1:
            print('Metric mean value (SE): {0:.3f} ({4:.4f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}'.format(m, m-float('NaN'), m+float('NaN'), len(data), float('NaN')))
        else:
            print('Metric mean value (SE): {0:.3f} ({4:.4f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}'.format(m, m-conf95, m+conf95, len(data), se))
        print('='*80)
        if args.all:
            for d in data:
                print(d)

if args.csv != '':
    columns = []
    for key, value in group:
        columns.append(key)
    columns.append('Mean')
    columns.append('SE')
    columns.append('Median')
    columns.append('CI lower')
    columns.append('CI upper')
    columns.append('n')
    df = pd.DataFrame(pandas_data, columns=columns)
    df.to_csv(args.csv)

