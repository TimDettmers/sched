import os
import glob
import numpy as np
import argparse
import re
from os.path import join

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('-f', '--folder-path', type=str, default=None, help='The folder to evaluate if running in folder mode.')
parser.add_argument('-r', '--recursive', action='store_true', help='Apply folder-path mode to all sub-directories')
parser.add_argument('--start', type=str, default='', help='String after which the test score appears.')
parser.add_argument('--end', type=str, default='\n', help='String before which the test score appears.')
parser.add_argument('--groupby', type=str, default='', help='Argument(s) which should be grouped by (without --). Multiple arguments separated with a comma.')
parser.add_argument('--orderby', type=str, default='', help='Argument(s) which should be ordered by (without --). Multiple arguments separated with a comma.')
parser.add_argument('--filter', type=str, default='', help='Argument(s) which should be kept by value (arg=value). Multiple arguments separated with a comma.')
parser.add_argument('--metric-lt', type=float, default=float("inf"), help='Only display aggregate results with a metric less than this value.')
parser.add_argument('--metric-gt', type=float, default=-float("inf"), help='Only display aggregate results with a metric greater than this value.')

args = parser.parse_args()


if args.recursive:
    folders = [x[0] for x in os.walk(args.folder_path)]

regex = re.compile(r'(?<={0}).*(?={1})'.format(args.start, args.end))


print(args.groupby.split(','))
groupby = set(args.groupby.split(','))
print('lr' in groupby)

data = []
fdata = {}
groups = {}
for folder in folders:
    fdata[folder] = []
    for log_name in glob.iglob(join(folder, '*.log')):
        with open(log_name, 'r') as f:
            config = None
            for line in f:
                if line.startswith('Namespace('):
                    matches = re.findall(r'([^,]+)=([^,]+)', line[len('Namespace('):])
                    config = []
                    for m in matches:
                        if m[0].strip() in groupby:
                            config.append((m[0].strip(), m[1].strip()))
                matches = re.findall(regex, line)
                if len(matches) > 0:
                    if config is None:
                        print('Config for {0} not found. Test metric: {1}'.format(log_name, matches[0]))
                        continue
                    config = tuple(config)
                    if config not in groups:
                        groups[config] = []

                    groups[config].append(float(matches[0]))

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
            print(fkey, key)
            k,v = fkey.split('=')
            if key == k.strip():
                filters[i] = v.strip()

for idx in order:
    keys = sorted(keys, key=lambda x: x[idx])

for group in keys:
    if any([v!=group[idx][1] for idx, v in filters.items()]): continue
    data = groups[group]
    if len(data) > 0:
        m = np.mean(data)
        if m > args.metric_lt: continue
        if m < args.metric_gt: continue
        se = np.std(data, ddof=1)/np.sqrt(len(data))
        conf95 = 1.96*se
        print('='*80)
        print('Summary for config {0}:'.format(group))
        print('Metric mean value (95%CI): {0:.3f} ({4:.3f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}'.format(m, m-conf95, m+conf95, len(data), conf95))
        print('='*80)
        print('')
