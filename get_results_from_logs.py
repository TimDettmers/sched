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
            for line in f:
                if line.startswith('Namespace('):
                    matches = re.findall(r'([^,]+)=([^,]+)', line[len('Namespace('):])
                    config = []
                    for m in matches:
                        if m[0].strip() in groupby:
                            config.append((m[0].strip(), m[1].strip()))
                matches = re.findall(regex, line)
                if len(matches) > 0:
                    config = tuple(config)
                    if config not in groups:
                        print(config)
                        groups[config] = []

                    groups[config].append(float(matches[0]))


for group, data in groups.items():
    if len(data) > 0:
        m = np.mean(data)
        se = np.std(data, ddof=1)/np.sqrt(len(data))
        conf95 = 1.96*se
        print('='*80)
        print('Summary for config {0}:'.format(group))
        print('Metric mean value: {0:.3f}. 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}'.format(m, m-conf95, m+conf95, len(data)))
        print('='*80)
        print('')
