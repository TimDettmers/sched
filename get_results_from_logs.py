import sys
import os
import math
import glob
import numpy as np
import argparse
import re
import difflib
import copy
from os.path import join
import pandas as pd
import operator

# output possible parameters configurations
# multiple metric via metric file
# aggregation mode:
# - max/min/average/last
# - early stopping
# regex: start, end, contains
# error analysis and exclusion
# csv output generation
# filter arguments
# filter by metric
# sort/group
# open files in vim
# change metric precision
# extra: automatic join, genetic/random search optimization

parser = argparse.ArgumentParser(description='Log file evaluator.')
parser.add_argument('-f', '--folder-path', type=str, default=None, help='The folder to evaluate if running in folder mode.')
parser.add_argument('--contains', type=str, default=None, help='The line of the test metric must contain this string.')
parser.add_argument('--start', type=str, default='', help='String after which the test score appears.')
parser.add_argument('--end', type=str, default='\n', help='String before which the test score appears.')
parser.add_argument('--groupby', nargs='+', type=str, default='', help='Argument(s) which should be grouped by. Multiple arguments separated with space.')
parser.add_argument('--filter', type=str, default='', help='Argument(s) which should be kept by value (arg=value). Multiple arguments separated with a comma.')
parser.add_argument('--all', action='store_true', help='Prints all individual scores.')
parser.add_argument('--csv', type=str, default=None, help='Prints all argparse arguments with differences.')
parser.add_argument('--lower-is-better', action='store_true', help='Whether a lower metric is better.')
parser.add_argument('--vim', action='store_true', help='Prints a vim command to open the files for the presented results')
parser.add_argument('--num-digits', type=int, default=3, help='The significant digits to display for the metric value')
parser.add_argument('--early-stopping-condition', type=str, default=None, help='If a line with the keyphrase occurs 3 times, the metric gathering is stopped for the log')
parser.add_argument('--grid-params', action='store_true', help='Outputs the different hyperparameters used in all configs')
parser.add_argument('--agg', type=str, default='last', choices=['mean', 'last', 'min', 'max'], help='How to aggregate the regex-matched scores. Default: Last')
parser.add_argument('--limits', nargs='+', type=int, default=None, help='Sets the [min, max] range of the metric value (two space separated values).')

args = parser.parse_args()
if args.limits is not None: args.limits = tuple(args.limits)

folders = [x[0] for x in os.walk(args.folder_path)]

regex = re.compile(r'(?<={0}).*(?={1})'.format(args.start, args.end))

def clean_string(key):
    key = key.strip()
    key = key.replace("'", '')
    key = key.replace('"', '')
    key = key.replace(']', '')
    key = key.replace('[', '')
    key = key.replace('(', '')
    key = key.replace(')', '')
    return key

configs = []
all_cols = set(['NAME'])
for folder in folders:
    for log_name in glob.iglob(join(folder, '*.log')):
        config = {'METRICS' : {'default' : []}, 'NAME' : ''}
        if os.stat(log_name.replace('.log','.err')).st_size > 0: config['has_error'] = True
        else: config['has_error'] = False

        with open(log_name, 'r') as f:
            has_config = False
            for line in f:
                if 'Namespace(' in line:
                    has_config = True
                    line = line[line.find('Namespace(')+len('Namespace('):]
                    matches = re.findall(r'(?!^\()([^=,]+)=([^\0]+?)(?=,[^,]+=|\)$)', line)
                    for m in matches:
                        key = clean_string(m[0])
                        value = clean_string(m[1])
                        all_cols.add(key)
                        config[key] = value
                    if args.grid_params:
                        # we just want the config, no metrics
                        break

                if args.contains is None or args.contains in line:
                    matches = re.findall(regex, line)
                else:
                    matches = []

                if len(matches) > 0:
                    if not has_config:
                        print('Config for {0} not found. Test metric: {1}'.format(log_name, matches[0]))
                        break
                    config['NAME'] = log_name

                    metric = float(matches[0])
                    config['METRICS']['default'].append(float(matches[0]))
        configs.append(config)

if args.grid_params:
    key2values = {}
    for config in configs:
        for key, value in config.items():
            if key == 'NAME': continue
            if key == 'METRICS': continue
            if key == 'has_error': continue
            if key not in key2values:
                key2values[key] = [value]
                continue
            else:
                exists = False
                for value2 in list(key2values[key]):
                    if value == value2: exists = True
                if not exists:
                    key2values[key].append(value)

    n = len(configs)
    print('')
    print('='*80)
    print('Hyperparameters:')
    print('='*80)
    for key, values in key2values.items():
        if len(values) == 1 or len(values) >= n-10: continue
        keyvalues = '{0}: '.format(key)
        keyvalues += '{' + ','.join(values) + '}'
        print(keyvalues)
    sys.exit()

data = []

for config in configs:
    x = np.array(config['METRICS']['default'])
    if args.agg == 'last': x = x[-1]
    elif args.agg == 'mean': x = np.mean(x)
    elif args.agg == 'min': x = np.min(x)
    elif args.agg == 'max': x = np.max(x)
    data.append(x)

if args.limits is not None:
    # remove all configs with data not in the metric limits [a, b]
    # also remove all nan values
    i = 0
    while i < len(configs):
        x = data[i]
        remove = False
        if math.isnan(x): remove = True
        if x < args.limits[0]: remove = True
        if x > args.limits[1]: remove = True

        if remove:
            data.pop(i)
            configs.pop(i)
            continue
        else:
            i += 1

idx = np.argsort(np.array(data))
if args.lower_is_better: idx = idx[::-1]

all_cols = list(all_cols)
pandas_data = []
for config, x in zip(configs, data):
    row = []
    for col in all_cols:
        if col in config:
            row.append(config[col])
        else:
            row.append('')
    row.append(x)
    pandas_data.append(row)

all_cols.append('default')
df = pd.DataFrame(pandas_data, columns=all_cols)

# compute general group by statistics: Mean, standard error, 95% confidence interval, sample size, and log file names
means = df.groupby(by=args.groupby)['default'].mean().to_frame('default').reset_index()
se95 = df.groupby(by=args.groupby)['default'].sem().to_frame('default').reset_index()
names = df.groupby(by=args.groupby)['NAME'].transform(lambda x: ', '.join(x)).to_frame().reset_index()
means['logfiles'] = df.groupby(by=args.groupby)['NAME'].transform(lambda x: ', '.join(x)).to_frame().reset_index()['NAME']
means['n'] = df.groupby(by=args.groupby)['default'].count().to_frame().reset_index()['default']
means['idx'] = df.groupby(by=args.groupby)['default'].apply(lambda x: x.index.tolist()).reset_index()['default']

means['lower'] = means['default']-(se95['default']*1.96)
means['upper'] = means['default']+(se95['default']*1.96)
means['se'] = se95['default']

if args.lower_is_better:
    means = means.sort_values(by='default', ascending=False)
else:
    means = means.sort_values(by='default', ascending=True)

print('='*80)
for index, row in means.iterrows():
    str_config = ''
    for k, v in row.iteritems():
        if k not in ['n', 'logfiles', 'idx', 'default', 'lower', 'upper', 'se']:
            str_config += '{0}: {1: <4}, '.format(k, v)


    print('Config: {0}'.format(str_config[:-2]))
    print(('Metric mean value (SE): {0:.' + str(args.num_digits) + 'f} ({4:.4f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}').format(row['default'], row['lower'], row['upper'], row['n'], row['se']))
    if args.vim:
        print('vim {0}'.format(row['logfiles']))

    if args.all:
        for idx in row['idx']:
            print(configs[idx]['NAME'], data[idx])
    print('='*80)

if args.vim and args.vim_mode == 'all':
    print('vim {0}'.format(' '.join(logs)))

if args.csv is not None:
    df.to_csv(args.csv)

