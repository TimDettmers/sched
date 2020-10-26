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
pd.set_option('display.max_colwidth', -1)

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
parser.add_argument('--contains', type=str, default='', help='The line of the test metric must contain this string.')
parser.add_argument('--start', type=str, default='', help='String after which the test score appears.')
parser.add_argument('--end', type=str, default='\n', help='String before which the test score appears.')
parser.add_argument('--groupby', nargs='+', type=str, default='', help='Argument(s) which should be grouped by. Multiple arguments separated with space.')
parser.add_argument('--filter', nargs='+', type=str, default='', help='Argument(s) which should be kept by value (arg=value). Multiple arguments separated with a space.')
parser.add_argument('--all', action='store_true', help='Prints all individual scores.')
parser.add_argument('--csv', type=str, default=None, help='Prints all argparse arguments with differences.')
parser.add_argument('--smaller-is-better', action='store_true', help='Whether a lower metric is better.')
parser.add_argument('--vim', action='store_true', help='Prints a vim command to open the files for the presented results')
parser.add_argument('--num-digits', type=int, default=4, help='The significant digits to display for the metric value')
parser.add_argument('--early-stopping-condition', type=str, default=None, help='If a line with the keyphrase occurs 3 times, the metric gathering is stopped for the log')
parser.add_argument('--diff', action='store_true', help='Outputs the different hyperparameters used in all configs')
parser.add_argument('--agg', type=str, default='last', choices=['mean', 'last', 'min', 'max'], help='How to aggregate the regex-matched scores. Default: Last')
parser.add_argument('--limits', nargs='+', type=int, default=None, help='Sets the [min, max] range of the metric value (two space separated values).')
parser.add_argument('--metric-file', type=str, default=None, help='A metric file which tracks multiple metrics as once.')

args = parser.parse_args()

metrics = None
if args.metric_file is not None:
    metrics = pd.read_csv(args.metric_file).fillna('')
    primary_metric = metrics.iloc[0]['name'] if metrics is not None else 'default'
    smaller_is_better = metrics.iloc[0]['smaller_is_better'] == 1
    metrics = metrics.to_dict('records')
else:
    primary_metric = 'default'
    smaller_is_better = args.smaller_is_better

if args.limits is not None: args.limits = tuple(args.limits)

folders = [x[0] for x in os.walk(args.folder_path)]


if metrics is not None:
    for metric in metrics:
        regex = re.compile(r'(?<={0}).*(?={1})'.format(metric['start_regex'], metric['end_regex']))
        metric['regex'] = regex
else:
    regex = re.compile(r'(?<={0}).*(?={1})'.format(args.start, args.end))
    metrics = [{'name' : 'default', 'regex' : regex, 'contains' : args.contains, 'agg' : args.agg }]

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
        config = {'METRICS' : {}, 'NAME' : log_name}
        for metric in metrics:
            config['METRICS'][metric['name']] = []
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
                    if args.diff:
                        # we just want the config, no metrics
                        break

                for metric in metrics:
                    contains = metric['contains']
                    if contains != '' and not contains in line: continue
                    regex = metric['regex']
                    name = metric['name']
                    matches = re.findall(regex, line)
                    if len(matches) > 0:
                        if not has_config:
                            print('Config for {0} not found. Test metric: {1}'.format(log_name, matches[0]))
                            break
                        if name not in config['METRICS']: config['METRICS'][name] = []
                        config['METRICS'][name].append(float(matches[0]))
                        break

        configs.append(config)

if args.diff:
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

for config in configs:
    for metric in metrics:
        name = metric['name']
        x = np.array(config['METRICS'][name])
        if x.size == 0: continue
        if metric['agg'] == 'last': x = x[-1]
        elif metric['agg'] == 'mean': x = np.mean(x)
        elif metric['agg'] == 'min': x = np.nanmin(x)
        elif metric['agg'] == 'max': x = np.nanmax(x)
        elif metric['agg'] == 'idx':
            name2 = metric['reference_metric_name']
            x2 = config['METRICS'][name2]
            if len(x2) > len(x): x2 = x2[:len(x)]
            if smaller_is_better:
                idx = np.argmin(x2)
            else:
                idx = np.argmax(x2)
            x = x[idx]
        elif metric['agg'] == 'early_stop':
            name2 = metric['reference_metric_name']
            x2 = config['METRICS'][name2]
            k = 0
            prev = None
            for idx, val in enumerate(x2):
                if prev is None: prev = val
                if val > prev and smaller_is_better: k += 1
                if val < prev and not smaller_is_better: k += 1
                if k == 3: break
                prev = val

            # this condition occurs if we have metrics which only happen after the second epoch
            if idx >= x.size: idx = x.size-1
            x = x[idx]
        config[name] = x

# build dictionary of filter values
filters = {}
for keyvalue in args.filter:
    key, value = keyvalue.strip().split('=')
    filters[key] = value
args.filter = filters

# remove all configs with data not in the metric limits [a, b]
# remove all data that does not conform to the filter
# also remove all nan values
i = 0
while i < len(configs):
    remove = False
    cfg = configs[i]
    # removal based on not satisfying limits
    if primary_metric in cfg:
        x = cfg[primary_metric]
        if math.isnan(x): remove = True
        if args.limits is not None:
            if x < args.limits[0]: remove = True
            if x > args.limits[1]: remove = True
    else:
        remove = True

    # removal based on not satisfying filter value
    for key, value in cfg.items():
        if key in args.filter:
            if value != args.filter[key]: remove = True

    if remove:
        configs.pop(i)
        continue
    else:
        i += 1

#idx = np.argsort(np.array(data))
#if args.lower_is_better: idx = idx[::-1]

all_cols.update([m['name'] for m in metrics])
all_cols = list(all_cols)
pandas_data = []
for i, config in enumerate(configs):
    row = []
    for col in all_cols:
        if col in config:
            row.append(config[col])
        else:
            row.append('NaN')
    pandas_data.append(row)

df = pd.DataFrame(pandas_data, columns=all_cols)

for col in args.groupby:
    if col not in all_cols:
        print('Column {0} does not exist'.format(col))
    assert col in all_cols

if df.size == 0:
    print('No logged data found!')
    print('Last log file that was processes: {0}'.format(log_name))
    sys.exit()

# compute general group by statistics: Mean, standard error, 95% confidence interval, sample size, and log file names
output = df.groupby(by=args.groupby)[primary_metric].mean().to_frame().reset_index()
output['logfiles']= df.groupby(by=args.groupby)['NAME'].transform(lambda x: ' '.join(x)).to_frame()['NAME']

for metric in metrics:
    name = metric['name']
    df[name] = df[name].astype(np.float32)
    output[name] = df.groupby(by=args.groupby)[name].mean().to_frame(name).reset_index()[name]
    se95 = df.groupby(by=args.groupby)[name].sem().to_frame(name).reset_index()
    output['{0}_lower'.format(name)] = output[name]-(se95[name]*1.96)
    output['{0}_upper'.format(name)] = output[name]+(se95[name]*1.96)
    output['{0}_se'.format(name)] = se95[name]
    output['n'] = df.groupby(by=args.groupby)[name].size().to_frame().reset_index()[name]
    output['idx'] = df.groupby(by=args.groupby)[name].apply(lambda x: x.index.tolist()).to_frame().reset_index()[name]


if smaller_is_better:
    output = output.sort_values(by=primary_metric, ascending=False)
else:
    output = output.sort_values(by=primary_metric, ascending=True)

print('='*80)
banned = set(['n', 'NAME', 'logfiles', 'idx', 'default'])
banned.update([m['name'] for m in metrics])
banned.update(['{0}_lower'.format(m['name']) for m in metrics])
banned.update(['{0}_upper'.format(m['name']) for m in metrics])
banned.update(['{0}_se'.format(m['name']) for m in metrics])
for index, row in output.iterrows():
    str_config = ''
    for k, v in row.iteritems():
        if k in banned: continue
        str_config += '{0}: {1: <4}, '.format(k, v)


    print('Config: {0}'.format(str_config[:-2]))
    for metric in metrics:
        name = metric['name']
        value = row[name]
        lower = row['{0}_lower'.format(name)]
        upper = row['{0}_upper'.format(name)]
        se = row['{0}_se'.format(name)]
        n = int(row['n'])
        print(('{5} mean (SE): {0:.' + str(args.num_digits) + 'f} ({4:.4f}). 95% CI ({1:.3f}, {2:.3f}). Sample size: {3}').format(value, lower, upper, n, se, name))
    if args.vim:
        files = ' '.join([df.iloc[idx]['NAME'] for idx in row['idx']])
        print('vim {0}'.format(files))

    if args.all:
        for idx in row['idx']:
            print(configs[idx]['NAME'], configs[idx][primary_metric])
    print('='*80)

if args.csv is not None:
    if not os.path.exists(os.path.dirname(args.csv)):
        os.makedirs(os.path.dirname(args.csv))
    metrics = set([m['name'] for m in metrics])
    to_drop = [col for col in df.columns if col not in args.groupby and col not in metrics]
    df = df.drop(columns=to_drop)
    df.to_csv(args.csv, sep=';', index=False)

