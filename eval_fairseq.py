import os
import glob
import ntpath
import subprocess
import re
import shlex
import argparse

from os.path import join

parser = argparse.ArgumentParser(description='Log file evaluator.')
parser.add_argument('-f', '--folder-path', type=str, default=None, help='The folder with logfiles of models to evaluate.')
parser.add_argument('--finished-contains', type=str, default='done training', help='The string to recognize if a logfile contains a complete training run.')
parser.add_argument('--last-checkpoint', type=str, default='checkpoint_last.pt', help='The name of the last checkpoint file.')
parser.add_argument('--best-checkpoint', type=str, default='checkpoint_best.pt', help='The name of the best checkpoint file.')
parser.add_argument('--start', type=str, default='', help='String after which the checkpoint path.')
parser.add_argument('--end', type=str, default='\n', help='String before which the checkpoint path.')
parser.add_argument('--dry', action='store_true', help='Prints the commands that would be executed without executing them')
parser.add_argument('--append', action='store_true', help='Append the evaluation data to the log file instead of creating a new one')
parser.add_argument('--out', default=None, type=str, help='The output folder')
parser.add_argument('--args', type=str, default='--special-eval', help='Additional args for fairseq.')
parser.add_argument('--fairseq-path', type=str, default='/private/home/timdettmers/git/fairseq_private', help='The path to the fairseq source.')
parser.add_argument('--filter', nargs='+', type=str, default='', help='Only evaluate configs with these key-value parameters. Space separated key-values.')

args = parser.parse_args()
if args.out is None and not args.append:
    print('Either set the output path --out or set the --append option to append to the log file.')
    os.exit()

if args.out is not None and not os.path.exists(args.out):
    os.makedirs(args.out)

def clean_string(key):
    key = key.strip()
    key = key.replace("'", '')
    key = key.replace('"', '')
    key = key.replace(']', '')
    key = key.replace('[', '')
    key = key.replace('(', '')
    key = key.replace(')', '')
    return key

def execute_and_return(strCMD):
    proc = subprocess.Popen(shlex.split(strCMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    return out, err


folders = [x[0] for x in os.walk(args.folder_path)]
regex = re.compile(r'(?<={0}).*(?={1})'.format(args.start, args.end))

configs = []
all_cols = set(['NAME'])
for folder in folders:
    files = list(glob.iglob(join(folder, '*.log')))
    n = len(files)
    i = 0
    for log_name in files:
        config = {}
        i += 1
        with open(log_name) as f:
            lines = f.readlines()

        finished = False
        last_ckpt = None
        best_ckpt = None
        namespace = None
        for line in lines[::-1]:
            if args.finished_contains in line: finished = True
            if args.last_checkpoint in line:
                matches = re.findall(regex, line)
                if len(matches) > 0:
                    last_ckpt = matches[0].strip()
            if args.best_checkpoint in line:
                matches = re.findall(regex, line)
                if len(matches) > 0:
                    best_ckpt = matches[0].strip()

            if 'Namespace(' in line:
                namespace = line

                line = line[line.find('Namespace(')+len('Namespace('):]
                matches = re.findall(r'(?!^\()([^=,]+)=([^\0]+?)(?=,[^,]+=|\)$)', line)
                for m in matches:
                    key = clean_string(m[0])
                    value = clean_string(m[1])
                    config[key] = value

        if last_ckpt is None and best_ckpt is not None: last_ckpt = best_ckpt
        if last_ckpt is None or namespace is None or not finished: continue
        if 'data' not in config:
            print('Dataset not found! Skipping this log file: {0}'.format(log_name))
        if len(args.filter) > 0:
            execute = True
            for keyvalue in args.filter:
                key, value = keyvalue.split('=')
                key = key.strip()
                value = value.strip()
                if key not in config: execute = False
                else:
                    if value != config[key]: execute = False
            if not execute: continue

        cmd = 'fairseq-eval-lm --path {0} --max-tokens 4096 --skip-invalid-size-inputs-valid-test --log-format simple --log-interval 100 {2} {1}'.format(best_ckpt if best_ckpt is not None else last_ckpt, join(args.fairseq_path, config['data']), args.args)

        if args.dry:
            print(cmd)
        else:
            print('Executing command {0}/{1}'.format(i, n+1))
            out, err = execute_and_return(cmd)
            out = out + '\n' + err

            if 'Traceback' in out:
                print('ERROR!')
                print(log_name)
                print(out)
                print(cmd)
            else:
                if args.append:
                    with open(log_name, 'a') as g:
                        g.write('\n')
                        g.write(out)
                else:
                    with open(join(args.out, ntpath.basename(log_name)), 'w') as g:
                        g.write('\n')
                        g.write(namespace + '\n')
                        g.write(out)







