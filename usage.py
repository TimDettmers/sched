import os
import subprocess
import shlex
import re
import datetime
import time

def execute_and_return(strCMD):
    proc = subprocess.Popen(shlex.split(strCMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    return out, err

label2column = {}
label2column['usage'] = -7
label2column['free'] = -3


def get_data(column=-7):
    out, err = execute_and_return('clust')
    usage = out.split('\n')[column]
    queues = usage.split('|')[1:]
    data = [str(datetime.datetime.now()), str(datetime.datetime.now().weekday())]
    for q in queues:
        splits = q.split('  ')
        if len(splits) == 2:
            data.append(splits[0].strip())
        elif len(splits) == 3:
            data.append(splits[1].strip())
    return data

iters = 0
usage_path = './usage_report.txt'
free_path = './free_report.txt'
header = ['date', 'weekday', 'total', 'learnfair', 'learnlab', 'devlab', 'prioritylab', 'scavenge', 'other']

while True:
    if iters == 0:
        with open(usage_path, 'w') as g:
            g.write(','.join(header)+'\n')

        with open(free_path, 'w') as g:
            g.write(','.join(header)+'\n')

    data = get_data(label2column['usage'])

    with open(usage_path, 'a') as g:
        g.write(','.join(data)+'\n')

    data = get_data(label2column['free'])

    with open(free_path, 'a') as g:
        g.write(','.join(data)+'\n')

    iters += 1
    time.sleep(60*15) # every 15 minutes



