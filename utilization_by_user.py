import argparse
import subprocess
import shlex
import datetime

parser = argparse.ArgumentParser('Script to restart failed or timeouted jobs.')
parser.add_argument('--days', type=int, default=30, help='How many days to look into the past')

args = parser.parse_args()

def execute_and_return(strCMD):
    proc = subprocess.Popen(strCMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    return out, err

cmd = 'sacct --noheader --allusers -X -S {0} --format="partition,allocgres,user,elapsed,start"'.format((datetime.datetime.today()-datetime.timedelta(days=args.days)).strftime('%Y-%m-%d'))


print(cmd)
out, err = execute_and_return(cmd)

partitions = {}
lines = out.split('\n')
for line in lines:
    values = [val for val in line.split(' ') if val != '']
    if len(values) == 5:
        p, gpus, user, elapsed, start = values
        if p not in partitions:
            partitions[p] = {}
        if user not in partitions[p]:
            partitions[p][user] = 0

        days = 0
        if '-' in elapsed:
            days = int(elapsed[:elapsed.index('-')])
            elapsed = elapsed[elapsed.index('-')+1:]
        hours = int(elapsed[0:2])
        minutes = int(elapsed[3:4])
        seconds = int(elapsed[6:7])

        GPU_hours = (days*24) + hours + (minutes/60) + seconds/(60*60)
        GPU_hours *= int(gpus[4])

        partitions[p][user] += GPU_hours


    else:
        # no GPUs used
        continue


for p in partitions:
    print('Partition {0}:'.format(p))
    for user in partitions[p]:
        print('\t{0:>11}: {1:>5} GPU hours'.format(user, str(int(partitions[p][user]))))
