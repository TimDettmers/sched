import argparse
import subprocess
import shlex

parser = argparse.ArgumentParser('Script to restart failed or timeouted jobs.')
parser.add_argument('--startid', type=int, required=True, help='Restart all failed or timeouted jobs with this jobid or greater.')
parser.add_argument('--user', type=str, required=True, help='The username for the restart.')
parser.add_argument('--dry', action='store_true', help='Dry run the scripts to execute')
parser.add_argument('--state', type=str, default='' ,help='If set only restarts jobs with a specific status: {FAILED,PREEMPTED,TIMEOUT}.')

args = parser.parse_args()

def execute_and_return(strCMD):
    proc = subprocess.Popen(strCMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    return out, err


cmd = 'sacct -X -u {0} --format="Jobid,State,JobName%250,NodeList" --noheader'.format(args.user)

out, err = execute_and_return(cmd)

if len(err) > 0:
    print(err)
    exit()

lines = out.split('\n')

states = set(['FAILED', 'TIMEOUT', 'PREEMPTED'])

banned = set()
restarts = set()
script2data = {}
for l in lines:
    data = [col for col in l.split(' ') if len(col) > 0]
    jobid = int(data[0])
    state = data[1]
    script = data[2]
    node = data[3]
    if script in script2data and state in ['RUNNING', 'COMPLETED', 'PENDING']:
        # job already restarted successfully, no action needed, remove job
        restarts.discard(script)
        script2data.pop(script)
    if state not in states: continue
    # add nodes that failed in the past even though the job to restart might not have failed on it
    if state == 'FAILED': banned.add(node)
    if jobid < args.startid: continue
    if args.state != '' and state != args.state: continue
    restarts.add(script)
    script2data[script] = (jobid, state, node)

print('Banned nodes: {0}'.format(','.join(banned)))
if args.dry:
    print('')
    print('='*80)
    print('Restarting the following {0} jobs...'.format(len(restarts)))
    print('='*80)
    print('')
for script in restarts:
    cmd = 'sbatch --exclude={1} {0}'.format(script, ','.join(banned))
    if not args.dry:
        print('Restarting script: {0}'.format(script))
        data = script2data[script]
        print('Originally: Job {0} with State {1} on NodeList {2}'.format(*data))
        out, err = execute_and_return(cmd)
        if len(err) > 0:
            print('Error in sbatch call: {0}'.format(err))
    else:
        data = script2data[script]
        print('Originally: Job {0} with State {1} on NodeList {2}'.format(*data))
        print(cmd)

