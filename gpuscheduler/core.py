import os
import subprocess
import shlex
import re
import pandas as pd
import threading
import uuid
import operator
import datetime
import shutil
import time
import numpy as np
import copy
import hashlib

from queue import Queue
from os.path import join
from aenum import Enum

MEM_THRESHOLD_AVAILABLE = 300
UTILIZARTION_THERESHOLD_AVAILABLE = 5

class HostState(Enum):
    unknown = 0
    available = 1
    error = 2

class GPUStatus(Enum):
    unknown = 0
    available = 1
    busy = 2

def cmd_over_ssh(name, script):
    strCMD = 'ssh {0} bash -l {1}'.format(name, script)
    return execute_and_return(strCMD)

def execute(strCMD):
    try:
        return subprocess.check_output(strCMD, shell=True, universal_newlines=True)
    except:
        return None

def cmds_to_hash(cmds, nested=False):
    strval = ''
    if nested:
        for cmds2 in cmds:
            for cmd in cmds2:
                strval += cmd
    else:
        for cmd in cmds:
            strval += cmd

    return hashlib.md5(strval.encode('utf-8')).hexdigest()

def execute_blocking(strCMD):
    proc = subprocess.Popen(strCMD, shell=True, universal_newlines=True)
    proc.wait()
    proc.terminate()

def execute_and_return(strCMD):
    proc = subprocess.Popen(shlex.split(strCMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    return out, err

gpu_name2fp16 = {}
gpu_name2fp16['TITAN V'] = True
gpu_name2fp16['GeForce RTX 2080 Ti'] = True
gpu_name2fp16['TITAN RTX'] = True

gpu_name2performance_class = {}
gpu_name2performance_class['TITAN V'] = 10
gpu_name2performance_class['TITAN RTX'] = 10
gpu_name2performance_class['GeForce RTX 2080 Ti'] = 8
gpu_name2performance_class['TITAN X (Pascal)'] = 5
gpu_name2performance_class['TITAN Xp'] = 4
gpu_name2performance_class['GeForce GTX TITAN X'] = 3
gpu_name2performance_class['GeForce GTX 1080 Ti'] = 3


class GPUWorker(threading.Thread):
    def __init__(self, scheduler, local_config, config_folder, logdir, host_name, host_config, device_id, job, idx, cmds):
        super(GPUWorker, self).__init__()
        self.scheduler = scheduler
        self.isDaemon = False
        self.job = job
        self.idx = idx
        self.cfg = host_config
        self.device_id = device_id
        self.host_name = host_name
        self.logdir = logdir
        self.init_dir = config_folder
        self.prefix = '{0}.{1}'.format(host_name, device_id)
        self.additional_cmds = cmds
        self.local_config = local_config

    def construct_init_file(self):
        if not os.path.exists('/tmp/gpuscheduler'): os.mkdir('/tmp/gpuscheduler')
        if not os.path.exists(join('/tmp/gpuscheduler', str(self.idx))): os.mkdir(join('/tmp/gpuscheduler', str(self.idx)))

        print('{0}: Constructing init file...'.format(self.prefix))
        init_path = join('/tmp/gpuscheduler/', str(self.idx), 'init_{0}.sh'.format(self.idx))
        shutil.copyfile(join(self.init_dir, 'init.sh'), init_path)
        if self.cfg['conda_path'] != 'anaconda3':
            new_env = "sed -i 's/anaconda3/{0}/g' {1}".format(self.cfg['conda_path'], init_path)
            print('{0}: Executing change in conda path variable...'.format(self.prefix))
            execute(new_env)

        work_dir_remote = join(self.cfg['GIT_HOME'], self.job['work_dir'])
        repo_local = join(self.local_config['GIT_HOME'], self.job['repo_dir'])
        repo_remote = join(self.cfg['GIT_HOME'])

        with open(init_path, 'a') as f:
            f.write('export GIT_HOME={0}\n'.format(self.cfg['GIT_HOME']))
            f.write('cd {0}\n'.format(work_dir_remote))
            if self.cfg['conda_env'] != 'base':
                f.write('source activate {0}\n'.format(self.cfg['conda_env']))
            for cmd in self.additional_cmds:
                f.write('{0}\n'.format(cmd))
            f.write('CUDA_VISIBLE_DEVICES={0} {1}\n'.format(self.device_id, self.job['cmd']))
        time.sleep(1)
        cmd = 'scp {0} {1}:~/'.format(init_path, self.host_name)
        print('{0}: Transfering init file...'.format(self.prefix))
        execute(cmd)

        print('Performing rsync...')
        rsync = 'rsync --update -raz --progress --max-size=10m {0} {1}:{2}/'.format(repo_local, self.host_name, repo_remote)
        execute_blocking(rsync)

        time.sleep(2)


    def create_log_path(self, path):
        path = os.path.normpath(path)
        paths = path.split(os.sep)
        full_path = self.logdir
        for p in paths:
            full_path = join(full_path, p)
            if not os.path.exists(full_path): os.mkdir(full_path)


    def run(self):
        if not os.path.exists(self.logdir): os.mkdir(self.logdir)
        if not os.path.exists(join(self.logdir, 'errors')): os.mkdir(join(self.logdir, 'errors'))
        print('Started worker {0} on Host {1} for GPU {2}'.format(self.idx, self.host_name, self.device_id))
        self.construct_init_file()
        print('Executing on {0}:{1}...'.format(self.host_name, self.device_id))
        out, err = cmd_over_ssh(self.host_name, 'init_{0}.sh'.format(self.idx))
        log_name = str(uuid.uuid4()) + '.log'
        if len(err) > 0 and 'warning' not in err.lower():
            print('{1}: ERROR: {0}'.format(err, self.prefix))
            err_file_path = join(self.logdir, 'errors', log_name)
            with open(err_file_path, 'w') as f:
                f.write(out)

        path = self.job['path']
        self.create_log_path(path)
        file_path = join(self.logdir, path, log_name)
        with open(file_path, 'w') as f:
            f.write(out)

        if len(err) > 0 and not 'warning' in err.lower():
            print('{0}: Finish task with errors! Writing stdout data to {1} and error to {2}...'.format(self.prefix, file_path, err_file_path))
        else:
            print('{0}: Finish task successfully! Writing data to {1}...'.format(self.prefix, file_path))


class GantryScheduler(object):
    def __init__(self, config_path, cluster, budget, workspace, weka=None, image='ai2/cuda11.8-dev-ubuntu20.04'):
        self.jobs = []
        self.config = {}
        self.init_with_config(config_path)
        self.cluster = cluster
        self.budget = budget
        self.weka = weka
        self.workspace = workspace
        self.image = image

    def init_with_config(self, config_path):
        with open(config_path) as f:
            for line in f:
                name, value = line.split(' ')
                self.config[name.strip()] = value.strip()

    def update_host_config(self, name, mem_threshold, util_threshold):
        pass


    def add_job(self, path, cmds, time_hours=0, fp16=False, gpus=1, mem=32, cores=6, constraint='', exclude='', time_minutes=0):
        if constraint != '': raise NotImplementedError('contraint not supported by beaker')
        if time_minutes != 0: raise NotImplementedError('time limits are not supported by beaker')
        if time_hours != 0: raise NotImplementedError('time limits are not supported by beaker')
        if exclude != '': raise NotImplementedError('exclude are not supported by beaker')
        self.jobs.append([path, cmds, time_hours, fp16, gpus, mem, cores, constraint, exclude, time_minutes])

    def run_jobs(self, preemptible=True, as_array=True, sleep_delay_seconds=0, single_process=False, log_id=None, skip_cmds=0, comment=None, begin=None, gpus_per_node=8, requeue=False, requeue_length_hours=4, priority='low'):

        if not os.path.exists(self.config['SCRIPT_HISTORY']):
            os.makedirs(self.config['SCRIPT_HISTORY'])

        print('processing cmds...')
        file_contents = ''

        strval = self.jobs[0][1]
        if not isinstance(strval, str): strval = strval[0]
        init_hash = cmds_to_hash(self.jobs[0][1], nested=True)
        init_file = join(self.config['SCRIPT_HISTORY'], 'init_{0}.sh'.format(init_hash))

        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#\n')
        for i, (path, cmds, time_hours, fp16, gpus, mem, cores, constraint, exclude, time_minutes) in enumerate(self.jobs):
            if not as_array:
                if i % 10 == 0 and i > 0: print('Processing cmd no ', i)

            if not isinstance(cmds, list): cmds = [cmds]

            cmd_hash = cmds_to_hash(cmds)
            run_file = join(self.config['SCRIPT_HISTORY'], f'run_{cmd_hash}.sh')
            if gpus_per_node == 0:
                nodes = 1
            else:
                nodes = gpus // gpus_per_node
                nodes += 1 if (gpus % gpus_per_node) > 0 else 0
            if nodes == 0: nodes = 1
            gpus = gpus_per_node if gpus > gpus_per_node else gpus
            if not isinstance(cmds, list): cmds = [cmds]
            log_folder = join(join(self.config['LOG_HOME'], path))
            log_file = join(log_folder, f'{cmd_hash}.log')
            os.makedirs(log_folder, exist_ok=True)
            if gpus > 8: raise NotImplementedError('Multi-node jobs are currently not supported')

            if isinstance(self.cluster, list):
                cluster = ''
                for c in self.cluster:
                    cluster += f'--cluster {c} '
            else:
                cluster = '--cluster {self.cluster}'

            gpus = '' if gpus == 0 else '--gpus {gpus}'
            cores = '' if cores == 0 else '--cpus {cores}'
            lines.append((f'gantry run --host-networking --allow-dirty {cores} {gpus} --workspace {self.workspace}'
                    f' {cluster} {"--preemptible" if preemptible else ""} --priority {priority}'
                    f' {f"--weka={self.weka}" if self.weka is not None else ""} --beaker-image {self.image}'
                    f' --no-python --budget {self.budget} -n {join(path, cmd_hash+".log").replace("/", "_")} -- bash {run_file} &\n\n'))
            lines.append('sleep 0.1\n')

            with open(run_file, 'w') as g:
                g.write('#!/bin/bash\n')
                g.write('#\n')
                g.write('export PATH="{0}:$PATH"'.format(join(self.config['ANACONDA_HOME'], 'bin')) + '\n')
                g.write('\n')
                for cmd_no, cmd in enumerate(cmds[skip_cmds:]):
                    #g.write(f'echo "{cmd}"\n')
                    if ('export' in cmd) or ('cd' in cmd) or ('curl' in cmd):
                        g.write(cmd + '\n')
                    else:
                        g.write(f'echo {cmd} \n')
                        g.write(cmd + f' 2>&1 | tee {log_file} \n')


        print('writing init file to:')
        print(init_file)
        with open(init_file, 'w') as f:
            f.writelines(lines)

        print('executing ...')
        out, err = execute_and_return('bash {0}'.format(init_file))
        if err != '':
            print(err)






class HyakScheduler(object):
    def __init__(self, config_folder='./config', verbose=False, account='cse', partition='cse-gpu', use_gres=False):
        self.jobs = []
        self.verbose = verbose
        self.config = {}
        self.remap = {}
        self.init_with_config(config_folder)
        self.config['account'] = account
        self.config['partition'] = partition
        self.use_gres = use_gres

    def init_with_config(self, config_folder):
        with open(join(config_folder, 'slurm_config.cfg')) as f:
            for line in f:
                name, value = line.split(' ')
                self.config[name.strip()] = value.strip()

    def update_host_config(self, name, mem_threshold, util_threshold):
        pass



    def add_job(self, path, repo_dir, work_dir, cmds, time_hours, fp16=False, gpus=1, mem=32, cores=6, constraint='', exclude='', time_minutes=0):
        self.jobs.append([path, work_dir, cmds, time_hours, fp16, gpus, mem, cores, constraint, exclude, time_minutes])
        if self.verbose:
            print('#SBATCH --time={0:02d}:{1:02d}:00'.format(time_hours, time_minutes))

    def run_jobs(self, as_array=True, sleep_delay_seconds=0, single_process=False, log_id=None, skip_cmds=0, comment=None, begin=None, gpus_per_node=8, requeue=False, requeue_length_hours=4):

        array_preamble = []


        strval = self.jobs[0][2]
        array_id = cmds_to_hash(strval) if log_id is None else log_id

        array_file = join(self.config['SCRIPT_HISTORY'], 'array_init_{0}.sh'.format(array_id))
        array_job_list = join(self.config['SCRIPT_HISTORY'], 'array_jobs_{0}.sh'.format(array_id))
        script_list = []
        print('processing cmds...')
        file_contents = ''
        for i, (path, work_dir, cmds, time_hours, fp16, gpus, mem, cores, constraint, exclude, time_minutes) in enumerate(self.jobs):
            if not as_array:
                if i % 10 == 0 and i > 0: print('Processing cmd no ', i)
            if gpus_per_node == 0:
                nodes = 1
            else:
                nodes = gpus // gpus_per_node
                nodes += 1 if (gpus % gpus_per_node) > 0 else 0
            if nodes == 0: nodes = 1
            gpus = gpus_per_node if gpus > gpus_per_node else gpus
            if not isinstance(cmds, list): cmds = [cmds]
            lines = []
            script_file = join(self.config['SCRIPT_HISTORY'], 'init_{0}_{1}.sh'.format(array_id, i))

            script_list.append(script_file)
            log_path = join(join(self.config['LOG_HOME'], path))
            lines.append('#!/bin/bash')
            lines.append('#')
            lines.append('#SBATCH --job-name={0}'.format(script_file))
            if self.config['account'] != '':
                lines.append('#SBATCH --account={0}'.format(self.config['account']))
            lines.append('#SBATCH --partition={0}'.format(self.config['partition']))
            lines.append('#')
            lines.append('#SBATCH --nodes={0}'.format(nodes))
            if single_process:
                lines.append('#SBATCH --ntasks-per-node=1')
                lines.append('#SBATCH --cpus-per-task={0}'.format(cores*(gpus if gpus != 0 else 1)))
            else:
                lines.append('#SBATCH --ntasks-per-node={0}'.format(gpus if gpus != 0 else 1))
                lines.append('#SBATCH --cpus-per-task={0}'.format(cores))
            lines.append('#SBATCH --time={0:02d}:{1:02}:00'.format(time_hours, time_minutes))
            if gpus > 0:
                if self.use_gres:
                    lines.append('#SBATCH --gres=gpu:{0}'.format(gpus))
                else:
                    lines.append('#SBATCH --gpus-per-node={0}'.format(gpus))
            lines.append('#SBATCH --mem={0}G'.format(mem))
            if len(constraint) > 0:
                lines.append('#SBATCH --constraint={0}'.format(constraint))
            if exclude != '':
                lines.append('#SBATCH --exclude={0}'.format(exclude))
            if comment is not None:
                lines.append('#SBATCH --comment={0}'.format(comment))
            if begin is not None:
                lines.append('#SBATCH --begin={0}'.format(begin))

            lines.append('#')
            lines.append('#SBATCH --open-mode=append')
            lines.append('#SBATCH --chdir={0}'.format(join(self.config['GIT_HOME'], work_dir)))
            lines.append('#SBATCH --output={0}'.format(join(log_path, array_id + '_{0}.log'.format(i))))
            lines.append('#SBATCH --error={0}'.format(join(log_path, array_id + '_{0}.err'.format(i))))
            lines.append('')
            lines.append('export PATH=$PATH:{0}'.format(join(self.config['ANACONDA_HOME'], 'bin')))
            for cmd_no, cmd in enumerate(cmds[skip_cmds:]):
                lines.append(cmd)

            if len(array_preamble) == 0:
                array_preamble = copy.deepcopy(lines[:-(1*len(cmds[skip_cmds:]) + 1)])
                array_preamble[2] = '#SBATCH --job-name={0}'.format(array_job_list)
                array_preamble[-3] = '#SBATCH --output={0}'.format(join(log_path, array_id + '_%a.log'))
                array_preamble[-2] = '#SBATCH --error={0}'.format(join(log_path, array_id + '_%a.err'))
                array_preamble.append('#SBATCH --array=0-{0}'.format(len(self.jobs)-1))
                array_preamble.append('')
                array_preamble.append('export PATH=$PATH:{0}'.format(join(self.config['ANACONDA_HOME'], 'bin')))

            if not os.path.exists(log_path):
                print('Creating {0}'.format(log_path))
                os.makedirs(log_path)

            if not os.path.exists(self.config['SCRIPT_HISTORY']):
                print('Creating {0}'.format(self.config['SCRIPT_HISTORY']))
                os.makedirs(self.config['SCRIPT_HISTORY'])


            if not as_array:
                print('Writing job file to: {0}'.format(script_file))
                with open(script_file, 'w') as f:
                    for line in lines:
                        f.write('{0}\n'.format(line))
                if not requeue:
                    time.sleep(0.05)
                    out, err = execute_and_return('sbatch {0}'.format(script_file))
                    if err != '':
                        print(err)
                else:
                    num_requeues = int((time_hours+(time_minutes/60)+requeue_length_hours-0.01)/requeue_length_hours)
                    bid, err = execute_and_return('sbatch --parsable {0}'.format(script_file))
                    for j in range(num_requeues-1):
                        print(num_requeues, bid)
                        if err != '':
                            print(err)
                            break
                        time.sleep(0.05)
                        bid, err = execute_and_return(f'sbatch --parsable --dependency=afterany:{bid} {script_file}')

        if as_array:
            print('creating array...')
            array_lines = []
            array_lines.append('')
            array_lines.append('')
            array_lines.append('echo $SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID'.format(cmd_no))
            for i, (path, work_dir, cmds, time_hours, fp16, gpus, mem, cores, constraint, exclude, time_minutes) in enumerate(self.jobs):
                bare_script_file = join(self.config['SCRIPT_HISTORY'], 'init_bare_{0}_{1}.sh'.format(array_id, i))
                bare_lines = []
                bare_lines.append('#!/bin/bash')
                bare_lines.append('#')
                bare_lines.append('export PATH=$PATH:{0}'.format(join(self.config['ANACONDA_HOME'], 'bin')))
                for cmd_no, cmd in enumerate(cmds[skip_cmds:]):
                    bare_lines.append(cmd)
                with open(bare_script_file, 'w') as f:
                    for line in bare_lines:
                        f.write('{0}\n'.format(line))

                if i == 0:
                    array_lines.append('if [[ $SLURM_ARRAY_TASK_ID -eq {0} ]]'.format(i))
                    array_lines.append('then')
                else:
                    array_lines.append('elif [[ $SLURM_ARRAY_TASK_ID -eq {0} ]]'.format(i))
                    array_lines.append('then')
                if sleep_delay_seconds > 0:
                    array_lines.append('\t sleep {0}'.format(i*sleep_delay_seconds))

                array_lines.append('\t srun bash ' + bare_script_file)

            array_lines.append('else')
            array_lines.append('\t echo $SLURM_ARRAY_TASK_ID')
            array_lines.append('fi')


            array_lines = array_preamble + array_lines
            print('Writing array file to: {0}'.format(array_file))
            with open(array_file, 'w') as f:
                for line in array_lines:
                    f.write('{0}\n'.format(line))

            print('Writing job list to: {0}'.format(array_job_list))
            with open(array_job_list, 'w') as f:
                for line in script_list:
                    f.write('{0}\n'.format(line))

            if not requeue:
                out, err = execute_and_return('sbatch {0}'.format(array_file))
                if err != '':
                    print(err)
            else:
                raise NotImplementedError('Requeue does not work for array jobs!')






class SshScheduler(object):

    """Execute tasks over ssh in the background."""

    def __init__(self, config_folder='./config', verbose=False):
        self.queue = Queue()
        self.host_data = None

        self.last_polled = datetime.datetime.now() - datetime.timedelta(hours=1)

        self.host2config = self.init_hosts(config_folder)
        self.config_folder = config_folder
        self.verbose = verbose
        self.local_config = {}
        self.remap = {}
        self.init_with_config(config_folder)
        self.init_remap(config_folder)

    def init_with_config(self, config_folder):
        with open(join(config_folder, 'ssh_config.cfg')) as f:
            for line in f:
                name, value = line.split(' ')
                self.local_config[name.strip()] = value.strip()

    def init_remap(self, config_folder):
        with open(join(config_folder, 'remap.txt')) as f:
            for line in f:
                host, a, b = line.split(',')
                self.remap[(host.strip(), int(a.strip()))] = int(b.strip())



    def init_hosts(self, config_folder):
        """Parses host config file and their git paths.

        :config_folder: Path to the config folder
        :returns: Dictionart host2config

        """
        host2config = {}

        with open(join(config_folder, 'hosts.txt')) as f:
            data = pd.read_csv(f)
            self.host_data = data
            for index, row in data.iterrows():
                name = row['ssh name']
                host2config[name] = {}
                host2config[name]['GIT_HOME'] = row['git path']
                host2config[name]['conda_env'] = row['conda env']
                host2config[name]['conda_path'] = row['conda path']
                host2config[name]['priority'] = int(row['priority'])
                host2config[name]['min_free'] = int(row['min free gpus'])
                host2config[name]['status'] = HostState.unknown
                host2config[name]['mem_threshold'] = MEM_THRESHOLD_AVAILABLE
                host2config[name]['util_threshold'] = UTILIZARTION_THERESHOLD_AVAILABLE
        return host2config

    def update_host_config(self, name, mem_threshold, util_threshold):
        """Updates particular config values for particular hosts."""
        self.host2config[name]['mem_threshold'] = mem_threshold
        self.host2config[name]['util_threshold'] = util_threshold


    def poll_gpu_status(self):
        """Polls GPU status (if a GPU is used etc)."""
        print('Polling a total of {0} hosts...'.format(len(self.host2config)))
        for i, host in enumerate(self.host2config):
            if self.verbose:
                print('Polling host {0} ...'.format(host))
            if i > 0 and i % 3 == 0: print('{0}/{1}'.format(i, len(self.host2config)))
            strCMD = 'ssh {0} "nvidia-smi -q -x"'.format(host)
            out, err = execute_and_return(strCMD)
            if err != '':
                self.host2config[host]['status'] = HostState.unknown
                self.host2config[host]['num_available'] = 0
                if self.verbose:
                    print('Error in nvidia-smi call!')
                    print(err)
                continue
            gpus, num_available = self.parse_nvidia_smi(out, host)
            self.host2config[host]['gpus'] = gpus
            self.host2config[host]['status'] = HostState.available
            self.host2config[host]['num_available'] = max(num_available-self.host2config[host]['min_free'], 0)
        self.last_polled = datetime.datetime.now()

    def get_total_available(self):
        """Gets the total amount of GPUs available after min free threshold."""
        if (datetime.datetime.now() - self.last_polled).total_seconds() > 60:
            self.poll_gpu_status()
        total_available = 0
        total_available_fp16 = 0
        for host, config in self.host2config.items():
            if config['status'] != HostState.available:
                print('Host {0} is down.'.format(host))
                continue
            num_available = config['num_available']
            min_free = config['min_free']

            total_available += num_available
            total_fp16 = 0
            for gpu in config['gpus']:
                if gpu['status'] == GPUStatus.available and gpu['fp16']:
                    total_fp16 += 1
            avail_fp16 = min(num_available, total_fp16)
            total_available_fp16 += avail_fp16

            print('Host: {0}. Available 16-bit: {2}. Total available {1}.'.format(host, num_available, min(num_available, avail_fp16)))

        print('A total of {0} GPUs are available on {1} total hosts.'.format(total_available, len(self.host2config)))
        print('Of these GPUs a total of {0} have 16-bit capability (tensor cores).'.format(total_available_fp16))
        return total_available


    def parse_nvidia_smi(self, text, host):
        """Parses nvidia-smi output."""
        from bs4 import BeautifulSoup
        xml_soup = BeautifulSoup(text, 'xml')
        gpus = []
        num_available = 0
        for gpu_node in xml_soup.find_all('gpu'):
            gpu = {}
            gpu['name'] = gpu_node.product_name.text
            gpu['device_id'] = int(gpu_node.minor_number.text)
            gpu['utilization'] = int(gpu_node.gpu_util.text[:-2])
            gpu['total_mem'] = int(gpu_node.fb_memory_usage.total.text[:-4])
            gpu['used_mem'] = int(gpu_node.fb_memory_usage.used.text[:-4])
            gpu['free_mem'] = int(gpu_node.fb_memory_usage.free.text[:-4])
            gpu['fp16'] = gpu['name'] in gpu_name2fp16
            gpu['status'] = self.determine_gpu_status(gpu, host)
            if gpu['status'] == GPUStatus.available: num_available += 1
            if gpu['name'] in gpu_name2performance_class:
                gpu['performance'] = gpu_name2performance_class[gpu['name']]
            else:
                print(gpu['name'], 'NOT IN PERFORMANCE CLASS. PLEASE ADD NOW.')
                gpu['performance'] = 5
            gpus.append(gpu)
        return gpus, num_available


    def determine_gpu_status(self, gpu, host):
        """Determines is a GPU is available for the queue."""
        mem_threshold = self.host2config[host]['mem_threshold']
        util_threshold = self.host2config[host]['util_threshold']
        if self.verbose:
            print('device {0} on host {1} has {2} mem and {3} util -> status {4}'.format(
                gpu['device_id'], host, gpu['used_mem'], gpu['utilization'],
                gpu['used_mem'] < mem_threshold and gpu['utilization'] < util_threshold))
        if gpu['used_mem'] < mem_threshold and gpu['utilization'] < util_threshold:
            return GPUStatus.available
        else:
            return GPUStatus.busy

    def add_job(self, path, repo_dir, work_dir, cmd, fp16=False, gpus=1, cores=None):
        """Adds a job to execute.

        :path: Sub-folder path for the log file.
        :fp16: If the job requires 16-bit capabilities.
        :gpus: The number of GPUs required for the job.

        """
        job = {}
        job['path'] = path
        job['work_dir'] = work_dir
        job['repo_dir'] = repo_dir
        job['cmd'] = cmd
        job['fp16'] = fp16
        job['gpus'] = gpus
        self.queue.put(job)

    def get_gpu_priority_list(self):
        """Gets the hosts and GPUs on which to execute first."""
        priority_list = []
        host_data = self.host_data.sort_values(by=['priority'], ascending=False)
        for idx, host in host_data.iterrows():
            name = host['ssh name']
            if self.host2config[name]['status'] != HostState.available: continue
            gpus = self.host2config[name]['gpus']
            num_available = self.host2config[name]['num_available']
            ids = []
            performance = []
            fp16s = []
            for gpu in gpus:
                if gpu['status'] != GPUStatus.available: continue
                ids.append(gpu['device_id'])
                performance.append(gpu['performance'])
                fp16s.append(gpu['fp16'])
            if len(ids) == 0: continue
            performance, ids = zip(*sorted(zip(performance, ids), reverse=True))
            for device_id, fp16 in zip(ids[:num_available], fp16s[:num_available]):
                priority_list.append((name, device_id, fp16))
        return priority_list


    def run_jobs(self, cmds=[], host2cmd_adds={}):
        gpus_available = self.get_total_available()

        while self.queue.qsize() > 0:
            workers = []
            print('Total jobs left: {0}.'.format(self.queue.qsize()))
            print('Calculating priority list...')
            priority_list = self.get_gpu_priority_list()
            print('{0}: Starting jobs...'.format(datetime.datetime.now()))
            for i in range(min(gpus_available, self.queue.qsize(), len(priority_list))):
                host, device_id, fp16 = priority_list[i]
                if (host, device_id) in self.remap:
                    device_id = self.remap[(host, device_id)]
                job = self.queue.get()

                if host in host2cmd_adds:
                    job['cmd'] += host2cmd_adds[host]

                workers.append(GPUWorker(self, self.local_config, self.config_folder, self.local_config['LOG_HOME'], host, self.host2config[host], device_id, job, i, cmds))

            for worker in workers:
                time.sleep(5)
                worker.start()

            if self.queue.qsize() > 0:
                for i in range(5):
                    time.sleep(20 + np.random.randint(1, 7)) # wait for 20 seconds + some random amount of time
                    if self.queue.qsize() == 0: break
                if self.queue.qsize() > 0:
                    self.poll_gpu_status()
                    print('Getting total available...')
                    gpus_available = self.get_total_available()
            else:
                for worker in workers:
                    if self.verbose:
                        print('Waiting for worker: {0}'.format(worker.idx))
                    worker.join()


