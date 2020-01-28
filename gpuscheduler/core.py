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

from queue import Queue
from bs4 import BeautifulSoup
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
    def __init__(self, scheduler, config_folder, logdir, host_name, host_config, device_id, job, idx, cmds):
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
        with open(init_path, 'a') as f:
            work_dir = join(self.cfg['GIT_HOME'], self.job['work_dir'])
            f.write('export GIT_HOME={0}\n'.format(self.cfg['GIT_HOME']))
            f.write('cd {0}\n'.format(work_dir))
            if self.cfg['conda_env'] != 'base':
                f.write('source activate {0}\n'.format(self.cfg['conda_env']))
            for cmd in self.additional_cmds:
                f.write('{0}\n'.format(cmd))
            f.write('CUDA_VISIBLE_DEVICES={0} {1}\n'.format(self.device_id, self.job['cmd']))
        time.sleep(1)
        cmd = 'scp {0} {1}:~/'.format(init_path, self.host_name)
        print('{0}: Transfering init file...'.format(self.prefix))
        execute(cmd)
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
        if len(err) > 0:
            print('{1}: ERROR: {0}'.format(err, self.prefix))
            err_file_path = join(self.logdir, 'errors', log_name)
            with open(err_file_path, 'w') as f:
                f.write(out)

        path = self.job['path']
        self.create_log_path(path)
        file_path = join(self.logdir, path, log_name)
        with open(file_path, 'w') as f:
            f.write(out)

        if len(err) > 0:
            print('{0}: Finish task with errors! Writing stdout data to {1} and error to {2}...'.format(self.prefix, file_path, err_file_path))
        else:
            print('{0}: Finish task successfully! Writing data to {1}...'.format(self.prefix, file_path))


class HyakScheduler(object):
    def __init__(self, config_folder, verbose=False, account='cse', partition='cse-gpu'):
        self.jobs = []
        self.verbose = verbose
        self.config = {}
        self.init_with_config(config_folder)
        self.config['account'] = account
        self.config['partition'] = partition

    def init_with_config(self, config_folder):
        with open(join(config_folder, 'config.cfg')) as f:
            for line in f:
                name, value = line.split(' ')
                self.config[name.strip()] = value.strip()
            

    def update_host_config(self, name, mem_threshold, util_threshold):
        pass



    def add_job(self, path, work_dir, cmd, time_hours, fp16=False, gpus=1, mem=32, cores=6):
        self.jobs.append([path, work_dir, cmd, time_hours, fp16, gpus, mem, cores])
        if self.verbose:
            print('#SBATCH --time={0:02d}:00:00'.format(time_hours))

    def run_jobs(self, logdir, cmds=[], add_fp16=False, host2cmd_adds={}, remap={}):
        for i, (path, work_dir, cmd, time_hours, fp16, gpus, mem, cores) in enumerate(self.jobs):
            lines = []
            logid = str(uuid.uuid4())
            log_path = join(join(logdir, path))
            lines.append('#!/bin/bash')
            lines.append('#')
            lines.append('#SBATCH --job-name={0}'.format(join(path, logid)))
            lines.append('#SBATCH --account={0}'.format(self.config['account']))
            lines.append('#SBATCH --partition={0}'.format(self.config['partition']))
            lines.append('#')
            lines.append('#SBATCH --nodes=1')
            lines.append('#SBATCH --ntasks-per-node=1')
            lines.append('#SBATCH --cpus-per-task={0}'.format(cores))
            lines.append('#SBATCH --time={0:02d}:00:00'.format(time_hours))
            lines.append('#SBATCH --gres=gpu:{0}'.format(gpus))
            lines.append('#SBATCH --mem={0}G'.format(mem))
            lines.append('#')
            lines.append('#SBATCH --chdir={0}'.format(join(self.config['GIT_HOME'], work_dir)))
            lines.append('#SBATCH --output={0}'.format(join(log_path, logid + '.log')))
            lines.append('#SBATCH --error={0}'.format(join(log_path, logid + '.err')))
            lines.append('')
            lines.append('export PATH=$PATH:{0}'.format(join(self.config['ANACONDA_HOME'], 'bin')))
            lines.append(cmd)

            if not os.path.exists(log_path):
                print('Creating {0}'.format(log_path))
                os.makedirs(log_path, exist_ok=True)


            with open('/tmp/init_{0}.sh'.format(i), 'w') as f:
                for line in lines:
                    f.write('{0}\n'.format(line))

            time.sleep(0.05)
            out, err = execute_and_return('sbatch /tmp/init_{0}.sh'.format(i))
            if err != '':
                print(err)






class SshScheduler(object):

    """Execute tasks over ssh in the background."""

    def __init__(self, config_folder, verbose=False):
        self.queue = Queue()
        self.host_data = None
        self.last_polled = datetime.datetime.now() - datetime.timedelta(hours=1)

        self.host2config = self.init_hosts(config_folder)
        self.config_folder = config_folder
        self.verbose = verbose

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
            print('Host: {0}. Available 16-bit: {2}. Total available {1}.'.format(host, min(num_available, avail_fp16), num_available))

        print('A total of {0} GPUs are available on {1} total hosts.'.format(total_available, len(self.host2config)))
        print('Of these GPUs a total of {0} have 16-bit capability (tensor cores).'.format(total_available_fp16))
        return total_available


    def parse_nvidia_smi(self, text, host):
        """Parses nvidia-smi output."""
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

    def add_job(self, path, work_dir, cmd, fp16=False, gpus=1, cores=None):
        """Adds a job to execute.

        :path: Sub-folder path for the log file.
        :fp16: If the job requires 16-bit capabilities.
        :gpus: The number of GPUs required for the job.

        """
        job = {}
        job['path'] = path
        job['work_dir'] = work_dir
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


    def run_jobs(self, logdir, cmds=[], add_fp16=False, host2cmd_adds={}, remap={}):
        gpus_available = self.get_total_available()

        while self.queue.qsize() > 0:
            workers = []
            print('Total jobs left: {0}.'.format(self.queue.qsize()))
            print('Calculating priority list...')
            priority_list = self.get_gpu_priority_list()
            print('{0}: Starting jobs...'.format(datetime.datetime.now()))
            for i in range(min(gpus_available, self.queue.qsize(), len(priority_list))):
                host, device_id, fp16 = priority_list[i]
                if (host, device_id) in remap:
                    device_id = remap[(host, device_id)]
                job = self.queue.get()
                if add_fp16 and fp16:
                    job['cmd'] += ' --fp16'

                if host in host2cmd_adds:
                    job['cmd'] += host2cmd_adds[host]

                if job['fp16'] and not fp16:
                    self.queue.put(job)
                    continue
                workers.append(GPUWorker(self, self.config_folder, logdir, host, self.host2config[host], device_id, job, i, cmds))

            for worker in workers:
                time.sleep(5)
                worker.start()

            if self.queue.qsize() > 0:
                for i in range(5):
                    time.sleep(60 + np.random.randint(1, 7)) # wait for 5 minutes + some random amount of time
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


