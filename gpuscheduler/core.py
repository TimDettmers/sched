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

from queue import Queue
from bs4 import BeautifulSoup
from os.path import join
from aenum import Enum

MEM_THRESHOLD_AVAILABLE = 1100
UTILIZARTION_THERESHOLD_AVAILABLE = 30

class HostState(Enum):
    unknown = 0
    available = 1

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

gpu_name2performance_class = {}
gpu_name2performance_class['TITAN V'] = 10
gpu_name2performance_class['GeForce RTX 2080 Ti'] = 8
gpu_name2performance_class['TITAN X (Pascal)'] = 5
gpu_name2performance_class['TITAN Xp'] = 4
gpu_name2performance_class['GeForce GTX TITAN X'] = 3
gpu_name2performance_class['GeForce GTX 1080 Ti'] = 3


class GPUWorker(threading.Thread):
    def __init__(self, config_folder, logdir, host_name, host_config, device_id, job, idx):
        super(GPUWorker, self).__init__()
        self.isDaemon = False
        self.job = job
        self.idx = idx
        self.cfg = host_config
        self.device_id = device_id
        self.host_name = host_name
        self.logdir = logdir
        self.init_dir = config_folder

    def construct_init_file(self):
        if not os.path.exists('/tmp/gpuscheduler'): os.mkdir('/tmp/gpuscheduler')
        if not os.path.exists(join('/tmp/gpuscheduler', str(self.idx))): os.mkdir(join('/tmp/gpuscheduler', str(self.idx)))

        print('Constructing init file...')
        init_path = join('/tmp/gpuscheduler/', str(self.idx), 'init_{0}.sh'.format(self.idx))
        shutil.copyfile(join(self.init_dir, 'init.sh'), init_path)
        with open(init_path, 'a') as f:
            work_dir = join(self.cfg['GIT_HOME'], self.job['work_dir'])
            f.write('cd {0}\n'.format(work_dir))
            f.write('CUDA_VISIBLE_DEVICES={0} {1}\n'.format(self.device_id, self.job['cmd']))
        cmd = 'scp {0} {1}:~/'.format(init_path, self.host_name)
        print('Transfering init file')
        execute(cmd)




    def run(self):
        print('Started worker {0} on Host {1} for GPU {2}'.format(self.idx, self.host_name, self.device_id))
        self.construct_init_file()
        print('Executing on {0}:{1}...'.format(self.host_name, self.device_id))
        out, err = cmd_over_ssh(self.host_name, 'init_{0}.sh'.format(self.idx))
        if len(err) > 0:
            print('ERROR: {0}'.format(err))
        else:
            group, subgroup, name = self.job['group'], self.job['subgroup'], self.job['name']
            if not os.path.exists(self.logdir): os.mkdir(self.logdir)
            if not os.path.exists(join(self.logdir, group)): os.mkdir(join(self.logdir, group))
            if not os.path.exists(join(self.logdir, group, subgroup)): os.mkdir(join(self.logdir, group, subgroup))
            if not os.path.exists(join(self.logdir, group, subgroup, name)): os.mkdir(join(self.logdir, group, subgroup, name))
            file_path = join(self.logdir, group, subgroup, name, str(uuid.uuid4()) + '.log')
            with open(file_path, 'w') as f:
                f.write(out)




class Scheduler(object):

    """Execute tasks over ssh in the background."""

    def __init__(self, config_folder):
        self.queue = Queue()
        self.host_data = None
        self.last_polled = datetime.datetime.now() - datetime.timedelta(hours=1)

        self.host2config = self.init_hosts(config_folder)
        self.config_folder = config_folder

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
                host2config[name]['priority'] = int(row['priority'])
                host2config[name]['max_usage'] = int(row['max gpu usage'])
                host2config[name]['min_free'] = int(row['min free gpus'])
                host2config[name]['status'] = HostState.unknown
        return host2config


    def poll_gpu_status(self):
        """Polls GPU status (if a GPU is used etc)."""
        print('Polling a total of {0} hosts...'.format(len(self.host2config)))
        for i, host in enumerate(self.host2config):
            if i > 0 and i % 3 == 0: print('{0}/{1}'.format(i, len(self.host2config)))
            strCMD = 'ssh {0} "nvidia-smi -q -x"'.format(host)
            out, err = execute_and_return(strCMD)
            if err != '':
                self.host2config[host]['status'] = HostState.unknown
                self.host2config[host]['num_available'] = 0
                continue
            gpus, num_available = self.parse_nvidia_smi(out)
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
            num_available = config['num_available']
            max_usage = config['max_usage']
            min_free = config['min_free']

            total_available += min(max_usage, num_available)
            total_fp16 = 0
            for gpu in config['gpus']:
                if gpu['status'] == GPUStatus.available and gpu['fp16']:
                    total_fp16 += 1
            avail_fp16 = min(min(max_usage, num_available), total_fp16)
            total_available_fp16 += avail_fp16
            print('Host: {0}. Available 16-bit: {2}. Total available {1}.'.format(host, min(max_usage, num_available), avail_fp16))


        print('A total of {0} GPUs are available on {1} total hosts.'.format(total_available, len(self.host2config)))
        print('Of these GPUs a total of {0} have 16-bit capability (tensor cores).'.format(total_available_fp16))
        return total_available


    def parse_nvidia_smi(self, text):
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
            gpu['status'] = self.determine_gpu_status(gpu)
            if gpu['status'] == GPUStatus.available: num_available += 1
            if gpu['name'] in gpu_name2performance_class:
                gpu['performance'] = gpu_name2performance_class[gpu['name']]
            else:
                print(gpu['name'], 'NOT IN PERFORMANCE CLASS. PLEASE ADD NOW.')
                gpu['performance'] = 5
            gpus.append(gpu)
        return gpus, num_available


    def determine_gpu_status(self, gpu):
        """Determines is a GPU is available for the queue."""
        if gpu['used_mem'] < MEM_THRESHOLD_AVAILABLE and gpu['utilization'] < UTILIZARTION_THERESHOLD_AVAILABLE:
            return GPUStatus.available
        else:
            return GPUStatus.busy

    def add_job(self, group, subgroup, name, work_dir, cmd, fp16=False, gpus=1):
        """Adds a job to execute.

        :group: The main folder.
        :subgroup: The sub-folder.
        :name: The log files name.
        :fp16: If the job requires 16-bit capabilities.
        :gpus: The number of GPUs required for the job.

        """
        job = {}
        job['group'] = group
        job['subgroup'] = subgroup
        job['name'] = name
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
            gpus = self.host2config[name]['gpus']
            num_available = self.host2config[name]['num_available']
            ids = []
            performance = []
            for gpu in gpus:
                if gpu['status'] != GPUStatus.available: continue
                ids.append(gpu['device_id'])
                performance.append(gpu['performance'])
            if len(ids) == 0: continue
            performance, ids = zip(*sorted(zip(performance, ids), reverse=True))
            for device_id in ids[:num_available]:
                priority_list.append((name, device_id))
        return priority_list


    def run_jobs(self, logdir):
        gpus_available = self.get_total_available()
        priority_list = self.get_gpu_priority_list()

        while self.queue.qsize() > 0:
            workers = []
            for i in range(min(gpus_available, self.queue.qsize())):
                host, device_id = priority_list[i]
                job = self.queue.get()
                workers.append(GPUWorker(self.config_folder, logdir, host, self.host2config[host], device_id, job, i))

            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            gpus_available = self.get_total_available()

