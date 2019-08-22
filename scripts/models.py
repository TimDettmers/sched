import itertools
import gpuscheduler
from sparselearning import funcs

s = gpuscheduler.Scheduler('/home/tim/data/git/sched/config/')

s.update_host_config('home', mem_threshold=1100, util_threshold=25)
s.update_host_config('office', mem_threshold=1100, util_threshold=10)
#s.update_host_config('loki', mem_threshold=2000, util_threshold=35)
#s.update_host_config('mandar', mem_threshold=2000, util_threshold=35)
#s.update_host_config('eunsol', mem_threshold=2000, util_threshold=35)

num_seeds = 6
# promising
#global_magnitude/momentum/momentum
#global_magnitude/momentum/magnitude
#global_magnitude/momentum/none
cmd_mnist = 'OMP_NUM_THREADS=1 python main.py --model lenet300-100 --epochs 100 --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose'
cmd_cifar = 'OMP_NUM_THREADS=1 python main.py --model {4} --decay_frequency 30000 --batch-size 128 --data cifar --epochs 250 --density {5} --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose {6}'


models_and_densities = [('vgg-d', 0.05, False), ('wrn-16-8', 0.05, False), ('alexnet-s', 0.10, False)]
jobs = []
for seed in range(num_seeds):
    for (model, density, fp16) in models_and_densities:
        prune = 'global_magnitude'
        growth = 'momentum'
        redist = 'momentum'
        jobs.append(['{3}/{0}/{1}/{2}'.format(prune,growth,redist, model), 'sparse_learning/mnist_cifar/', cmd_cifar.format(prune, growth, redist, seed, model, density, '--fp16' if fp16 else '')])

print(len(jobs))


for job in jobs:
    s.add_job(*job)

s.run_jobs('/home/tim/logs/')
