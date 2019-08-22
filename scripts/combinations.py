import itertools
import gpuscheduler
from sparselearning import funcs

combinations = list(itertools.product(['magnitude_variance']+list(funcs.prune_funcs.keys()), funcs.growth_funcs, ['variance'] + list(funcs.redistribution_funcs.keys())))

print(combinations)
print(len(combinations))

s = gpuscheduler.Scheduler('/home/tim/data/git/sched/config/')


s.update_host_config('home', mem_threshold=900, util_threshold=20)
s.update_host_config('office', mem_threshold=900, util_threshold=35)
#s.update_host_config('loki', mem_threshold=2000, util_threshold=35)
#s.update_host_config('mandar', mem_threshold=2000, util_threshold=35)
#s.update_host_config('eunsol', mem_threshold=2000, util_threshold=35)

num_seeds = 10
# promising
#global_magnitude/momentum/magnitude
#global_magnitude/momentum/none

jobs = []
#for i, (prune, growth, redist) in enumerate(combinations[80:]):
    #print(i, prune, growth, redist)
for i in range(1):
    prune = 'global_magnitude'
    growth = 'momentum_neuron'
    redist = 'none'
    for seed in range(num_seeds):
        jobs.append(['{0}/{1}/{2}'.format(prune,growth,redist), 'sparse_learning/mnist_cifar/', 'OMP_NUM_THREADS=1 python main.py --model lenet300-100 --epochs 100 --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose'.format(prune, growth, redist, seed)])

#print(jobs)
print(len(jobs))


for job in jobs:
    s.add_job(*job)
s.run_jobs('/home/tim/logs/')
