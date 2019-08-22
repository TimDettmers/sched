import itertools
import gpuscheduler
from sparselearning import funcs

s = gpuscheduler.Scheduler('/home/tim/data/git/sched/config/')

s.update_host_config('home', mem_threshold=2500, util_threshold=30)
#s.update_host_config('office', mem_threshold=2500, util_threshold=25)
#s.update_host_config('ari', mem_threshold=2500, util_threshold=25)

cmd_mnist = 'OMP_NUM_THREADS=1 python main.py --model {4} --density {5} --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose{6}'
cmd_cifar = 'OMP_NUM_THREADS=1 python main.py --model {4} --decay_frequency 30000 --batch-size 128 --data cifar --epochs 250 --density {5} --seed {3} --prune {0} --growth {1} --redistribution {2} --verbose{6} --max-threads 2'

num_seeds = 6
cmd = cmd_cifar
seed_offset = 0

densities = [0.25]
jobs = []
for seed in range(num_seeds):
    for density in densities:
        model = 'vgg-d'
        fp16 = True
        prune = 'magnitude'
        growth = 'momentum'
        redist = 'momentum'
        jobs.append(['dense_equivalent/{0}/{1}/fp16'.format(model, density), 'sparse_learning/mnist_cifar/', cmd.format(prune, growth, redist, seed+seed_offset, model, density, ' --fp16' if fp16 else ''), fp16])
            #jobs.append(['fp16/only/O2/{0}/{1}'.format(model, 'fp16' if fp16 else 'fp32'), 'sparse_learning/mnist_cifar/', cmd.format(prune, growth, redist, seed+seed_offset, model, density, ' --fp16' if fp16 else ''), fp16])

print(jobs[0])
print(len(jobs))

for job in jobs:
    s.add_job(*job)

cmds = []
#cmds = ['git stash', 'git checkout 9bf460346ae133d5632066c4364e7d70437a1559'] # O1
#cmds = ['git stash', 'git checkout 85e6f84d7f5c2e92752f87994e1a71ffca4973d9'] # O2
#cmds = ['git stash', 'git checkout 24f59a80352c512106d0f3134fcf71b49ed6065e'] # O2 no float loss
#cmds = ['git stash', 'git checkout master', 'git pull']
cmds = ['git stash', 'git checkout master', 'git pull', 'git checkout fully_sparse_dynamic']
s.run_jobs('/home/tim/logs/', cmds=cmds, add_fp16=True)
