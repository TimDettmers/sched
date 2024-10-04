import beaker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--only_queued', action='store_true')
parser.add_argument('--all', action='store_true')

args = parser.parse_args()

BSESS = beaker.Beaker.from_env()
bclient = beaker.services.JobClient(BSESS)

for job in bclient.list(author='timd'):
    if args.only_queued:
        if job.status.started is None:
            print(f'cancelling {job.id} ...')
            bclient.stop(job.id)
    elif args.all:
            print(f'cancelling {job.id} ...')
            bclient.stop(job.id)
    else:
        if job.kind=='execution':
            print(f'cancelling {job.id} ...')
            bclient.stop(job.id)
