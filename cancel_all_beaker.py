import beaker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--only_queued', action='store_true')
parser.add_argument('--all', action='store_true')
parser.add_argument('--list', action='store_true')
parser.add_argument('--gpu_count', type=int, default=None)
parser.add_argument('--cpu_count', type=int, default=None)

args = parser.parse_args()

BSESS = beaker.Beaker.from_env()
bclient = beaker.services.JobClient(BSESS)

for job in bclient.list(author='timd'):
    if args.list:
        print(job.id, job.kind, job.requests.gpu_count if job.requests is not None else '', job.requests.cpu_count if job.requests is not None else '')
    elif args.all:
        print(f'cancelling {job.id} ...')
        bclient.stop(job)
    else:
        if args.only_queued:
            if job.status.started is None:
                print(f'cancelling {job.id} ...')
                bclient.stop(job.id)
                print(f'cancelling {job.id} ...')
                bclient.stop(job.id)
        else:
            if job.kind=='execution':
                if args.gpu_count is not None:
                    if job.requests.gpu_count == args.gpu_count:
                        print(f'cancelling {job.id} with {args.gpu_count} gpus...')
                        bclient.stop(job.id)
                elif args.cpu_count is not None:
                    if job.requests.cpu_count == args.cpu_count:
                        print(f'cancelling {job.id} with {args.cpu_count} cpus...')
                        bclient.stop(job.id)
                else:
                    print(f'cancelling {job.id} ...')
                    bclient.stop(job.id)
