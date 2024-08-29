import os
import json
import torch
import argparse
import uuid; uuid.uuid4().hex.upper()


parser = argparse.ArgumentParser('some parser')
parser.add_argument('--comment', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=87878766)
parser.add_argument('--checkpoint_dir', type=str, default='')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

past_i = 0
values = []
if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint_50.pt')):
    print('loading checkpoint')
    values, past_i = json.load(open(os.path.join(args.checkpoint_dir, 'checkpoint_50.pt')))

batch_size = 128
h = 1024

b = torch.randn(batch_size, h, h, device='cuda')
w = torch.randn(h, h, device='cuda')

#checkpoint_val = str(uuid.uuid4().hex.upper())
out = torch.matmul(b, w)
for i in range(50000):
    if i % 100 == 0:
        print(f'Step: {i}')
    out = torch.matmul(out, w)

    #if past_i > i: continue
    #val = torch.rand(10, 10).mean().item()
    #values.append(val)
    #print(f'Step: {i}')
    #print(f"Metric value: {val}")
    #if i == 50:
    #    if args.checkpoint_dir is not None:
    #        print('saving checkpoint')
    #        if not os.path.exists(args.checkpoint_dir):
    #            os.makedirs(args.checkpoint_dir)
    #        with open(os.path.join(args.checkpoint_dir, 'checkpoint_50.pt'), 'w') as f:
    #            f.write(json.dumps([values, i]))


