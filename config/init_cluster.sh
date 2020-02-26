#!/bin/bash
#
#SBATCH --job-name=sparse_momentum_imagenet_0.2
#SBATCH --account=cse
#SBATCH --partition=cse-int-gpu
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:4

## Memory per node
#SBATCH --mem=64G
## Specify the working directory for this job
#SBATCH --chdir=/usr/lusers/dettmers/git/sparse_learning/imagenet/partially_dense/
#SBATCH --output=/usr/lusers/dettmers/logs/sparse_momentum_0.2.out
#SBATCH -e /usr/lusers/dettmers/logs/sparse_momentum_0.2.err
