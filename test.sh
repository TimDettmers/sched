#!/bin/bash
#
#SBATCH --job-name=/checkpoint/timdettmers/sbatch_scripts/init_c39b4728c3e7ed300472fb7c8d25c3e7_0.sh
#SBATCH --partition=dev
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=8
#SBATCH --mem=256G
#SBATCH --constraint=volta
#
#SBATCH --open-mode=append
#SBATCH --chdir=/private/home/timdettmers/git/mesh/
#SBATCH --output=/private/home/timdettmers/logs/moe/tf/baseline/4k/c39b4728c3e7ed300472fb7c8d25c3e7_0.log
#SBATCH --error=/private/home/timdettmers/logs/moe/tf/baseline/4k/c39b4728c3e7ed300472fb7c8d25c3e7_0.err

export PATH=$PATH:/private/home/timdettmers/anaconda3/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/private/home/timdettmers/local/lib64
echo "cmd0"
export ld_library_path=$ld_library_path:/private/home/timdettmers/local/lib64
source /private/home/timdettmers/.bashrc
source activate tf
echo $LD_LIBRARY_PATH
echo "cmd1"
python mesh_tensorflow/transformer/main.py  --gin_file=mesh_tensorflow/transformer/gin/defaults.gin --gin_file=mesh_tensorflow/transformer/gin/layouts/8dp_gpu_minxu.gin --gin_file=mesh_tensorflow/transformer/gin/models/lm_moe_minxu.gin  --gin_file=mesh_tensorflow/transformer/gin/problems/t2t_lm1b_minxu.gin  --gin_param "run.mode='train'"   --model-dir /checkpoint/timdettmers/moe/tf/baseline/4k/620c361b2700ee26d77df41090411803 
