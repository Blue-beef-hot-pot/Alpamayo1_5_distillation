#!/bin/bash
#SBATCH --job-name=alpamayo_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100x
#SBATCH --output=log/result/test_result_%j.out
#SBATCH --error=log/err/%j.err

source a1_5_dist_venv/bin/activate
source export_env.sh

#单机多卡的话使用多少张卡就设置为几
export WORLD_SIZE=1

python src/alpamayo1_5/test_inference.py