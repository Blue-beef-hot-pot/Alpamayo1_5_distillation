#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SLURM submission script for pipeline-parallel distillation training.
#
# Usage (SLURM):  sbatch scripts/submit_pipeline.sh
# Usage (local):  bash scripts/submit_pipeline.sh

#SBATCH --job-name=alpamayo_distill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=log/result/distill_pipeline_%j.out
#SBATCH --error=log/err/%j.err

set -euo pipefail

# Activate virtual environment
source a1_5_venv/bin/activate

# Set up distributed environment (required by torchrun)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WORLD_SIZE=${WORLD_SIZE:-4}

# Fallback: use flash-attn-free install if flash-attn is unavailable
uv sync --active --no-install-package flash-attn 2>/dev/null || uv sync --active

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    scripts/train_distill_pipeline.py \
    --config-name=distill_pipeline
