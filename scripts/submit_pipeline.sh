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
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=log/result/distill_pipeline_%j.out
#SBATCH --error=log/err/%j.err

set -euo pipefail

# Activate virtual environment
source a1_5_venv/bin/activate

# Spawned workers use these rendezvous settings.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# Fallback: use flash-attn-free install if flash-attn is unavailable
uv sync --active --no-install-package flash-attn 2>/dev/null || uv sync --active

python scripts/train_distill_pipeline.py \
    --config-name=distill_pipeline
