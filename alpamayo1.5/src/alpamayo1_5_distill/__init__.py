# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 1.5 Distillation package.

Provides the distilled student model (2B VLM + 4-step flow matching),
teacher loading utilities, and distillation loss functions.
"""

from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.teacher import load_teacher, teacher_forward
from alpamayo1_5_distill.distill_loss import DistillationLoss

__all__ = [
    "Alpamayo1_5_DistilledConfig",
    "Alpamayo1_5_Distilled",
    "load_teacher",
    "teacher_forward",
    "DistillationLoss",
]
