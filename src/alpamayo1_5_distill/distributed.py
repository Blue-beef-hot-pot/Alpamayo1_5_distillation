# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed training utilities for pipeline parallelism.

Provides DDP setup, process group management, and StudentWithLoss wrapper
that combines the student model and distillation loss for DDP wrapping.
"""

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from alpamayo1_5_distill.distill_loss import DistillationLoss
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.student_forward import student_forward

logger = logging.getLogger(__name__)


class StudentWithLoss(torch.nn.Module):
    """Wrapper combining student model and distillation loss for DDP.

    The forward method runs student_forward + distill_loss and returns the
    loss dict. This allows DDP to track the full computation graph for
    gradient synchronization.
    """

    def __init__(
        self,
        student: Alpamayo1_5_Distilled,
        distill_loss: DistillationLoss,
        num_traj_samples: int = 6,
        collect_vlm_logits: bool = True,
        collect_expert_hiddens: bool = True,
        collect_vlm_hiddens: bool = True,
    ) -> None:
        super().__init__()
        self.student = student
        self.distill_loss = distill_loss
        self.num_traj_samples = num_traj_samples
        self.collect_vlm_logits = collect_vlm_logits
        self.collect_expert_hiddens = collect_expert_hiddens
        self.collect_vlm_hiddens = collect_vlm_hiddens

    def forward(
        self,
        model_inputs: dict[str, Any],
        teacher_sequences: torch.Tensor,
        teacher_vlm_logits: torch.Tensor | None,
        teacher_vlm_hiddens: list[torch.Tensor],
        teacher_expert_hiddens: list[list[torch.Tensor]],
        teacher_traj: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Run student forward + distillation loss.

        Args:
            model_inputs: Tokenized inputs and ego history data.
            teacher_sequences: Teacher-generated token IDs for teacher-forcing.
            teacher_vlm_logits: Teacher VLM logits (may be None).
            teacher_vlm_hiddens: Per-layer teacher VLM hidden states.
            teacher_expert_hiddens: Per-step per-layer teacher Expert hidden states.
            teacher_traj: Teacher sampled trajectory (may be None).

        Returns:
            Dict of loss values including "total".
        """
        student_out = student_forward(
            self.student,
            model_inputs,
            teacher_sequences=teacher_sequences,
            num_traj_samples=self.num_traj_samples,
            collect_vlm_logits=self.collect_vlm_logits,
            collect_expert_hiddens=self.collect_expert_hiddens,
            collect_vlm_hiddens=self.collect_vlm_hiddens,
        )

        losses = self.distill_loss(
            student_vlm_logits=student_out.vlm_logits,
            teacher_vlm_logits=teacher_vlm_logits,
            student_expert_hiddens=student_out.expert_hiddens_all_steps,
            teacher_expert_hiddens=teacher_expert_hiddens,
            student_vlm_hiddens=student_out.vlm_hiddens,
            teacher_vlm_hiddens=teacher_vlm_hiddens,
            student_traj=student_out.sampled_traj,
            teacher_traj=teacher_traj,
        )

        return losses


def setup_distributed(
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> tuple[int, int]:
    """Initialize NCCL process group and set CUDA device."""
    if rank is not None:
        os.environ["RANK"] = str(rank)
    if world_size is not None:
        os.environ["WORLD_SIZE"] = str(world_size)
    if local_rank is not None:
        os.environ["LOCAL_RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    logger.info("Initialized distributed: rank %d / %d", rank, world_size)
    return rank, world_size


def create_student_group(student_ranks: list[int]) -> dist.ProcessGroup | None:
    """Create a process group for student DDP ranks."""
    rank = dist.get_rank()
    group = dist.new_group(ranks=student_ranks)
    return group if rank in student_ranks else None


def freeze_student_visual_tower(student: Alpamayo1_5_Distilled) -> None:
    visual = student.vlm.model.visual
    visual.eval()
    for param in visual.parameters():
        param.requires_grad_(False)


def wrap_student_ddp(
    student: Alpamayo1_5_Distilled,
    distill_loss: DistillationLoss,
    student_group: dist.ProcessGroup,
    num_traj_samples: int = 6,
    collect_vlm_logits: bool = True,
    collect_expert_hiddens: bool = True,
    collect_vlm_hiddens: bool = True,
) -> DDP:
    """Wrap student + distill_loss in DDP for gradient synchronization.

    Args:
        student: The student model.
        distill_loss: The distillation loss module.
        student_group: Process group containing student ranks only.
        num_traj_samples: Number of trajectory samples.
        collect_vlm_logits: Whether VLM logits KD is active.
        collect_expert_hiddens: Whether Expert hidden KD is active.
        collect_vlm_hiddens: Whether VLM hidden KD is active.

    Returns:
        DDP-wrapped StudentWithLoss.
    """
    freeze_student_visual_tower(student)
    model = StudentWithLoss(
        student,
        distill_loss,
        num_traj_samples=num_traj_samples,
        collect_vlm_logits=collect_vlm_logits,
        collect_expert_hiddens=collect_expert_hiddens,
        collect_vlm_hiddens=collect_vlm_hiddens,
    )
    ddp_model = DDP(
        model,
        process_group=student_group,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    return ddp_model
