# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation losses for Alpamayo 1.5 knowledge distillation.

Three complementary loss signals:
1. VLM Logits KD — KL divergence between teacher and student VLM output distributions
2. Expert Hidden KD — MSE between teacher and student Expert hidden states
   (with layer mapping when layer counts differ, e.g., 36 -> 24,
   and learnable projection when hidden dimensions differ, e.g., 4096 -> 1536)
3. Trajectory L2 — MSE between teacher and student predicted trajectories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


def _compute_layer_mapping(
    n_teacher_layers: int, n_student_layers: int,
) -> list[int]:
    """Compute which teacher layers to align with each student layer.

    Uniformly samples n_student_layers indices from [0, n_teacher_layers).

    Args:
        n_teacher_layers: Number of layers in the teacher Expert.
        n_student_layers: Number of layers in the student Expert.

    Returns:
        List of teacher layer indices, one per student layer.
    """
    if n_teacher_layers == n_student_layers:
        return list(range(n_teacher_layers))
    return [
        round(i * (n_teacher_layers - 1) / (n_student_layers - 1))
        for i in range(n_student_layers)
    ]


def vlm_logits_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence loss between student and teacher VLM logits.

    Args:
        student_logits: Student VLM logits [B, L, V].
        teacher_logits: Teacher VLM logits [B, L, V].
        temperature: Softmax temperature (higher = softer distributions).
        mask: Optional mask [B, L] where 1 = valid, 0 = padding.

    Returns:
        Scalar KL divergence loss.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # [B, L]

    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp(min=1)
    return kl.mean()


def expert_hidden_kd_loss(
    student_hiddens: list[torch.Tensor],
    teacher_hiddens: list[torch.Tensor],
    layer_mapping: list[int] | None = None,
    hidden_proj: nn.Module | None = None,
) -> torch.Tensor:
    """MSE loss between student and teacher Expert hidden states.

    When the teacher has more layers than the student, a layer mapping
    selects which teacher layers to compare against. When hidden dimensions
    differ, a learnable linear projection (hidden_proj) is used instead
    of truncation.

    Args:
        student_hiddens: Per-layer hidden states from student, each [B, T, D_s].
        teacher_hiddens: Per-layer hidden states from teacher, each [B, T, D_t].
        layer_mapping: Optional list mapping student layer index to teacher
            layer index. Computed automatically if None.
        hidden_proj: Optional nn.Linear(D_t, D_s) to project teacher hiddens
            into the student's hidden space before computing MSE.

    Returns:
        Scalar MSE loss averaged over matched layers.
    """
    if not student_hiddens or not teacher_hiddens:
        device = student_hiddens[0].device if student_hiddens else "cpu"
        return torch.zeros((), device=device)

    n_student = len(student_hiddens)
    n_teacher = len(teacher_hiddens)

    if layer_mapping is None:
        layer_mapping = _compute_layer_mapping(n_teacher, n_student)

    assert len(layer_mapping) == n_student, (
        f"layer_mapping length ({len(layer_mapping)}) must match "
        f"student layers ({n_student})"
    )

    total_loss = torch.zeros((), device=student_hiddens[0].device)
    for s_idx, t_idx in enumerate(layer_mapping):
        s_hidden = student_hiddens[s_idx]
        t_hidden = teacher_hiddens[t_idx]

        # Project teacher hidden to student dimension if needed
        if s_hidden.shape[-1] != t_hidden.shape[-1]:
            if hidden_proj is not None:
                t_hidden = hidden_proj(t_hidden)
            else:
                # Fallback to truncation when no projection is provided
                min_dim = min(s_hidden.shape[-1], t_hidden.shape[-1])
                s_hidden = s_hidden[..., :min_dim]
                t_hidden = t_hidden[..., :min_dim]

        total_loss = total_loss + F.mse_loss(s_hidden, t_hidden.detach())

    return total_loss / n_student


def trajectory_l2_loss(
    student_traj: torch.Tensor,
    teacher_traj: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between student and teacher predicted trajectories.

    Args:
        student_traj: Student trajectory predictions [B, ..., T, C].
        teacher_traj: Teacher trajectory predictions [B, ..., T, C].

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(student_traj, teacher_traj.detach())


class DistillationLoss(nn.Module):
    """Combined distillation loss with configurable weights.

    Extends nn.Module to hold learnable parameters (hidden projection)
    that must be included in the optimizer and saved with checkpoints.

    Args:
        vlm_logits_weight: Weight for VLM logits KD loss.
        expert_hidden_weight: Weight for Expert hidden state KD loss.
        trajectory_l2_weight: Weight for trajectory L2 loss.
        temperature: Softmax temperature for VLM logits KD.
        teacher_hidden_dim: Teacher Expert hidden dimension (e.g., 4096).
        student_hidden_dim: Student Expert hidden dimension (e.g., 1536).
    """

    def __init__(
        self,
        vlm_logits_weight: float = 1.0,
        expert_hidden_weight: float = 0.5,
        trajectory_l2_weight: float = 1.0,
        temperature: float = 2.0,
        teacher_hidden_dim: int | None = None,
        student_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.vlm_logits_weight = vlm_logits_weight
        self.expert_hidden_weight = expert_hidden_weight
        self.trajectory_l2_weight = trajectory_l2_weight
        self.temperature = temperature

        # Learnable projection for aligning teacher expert hiddens to student space
        if (
            teacher_hidden_dim is not None
            and student_hidden_dim is not None
            and teacher_hidden_dim != student_hidden_dim
        ):
            self.hidden_proj = nn.Linear(teacher_hidden_dim, student_hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.hidden_proj.weight)
        else:
            self.hidden_proj = None

    def forward(
        self,
        student_vlm_logits: torch.Tensor | None = None,
        teacher_vlm_logits: torch.Tensor | None = None,
        student_expert_hiddens: list[torch.Tensor] | None = None,
        teacher_expert_hiddens: list[torch.Tensor] | None = None,
        student_traj: torch.Tensor | None = None,
        teacher_traj: torch.Tensor | None = None,
        vlm_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all distillation losses.

        Args:
            student_vlm_logits: Student VLM logits [B, L, V].
            teacher_vlm_logits: Teacher VLM logits [B, L, V].
            student_expert_hiddens: Per-layer student Expert hiddens.
            teacher_expert_hiddens: Per-layer teacher Expert hiddens.
            student_traj: Student trajectory [B, ..., T, C].
            teacher_traj: Teacher trajectory [B, ..., T, C].
            vlm_mask: Mask for VLM logits [B, L].

        Returns:
            Dict with individual losses and total weighted loss.
        """
        # Determine device from any available input tensor
        device = self._detect_device(
            student_vlm_logits, teacher_vlm_logits, student_traj, teacher_traj,
            student_expert_hiddens, teacher_expert_hiddens,
        )

        losses = {}

        if (
            self.vlm_logits_weight > 0
            and student_vlm_logits is not None
            and teacher_vlm_logits is not None
        ):
            # VLM logits must share the same vocabulary dimension
            if student_vlm_logits.shape[-1] != teacher_vlm_logits.shape[-1]:
                losses["vlm_logits_kd"] = torch.zeros((), device=device)
            else:
                losses["vlm_logits_kd"] = vlm_logits_kd_loss(
                    student_vlm_logits, teacher_vlm_logits,
                    temperature=self.temperature, mask=vlm_mask,
                )
        else:
            losses["vlm_logits_kd"] = torch.zeros((), device=device)

        if (
            self.expert_hidden_weight > 0
            and student_expert_hiddens
            and teacher_expert_hiddens
        ):
            losses["expert_hidden_kd"] = expert_hidden_kd_loss(
                student_expert_hiddens, teacher_expert_hiddens,
                hidden_proj=self.hidden_proj,
            )
        else:
            losses["expert_hidden_kd"] = torch.zeros((), device=device)

        if (
            self.trajectory_l2_weight > 0
            and student_traj is not None
            and teacher_traj is not None
        ):
            losses["trajectory_l2"] = trajectory_l2_loss(
                student_traj, teacher_traj,
            )
        else:
            losses["trajectory_l2"] = torch.zeros((), device=device)

        losses["total"] = (
            self.vlm_logits_weight * losses["vlm_logits_kd"]
            + self.expert_hidden_weight * losses["expert_hidden_kd"]
            + self.trajectory_l2_weight * losses["trajectory_l2"]
        )

        return losses

    @staticmethod
    def _detect_device(*tensors_or_lists) -> torch.device:
        """Detect device from any available tensor or list of tensors."""
        for item in tensors_or_lists:
            if isinstance(item, torch.Tensor):
                return item.device
            if isinstance(item, list) and item and isinstance(item[0], torch.Tensor):
                return item[0].device
        return torch.device("cpu")
