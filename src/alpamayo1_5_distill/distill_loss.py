# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation losses for Alpamayo 1.5 knowledge distillation.

Four complementary loss signals:
1. VLM Logits KD — KL divergence between teacher and student VLM output distributions
2. Expert Hidden KD — MSE between teacher and student Expert hidden states across
   all diffusion steps (with layer mapping 36->24, grouped learnable projections
   4096->1536, and margin ReLU to suppress weak teacher activations)
3. VLM Hidden KD — MSE between teacher and student VLM per-layer hidden states
   (with grouped projections and margin ReLU, same as Expert Hidden KD)
4. Trajectory L2 — MSE between teacher and student predicted trajectories
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


def _compute_step_mapping(
    n_teacher_steps: int, n_student_steps: int,
) -> list[int]:
    """Compute which teacher diffusion steps to align with each student step.

    Uniformly samples n_student_steps indices from [0, n_teacher_steps).
    E.g., 10 teacher steps -> 4 student steps maps to [0, 3, 6, 9].

    Args:
        n_teacher_steps: Number of diffusion steps in the teacher.
        n_student_steps: Number of diffusion steps in the student.

    Returns:
        List of teacher step indices, one per student step.
    """
    if n_teacher_steps == n_student_steps:
        return list(range(n_teacher_steps))
    if n_student_steps == 1:
        return [n_teacher_steps - 1]
    return [
        round(i * (n_teacher_steps - 1) / (n_student_steps - 1))
        for i in range(n_student_steps)
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


def _layer_group_idx(layer_idx: int, num_layers: int, num_groups: int) -> int:
    """Return which group (0..num_groups-1) a layer belongs to."""
    return min(layer_idx * num_groups // num_layers, num_groups - 1)


def hidden_kd_loss(
    student_hiddens: list[torch.Tensor],
    teacher_hiddens: list[torch.Tensor],
    layer_mapping: list[int] | None = None,
    hidden_projs: nn.ModuleList | None = None,
    margins: nn.ParameterList | None = None,
    num_groups: int = 3,
) -> torch.Tensor:
    """MSE loss between student and teacher hidden states (Expert or VLM).

    When the teacher has more layers than the student, a layer mapping
    selects which teacher layers to compare against. When hidden dimensions
    differ, grouped learnable linear projections are used (one per layer
    group: shallow/mid/deep) with margin ReLU to suppress weak teacher
    activations (Heo et al., ICCV 2019).

    Args:
        student_hiddens: Per-layer hidden states from student, each [B, T, D_s].
        teacher_hiddens: Per-layer hidden states from teacher, each [B, T, D_t].
        layer_mapping: Optional list mapping student layer index to teacher
            layer index. Computed automatically if None.
        hidden_projs: Optional nn.ModuleList of nn.Linear(D_t, D_s), one per
            projection group, to project teacher hiddens into student space.
        margins: Optional nn.ParameterList of margin scalars, one per group.
            Applied as F.relu(proj(t) + margin) to filter weak activations.
        num_groups: Number of projection groups (e.g., 3 for shallow/mid/deep).

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
            if hidden_projs is not None:
                g_idx = _layer_group_idx(s_idx, n_student, num_groups)
                t_hidden = hidden_projs[g_idx](t_hidden)
                # Margin ReLU: suppress weak teacher activations so student
                # doesn't waste capacity fitting noise (Heo et al., ICCV 2019)
                if margins is not None:
                    t_hidden = F.relu(t_hidden + margins[g_idx])
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

    Extends nn.Module to hold learnable parameters (grouped projections,
    margin scalars) that must be included in the optimizer and saved with
    checkpoints.

    Args:
        vlm_logits_weight: Weight for VLM logits KD loss.
        expert_hidden_weight: Weight for Expert hidden state KD loss
            (averaged across matched diffusion steps).
        vlm_hidden_weight: Weight for VLM hidden state KD loss.
        trajectory_l2_weight: Weight for trajectory L2 loss.
        temperature: Softmax temperature for VLM logits KD.
        teacher_hidden_dim: Teacher hidden dimension (e.g., 4096).
            Applies to both Expert and VLM (Expert is derived from VLM text_config).
        student_hidden_dim: Student hidden dimension (e.g., 1536).
        num_projection_groups: Number of grouped projections for hidden KD
            (e.g., 3 for shallow/mid/deep). Each group shares one
            nn.Linear projection and one margin parameter.
        margin_init: Initial value for margin ReLU (negative, e.g., -0.5).
            Larger negative values suppress more weak teacher activations.
    """

    def __init__(
        self,
        vlm_logits_weight: float = 1.0,
        expert_hidden_weight: float = 0.5,
        vlm_hidden_weight: float = 0.5,
        trajectory_l2_weight: float = 1.0,
        temperature: float = 2.0,
        teacher_hidden_dim: int | None = None,
        student_hidden_dim: int | None = None,
        num_projection_groups: int = 3,
        margin_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.vlm_logits_weight = vlm_logits_weight
        self.expert_hidden_weight = expert_hidden_weight
        self.vlm_hidden_weight = vlm_hidden_weight
        self.trajectory_l2_weight = trajectory_l2_weight
        self.temperature = temperature
        self.num_projection_groups = num_projection_groups

        need_projection = (
            teacher_hidden_dim is not None
            and student_hidden_dim is not None
            and teacher_hidden_dim != student_hidden_dim
        )

        # Grouped projections for Expert hidden KD (shallow/mid/deep)
        # with margin ReLU (Heo et al., ICCV 2019)
        if need_projection:
            self.expert_hidden_projs = nn.ModuleList([
                nn.Linear(teacher_hidden_dim, student_hidden_dim, bias=False)
                for _ in range(num_projection_groups)
            ])
            for proj in self.expert_hidden_projs:
                nn.init.xavier_uniform_(proj.weight)
            self.expert_margins = nn.ParameterList([
                nn.Parameter(torch.tensor(margin_init))
                for _ in range(num_projection_groups)
            ])

            # Grouped projections for VLM hidden KD (same dimensions as Expert)
            self.vlm_hidden_projs = nn.ModuleList([
                nn.Linear(teacher_hidden_dim, student_hidden_dim, bias=False)
                for _ in range(num_projection_groups)
            ])
            for proj in self.vlm_hidden_projs:
                nn.init.xavier_uniform_(proj.weight)
            self.vlm_margins = nn.ParameterList([
                nn.Parameter(torch.tensor(margin_init))
                for _ in range(num_projection_groups)
            ])
        else:
            self.expert_hidden_projs = None
            self.expert_margins = None
            self.vlm_hidden_projs = None
            self.vlm_margins = None

    def forward(
        self,
        student_vlm_logits: torch.Tensor | None = None,
        teacher_vlm_logits: torch.Tensor | None = None,
        student_expert_hiddens: list[list[torch.Tensor]] | None = None,
        teacher_expert_hiddens: list[list[torch.Tensor]] | None = None,
        student_vlm_hiddens: list[torch.Tensor] | None = None,
        teacher_vlm_hiddens: list[torch.Tensor] | None = None,
        student_traj: torch.Tensor | None = None,
        teacher_traj: torch.Tensor | None = None,
        vlm_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all distillation losses.

        Args:
            student_vlm_logits: Student VLM logits [B, L, V].
            teacher_vlm_logits: Teacher VLM logits [B, L, V].
            student_expert_hiddens: Per-step per-layer student Expert hiddens,
                outer list = diffusion steps, inner list = layers.
            teacher_expert_hiddens: Per-step per-layer teacher Expert hiddens,
                outer list = diffusion steps, inner list = layers.
            student_vlm_hiddens: Per-layer student VLM hiddens.
            teacher_vlm_hiddens: Per-layer teacher VLM hiddens.
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
            student_vlm_hiddens, teacher_vlm_hiddens,
        )

        losses = {}

        # VLM Logits KD
        if (
            self.vlm_logits_weight > 0
            and student_vlm_logits is not None
            and teacher_vlm_logits is not None
        ):
            if student_vlm_logits.shape[-1] != teacher_vlm_logits.shape[-1]:
                losses["vlm_logits_kd"] = torch.zeros((), device=device)
            else:
                losses["vlm_logits_kd"] = vlm_logits_kd_loss(
                    student_vlm_logits, teacher_vlm_logits,
                    temperature=self.temperature, mask=vlm_mask,
                )
        else:
            losses["vlm_logits_kd"] = torch.zeros((), device=device)

        # Expert Hidden KD (Full Diffusion Trajectory — all steps)
        if (
            self.expert_hidden_weight > 0
            and student_expert_hiddens
            and teacher_expert_hiddens
        ):
            n_teacher_steps = len(teacher_expert_hiddens)
            n_student_steps = len(student_expert_hiddens)
            step_mapping = _compute_step_mapping(n_teacher_steps, n_student_steps)

            expert_hidden_loss = torch.zeros((), device=device)
            for s_step, t_step in enumerate(step_mapping):
                expert_hidden_loss = expert_hidden_loss + hidden_kd_loss(
                    student_expert_hiddens[s_step],
                    teacher_expert_hiddens[t_step],
                    hidden_projs=self.expert_hidden_projs,
                    margins=self.expert_margins,
                    num_groups=self.num_projection_groups,
                )
            losses["expert_hidden_kd"] = expert_hidden_loss / len(step_mapping)
        else:
            losses["expert_hidden_kd"] = torch.zeros((), device=device)

        # VLM Hidden KD
        if (
            self.vlm_hidden_weight > 0
            and student_vlm_hiddens
            and teacher_vlm_hiddens
        ):
            losses["vlm_hidden_kd"] = hidden_kd_loss(
                student_vlm_hiddens, teacher_vlm_hiddens,
                hidden_projs=self.vlm_hidden_projs,
                margins=self.vlm_margins,
                num_groups=self.num_projection_groups,
            )
        else:
            losses["vlm_hidden_kd"] = torch.zeros((), device=device)

        # Trajectory L2
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
            + self.vlm_hidden_weight * losses["vlm_hidden_kd"]
            + self.trajectory_l2_weight * losses["trajectory_l2"]
        )

        return losses

    @staticmethod
    def _detect_device(*tensors_or_lists) -> torch.device:
        """Detect device from any available tensor or nested list of tensors."""
        for item in tensors_or_lists:
            if isinstance(item, torch.Tensor):
                return item.device
            if isinstance(item, list) and item:
                # Handle list[Tensor] or list[list[Tensor]]
                first = item[0]
                if isinstance(first, torch.Tensor):
                    return first.device
                if isinstance(first, list) and first and isinstance(first[0], torch.Tensor):
                    return first[0].device
        return torch.device("cpu")
