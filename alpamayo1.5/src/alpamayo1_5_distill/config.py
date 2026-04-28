# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for Alpamayo 1.5 Distilled model."""

from typing import Any

from alpamayo1_5.config import Alpamayo1_5Config


class Alpamayo1_5_DistilledConfig(Alpamayo1_5Config):
    """Configuration for the Alpamayo 1.5 Distilled model.

    Key differences from the original Alpamayo 1.5:
    - VLM backbone: Qwen3-VL-2B (24 layers, hidden_size=1536)
    - Expert: derived from 2B VLM text_config (same architecture as VLM)
    - Flow Matching: 4 inference steps (vs 10 in original)
    """

    model_type = "alpamayo1_5_distilled"

    def __init__(
        self,
        vlm_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        diffusion_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        teacher_model_name: str = "nvidia/Alpamayo-1.5-10B",
        distill_loss_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if diffusion_cfg is None:
            diffusion_cfg = {
                "_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching",
                "num_inference_steps": 4,
                "int_method": "euler",
            }
        if action_in_proj_cfg is None:
            action_in_proj_cfg = {
                "_target_": "alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2",
                "num_enc_layers": 4,
                "hidden_size": 1024,
                "num_fourier_feats": 20,
                "max_freq": 100.0,
            }
        if distill_loss_weights is None:
            distill_loss_weights = {
                "vlm_logits": 1.0,
                "expert_hidden": 0.5,
                "trajectory_l2": 1.0,
            }

        self.teacher_model_name = teacher_model_name
        self.distill_loss_weights = distill_loss_weights

        super().__init__(
            vlm_name_or_path=vlm_name_or_path,
            diffusion_cfg=diffusion_cfg,
            action_in_proj_cfg=action_in_proj_cfg,
            **kwargs,
        )
