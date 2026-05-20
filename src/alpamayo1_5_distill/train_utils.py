# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared training utilities used by both single-GPU and pipeline training scripts."""

from typing import Any

import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig

from omegaconf import DictConfig


def build_student_config(cfg: DictConfig) -> Alpamayo1_5_DistilledConfig:
    """Build student config from Hydra config."""
    diffusion_cfg = {
        "_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching",
        "num_inference_steps": cfg.student.get("diffusion_steps", 4),
        "int_method": "euler",
    }
    action_in_proj_cfg = {
        "_target_": "alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2",
        "num_enc_layers": 4,
        "hidden_size": 1024,
        "num_fourier_feats": 20,
        "max_freq": 100.0,
    }
    return Alpamayo1_5_DistilledConfig(
        vlm_name_or_path=cfg.student.vlm_name_or_path,
        diffusion_cfg=diffusion_cfg,
        action_in_proj_cfg=action_in_proj_cfg,
        teacher_model_name=cfg.teacher.model_name,
        distill_loss_weights={
            "vlm_logits": cfg.loss.vlm_logits_weight,
            "expert_hidden": cfg.loss.expert_hidden_weight,
            "trajectory_l2": cfg.loss.trajectory_l2_weight,
        },
        attn_implementation=cfg.student.get("attn_implementation", "flash_attention_2"),
    )


def build_dataloader(cfg: DictConfig):
    """Yield clip data dicts for training."""
    clip_ids = cfg.data.get("clip_ids")
    if not clip_ids:
        clip_ids = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]
    for clip_id in clip_ids:
        data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
        yield data


def prepare_model_inputs(data: dict, processor, device: str) -> dict:
    """Tokenize image/text inputs and build the model_inputs dict."""
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def repeat_visual_inputs(
    tokenized_data: dict[str, Any], batch_size: int, num_traj_samples: int,
) -> dict[str, torch.Tensor]:
    """Repeat visual tensors to match num_return_sequences batch expansion."""
    visual_kwargs: dict[str, Any] = {}
    for key in ("pixel_values", "image_grid_thw", "image_grid_thw_batch"):
        if key in tokenized_data:
            val = tokenized_data[key]
            if isinstance(val, torch.Tensor) and val.shape[0] == batch_size:
                val = val.repeat_interleave(num_traj_samples, dim=0)
            visual_kwargs[key] = val
    return visual_kwargs


def shallow_copy_data(data: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy data dict, isolating only tokenized_data for safe mutation."""
    data = dict(data)
    if "tokenized_data" in data:
        data["tokenized_data"] = dict(data["tokenized_data"])
    return data
