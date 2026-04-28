# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 1.5 Distillation Training Script.

Usage:
    python scripts/train_distill.py --config-name=distill

Or with overrides:
    python scripts/train_distill.py --config-name=distill training.batch_size=2 training.num_epochs=20
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure the src directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig
from alpamayo1_5_distill.distill_loss import DistillationLoss
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.teacher import load_teacher, teacher_forward

logger = logging.getLogger(__name__)


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
    """Build a simple dataloader that yields clip data.

    In a full implementation, this would iterate over all clips in the dataset.
    For now, it yields a single example clip for development/testing.
    """
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    logger.info(f"Loading data for clip_id: {clip_id}")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    yield data


@hydra.main(config_path="../configs", config_name="distill", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run distillation training."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load teacher
    device = "cuda"
    teacher = load_teacher(
        model_name=cfg.teacher.model_name,
        device=device,
        dtype=getattr(torch, cfg.teacher.dtype),
    )
    logger.info("Teacher model loaded: %s", cfg.teacher.model_name)

    # 2) Build student
    student_config = build_student_config(cfg)
    student = Alpamayo1_5_Distilled(student_config).to(device)
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(
        "Student model created: %s total params, %s trainable",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    processor = helper.get_processor(student.tokenizer)

    # 3) Build loss, optimizer, scheduler
    distill_loss = DistillationLoss(
        vlm_logits_weight=cfg.loss.vlm_logits_weight,
        expert_hidden_weight=cfg.loss.expert_hidden_weight,
        trajectory_l2_weight=cfg.loss.trajectory_l2_weight,
        temperature=cfg.loss.temperature,
    )

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=student.parameters())
    scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)

    # 4) Training loop
    global_step = 0
    best_loss = float("inf")

    for epoch in range(cfg.training.num_epochs):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(build_dataloader(cfg)):
            # Prepare inputs
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
            model_inputs = helper.to_device(model_inputs, device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = teacher_forward(
                    teacher,
                    model_inputs,
                    top_p=cfg.teacher.top_p,
                    temperature=cfg.teacher.temperature,
                    num_traj_samples=cfg.teacher.num_traj_samples,
                    max_generation_length=cfg.teacher.max_generation_length,
                    collect_expert_hiddens=cfg.teacher.collect_expert_hiddens,
                )

            # Student forward (with grad)
            # TODO: implement student_forward_with_hiddens analogous to teacher_forward
            # For now, compute trajectory loss only
            with torch.autocast("cuda", dtype=getattr(torch, cfg.training.mixed_precision)):
                student_pred_xyz, student_pred_rot = student.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=cfg.teacher.top_p,
                    temperature=cfg.teacher.temperature,
                    num_traj_samples=cfg.teacher.num_traj_samples,
                    max_generation_length=cfg.teacher.max_generation_length,
                )

                losses = distill_loss(
                    student_traj=student_pred_xyz,
                    teacher_traj=teacher_out.pred_xyz,
                )

            loss = losses["total"]
            loss_scaled = loss / cfg.training.gradient_accumulation_steps
            loss_scaled.backward()

            if (global_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                if cfg.training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), cfg.training.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if batch_idx % 10 == 0:
                loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
                logger.info(
                    "Epoch %d, Batch %d, Step %d — %s",
                    epoch, batch_idx, global_step, loss_str,
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d complete — avg loss: %.4f", epoch, avg_loss)

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
            ckpt_path = output_dir / f"epoch_{epoch + 1}"
            student.save_pretrained(str(ckpt_path))
            student.tokenizer.save_pretrained(str(ckpt_path))
            logger.info("Checkpoint saved: %s", ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best"
            student.save_pretrained(str(best_path))
            student.tokenizer.save_pretrained(str(best_path))
            logger.info("New best model saved: %s", best_path)

    # Save final model
    final_path = output_dir / "final"
    student.save_pretrained(str(final_path))
    student.tokenizer.save_pretrained(str(final_path))
    logger.info("Final model saved: %s", final_path)


if __name__ == "__main__":
    main()
