# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged Alpamayo 1.5 Distillation Training Script.

Two-stage training:
- Stage 1 (VLM): Distill VLM text decoder with VLM Logits KD + VLM Hidden KD
- Stage 2 (Expert): Distill Expert + trajectory with Expert Hidden KD + Traj L2

Usage:
    # Single GPU
    python scripts/train_distill_staged.py --config-name=distill_staged

    # Multi-GPU DDP (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train_distill_staged.py --config-name=distill_staged

    # With local cache
    torchrun --nproc_per_node=4 scripts/train_distill_staged.py --config-name=distill_staged data.cache_dir=./.cache/

    # Resume from stage checkpoint
    torchrun --nproc_per_node=4 scripts/train_distill_staged.py --config-name=distill_staged \\
        training.resume_from_checkpoint=outputs/distilled_staged/stage_vlm_final
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5_distill.checkpoint import load_training_state, save_training_checkpoint
from alpamayo1_5_distill.distill_loss import DistillationLoss
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.student_forward import student_forward
from alpamayo1_5_distill.teacher import load_teacher, teacher_forward
from alpamayo1_5_distill.train_utils import (
    build_dataloader,
    build_student_config,
    prepare_model_inputs,
    resolve_clip_samples,
)
from alpamayo1_5_distill.distributed import (
    setup_distributed,
    setup_stage_vlm,
    setup_stage_expert,
)

logger = logging.getLogger(__name__)


def patch_conv3d_for_a100(model) -> None:
    """Patch Qwen3-VL patch_embed Conv3D to avoid cuDNN CUDNN_STATUS_INTERNAL_ERROR.

    On some cuDNN/A100 combos, Conv3D in bfloat16 triggers cuDNN internal error.
    Since kernel_size == stride in patch_embed, each patch is projected independently,
    so Conv3D is equivalent to F.linear (matrix multiply via cuBLAS).
    """
    if not hasattr(model, 'vlm') or not hasattr(model.vlm, 'model'):
        return
    if not hasattr(model.vlm.model, 'visual'):
        return

    _pe = model.vlm.model.visual.patch_embed
    _proj = _pe.proj  # nn.Conv3d

    def _fwd(self, hidden_states):
        C = self.in_channels
        T = self.temporal_patch_size
        H = self.patch_size
        W = self.patch_size
        hidden_states = hidden_states.view(-1, C * T * H * W)
        # Get weight and bias on the same device as input
        weight_2d = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
        bias = self.proj.bias
        return torch.nn.functional.linear(hidden_states, weight_2d, bias)

    _pe.forward = _fwd.__get__(_pe, type(_pe))
    logger.info("Patched Conv3D -> F.linear for model")


def save_stage_progress(output_dir: Path, stage_name: str, epoch: int, global_step: int) -> None:
    """Save stage progress for resume support."""
    progress = {
        "current_stage": stage_name,
        "epoch": epoch,
        "global_step": global_step,
    }
    with open(output_dir / "stage_progress.json", "w") as f:
        json.dump(progress, f)


def load_stage_progress(output_dir: Path) -> dict | None:
    """Load stage progress for resume."""
    progress_file = output_dir / "stage_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return None


def train_stage(
    stage_name: str,
    stage_cfg: DictConfig,
    cfg: DictConfig,
    student: Alpamayo1_5_Distilled,
    teacher: Alpamayo1_5_Distilled,
    distill_loss: DistillationLoss,
    processor,
    device: str,
    output_dir: Path,
    rank: int,
    start_epoch: int = 0,
    global_step: int = 0,
) -> tuple[int, int]:
    """Train one stage.

    Args:
        stage_name: "vlm" or "expert"
        stage_cfg: Stage-specific config (loss, optimizer, scheduler, num_epochs)
        cfg: Full config
        student: Student model
        teacher: Teacher model
        distill_loss: Distillation loss module
        processor: Tokenizer/processor
        device: CUDA device
        output_dir: Output directory
        rank: Distributed rank
        start_epoch: Starting epoch (for resume)
        global_step: Starting global step (for resume)

    Returns:
        (last_epoch, global_step)
    """
    # Setup frozen/trainable modules for this stage
    if stage_name == "vlm":
        setup_stage_vlm(student, distill_loss)
        collect_vlm_logits = stage_cfg.loss.vlm_logits_weight > 0
        collect_expert_hiddens = False
        collect_vlm_hiddens = stage_cfg.loss.vlm_hidden_weight > 0
    else:
        setup_stage_expert(student, distill_loss)
        collect_vlm_logits = False
        collect_expert_hiddens = stage_cfg.loss.expert_hidden_weight > 0
        collect_vlm_hiddens = False

    # Update loss weights
    distill_loss.vlm_logits_weight = stage_cfg.loss.vlm_logits_weight
    distill_loss.expert_hidden_weight = stage_cfg.loss.expert_hidden_weight
    distill_loss.vlm_hidden_weight = stage_cfg.loss.vlm_hidden_weight
    distill_loss.trajectory_l2_weight = stage_cfg.loss.trajectory_l2_weight
    distill_loss.temperature = stage_cfg.loss.temperature

    # Build optimizer with only trainable parameters
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    trainable_params += [p for p in distill_loss.parameters() if p.requires_grad]
    optimizer = hydra.utils.instantiate(stage_cfg.optimizer, params=trainable_params)

    # Build scheduler
    num_samples = len(resolve_clip_samples(cfg, epoch=0))
    total_steps = (num_samples * stage_cfg.num_epochs) // cfg.training.gradient_accumulation_steps
    scheduler_cfg = OmegaConf.to_container(stage_cfg.scheduler, resolve=True)
    scheduler_cfg["T_max"] = total_steps
    scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)

    logger.info(
        "Stage %s: %d epochs, %d samples/epoch, %d optimizer steps, lr=%.2e",
        stage_name, stage_cfg.num_epochs, num_samples, total_steps,
        stage_cfg.optimizer.lr,
    )

    # Resume if applicable
    best_loss = float("inf")
    if start_epoch > 0:
        logger.info("Resuming stage %s from epoch %d", stage_name, start_epoch)

    optimizer.zero_grad()

    # Training loop
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, stage_cfg.num_epochs):
        last_epoch = epoch
        student.train()
        distill_loss.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(build_dataloader(cfg, epoch=epoch)):
            model_inputs = prepare_model_inputs(data, processor, device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = teacher_forward(
                    teacher,
                    model_inputs,
                    top_p=cfg.teacher.top_p,
                    temperature=cfg.teacher.temperature,
                    num_traj_samples=cfg.teacher.num_traj_samples,
                    max_generation_length=cfg.teacher.max_generation_length,
                    collect_expert_hiddens=collect_expert_hiddens,
                    collect_vlm_hiddens=collect_vlm_hiddens,
                )

            del data
            torch.cuda.empty_cache()

            # Student forward with teacher-forcing
            with torch.autocast("cuda", dtype=getattr(torch, cfg.training.mixed_precision)):
                student_out = student_forward(
                    student,
                    model_inputs,
                    teacher_sequences=teacher_out.sequences,
                    teacher=teacher,
                    num_traj_samples=cfg.teacher.num_traj_samples,
                    collect_vlm_logits=collect_vlm_logits,
                    collect_expert_hiddens=collect_expert_hiddens,
                    collect_vlm_hiddens=collect_vlm_hiddens,
                )

                losses = distill_loss(
                    student_vlm_logits=student_out.vlm_logits,
                    teacher_vlm_logits=teacher_out.vlm_logits,
                    student_expert_hiddens=student_out.expert_hiddens_all_steps,
                    teacher_expert_hiddens=teacher_out.expert_hiddens_all_steps,
                    student_vlm_hiddens=student_out.vlm_hiddens,
                    teacher_vlm_hiddens=teacher_out.vlm_hiddens,
                    student_traj=student_out.sampled_traj,
                    teacher_traj=teacher_out.sampled_traj,
                )

            loss = losses["total"]
            loss_scaled = loss / cfg.training.gradient_accumulation_steps
            loss_scaled.backward()

            if (global_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                if cfg.training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0 and batch_idx % 10 == 0:
                loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
                logger.info(
                    "Stage %s | Epoch %d | Batch %d | Step %d — %s",
                    stage_name, epoch, batch_idx, global_step, loss_str,
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        if rank == 0:
            logger.info("Stage %s | Epoch %d complete — avg loss: %.4f", stage_name, epoch, avg_loss)

            # Save checkpoint
            if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
                ckpt_dir = output_dir / f"stage_{stage_name}_epoch_{epoch + 1}"
                save_training_checkpoint(
                    ckpt_dir,
                    student=student,
                    distill_loss=distill_loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                )
                save_stage_progress(output_dir, stage_name, epoch, global_step)
                logger.info("Checkpoint saved: %s", ckpt_dir)

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_training_checkpoint(
                    output_dir / f"stage_{stage_name}_best",
                    student=student,
                    distill_loss=distill_loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                )

    # Save final stage checkpoint
    if rank == 0:
        save_training_checkpoint(
            output_dir / f"stage_{stage_name}_final",
            student=student,
            distill_loss=distill_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=last_epoch,
            global_step=global_step,
            best_loss=best_loss,
        )
        save_stage_progress(output_dir, stage_name, last_epoch, global_step)
        logger.info("Stage %s final checkpoint saved", stage_name)

    return last_epoch, global_step


@hydra.main(config_path="../configs", config_name="distill_staged", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run staged distillation training."""
    # Setup distributed
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif "RANK" in os.environ:
        rank, world_size = setup_distributed()

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.training.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load teacher (each GPU has its own copy)
    teacher = load_teacher(
        model_name=cfg.teacher.model_name,
        device=device,
        dtype=getattr(torch, cfg.teacher.dtype),
    )
    patch_conv3d_for_a100(teacher)
    if rank == 0:
        logger.info("Teacher loaded: %s", cfg.teacher.model_name)

    # 2) Build student
    resume_path = cfg.training.get("resume_from_checkpoint")
    if resume_path:
        student = Alpamayo1_5_Distilled.from_pretrained(resume_path).to(device)
        if rank == 0:
            logger.info("Student loaded from checkpoint: %s", resume_path)
    else:
        student_config = build_student_config(cfg)
        student = Alpamayo1_5_Distilled(student_config).to(device)
    patch_conv3d_for_a100(student)

    total_params = sum(p.numel() for p in student.parameters())
    if rank == 0:
        logger.info("Student created: %s total params", f"{total_params:,}")

    processor = helper.get_processor(student.tokenizer)

    # 3) Build loss
    teacher_hidden_dim = cfg.teacher.get("hidden_dim", 4096)
    student_hidden_dim = cfg.student.get("hidden_dim", 2048)
    distill_loss = DistillationLoss(
        teacher_hidden_dim=teacher_hidden_dim,
        student_hidden_dim=student_hidden_dim,
    ).to(device)

    # 4) Determine which stages to run
    stages_to_run = ["vlm", "expert"]
    start_stage_idx = 0
    start_epoch = 0
    global_step = 0

    # Check for resume
    if resume_path:
        progress = load_stage_progress(Path(resume_path))
        if progress:
            completed_stage = progress["current_stage"]
            if completed_stage == "vlm":
                # Stage 1 done, start from stage 2
                start_stage_idx = 1
                start_epoch = 0
                global_step = progress["global_step"]
                if rank == 0:
                    logger.info("Resuming from Stage Expert (Stage VLM completed)")
            elif completed_stage == "expert":
                if rank == 0:
                    logger.info("All stages completed, nothing to resume")
                return

    # 5) Run stages
    for idx in range(start_stage_idx, len(stages_to_run)):
        stage_name = stages_to_run[idx]
        stage_cfg = cfg.stages[stage_name]

        if rank == 0:
            logger.info("=" * 60)
            logger.info("Starting Stage: %s", stage_name.upper())
            logger.info("=" * 60)

        last_epoch, global_step = train_stage(
            stage_name=stage_name,
            stage_cfg=stage_cfg,
            cfg=cfg,
            student=student,
            teacher=teacher,
            distill_loss=distill_loss,
            processor=processor,
            device=device,
            output_dir=output_dir,
            rank=rank,
            start_epoch=start_epoch if idx == start_stage_idx else 0,
            global_step=global_step if idx == start_stage_idx else 0,
        )

        # Reset for next stage
        start_epoch = 0

        if not cfg.training.get("auto_next_stage", True) and idx < len(stages_to_run) - 1:
            if rank == 0:
                logger.info("Stage %s complete. Set training.resume_from_checkpoint=%s to continue",
                            stage_name, output_dir / f"stage_{stage_name}_final")
            break

    if rank == 0:
        logger.info("Training complete!")


if __name__ == "__main__":
    import os

    # Support both single-GPU and torchrun DDP
    if "RANK" in os.environ and not dist.is_initialized():
        setup_distributed()

    main()
