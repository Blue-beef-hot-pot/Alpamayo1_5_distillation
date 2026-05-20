# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel distillation training for Alpamayo 1.5.

Runs on 4 GPUs: rank 0 runs teacher inference and dispatches results
round-robin to ranks 1-3, which run student DDP training.

Usage:
    torchrun --nproc_per_node=4 scripts/train_distill_pipeline.py --config-name=distill_pipeline

Or with overrides:
    torchrun --nproc_per_node=4 scripts/train_distill_pipeline.py \
        --config-name=distill_pipeline training.num_epochs=20
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5_distill.comm import recv_teacher_bundle, send_termination, send_teacher_bundle
from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig
from alpamayo1_5_distill.distill_loss import DistillationLoss
from alpamayo1_5_distill.distributed import (
    StudentWithLoss,
    create_student_group,
    setup_distributed,
    wrap_student_ddp,
)
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.teacher import load_teacher, teacher_forward
from alpamayo1_5_distill.train_utils import (
    build_dataloader,
    build_student_config,
    prepare_model_inputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank 0: Teacher orchestrator
# ---------------------------------------------------------------------------


def run_teacher_loop(cfg: DictConfig, rank: int) -> None:
    """Teacher loop: load clips, run teacher_forward, dispatch to student ranks."""
    device = f"cuda:{rank}"
    teacher = load_teacher(
        model_name=cfg.teacher.model_name,
        device=device,
        dtype=getattr(torch, cfg.teacher.dtype),
    )
    logger.info("[Rank 0] Teacher model loaded: %s", cfg.teacher.model_name)

    processor = helper.get_processor(teacher.tokenizer)
    num_student_ranks = cfg.pipeline.num_student_ranks
    grad_accum = cfg.training.gradient_accumulation_steps

    for epoch in range(cfg.training.num_epochs):
        clips = list(build_dataloader(cfg))
        n_clips = len(clips)

        # Pad to multiple of num_student_ranks * grad_accum so each student
        # rank gets exactly grad_accum micro-batches per optimizer step
        window = num_student_ranks * grad_accum
        if n_clips % window != 0:
            pad_count = window - (n_clips % window)
            clips = clips + clips[:pad_count]

        logger.info(
            "[Rank 0] Epoch %d: %d clips (padded to %d)", epoch, n_clips, len(clips)
        )

        for i, data in enumerate(clips):
            target_rank = 1 + (i % num_student_ranks)
            model_inputs = prepare_model_inputs(data, processor, device)

            with torch.no_grad():
                teacher_out = teacher_forward(
                    teacher,
                    model_inputs,
                    top_p=cfg.teacher.top_p,
                    temperature=cfg.teacher.temperature,
                    num_traj_samples=cfg.teacher.num_traj_samples,
                    max_generation_length=cfg.teacher.max_generation_length,
                    collect_expert_hiddens=cfg.teacher.collect_expert_hiddens,
                    collect_vlm_hiddens=cfg.loss.vlm_hidden_weight > 0,
                )

            send_teacher_bundle(model_inputs, teacher_out, dst=target_rank)
            del teacher_out, data

            if i % 10 == 0:
                logger.info("[Rank 0] Epoch %d: sent clip %d to rank %d", epoch, i, target_rank)

        # Send termination signal to all student ranks
        for r in range(1, 1 + num_student_ranks):
            send_termination(r, device=torch.device(device))

        dist.barrier()
        logger.info("[Rank 0] Epoch %d complete", epoch)

    logger.info("[Rank 0] Training complete")


# ---------------------------------------------------------------------------
# Ranks 1-3: Student DDP workers
# ---------------------------------------------------------------------------


def run_student_loop(cfg: DictConfig, rank: int, student_group) -> None:
    """Student loop: receive teacher output, run student forward/backward, DDP sync."""
    device = f"cuda:{rank}"

    # Build student
    student_config = build_student_config(cfg)
    student = Alpamayo1_5_Distilled(student_config).to(device)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info("[Rank %d] Student created: %s params", rank, f"{total_params:,}")

    # Build distillation loss
    teacher_hidden_dim = cfg.teacher.get("hidden_dim", 4096)
    student_hidden_dim = student.vlm.config.text_config.hidden_size
    distill_loss = DistillationLoss(
        vlm_logits_weight=cfg.loss.vlm_logits_weight,
        expert_hidden_weight=cfg.loss.expert_hidden_weight,
        vlm_hidden_weight=cfg.loss.vlm_hidden_weight,
        trajectory_l2_weight=cfg.loss.trajectory_l2_weight,
        temperature=cfg.loss.temperature,
        teacher_hidden_dim=teacher_hidden_dim,
        student_hidden_dim=student_hidden_dim,
    ).to(device)

    # Wrap in DDP
    ddp_model = wrap_student_ddp(
        student,
        distill_loss,
        student_group,
        num_traj_samples=cfg.teacher.num_traj_samples,
        collect_vlm_logits=cfg.loss.vlm_logits_weight > 0,
        collect_expert_hiddens=cfg.loss.expert_hidden_weight > 0,
        collect_vlm_hiddens=cfg.loss.vlm_hidden_weight > 0,
    )

    # Optimizer + scheduler
    all_params = list(ddp_model.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=all_params)

    clip_ids = cfg.data.get("clip_ids")
    num_clips_per_epoch = len(clip_ids) if clip_ids else 1
    grad_accum = cfg.training.gradient_accumulation_steps
    num_student_ranks = cfg.pipeline.num_student_ranks

    # Pad clip count same as teacher does
    window = num_student_ranks * grad_accum
    n_clips = num_clips_per_epoch
    if n_clips % window != 0:
        n_clips += window - (n_clips % window)

    clips_per_rank_per_epoch = n_clips // num_student_ranks
    total_optimizer_steps = (clips_per_rank_per_epoch * cfg.training.num_epochs) // grad_accum
    logger.info(
        "[Rank %d] Clips/rank/epoch: %d, total optimizer steps: %d",
        rank, clips_per_rank_per_epoch, total_optimizer_steps,
    )

    scheduler_cfg = OmegaConf.to_container(cfg.lr_scheduler, resolve=True)
    scheduler_cfg["T_max"] = total_optimizer_steps
    scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)

    # Training loop
    global_step = 0
    best_loss = float("inf")
    output_dir = Path(cfg.training.output_dir)
    save_rank = 1  # Only rank 1 saves checkpoints

    for epoch in range(cfg.training.num_epochs):
        ddp_model.train()
        epoch_loss = 0.0
        num_batches = 0
        micro_batch_count = 0
        optimizer.zero_grad()

        while True:
            bundle = recv_teacher_bundle(src=0, device=torch.device(device))
            if bundle is None:
                break

            model_inputs, teacher_dict = bundle

            with torch.autocast("cuda", dtype=getattr(torch, cfg.training.mixed_precision)):
                losses = ddp_model(
                    model_inputs=model_inputs,
                    teacher_sequences=teacher_dict["sequences"],
                    teacher_vlm_logits=teacher_dict["vlm_logits"],
                    teacher_vlm_hiddens=teacher_dict["vlm_hiddens"],
                    teacher_expert_hiddens=teacher_dict["expert_hiddens_all_steps"],
                    teacher_traj=teacher_dict["sampled_traj"],
                )

            loss = losses["total"]
            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            micro_batch_count += 1
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if micro_batch_count % grad_accum == 0:
                if cfg.training.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if num_batches % 10 == 0:
                loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
                logger.info(
                    "[Rank %d] Epoch %d, Batch %d, Step %d — %s",
                    rank, epoch, num_batches, global_step, loss_str,
                )

            # Free received tensors
            del model_inputs, teacher_dict

        dist.barrier()

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info("[Rank %d] Epoch %d complete — avg loss: %.4f", rank, epoch, avg_loss)

        # Checkpoint (only save_rank)
        if rank == save_rank:
            if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
                _save_checkpoint(ddp_model, optimizer, scheduler, epoch, global_step, output_dir)

            if avg_loss < best_loss:
                best_loss = avg_loss
                _save_checkpoint(ddp_model, optimizer, scheduler, epoch, global_step, output_dir / "best")
                logger.info("[Rank %d] New best model saved", rank)

    # Save final model
    if rank == save_rank:
        _save_checkpoint(ddp_model, optimizer, scheduler, epoch, global_step, output_dir / "final")
        logger.info("[Rank %d] Final model saved", rank)


def _save_checkpoint(
    ddp_model: StudentWithLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    output_dir: Path,
) -> None:
    """Save student model + distill_loss + optimizer + scheduler state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    student = ddp_model.module.student
    student.save_pretrained(str(output_dir))
    student.tokenizer.save_pretrained(str(output_dir))
    torch.save(
        {
            "distill_loss_state": ddp_model.module.distill_loss.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        output_dir / "training_state.pt",
    )
    logger.info("Checkpoint saved: %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="distill_pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run pipeline-parallel distillation training."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    rank, world_size = setup_distributed()
    assert world_size == 4, f"Expected 4 GPUs, got {world_size}"

    is_student = rank > 0
    student_group = create_student_group(cfg.pipeline.num_student_ranks) if is_student else None

    if rank == 0:
        run_teacher_loop(cfg, rank)
    else:
        run_student_loop(cfg, rank, student_group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
