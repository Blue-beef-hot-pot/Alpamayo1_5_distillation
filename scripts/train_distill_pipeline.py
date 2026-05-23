# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel distillation training for Alpamayo 1.5.

Spawns one process per pipeline worker: the configured teacher rank runs
teacher inference and dispatches results round-robin to all other ranks, which
run student DDP training.

Usage:
    python scripts/train_distill_pipeline.py --config-name=distill_pipeline

Or with overrides:
    python scripts/train_distill_pipeline.py --config-name=distill_pipeline \
        pipeline.num_processes=4 training.num_epochs=20
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5_distill.checkpoint import (
    load_training_state,
    read_checkpoint_progress,
    save_training_checkpoint,
)
from alpamayo1_5_distill.comm import recv_teacher_bundle, send_termination, send_teacher_bundle
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
    _build_avdi,
    build_student_config,
    load_clip_sample,
    prepare_model_inputs,
    resolve_clip_samples,
)

logger = logging.getLogger(__name__)


def infer_pipeline_process_count(configured: int | None, cuda_device_count: int) -> int:
    """Infer pipeline process count from config or available CUDA devices."""
    num_processes = configured if configured is not None else cuda_device_count
    if num_processes < 2:
        raise ValueError(
            "Pipeline training requires at least 2 processes: 1 teacher + >=1 student."
        )
    return num_processes


def infer_student_ranks(world_size: int, teacher_rank: int) -> list[int]:
    """Return all ranks except teacher_rank, validating the pipeline shape."""
    if world_size < 2:
        raise ValueError(
            "Pipeline training requires at least 2 processes: 1 teacher + >=1 student."
        )
    if teacher_rank < 0 or teacher_rank >= world_size:
        raise ValueError(f"teacher_rank={teacher_rank} must be in [0, {world_size})")
    return [rank for rank in range(world_size) if rank != teacher_rank]


# ---------------------------------------------------------------------------
# Rank 0: Teacher orchestrator
# ---------------------------------------------------------------------------


def run_teacher_loop(cfg: DictConfig, rank: int, student_ranks: list[int]) -> None:
    """Teacher loop: load clips, run teacher_forward, dispatch to student ranks."""
    device = f"cuda:{rank}"
    teacher = load_teacher(
        model_name=cfg.teacher.model_name,
        device=device,
        dtype=getattr(torch, cfg.teacher.dtype),
    )
    logger.info("[Teacher] Teacher model loaded: %s", cfg.teacher.model_name)

    processor = helper.get_processor(teacher.tokenizer)
    num_student_ranks = len(student_ranks)
    grad_accum = cfg.training.gradient_accumulation_steps
    avdi = _build_avdi(cfg.data.get("cache_dir"), cfg.data.get("revision"))
    resume_path = cfg.training.get("resume_from_checkpoint")
    start_epoch = 0
    if resume_path:
        start_epoch, _, _ = read_checkpoint_progress(resume_path)
        logger.info("[Teacher] Resuming from epoch %d", start_epoch)

    for epoch in range(start_epoch, cfg.training.num_epochs):
        samples = resolve_clip_samples(cfg, epoch=epoch, avdi=avdi)
        n_samples = len(samples)

        # Pad to multiple of num_student_ranks * grad_accum so each student
        # rank gets exactly grad_accum micro-batches per optimizer step
        window = num_student_ranks * grad_accum
        if n_samples % window != 0:
            pad_count = window - (n_samples % window)
            samples = samples + samples[:pad_count]

        logger.info(
            "[Teacher] Epoch %d: %d samples (padded to %d)", epoch, n_samples, len(samples)
        )

        for i, (clip_id, t0_us) in enumerate(samples):
            target_rank = student_ranks[i % num_student_ranks]
            data = load_clip_sample(cfg, avdi, clip_id, t0_us)
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
                logger.info("[Teacher] Epoch %d: sent sample %d to rank %d", epoch, i, target_rank)

        # Send termination signal to all student ranks
        for student_rank in student_ranks:
            send_termination(student_rank, device=torch.device(device))

        dist.barrier()
        logger.info("[Teacher] Epoch %d complete", epoch)

    logger.info("[Teacher] Training complete")


# ---------------------------------------------------------------------------
# Ranks 1-3: Student DDP workers
# ---------------------------------------------------------------------------


def run_student_loop(
    cfg: DictConfig, rank: int, student_group, student_ranks: list[int], teacher_rank: int
) -> None:
    """Student loop: receive teacher output, run student forward/backward, DDP sync."""
    device = f"cuda:{rank}"

    # Build student
    resume_path = cfg.training.get("resume_from_checkpoint")
    if resume_path:
        student = Alpamayo1_5_Distilled.from_pretrained(resume_path).to(device)
        logger.info("[Rank %d] Student loaded from checkpoint: %s", rank, resume_path)
    else:
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

    num_samples_per_epoch = len(resolve_clip_samples(cfg, epoch=0))
    grad_accum = cfg.training.gradient_accumulation_steps
    num_student_ranks = len(student_ranks)

    # Pad sample count same as teacher does
    window = num_student_ranks * grad_accum
    n_samples = num_samples_per_epoch
    if n_samples % window != 0:
        n_samples += window - (n_samples % window)

    samples_per_rank_per_epoch = n_samples // num_student_ranks
    total_optimizer_steps = (samples_per_rank_per_epoch * cfg.training.num_epochs) // grad_accum
    logger.info(
        "[Rank %d] Samples/rank/epoch: %d, total optimizer steps: %d",
        rank, samples_per_rank_per_epoch, total_optimizer_steps,
    )

    scheduler_cfg = OmegaConf.to_container(cfg.lr_scheduler, resolve=True)
    scheduler_cfg["T_max"] = total_optimizer_steps
    scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)

    # Training loop
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if resume_path:
        start_epoch, global_step, best_loss = load_training_state(
            resume_path,
            distill_loss=ddp_model.module.distill_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        logger.info(
            "[Rank %d] Resumed training from %s at epoch %d, global_step %d, best_loss %.4f",
            rank,
            resume_path,
            start_epoch,
            global_step,
            best_loss,
        )
    output_dir = Path(cfg.training.output_dir)
    save_rank = student_ranks[0]

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.training.num_epochs):
        last_epoch = epoch
        ddp_model.train()
        epoch_loss = 0.0
        num_batches = 0
        micro_batch_count = 0
        optimizer.zero_grad()

        while True:
            bundle = recv_teacher_bundle(src=teacher_rank, device=torch.device(device))
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
                _save_checkpoint(
                    ddp_model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_loss,
                    output_dir / f"epoch_{epoch + 1}",
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                _save_checkpoint(
                    ddp_model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_loss,
                    output_dir / "best",
                )
                logger.info("[Rank %d] New best model saved", rank)

    # Save final model
    if rank == save_rank:
        _save_checkpoint(
            ddp_model,
            optimizer,
            scheduler,
            last_epoch,
            global_step,
            best_loss,
            output_dir / "final",
        )
        logger.info("[Rank %d] Final model saved", rank)


def _save_checkpoint(
    ddp_model: StudentWithLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
    output_dir: Path,
) -> None:
    """Save a checkpoint from the wrapped student/loss module."""
    save_training_checkpoint(
        output_dir,
        student=ddp_model.module.student,
        distill_loss=ddp_model.module.distill_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_loss=best_loss,
    )
    logger.info("Checkpoint saved: %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pipeline_worker(local_rank: int, world_size: int, cfg: DictConfig) -> None:
    """Worker process launched by torch.multiprocessing.spawn."""
    os.environ.setdefault("MASTER_ADDR", cfg.pipeline.get("master_addr", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", str(cfg.pipeline.get("master_port", 29500)))

    rank, world_size = setup_distributed(
        rank=local_rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    teacher_rank = cfg.pipeline.get("teacher_rank", 0)
    student_ranks = infer_student_ranks(world_size, teacher_rank)
    student_group = create_student_group(student_ranks) if rank in student_ranks else None

    if rank == teacher_rank:
        run_teacher_loop(cfg, rank, student_ranks)
    else:
        run_student_loop(cfg, rank, student_group, student_ranks, teacher_rank)

    dist.destroy_process_group()


@hydra.main(config_path="../configs", config_name="distill_pipeline", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run pipeline-parallel distillation training."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    num_processes = infer_pipeline_process_count(
        cfg.pipeline.get("num_processes"), torch.cuda.device_count()
    )
    mp.spawn(pipeline_worker, args=(num_processes, cfg), nprocs=num_processes, join=True)


if __name__ == "__main__":
    main()
