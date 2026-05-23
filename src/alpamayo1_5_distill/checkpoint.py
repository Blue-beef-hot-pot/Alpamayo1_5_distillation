# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint state helpers for distillation training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

TRAINING_PROGRESS_FILENAME = "training_progress.json"
TRAINING_STATE_FILENAME = "training_state.pt"


def _training_state_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / TRAINING_STATE_FILENAME


def _training_progress_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / TRAINING_PROGRESS_FILENAME


def _progress_from_state(state: dict) -> tuple[int, int, float]:
    return int(state["epoch"]) + 1, int(state["global_step"]), float(
        state.get("best_loss", float("inf"))
    )


def save_training_checkpoint(
    output_dir: str | Path,
    student: torch.nn.Module,
    distill_loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
) -> None:
    """Save student weights, tokenizer, and complete training state."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(output_dir))
    student.tokenizer.save_pretrained(str(output_dir))
    save_training_state(
        output_dir,
        distill_loss=distill_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_loss=best_loss,
    )


def save_training_state(
    output_dir: str | Path,
    distill_loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
) -> None:
    """Save optimizer/scheduler/loss training state next to model weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = {"epoch": epoch, "global_step": global_step, "best_loss": best_loss}
    torch.save(
        {
            "distill_loss_state": distill_loss.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            **progress,
        },
        _training_state_path(output_dir),
    )
    _training_progress_path(output_dir).write_text(json.dumps(progress), encoding="utf-8")


def _load_state(checkpoint_dir: str | Path, device: str | torch.device) -> dict:
    state_path = _training_state_path(checkpoint_dir)
    try:
        return torch.load(state_path, map_location=device)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {TRAINING_STATE_FILENAME}: {state_path}") from error


def load_training_state(
    checkpoint_dir: str | Path,
    distill_loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str | torch.device,
) -> tuple[int, int, float]:
    """Load training state and return (start_epoch, global_step, best_loss)."""
    state = _load_state(checkpoint_dir, "cpu")
    distill_loss.load_state_dict(state["distill_loss_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)
    return _progress_from_state(state)


def read_checkpoint_progress(checkpoint_dir: str | Path) -> tuple[int, int, float]:
    """Read (start_epoch, global_step, best_loss) without model objects."""
    progress_path = _training_progress_path(checkpoint_dir)
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        progress = _load_state(checkpoint_dir, "cpu")
    return _progress_from_state(progress)
