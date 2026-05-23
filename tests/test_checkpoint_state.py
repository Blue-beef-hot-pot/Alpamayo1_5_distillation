# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch

from alpamayo1_5_distill.checkpoint import (
    TRAINING_PROGRESS_FILENAME,
    TRAINING_STATE_FILENAME,
    load_training_state,
    read_checkpoint_progress,
    save_training_state,
)


def _build_stateful_objects() -> tuple[
    torch.nn.Linear, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    x = torch.ones(1, 2)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return model, optimizer, scheduler


def test_save_and_load_training_state_restores_progress_and_states(tmp_path) -> None:
    model, optimizer, scheduler = _build_stateful_objects()
    saved_weight = model.weight.detach().clone()
    saved_optimizer_state = optimizer.state_dict()
    saved_scheduler_state = scheduler.state_dict()

    save_training_state(
        tmp_path,
        distill_loss=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=17,
        best_loss=0.123,
    )

    with torch.no_grad():
        model.weight.add_(10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    start_epoch, global_step, best_loss = load_training_state(
        tmp_path,
        distill_loss=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
    )

    assert start_epoch == 3
    assert global_step == 17
    assert best_loss == pytest.approx(0.123)
    assert torch.allclose(model.weight, saved_weight)
    loaded_optimizer_state = optimizer.state_dict()["state"]
    assert loaded_optimizer_state.keys() == saved_optimizer_state["state"].keys()
    for param_id, loaded_state in loaded_optimizer_state.items():
        for key, loaded_value in loaded_state.items():
            saved_value = saved_optimizer_state["state"][param_id][key]
            if isinstance(loaded_value, torch.Tensor):
                assert torch.allclose(loaded_value, saved_value)
            else:
                assert loaded_value == saved_value
    assert scheduler.state_dict() == saved_scheduler_state


def test_read_checkpoint_progress_does_not_require_model_objects(tmp_path) -> None:
    model, optimizer, scheduler = _build_stateful_objects()
    save_training_state(
        tmp_path, model, optimizer, scheduler, epoch=4, global_step=30, best_loss=0.5
    )

    assert (tmp_path / TRAINING_PROGRESS_FILENAME).exists()
    assert read_checkpoint_progress(tmp_path) == (5, 30, 0.5)


def test_read_checkpoint_progress_supports_legacy_state_only_checkpoint(tmp_path) -> None:
    model, optimizer, scheduler = _build_stateful_objects()
    torch.save(
        {
            "distill_loss_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": 4,
            "global_step": 30,
            "best_loss": 0.5,
        },
        tmp_path / TRAINING_STATE_FILENAME,
    )

    assert read_checkpoint_progress(tmp_path) == (5, 30, 0.5)


def test_load_training_state_defaults_missing_best_loss_to_infinity(tmp_path) -> None:
    model, optimizer, scheduler = _build_stateful_objects()
    torch.save(
        {
            "distill_loss_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": 1,
            "global_step": 9,
        },
        tmp_path / TRAINING_STATE_FILENAME,
    )

    start_epoch, global_step, best_loss = load_training_state(
        tmp_path, model, optimizer, scheduler, device="cpu"
    )

    assert start_epoch == 2
    assert global_step == 9
    assert math.isinf(best_loss)


def test_load_training_state_requires_training_state_file(tmp_path) -> None:
    model, optimizer, scheduler = _build_stateful_objects()

    with pytest.raises(FileNotFoundError, match=TRAINING_STATE_FILENAME):
        load_training_state(tmp_path, model, optimizer, scheduler, device="cpu")
