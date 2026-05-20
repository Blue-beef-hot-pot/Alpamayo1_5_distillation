# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-GPU communication utilities for pipeline parallelism.

Serializes TeacherOutput + model_inputs into flat tensors for NCCL send/recv,
and provides termination signal protocol for epoch boundaries.
"""

import json
import logging
from typing import Any

import torch
import torch.distributed as dist

from alpamayo1_5_distill.teacher import TeacherOutput

logger = logging.getLogger(__name__)

_TERMINATION_MAGIC = -1


def _tensor_meta(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}


def _make_tensor(desc: dict) -> torch.Tensor:
    return torch.empty(desc["shape"], dtype=getattr(torch, desc["dtype"]))


def serialize_teacher_bundle(
    model_inputs: dict[str, Any],
    teacher_out: TeacherOutput,
) -> tuple[list[torch.Tensor], dict]:
    """Flatten model_inputs + teacher_out into a flat tensor list + metadata.

    model_inputs after teacher_forward has input_ids already popped from
    tokenized_data (student uses teacher_out.sequences instead).
    """
    tensors: list[torch.Tensor] = []
    meta: dict[str, Any] = {"tokenized_keys": [], "all_tensor_descs": []}

    # model_inputs: tokenized_data (dict of tensors)
    td = model_inputs.get("tokenized_data", {})
    for key in sorted(td.keys()):
        val = td[key]
        if isinstance(val, torch.Tensor):
            meta["tokenized_keys"].append(key)
            tensors.append(val.contiguous())
            meta["all_tensor_descs"].append(_tensor_meta(val))

    # model_inputs: ego_history_xyz
    if "ego_history_xyz" in model_inputs:
        t = model_inputs["ego_history_xyz"].contiguous()
        tensors.append(t)
        meta["ego_history_xyz_idx"] = len(tensors) - 1
        meta["all_tensor_descs"].append(_tensor_meta(t))

    # model_inputs: ego_history_rot
    if "ego_history_rot" in model_inputs:
        t = model_inputs["ego_history_rot"].contiguous()
        tensors.append(t)
        meta["ego_history_rot_idx"] = len(tensors) - 1
        meta["all_tensor_descs"].append(_tensor_meta(t))

    # TeacherOutput: sequences
    t = teacher_out.sequences.contiguous()
    tensors.append(t)
    meta["sequences_idx"] = len(tensors) - 1
    meta["all_tensor_descs"].append(_tensor_meta(t))

    # TeacherOutput: vlm_logits (may be None)
    if teacher_out.vlm_logits is not None:
        meta["has_vlm_logits"] = True
        tensors.append(teacher_out.vlm_logits.contiguous())
        meta["vlm_logits_idx"] = len(tensors) - 1
        meta["all_tensor_descs"].append(_tensor_meta(teacher_out.vlm_logits))
    else:
        meta["has_vlm_logits"] = False

    # TeacherOutput: vlm_hiddens (list[Tensor])
    n_vlm_layers = len(teacher_out.vlm_hiddens)
    meta["n_vlm_layers"] = n_vlm_layers
    meta["vlm_hidden_start_idx"] = len(tensors)
    for h in teacher_out.vlm_hiddens:
        tensors.append(h.contiguous())
        meta["all_tensor_descs"].append(_tensor_meta(h))

    # TeacherOutput: expert_hiddens_all_steps (list[list[Tensor]])
    n_expert_steps = len(teacher_out.expert_hiddens_all_steps)
    meta["n_expert_steps"] = n_expert_steps
    meta["n_expert_layers_per_step"] = [len(step) for step in teacher_out.expert_hiddens_all_steps]
    meta["expert_step_start_idx"] = []
    for step_hiddens in teacher_out.expert_hiddens_all_steps:
        meta["expert_step_start_idx"].append(len(tensors))
        for h in step_hiddens:
            tensors.append(h.contiguous())
            meta["all_tensor_descs"].append(_tensor_meta(h))

    # TeacherOutput: sampled_traj
    if teacher_out.sampled_traj is not None:
        meta["has_sampled_traj"] = True
        tensors.append(teacher_out.sampled_traj.contiguous())
        meta["sampled_traj_idx"] = len(tensors) - 1
        meta["all_tensor_descs"].append(_tensor_meta(teacher_out.sampled_traj))
    else:
        meta["has_sampled_traj"] = False

    return tensors, meta


def deserialize_teacher_bundle(
    tensors: list[torch.Tensor],
    meta: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct model_inputs dict and teacher_out dict from flat tensors.

    Uses stored _idx fields from metadata for direct indexing instead of a
    sequential counter, making deserialization order-independent.
    """
    # model_inputs: tokenized_data
    tokenized_data = {
        key: tensors[i] for i, key in enumerate(meta["tokenized_keys"])
    }
    model_inputs: dict[str, Any] = {"tokenized_data": tokenized_data}

    # model_inputs: ego_history
    if "ego_history_xyz_idx" in meta:
        model_inputs["ego_history_xyz"] = tensors[meta["ego_history_xyz_idx"]]
    if "ego_history_rot_idx" in meta:
        model_inputs["ego_history_rot"] = tensors[meta["ego_history_rot_idx"]]

    # TeacherOutput: sequences
    teacher_dict: dict[str, Any] = {}
    teacher_dict["sequences"] = tensors[meta["sequences_idx"]]

    # TeacherOutput: vlm_logits
    if meta.get("has_vlm_logits", False):
        teacher_dict["vlm_logits"] = tensors[meta["vlm_logits_idx"]]
    else:
        teacher_dict["vlm_logits"] = None

    # TeacherOutput: vlm_hiddens
    n_vlm_layers = meta["n_vlm_layers"]
    start = meta["vlm_hidden_start_idx"]
    teacher_dict["vlm_hiddens"] = list(tensors[start : start + n_vlm_layers])

    # TeacherOutput: expert_hiddens_all_steps
    n_expert_steps = meta["n_expert_steps"]
    expert_hiddens_all_steps = []
    for s in range(n_expert_steps):
        start = meta["expert_step_start_idx"][s]
        n_layers = meta["n_expert_layers_per_step"][s]
        expert_hiddens_all_steps.append(list(tensors[start : start + n_layers]))
    teacher_dict["expert_hiddens_all_steps"] = expert_hiddens_all_steps

    # TeacherOutput: sampled_traj
    if meta.get("has_sampled_traj", False):
        teacher_dict["sampled_traj"] = tensors[meta["sampled_traj_idx"]]
    else:
        teacher_dict["sampled_traj"] = None

    return model_inputs, teacher_dict


def send_teacher_bundle(
    model_inputs: dict[str, Any],
    teacher_out: TeacherOutput,
    dst: int,
) -> None:
    """Serialize and send teacher bundle to dst rank."""
    tensors, meta = serialize_teacher_bundle(model_inputs, teacher_out)

    # Send metadata as uint8 byte tensor
    meta_bytes = json.dumps(meta).encode("utf-8")
    meta_tensor = torch.ByteTensor(list(meta_bytes)).to(tensors[0].device)
    meta_len = torch.tensor([len(meta_bytes)], dtype=torch.int64).to(tensors[0].device)

    dist.send(meta_len, dst=dst)
    dist.send(meta_tensor, dst=dst)

    # Send each tensor
    for t in tensors:
        dist.send(t.contiguous(), dst=dst)


def recv_teacher_bundle(
    src: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Receive and deserialize teacher bundle from src rank.

    Returns None if a termination signal is received instead.
    """
    # Receive metadata length
    meta_len = torch.tensor([0], dtype=torch.int64).to(device)
    dist.recv(meta_len, src=src)

    # Check for termination signal
    if meta_len.item() == _TERMINATION_MAGIC:
        return None

    # Receive metadata bytes
    meta_tensor = torch.ByteTensor([0] * meta_len.item()).to(device)
    dist.recv(meta_tensor, src=src)
    meta = json.loads(meta_tensor.cpu().numpy().tobytes().decode("utf-8"))

    # Receive tensors
    tensors: list[torch.Tensor] = []
    for desc in meta["all_tensor_descs"]:
        t = _make_tensor(desc).to(device)
        dist.recv(t, src=src)
        tensors.append(t)

    return deserialize_teacher_bundle(tensors, meta)


def send_termination(dst: int, device: torch.device = torch.device("cuda")) -> None:
    """Send termination signal to dst rank."""
    signal = torch.tensor([_TERMINATION_MAGIC], dtype=torch.int64).to(device)
    dist.send(signal, dst=dst)
