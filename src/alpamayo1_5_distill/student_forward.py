# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Student forward pass for distillation.

Supports two modes:
1. **Teacher-forcing** (training): teacher-generated token IDs are fed through
   ``student.vlm.forward()`` (not ``generate()``) with gradient enabled, so VLM
   Logits KD loss can backprop to student VLM parameters.
2. **Inference** (eval): student uses its own ``generate()`` — no gradient
   through VLM, matching the original Alpamayo1_5 pipeline.

Both modes share the Expert denoising step (always with gradient).
"""

import logging
from typing import Any

import einops
import torch

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5_distill.train_utils import repeat_visual_inputs, shallow_copy_data

logger = logging.getLogger(__name__)


class StudentOutput:
    """Container for student model outputs used in distillation."""

    def __init__(
        self,
        vlm_logits: torch.Tensor | None = None,
        vlm_hiddens: list[torch.Tensor] | None = None,
        expert_hiddens_all_steps: list[list[torch.Tensor]] | None = None,
        sampled_traj: torch.Tensor | None = None,
        pred_xyz: torch.Tensor | None = None,
        pred_rot: torch.Tensor | None = None,
        sequences: torch.Tensor | None = None,
        num_expert_layers: int | None = None,
    ) -> None:
        self.vlm_logits = vlm_logits
        self.vlm_hiddens = vlm_hiddens or []
        self.expert_hiddens_all_steps = expert_hiddens_all_steps or []
        self.sampled_traj = sampled_traj
        self.pred_xyz = pred_xyz
        self.pred_rot = pred_rot
        self.sequences = sequences
        self.num_expert_layers = num_expert_layers


def student_forward(
    student: Alpamayo1_5,
    data: dict[str, Any],
    teacher_sequences: torch.Tensor | None = None,
    top_p: float = 0.98,
    temperature: float = 0.6,
    num_traj_samples: int = 6,
    max_generation_length: int = 256,
    collect_vlm_logits: bool = True,
    collect_expert_hiddens: bool = True,
    collect_vlm_hiddens: bool = True,
) -> StudentOutput:
    """Run student forward pass and extract all distillation signals.

    When ``teacher_sequences`` is provided the VLM runs in **teacher-forcing**
    mode: the teacher-generated token IDs are passed through
    ``student.vlm.forward()`` with gradient, making VLM Logits KD and
    VLM Hidden KD differentiable.  Otherwise the student falls back to
    ``generate()`` (no gradient through VLM).

    Args:
        student: The student model (Alpamayo1_5_Distilled).
        data: Input data dict with tokenized_data, ego_history_xyz, ego_history_rot.
        teacher_sequences: Teacher-generated token IDs ``[B*, total_len]``
            for teacher-forcing.  If *None*, the student generates its own
            tokens (inference / eval mode).
        top_p: Top-p sampling parameter (inference mode only).
        temperature: Sampling temperature (inference mode only).
        num_traj_samples: Number of trajectory samples.
        max_generation_length: Max VLM generation tokens (inference mode only).
        collect_vlm_logits: Whether to collect VLM logits.
        collect_expert_hiddens: Whether to collect Expert hidden states.
        collect_vlm_hiddens: Whether to collect VLM hidden states.

    Returns:
        StudentOutput with all intermediate representations.
    """
    from alpamayo1_5.models.token_utils import (
        StopAfterEOS,
        replace_padding_after_eos,
        to_special_token,
    )
    from transformers import LogitsProcessorList, StoppingCriteriaList

    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor

    data = shallow_copy_data(data)
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B = ego_history_xyz.shape[0]
    tokenized_data = data["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")

    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = student.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    eos_token_id = student.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    pad_token_id = student.tokenizer.pad_token_id

    # ── VLM forward ─────────────────────────────────────────────────
    student_vlm_hiddens: list[torch.Tensor] = []

    if teacher_sequences is not None:
        # Teacher-forcing mode: differentiable VLM forward
        sequences = teacher_sequences.to(device)
        b_star = sequences.shape[0]

        full_attention_mask = (sequences != pad_token_id).long()

        visual_kwargs = repeat_visual_inputs(tokenized_data, B, num_traj_samples)

        # Forward with gradient — VLM Logits KD and VLM Hidden KD can backprop
        vlm_out = student.vlm(
            input_ids=sequences,
            attention_mask=full_attention_mask,
            use_cache=True,
            output_hidden_states=collect_vlm_hiddens,
            **visual_kwargs,
        )

        # Extract generation-position logits (shift by 1 for next-token prediction)
        if collect_vlm_logits:
            gen_len = sequences.shape[1] - prompt_len
            vlm_logits = vlm_out.logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
        else:
            vlm_logits = None

        # Collect VLM hidden states (exclude embedding layer output)
        student_vlm_hiddens: list[torch.Tensor] = []
        if collect_vlm_hiddens and vlm_out.hidden_states is not None:
            student_vlm_hiddens = list(vlm_out.hidden_states[1:])

        prompt_cache = vlm_out.past_key_values
        rope_deltas = student.vlm.model.rope_deltas

        # prefix_mask for Expert attention: repeat original prompt mask
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, num_traj_samples, dim=0)

    else:
        # Inference mode: generate() — no gradient through VLM
        generation_config = student.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = collect_vlm_logits
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = pad_token_id

        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=student.config.traj_token_start_idx,
                    traj_vocab_size=student.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = student.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )

        vlm_logits = None
        if collect_vlm_logits and hasattr(vlm_outputs, "logits") and vlm_outputs.logits:
            vlm_logits = torch.stack(vlm_outputs.logits, dim=1)  # [B*, gen_len, vocab]

        vlm_outputs.rope_deltas = student.vlm.model.rope_deltas
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        sequences = vlm_outputs.sequences
        prompt_cache = vlm_outputs.past_key_values
        rope_deltas = vlm_outputs.rope_deltas
        b_star = sequences.shape[0]

        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, num_traj_samples, dim=0)

    # ── Expert denoising (shared, always with gradient) ─────────────
    prefill_seq_len = prompt_cache.get_seq_length()
    n_diffusion_tokens = student.action_space.get_action_space_dims()[0]

    offset = student._find_eos_offset(
        sequences=sequences,
        eos_token_id=eos_token_id,
        device=device,
    )
    position_ids, attention_mask = student._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )

    forward_kwargs = {}
    if student.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    all_expert_hiddens: list[list[torch.Tensor]] = []

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        future_token_embeds = student.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(bs, n_diffusion_tokens, -1)

        prefill_len = prompt_cache.get_seq_length()

        if collect_expert_hiddens:
            expert_out = student.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                **forward_kwargs,
            )
            hiddens = list(expert_out.hidden_states[1:])
            all_expert_hiddens.append(hiddens)
        else:
            expert_out = student.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )

        prompt_cache.crop(prefill_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        pred = student.action_out_proj(last_hidden).view(
            -1, *student.action_space.get_action_space_dims()
        )
        return pred

    total_batch = B * num_traj_samples
    sampled_action = student.diffusion.sample(
        batch_size=total_batch,
        step_fn=step_fn,
        device=device,
        dtype=next(student.action_in_proj.parameters()).dtype,
        return_all_steps=False,
    )

    # ── Trajectory conversion ───────────────────────────────────────
    hist_xyz_rep = einops.repeat(
        ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_traj_samples
    )
    hist_rot_rep = einops.repeat(
        ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_traj_samples
    )
    pred_xyz, pred_rot = student.action_space.action_to_traj(
        sampled_action, hist_xyz_rep, hist_rot_rep
    )
    pred_xyz = einops.rearrange(pred_xyz, "(b n) ... -> b 1 n ...", n=num_traj_samples)
    pred_rot = einops.rearrange(pred_rot, "(b n) ... -> b 1 n ...", n=num_traj_samples)

    # Keep Expert hidden states from all diffusion steps
    num_expert_layers = None
    if all_expert_hiddens:
        num_expert_layers = len(all_expert_hiddens[-1])

    return StudentOutput(
        vlm_logits=vlm_logits,
        vlm_hiddens=student_vlm_hiddens,
        expert_hiddens_all_steps=all_expert_hiddens,
        sampled_traj=sampled_action,
        pred_xyz=pred_xyz,
        pred_rot=pred_rot,
        sequences=sequences,
        num_expert_layers=num_expert_layers,
    )
