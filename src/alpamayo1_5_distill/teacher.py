# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teacher model loading and inference for distillation.

Provides utilities to load the original Alpamayo 1.5 (8B) model and run
forward passes that extract intermediate representations (VLM logits,
VLM hidden states, Expert hidden states across all diffusion steps,
sampled trajectories) as soft labels for distillation.
"""

import logging
from typing import Any

import einops
import torch
from transformers import AutoConfig

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5_distill.train_utils import repeat_visual_inputs, shallow_copy_data

logger = logging.getLogger(__name__)


class TeacherOutput:
    """Container for teacher model outputs used in distillation.

    Attributes:
        vlm_logits: Logits from the VLM generation step, shape depends on generation.
        vlm_hiddens: Per-layer VLM hidden states, each [B*, T, hidden].
        expert_hiddens_all_steps: Per-step per-layer Expert hidden states,
            outer list = diffusion steps, inner list = layers,
            each [B*, T, hidden].
        sampled_traj: Sampled action trajectories, shape [B, n_sets, n_samples, T, 2].
        pred_xyz: Predicted xyz trajectories, shape [B, n_sets, n_samples, T, 3].
        pred_rot: Predicted rotation matrices, shape [B, n_sets, n_samples, T, 3, 3].
        sequences: Generated token IDs from teacher VLM, used for student teacher-forcing.
        cot: Chain-of-causation reasoning text per sample.
        num_expert_layers: Number of layers in the teacher Expert.
    """

    def __init__(
        self,
        vlm_logits: torch.Tensor | None = None,
        vlm_hiddens: list[torch.Tensor] | None = None,
        expert_hiddens_all_steps: list[list[torch.Tensor]] | None = None,
        sampled_traj: torch.Tensor | None = None,
        pred_xyz: torch.Tensor | None = None,
        pred_rot: torch.Tensor | None = None,
        sequences: torch.Tensor | None = None,
        cot: list[str] | None = None,
        num_expert_layers: int | None = None,
    ) -> None:
        self.vlm_logits = vlm_logits
        self.vlm_hiddens = vlm_hiddens or []
        self.expert_hiddens_all_steps = expert_hiddens_all_steps or []
        self.sampled_traj = sampled_traj
        self.pred_xyz = pred_xyz
        self.pred_rot = pred_rot
        self.sequences = sequences
        self.cot = cot
        self.num_expert_layers = num_expert_layers


def load_teacher(
    model_name: str = "nvidia/Alpamayo-1.5-10B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Alpamayo1_5:
    """Load the teacher Alpamayo 1.5 model for distillation.

    Args:
        model_name: HuggingFace model identifier or local path.
        device: Target device.
        dtype: Model dtype.

    Returns:
        The loaded teacher model in eval mode.
    """
    logger.info(f"Loading teacher model: {model_name}")
    model = Alpamayo1_5.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model


def _run_expert_with_hiddens(
    model: Alpamayo1_5,
    future_token_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_cache: Any,
    attention_mask: torch.Tensor,
    n_diffusion_tokens: int,
    forward_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run Expert forward pass and collect per-layer hidden states.

    This is a modified version of the step_fn in Alpamayo1_5 that additionally
    extracts hidden states from every Expert layer for distillation.

    Args:
        model: The Alpamayo1_5 model instance.
        future_token_embeds: Projected action embeddings [B*, T, hidden].
        position_ids: Position IDs for the Expert.
        prompt_cache: KV cache from the VLM.
        attention_mask: 4D attention mask for the Expert.
        n_diffusion_tokens: Number of diffusion tokens.
        forward_kwargs: Additional kwargs (e.g., is_causal).

    Returns:
        Tuple of (prediction tensor, list of per-layer hidden states).
    """
    b_star = future_token_embeds.shape[0]
    prefill_seq_len = prompt_cache.get_seq_length()

    expert_out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prompt_cache,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)

    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    pred = model.action_out_proj(last_hidden).view(
        -1, *model.action_space.get_action_space_dims()
    )

    # Collect hidden states from all layers (excluding embedding layer output)
    expert_hiddens = list(expert_out.hidden_states[1:])

    return pred, expert_hiddens


@torch.no_grad()
def teacher_forward(
    teacher: Alpamayo1_5,
    data: dict[str, Any],
    top_p: float = 0.98,
    temperature: float = 0.6,
    num_traj_samples: int = 6,
    max_generation_length: int = 256,
    collect_expert_hiddens: bool = True,
    collect_vlm_hiddens: bool = True,
) -> TeacherOutput:
    """Run teacher forward pass and extract soft labels for distillation.

    This mirrors ``sample_trajectories_from_data_with_vlm_rollout`` but
    additionally extracts VLM logits, VLM hidden states, and Expert hidden
    states across all diffusion steps.

    Args:
        teacher: The loaded teacher model (Alpamayo1_5).
        data: Input data dict with tokenized_data, ego_history_xyz, ego_history_rot.
        top_p: Top-p sampling parameter.
        temperature: Sampling temperature.
        num_traj_samples: Number of trajectory samples.
        max_generation_length: Max VLM generation tokens.
        collect_expert_hiddens: Whether to collect Expert hidden states
            (memory-intensive, disable for inference-only).
        collect_vlm_hiddens: Whether to collect VLM hidden states via
            an extra forward pass (memory-intensive).

    Returns:
        TeacherOutput with all intermediate representations.
    """
    from alpamayo1_5.models.token_utils import (
        StopAfterEOS,
        extract_text_tokens,
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
    input_ids = teacher.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device

    # 1) VLM autoregressive generation
    generation_config = teacher.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = teacher.tokenizer.pad_token_id

    eos_token_id = teacher.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=teacher.config.traj_token_start_idx,
                traj_vocab_size=teacher.config.traj_vocab_size,
            )
        ]
    )
    vlm_outputs = teacher.vlm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
        **tokenized_data,
    )

    # Extract VLM logits (stacked across generation steps)
    vlm_logits = None
    if hasattr(vlm_outputs, "logits") and vlm_outputs.logits is not None:
        vlm_logits = torch.stack(vlm_outputs.logits, dim=1)  # [B*, gen_len, vocab]

    vlm_outputs.rope_deltas = teacher.vlm.model.rope_deltas
    vlm_outputs.sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=teacher.tokenizer.pad_token_id,
    )

    # 1b) Collect VLM hidden states via a separate forward pass
    teacher_vlm_hiddens: list[torch.Tensor] = []
    if collect_vlm_hiddens:
        sequences = vlm_outputs.sequences
        b_star_vlm = sequences.shape[0]
        seq_attention_mask = (sequences != teacher.tokenizer.pad_token_id).long()

        visual_kwargs = repeat_visual_inputs(tokenized_data, B, num_traj_samples)

        vlm_fwd_out = teacher.vlm(
            input_ids=sequences,
            attention_mask=seq_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **visual_kwargs,
        )
        # Exclude embedding layer output (index 0)
        teacher_vlm_hiddens = list(vlm_fwd_out.hidden_states[1:])
        del vlm_fwd_out
        torch.cuda.empty_cache()

    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    b_star = vlm_outputs.sequences.shape[0]
    n_diffusion_tokens = teacher.action_space.get_action_space_dims()[0]

    offset = teacher._find_eos_offset(
        sequences=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        device=device,
    )
    prefix_mask = tokenized_data.get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = torch.repeat_interleave(prefix_mask, num_traj_samples, dim=0)

    position_ids, attention_mask = teacher._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=vlm_outputs.rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=device,
        dtype=next(teacher.action_in_proj.parameters()).dtype,
        prefix_mask=prefix_mask,
    )

    forward_kwargs = {}
    if teacher.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    # 2) Define denoising step function
    all_expert_hiddens: list[list[torch.Tensor]] = []

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        future_token_embeds = teacher.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(bs, n_diffusion_tokens, -1)

        if collect_expert_hiddens:
            pred, hiddens = _run_expert_with_hiddens(
                teacher, future_token_embeds, position_ids, prompt_cache,
                attention_mask, n_diffusion_tokens, forward_kwargs,
            )
            all_expert_hiddens.append(hiddens)
        else:
            prefill_len = prompt_cache.get_seq_length()
            expert_out = teacher.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_len)
            last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            pred = teacher.action_out_proj(last_hidden).view(
                -1, *teacher.action_space.get_action_space_dims()
            )
        return pred

    # 3) Diffusion sampling
    total_batch = B * num_traj_samples
    sampled_action = teacher.diffusion.sample(
        batch_size=total_batch,
        step_fn=step_fn,
        device=device,
        dtype=next(teacher.action_in_proj.parameters()).dtype,
        return_all_steps=False,
    )

    # 4) Convert to trajectories
    hist_xyz_rep = einops.repeat(
        ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_traj_samples
    )
    hist_rot_rep = einops.repeat(
        ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_traj_samples
    )
    pred_xyz, pred_rot = teacher.action_space.action_to_traj(
        sampled_action, hist_xyz_rep, hist_rot_rep
    )
    pred_xyz = einops.rearrange(pred_xyz, "(b n) ... -> b 1 n ...", n=num_traj_samples)
    pred_rot = einops.rearrange(pred_rot, "(b n) ... -> b 1 n ...", n=num_traj_samples)

    # 5) Extract CoT text
    cot = None
    extra = extract_text_tokens(teacher.tokenizer, vlm_outputs.sequences)
    if "cot" in extra:
        import numpy as np
        cot = np.array(extra["cot"]).reshape([B, num_traj_samples]).tolist()

    # 6) Keep Expert hidden states from all diffusion steps
    # all_expert_hiddens: [n_diff_steps][n_layers][B*, T, hidden]
    num_expert_layers = None
    if all_expert_hiddens:
        num_expert_layers = len(all_expert_hiddens[-1])

    return TeacherOutput(
        vlm_logits=vlm_logits,
        vlm_hiddens=teacher_vlm_hiddens,
        expert_hiddens_all_steps=all_expert_hiddens,
        sampled_traj=sampled_action,
        pred_xyz=pred_xyz,
        pred_rot=pred_rot,
        sequences=vlm_outputs.sequences,
        cot=cot,
        num_expert_layers=num_expert_layers,
    )
