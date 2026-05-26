# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark student model (Qwen3-VL-2B + random Expert) for Orin deployment.

Two inference modes:
  - CoC mode: VLM autoregressive generate → KV Cache → Expert 4-step denoising
  - Skip-CoC mode: prompt + <|traj_future_start|> → VLM prefill → KV Cache → Expert

Usage:
    python scripts/benchmark_student.py
    python scripts/benchmark_student.py --clip-id <uuid> --sample-step-us 100000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.models.token_utils import (
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.train_utils import (
    _build_avdi,
    build_student_config,
    load_clip_sample,
    prepare_model_inputs,
    resolve_clip_samples,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_student(device: torch.device, dtype: torch.dtype) -> Alpamayo1_5_Distilled:
    """Build student with pretrained Qwen3-VL-2B VLM + random Expert, both sdpa."""
    base_cfg = OmegaConf.create({
        "student": {
            "vlm_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
            "attn_implementation": "sdpa",
            "expert_attn_implementation": "sdpa",
        },
        "loss": {
            "vlm_logits_weight": 0.0,
            "expert_hidden_weight": 0.0,
            "vlm_hidden_weight": 0.0,
            "trajectory_l2_weight": 0.0,
        },
        "teacher": {"model_name": "nvidia/Alpamayo-1.5-10B"},
    })
    config = build_student_config(base_cfg)
    student = Alpamayo1_5_Distilled.from_pretrained_submodules(config)
    student = student.to(device=device, dtype=dtype)
    # Qwen3-VL-2B patch_embed Conv3D triggers CUDNN_STATUS_INTERNAL_ERROR
    # when running in bfloat16 on some cuDNN/A100 combos.
    # Workaround: run Conv3D in float32, convert output back to bfloat16.
    _pe = student.vlm.model.visual.patch_embed
    _proj = _pe.proj
    _proj.weight = torch.nn.Parameter(_proj.weight.to(dtype=torch.float32))
    if _proj.bias is not None:
        _proj.bias = torch.nn.Parameter(_proj.bias.to(dtype=torch.float32))
    def _fwd(self, hidden_states):
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=torch.float32)).view(-1, self.embed_dim)
        return hidden_states.to(dtype=torch.bfloat16)
    _pe.forward = _fwd.__get__(_pe, type(_pe))
    student.eval()
    return student


# ---------------------------------------------------------------------------
# CoC mode: VLM generate → Expert denoising
# ---------------------------------------------------------------------------


def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample a token with temperature scaling and top-p (nucleus) filtering."""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _extract_vlm_kwargs(tokenized_data: dict) -> dict:
    """Extract visual/attention kwargs from tokenized data for direct VLM forward."""
    vlm_kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "image_grid_thw_batch"):
        if key in tokenized_data:
            vlm_kwargs[key] = tokenized_data[key]
    if "attention_mask" in tokenized_data:
        vlm_kwargs["attention_mask"] = tokenized_data["attention_mask"]
    return vlm_kwargs


@torch.no_grad()
def run_coc_inference(
    student: Alpamayo1_5_Distilled,
    data: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    B = data["ego_history_xyz"].shape[0]
    tokenized_data = {k: v for k, v in data["tokenized_data"].items()}
    input_ids = tokenized_data.pop("input_ids")

    traj_data_fuse = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    input_ids = student.fuse_traj_tokens(input_ids, traj_data_fuse)
    torch.cuda.synchronize(device)
    # Validate token IDs are within student vocabulary bounds
    vocab_size = student.vlm.get_input_embeddings().num_embeddings
    max_id = int(input_ids.max().item())
    if max_id >= vocab_size:
        raise RuntimeError(
            f"fuse_traj_tokens produced token ID {max_id} >= vocab_size {vocab_size}. "
            f"Student vocabulary is too small for the fused trajectory tokens."
        )

    eos_id = student.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    pad_id = student.tokenizer.pad_token_id
    traj_token_offset = student.config.traj_token_start_idx
    traj_vocab_size = student.config.traj_vocab_size
    max_new_tokens = 256
    temperature = 0.6
    top_p = 0.98

    vlm_kwargs = _extract_vlm_kwargs(tokenized_data)

    # ── VLM generation (manual autoregressive, bypassing generate()) ──
    vlm_start = torch.cuda.Event(enable_timing=True)
    vlm_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    vlm_start.record()

    # Prefill
    with torch.autocast("cuda", dtype=dtype):
        vlm_out = student.vlm(input_ids=input_ids, use_cache=True, **vlm_kwargs)

    past_key_values = vlm_out.past_key_values
    generated_ids: list[torch.Tensor] = []
    eos_found = torch.zeros(B, dtype=torch.bool, device=device)

    for _step in range(max_new_tokens):
        logits = vlm_out.logits[:, -1, :].float()
        # ExpertLogitsProcessor: mask trajectory token logits
        logits[:, traj_token_offset : traj_token_offset + traj_vocab_size] = float("-inf")
        next_token = _top_p_sample(logits, temperature, top_p)  # [B, 1]
        generated_ids.append(next_token)

        # StopAfterEOS: stop one step after EOS is first seen
        eos_found = eos_found | (next_token.squeeze(-1) == eos_id)
        if eos_found.all():
            break

        # Decode step (no visual inputs after prefill)
        with torch.autocast("cuda", dtype=dtype):
            vlm_out = student.vlm(
                input_ids=next_token, past_key_values=past_key_values, use_cache=True,
            )

    vlm_end.record()
    torch.cuda.synchronize(device)
    vlm_ms = vlm_start.elapsed_time(vlm_end)
    n_generated = len(generated_ids)

    # Build full sequences tensor
    if generated_ids:
        generated = torch.cat(generated_ids, dim=-1)
        sequences = torch.cat([input_ids, generated], dim=-1)
    else:
        sequences = input_ids
    sequences = replace_padding_after_eos(
        token_ids=sequences, eos_token_id=eos_id, pad_token_id=pad_id,
    )

    torch.cuda.empty_cache()

    # ── Expert denoising ──
    prompt_cache = past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    b_star = sequences.shape[0]
    n_diff = student.action_space.get_action_space_dims()[0]

    offset = student._find_eos_offset(
        sequences=sequences, eos_token_id=eos_id, device=device,
    )
    prefix_mask = tokenized_data.get("attention_mask")

    position_ids, attn_mask = student._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=student.vlm.model.rope_deltas,
        kv_cache_seq_len=prefill_seq_len, n_diffusion_tokens=n_diff, b_star=b_star,
        device=device, dtype=next(student.action_in_proj.parameters()).dtype,
        prefix_mask=prefix_mask,
    )

    fwd_kwargs = {}
    if student.config.expert_non_causal_attention:
        fwd_kwargs["is_causal"] = False

    expert_step_ms: list[float] = []

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        step_s = torch.cuda.Event(enable_timing=True)
        step_e = torch.cuda.Event(enable_timing=True)
        step_s.record()

        bs = x.shape[0]
        future_embeds = student.action_in_proj(x, t)
        if future_embeds.dim() == 2:
            future_embeds = future_embeds.view(bs, n_diff, -1)
        prefill_len = prompt_cache.get_seq_length()
        expert_out = student.expert(
            inputs_embeds=future_embeds, position_ids=position_ids,
            past_key_values=prompt_cache, attention_mask=attn_mask,
            use_cache=True, **fwd_kwargs,
        )
        prompt_cache.crop(prefill_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diff:]
        pred = student.action_out_proj(last_hidden).view(
            -1, *student.action_space.get_action_space_dims()
        )

        step_e.record()
        torch.cuda.synchronize(device)
        expert_step_ms.append(step_s.elapsed_time(step_e))
        return pred

    diff_start = torch.cuda.Event(enable_timing=True)
    diff_end = torch.cuda.Event(enable_timing=True)
    diff_start.record()

    with torch.autocast("cuda", dtype=dtype):
        sampled_action = student.diffusion.sample(
            batch_size=b_star, step_fn=step_fn, device=device,
            dtype=next(student.action_in_proj.parameters()).dtype,
            return_all_steps=False,
        )
        ego_hist_xyz = data["ego_history_xyz"]
        ego_hist_rot = data["ego_history_rot"]
        pred_xyz, pred_rot = student.action_space.action_to_traj(
            sampled_action, ego_hist_xyz[:, -1], ego_hist_rot[:, -1],
        )

    diff_end.record()
    torch.cuda.synchronize(device)
    diff_total_ms = diff_start.elapsed_time(diff_end)
    expert_total = float(sum(expert_step_ms))
    traj_conv_ms = diff_total_ms - expert_total

    return {
        "vlm_ms": vlm_ms,
        "n_tokens": n_generated,
        "expert_step_ms": expert_step_ms,
        "expert_total_ms": expert_total,
        "traj_conv_ms": max(traj_conv_ms, 0.0),
        "diffusion_total_ms": diff_total_ms,
        "total_ms": vlm_ms + diff_total_ms,
    }


# ---------------------------------------------------------------------------
# Skip-CoC mode: prefill-only VLM → Expert denoising
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_skip_coc_inference(
    student: Alpamayo1_5_Distilled,
    data: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    B = data["ego_history_xyz"].shape[0]
    tokenized_data = {k: v for k, v in data["tokenized_data"].items()}
    input_ids = tokenized_data.pop("input_ids")

    traj_data_fuse = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    input_ids = student.fuse_traj_tokens(input_ids, traj_data_fuse)

    # Build skip-CoC sequence: prompt up to <|cot_start|> + <|traj_future_start|>
    cot_start_id = student.special_token_ids["cot_start"]
    traj_future_start_id = student.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )

    # Strip potential left-padding from input_ids
    attn = tokenized_data.get("attention_mask")
    if attn is not None:
        valid = attn[0].bool()
        ids_clean = input_ids[0][valid]
    else:
        ids_clean = input_ids[0]

    cot_positions = (ids_clean == cot_start_id).nonzero(as_tuple=True)[0]
    if len(cot_positions) == 0:
        raise RuntimeError(
            f"<|cot_start|> (id={cot_start_id}) not found in input_ids. "
            f"Last 5 token ids: {ids_clean[-5:].tolist()}"
        )
    cot_pos = cot_positions[-1].item()

    new_ids = torch.cat([
        ids_clean[:cot_pos + 1],
        torch.tensor([traj_future_start_id], device=device),
    ]).unsqueeze(0)  # [1, L_new]
    new_mask = torch.ones_like(new_ids)

    eos_id = traj_future_start_id
    pad_id = student.tokenizer.pad_token_id
    n_diff = student.action_space.get_action_space_dims()[0]

    # Extract visual kwargs from tokenized_data for VLM forward
    visual_kwargs = {}
    for key in ("pixel_values", "image_grid_thw", "image_grid_thw_batch"):
        if key in tokenized_data:
            val = tokenized_data[key]
            visual_kwargs[key] = val

    # ── VLM prefill ──
    vlm_pf_start = torch.cuda.Event(enable_timing=True)
    vlm_pf_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    vlm_pf_start.record()

    with torch.autocast("cuda", dtype=dtype):
        vlm_out = student.vlm(
            input_ids=new_ids,
            attention_mask=new_mask,
            use_cache=True,
            **visual_kwargs,
        )

    vlm_pf_end.record()
    torch.cuda.synchronize(device)
    vlm_prefill_ms = vlm_pf_start.elapsed_time(vlm_pf_end)
    del vlm_out.logits
    torch.cuda.empty_cache()

    prompt_cache = vlm_out.past_key_values
    kv_len = prompt_cache.get_seq_length()
    rope_deltas = student.vlm.model.rope_deltas

    offset = torch.full((1,), new_ids.shape[1], device=device, dtype=torch.long)
    position_ids, attn_mask = student._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas,
        kv_cache_seq_len=kv_len, n_diffusion_tokens=n_diff, b_star=1,
        device=device, dtype=next(student.action_in_proj.parameters()).dtype,
        prefix_mask=None,  # no padding in the constructed sequence
    )

    fwd_kwargs = {}
    if student.config.expert_non_causal_attention:
        fwd_kwargs["is_causal"] = False

    # ── Expert denoising ──
    expert_step_ms: list[float] = []

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        step_s = torch.cuda.Event(enable_timing=True)
        step_e = torch.cuda.Event(enable_timing=True)
        step_s.record()

        bs = x.shape[0]
        future_embeds = student.action_in_proj(x, t)
        if future_embeds.dim() == 2:
            future_embeds = future_embeds.view(bs, n_diff, -1)
        prefill_len = prompt_cache.get_seq_length()
        expert_out = student.expert(
            inputs_embeds=future_embeds, position_ids=position_ids,
            past_key_values=prompt_cache, attention_mask=attn_mask,
            use_cache=True, **fwd_kwargs,
        )
        prompt_cache.crop(prefill_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diff:]
        pred = student.action_out_proj(last_hidden).view(
            -1, *student.action_space.get_action_space_dims()
        )

        step_e.record()
        torch.cuda.synchronize(device)
        expert_step_ms.append(step_s.elapsed_time(step_e))
        return pred

    diff_start = torch.cuda.Event(enable_timing=True)
    diff_end = torch.cuda.Event(enable_timing=True)
    diff_start.record()

    with torch.autocast("cuda", dtype=dtype):
        sampled_action = student.diffusion.sample(
            batch_size=1, step_fn=step_fn, device=device,
            dtype=next(student.action_in_proj.parameters()).dtype,
            return_all_steps=False,
        )
        ego_hist_xyz = data["ego_history_xyz"]
        ego_hist_rot = data["ego_history_rot"]
        pred_xyz, pred_rot = student.action_space.action_to_traj(
            sampled_action, ego_hist_xyz[:, -1], ego_hist_rot[:, -1],
        )

    diff_end.record()
    torch.cuda.synchronize(device)
    diff_total_ms = diff_start.elapsed_time(diff_end)
    expert_total = float(sum(expert_step_ms))
    traj_conv_ms = diff_total_ms - expert_total

    return {
        "vlm_ms": vlm_prefill_ms,
        "n_tokens": 0,
        "expert_step_ms": expert_step_ms,
        "expert_total_ms": expert_total,
        "traj_conv_ms": max(traj_conv_ms, 0.0),
        "diffusion_total_ms": diff_total_ms,
        "total_ms": vlm_prefill_ms + diff_total_ms,
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _stats_str(values: list[float]) -> str:
    arr = np.array(values)
    return (
        f"mean={arr.mean():.1f}  std={arr.std():.1f}  "
        f"min={arr.min():.1f}  max={arr.max():.1f} ms"
    )


def print_final_stats(label: str, results: list[dict]) -> None:
    print()
    print(f"--- {label} ---")
    total = [r["total_ms"] for r in results]
    print(f"  End-to-end       : {_stats_str(total)}")
    vlm = [r["vlm_ms"] for r in results]
    if results and results[0]["n_tokens"] > 0:
        avg_tok = int(np.mean([r["n_tokens"] for r in results]))
        print(f"  VLM generate     : {_stats_str(vlm)}  (avg {avg_tok} tokens)")
    else:
        print(f"  VLM prefill      : {_stats_str(vlm)}")
    expert = [r["expert_total_ms"] for r in results]
    print(f"  Expert 4-step    : {_stats_str(expert)}")
    traj = [r["traj_conv_ms"] for r in results]
    print(f"  Traj conversion  : {_stats_str(traj)}")

    # Per-frame detail table
    print()
    print(f"  Per-frame detail ({label}):")
    header = f"  {'frame':>5s}  {'vlm_ms':>8s}  {'expert_ms':>10s}  {'traj_ms':>8s}  {'total_ms':>9s}"
    if results and results[0]["n_tokens"] > 0:
        header += f"  {'tokens':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, r in enumerate(results):
        line = (f"  {i+1:5d}  {r['vlm_ms']:8.1f}  {r['expert_total_ms']:10.1f}  "
                f"{r['traj_conv_ms']:8.1f}  {r['total_ms']:9.1f}")
        if r["n_tokens"] > 0:
            line += f"  {r['n_tokens']:7d}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark student model inference for Orin deployment",
    )
    parser.add_argument("--clip-id", type=str, default=None,
                        help="Specific clip UUID (default: auto-detect from cache)")
    parser.add_argument("--cache-dir", type=str, default="./.cache/")
    parser.add_argument("--sample-step-us", type=int, default=100_000,
                        help="Sampling interval in microseconds (default 100000 = 0.1s)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ── Load student model ──
    print("=" * 70)
    print("Loading student: pretrained Qwen3-VL-2B VLM + random Expert (sdpa)")
    student = build_student(device, dtype)
    total_m = sum(p.numel() for p in student.parameters()) / 1e6
    vlm_m = sum(p.numel() for p in student.vlm.parameters()) / 1e6
    expert_m = sum(p.numel() for p in student.expert.parameters()) / 1e6
    print(f"  VLM params: {vlm_m:.0f}M  Expert params: {expert_m:.0f}M  Total: {total_m:.0f}M")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Diffusion: {student.diffusion.num_inference_steps}-step Euler")
    torch.backends.cudnn.benchmark = True

    # ── Resolve clip samples ──
    data_cfg: dict = {
        "cache_dir": args.cache_dir,
        "sample_step_us": args.sample_step_us,
        "history_us": 1_500_000,
        "future_us": 6_400_000,
        "shuffle": False,
        "seed": 42,
        "revision": "b719eea7f0a63619ef51ec7f54178af0937ef050",
    }
    if args.clip_id is not None:
        data_cfg["clip_ids"] = [args.clip_id]
    cfg = OmegaConf.create({"data": data_cfg})
    samples = resolve_clip_samples(cfg, epoch=0)
    if len(samples) == 0:
        print("ERROR: No valid samples found for this clip.")
        sys.exit(1)

    avdi = _build_avdi(args.cache_dir, cfg.data.revision)
    processor = helper.get_processor(student.tokenizer)

    coc_results: list[dict] = []
    skip_results: list[dict] = []

    # ── Warmup ──
    torch.cuda.empty_cache()
    data = load_clip_sample(cfg, avdi, samples[0][0], samples[0][1])
    model_inputs = prepare_model_inputs(data, processor, device)
    print("Warmup (1 frame, both modes)...")
    run_coc_inference(student, model_inputs, device, dtype)
    run_skip_coc_inference(student, model_inputs, device, dtype)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    del data, model_inputs

    # ── Benchmark loop ──
    mem_alloc = 0.0
    for i, (clip_id, t0_us) in enumerate(samples):
        data = load_clip_sample(cfg, avdi, clip_id, t0_us)
        model_inputs = prepare_model_inputs(data, processor, device)

        print(f"\nFrame {i + 1}/{len(samples)} (t0={t0_us / 1e6:.1f}s):")

        r_coc = run_coc_inference(student, model_inputs, device, dtype)
        coc_results.append(r_coc)
        tok_str = f" {r_coc['n_tokens']} tokens" if r_coc["n_tokens"] > 0 else ""
        print(f"  CoC mode       : {r_coc['total_ms']:.1f} ms  "
              f"(VLM: {r_coc['vlm_ms']:.1f}{tok_str}, Expert: {r_coc['expert_total_ms']:.1f})")

        r_skip = run_skip_coc_inference(student, model_inputs, device, dtype)
        skip_results.append(r_skip)
        print(f"  Skip-CoC mode  : {r_skip['total_ms']:.1f} ms  "
              f"(VLM prefill: {r_skip['vlm_ms']:.1f}, Expert: {r_skip['expert_total_ms']:.1f})")

        del data, model_inputs

        if i == 0:
            mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # ── Final stats ──
    print()
    print("=" * 70)
    print(f"  Peak GPU memory allocated: {mem_alloc:.2f} GB")
    print("=" * 70)

    print_final_stats("CoC mode (autoregressive generate)", coc_results)
    print_final_stats("Skip-CoC mode (prefill only)", skip_results)

    # ── Comparison summary ──
    coc_total = np.array([r["total_ms"] for r in coc_results])
    skip_total = np.array([r["total_ms"] for r in skip_results])
    speedup = coc_total.mean() / skip_total.mean()
    print()
    print(f"  Skip-CoC speedup: {speedup:.2f}x faster end-to-end")


if __name__ == "__main__":
    main()
