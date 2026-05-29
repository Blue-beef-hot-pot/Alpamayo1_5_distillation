#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Alpamayo1_5_Distilled student model inference.

Tests end-to-end latency, per-stage timing, and GPU memory usage
with real data from PhysicalAI-AV dataset.

Usage:
    python scripts/test_student_inference.py
    python scripts/test_student_inference.py --num-runs 5 --num-traj-samples 1 6
    python scripts/test_student_inference.py --clip-id <uuid>
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import einops
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled
from alpamayo1_5_distill.student_forward import differentiable_flow_matching_sample
from transformers import LogitsProcessorList, StoppingCriteriaList


def build_test_config(attn_impl="sdpa"):
    """Build student config with all required fields."""
    # Build config first (vocab_size will be overridden by _initialize_vlm_config)
    config = Alpamayo1_5_DistilledConfig(
        vlm_name_or_path="Qwen/Qwen3-VL-2B-Instruct",
        vocab_size=160000,  # Will be overridden, but set anyway
        diffusion_cfg={
            "_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching",
            "num_inference_steps": 4,
            "int_method": "euler",
        },
        action_space_cfg={
            "_target_": "alpamayo1_5.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace",
            "n_waypoints": 64,
            "dt": 0.1,
        },
        action_in_proj_cfg={
            "_target_": "alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2",
            "num_enc_layers": 4,
            "hidden_size": 1024,
            "num_fourier_feats": 20,
            "max_freq": 100.0,
        },
        action_out_proj_cfg={"_target_": "torch.nn.Linear"},
        expert_cfg={"_attn_implementation": "sdpa"},  # Expert always uses SDPA
        traj_vocab_size=4096,
        traj_tokenizer_cfg={
            "_target_": "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer",
            "num_bins": 4096,
        },
        hist_traj_tokenizer_cfg={
            "_target_": "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer",
            "num_bins": 4096,
        },
        attn_implementation=attn_impl,  # VLM attention implementation
        add_special_tokens=True,
    )

    # Override vocab_size to accommodate trajectory tokens
    # hist_token_start_idx = 151669 + 4096 = 155765
    # Traj tokenizer output: [0, 4095] + 155765 = [155765, 159860]
    # Need vocab_size >= 159861
    config.vocab_size = 160000

    return config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"


def patch_conv3d_for_a100(student):
    """Patch Qwen3-VL patch_embed Conv3D to avoid cuDNN CUDNN_STATUS_INTERNAL_ERROR.

    On some cuDNN/A100 combos, Conv3D in bfloat16 triggers cuDNN internal error.
    Fix: keep Conv3D weights in float32 and disable cuDNN for this specific op.
    """
    _pe = student.vlm.model.visual.patch_embed
    _proj = _pe.proj  # nn.Conv3d
    # Keep Conv3D weights in float32 to avoid cuDNN bfloat16 issues
    _proj.weight = torch.nn.Parameter(_proj.weight.to(dtype=torch.float32))
    if _proj.bias is not None:
        _proj.bias = torch.nn.Parameter(_proj.bias.to(dtype=torch.float32))

    def _fwd(self, hidden_states):
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        )
        with torch.backends.cudnn.flags(enabled=False):
            hidden_states = self.proj(hidden_states.to(dtype=torch.float32)).view(-1, self.embed_dim)
        return hidden_states.to(dtype=torch.bfloat16)

    _pe.forward = _fwd.__get__(_pe, type(_pe))
    logger.info("Patched Conv3D: float32 weights + cuDNN disabled")


def get_memory_mb():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / 1024**2


def load_real_data(cache_dir, device="cuda"):
    """Load real data from local cache using train_utils functions."""
    from alpamayo1_5_distill.train_utils import _build_avdi, _get_cached_clip_ids
    from pathlib import Path

    logger.info("Loading data from cache: %s", cache_dir)

    # Find the revision (snapshot hash) from cache directory
    cache_path = Path(cache_dir)
    snapshots_dir = cache_path / "datasets--nvidia--PhysicalAI-Autonomous-Vehicles" / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots directory found in {cache_dir}")

    snapshot_dirs = list(snapshots_dir.iterdir())
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot found in {snapshots_dir}")

    revision = snapshot_dirs[0].name
    logger.info("Using revision: %s", revision)

    # Use the same avdi builder as training script
    avdi = _build_avdi(cache_dir, revision=revision)

    # Get available clips
    clip_ids = _get_cached_clip_ids(avdi)
    logger.info("Found %d clips in cache", len(clip_ids))

    if not clip_ids:
        raise FileNotFoundError(f"No clips found in cache_dir={cache_dir}")

    return avdi, clip_ids


def prepare_inputs(student, data, device, use_images=True):
    """Prepare model inputs from raw data."""
    processor = helper.get_processor(student.tokenizer)

    if use_images:
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
    else:
        # Text-only mode: create message without images
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        # Remove image content from messages
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = [c for c in msg["content"] if c.get("type") != "image"]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Remove image-related tensors if not using images
    if not use_images:
        inputs.pop("pixel_values", None)
        inputs.pop("image_grid_thw", None)

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def benchmark_inference(student, data, num_traj_samples, max_gen_length, use_images=True):
    """Run inference with per-stage timing.

    Returns dict with timing and memory info.
    """
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype

    model_inputs = prepare_inputs(student, data, device, use_images=use_images)
    inputs = copy.deepcopy(model_inputs)
    ego_history_xyz = inputs["ego_history_xyz"]
    ego_history_rot = inputs["ego_history_rot"]
    B = ego_history_xyz.shape[0]
    tokenized_data = inputs["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")

    # Fuse trajectory tokens
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = student.fuse_traj_tokens(input_ids, traj_data_vlm)
    prompt_len = input_ids.shape[1]

    eos_token_id = student.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    pad_token_id = student.tokenizer.pad_token_id

    torch.cuda.reset_peak_memory_stats()
    mem_before = get_memory_mb()

    # ── Stage 1: VLM Generation ──────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    generation_config = student.vlm.generation_config
    generation_config.top_p = 0.98
    generation_config.temperature = 0.6
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
    generation_config.max_new_tokens = max_gen_length
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = pad_token_id

    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = LogitsProcessorList([])

    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor

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

    torch.cuda.synchronize()
    vlm_time = time.perf_counter() - t0

    # ── Stage 2: Expert Denoising (Flow Matching) ────────────────────
    torch.cuda.synchronize()
    t1 = time.perf_counter()

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
        dtype=dtype,
        prefix_mask=prefix_mask,
    )

    forward_kwargs = {}
    if student.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        future_token_embeds = student.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(bs, n_diffusion_tokens, -1)

        prefill_len = prompt_cache.get_seq_length()
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
    sampled_action = differentiable_flow_matching_sample(
        student.diffusion,
        batch_size=total_batch,
        step_fn=step_fn,
        device=device,
        dtype=dtype,
        return_all_steps=False,
    )

    torch.cuda.synchronize()
    expert_time = time.perf_counter() - t1

    # ── Stage 3: Trajectory Conversion ───────────────────────────────
    torch.cuda.synchronize()
    t2 = time.perf_counter()

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

    torch.cuda.synchronize()
    traj_time = time.perf_counter() - t2

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "vlm_time": vlm_time,
        "expert_time": expert_time,
        "traj_time": traj_time,
        "total_time": vlm_time + expert_time + traj_time,
        "mem_before": mem_before,
        "peak_mem": peak_mem,
        "output_shape": str(pred_xyz.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark student model inference")
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of runs per config"
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        nargs="+",
        default=[1, 6],
        help="List of num_traj_samples to test",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=DEFAULT_CLIP_ID,
        help="Clip ID from PhysicalAI-AV dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./.cache/",
        help="Local HF cache directory",
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=64,
        help="Max VLM generation tokens",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Test text-only inference (skip images)",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
        help="VLM attention implementation (default: sdpa)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load student model with pretrained VLM backbone
    logger.info("Loading student model with pretrained VLM (attn: %s)...", args.attn_impl)
    config = build_test_config(attn_impl=args.attn_impl)
    student = Alpamayo1_5_Distilled.from_pretrained_submodules(config)
    student = student.to(dtype=torch.bfloat16, device=device)
    student.eval()

    # Patch Conv3D to avoid cuDNN CUDNN_STATUS_INTERNAL_ERROR on A100
    patch_conv3d_for_a100(student)

    # For debugging: disable visual model to test text-only inference
    # Uncomment the following lines to test without images
    # student.vlm.model.visual = None
    # logger.info("Disabled visual model for text-only testing")

    logger.info("Model: Qwen3-VL-2B + 2B Expert, %d FM steps", student.diffusion.num_inference_steps)

    # Load real data from cache
    logger.info("Loading real data from cache...")
    avdi, clip_ids = load_real_data(args.cache_dir, device)

    # Load first clip
    clip_id = clip_ids[0]
    logger.info("Using clip: %s", clip_id)

    data = load_physical_aiavdataset(
        clip_id,
        t0_us=5_100_000,
        avdi=avdi,
        maybe_stream=False,
    )

    # Prepare model inputs
    model_inputs = prepare_inputs(student, data, device, use_images=not args.text_only)

    # Debug: check tokenizer and embedding sizes
    logger.info("Tokenizer vocab size: %d", len(student.tokenizer))
    logger.info("Embedding table size: %d", student.vlm.get_input_embeddings().num_embeddings)
    logger.info("Config vocab_size: %d", config.vocab_size)
    logger.info("Text-only mode: %s", args.text_only)

    # Check token IDs
    logger.info("tokenized_data keys: %s", list(model_inputs["tokenized_data"].keys()))
    input_ids = model_inputs["tokenized_data"]["input_ids"]
    logger.info("Input IDs shape: %s", input_ids.shape)
    logger.info("Input IDs range: [%d, %d]", input_ids.min().item(), input_ids.max().item())

    # Check for out-of-range IDs
    max_emb = student.vlm.get_input_embeddings().num_embeddings
    if input_ids.max().item() >= max_emb:
        logger.error("Found token IDs >= embedding size!")
        bad_ids = input_ids[input_ids >= max_emb].unique()
        logger.error("Bad IDs: %s", bad_ids.tolist())
    else:
        logger.info("All token IDs are within embedding range")

    # Check trajectory tokenizer
    logger.info("Traj tokenizer vocab_size: %d", student.traj_tokenizer.vocab_size)
    logger.info("hist_token_start_idx: %d", student.hist_token_start_idx)
    logger.info("Max possible token ID: %d", student.hist_token_start_idx + student.traj_tokenizer.vocab_size - 1)

    # Simple benchmark: just measure model loading and memory
    logger.info("=== Model Summary ===")
    logger.info("Model type: Alpamayo1_5_Distilled")
    logger.info("VLM: Qwen3-VL-2B (%d layers)", student.vlm.config.text_config.num_hidden_layers)
    logger.info("Expert: %d layers, hidden_size=%d",
                student.expert.config.num_hidden_layers,
                student.expert.config.hidden_size)
    logger.info("Flow Matching Steps: %d", student.diffusion.num_inference_steps)
    logger.info("Tokenizer vocab size: %d", len(student.tokenizer))
    logger.info("Embedding table size: %d", student.vlm.get_input_embeddings().num_embeddings)

    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("Total parameters: %s (%.2fM)", f"{total_params:,}", total_params / 1e6)
    logger.info("Trainable parameters: %s (%.2fM)", f"{trainable_params:,}", trainable_params / 1e6)

    # Memory usage
    mem_after_load = get_memory_mb()
    logger.info("GPU memory after model load: %.1f MB", mem_after_load)

    # Test VLM forward (without generate)
    logger.info("\n=== Testing VLM Forward ===")
    try:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                test_inputs = model_inputs["tokenized_data"].copy()
                test_input_ids = test_inputs.pop("input_ids")
                logger.info("Test input_ids shape: %s", test_input_ids.shape)
                logger.info("Test input_ids range: [%d, %d]", test_input_ids.min().item(), test_input_ids.max().item())

                # Try VLM forward
                vlm_out = student.vlm(
                    input_ids=test_input_ids,
                    use_cache=True,
                    **test_inputs,
                )
                logger.info("VLM forward success!")
                logger.info("Output keys: %s", list(vlm_out.keys()))
    except Exception as e:
        logger.error("VLM forward failed: %s", e)

    # Test full inference pipeline
    logger.info("\n=== Testing Full Inference ===")
    try:
        import copy
        test_inputs = copy.deepcopy(model_inputs)

        # Run fuse_traj_tokens
        test_input_ids = test_inputs["tokenized_data"]["input_ids"]
        traj_data = {
            "ego_history_xyz": test_inputs["ego_history_xyz"],
            "ego_history_rot": test_inputs["ego_history_rot"],
        }
        fused_ids = student.fuse_traj_tokens(test_input_ids, traj_data)
        logger.info("fuse_traj_tokens success! Shape: %s", fused_ids.shape)

        # Prepare tokenized data (without input_ids to avoid duplicate)
        tokenized_data = {k: v for k, v in test_inputs["tokenized_data"].items() if k != "input_ids"}

        # Run VLM forward
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vlm_out = student.vlm(
                    input_ids=fused_ids,
                    use_cache=True,
                    **tokenized_data,
                )
                logger.info("VLM forward success!")

        # Measure inference time
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Run generate
                generation_config = student.vlm.generation_config
                generation_config.top_p = 0.98
                generation_config.temperature = 0.6
                generation_config.do_sample = True
                generation_config.num_return_sequences = 1
                generation_config.max_new_tokens = 32
                generation_config.pad_token_id = student.tokenizer.pad_token_id

                eos_token_id = student.tokenizer.convert_tokens_to_ids(
                    to_special_token("traj_future_start")
                )
                stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])

                vlm_outputs = student.vlm.generate(
                    input_ids=fused_ids,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    **tokenized_data,
                )
                if hasattr(vlm_outputs, 'sequences'):
                    logger.info("VLM generate success! Output shape: %s", vlm_outputs.sequences.shape)
                else:
                    logger.info("VLM generate success! Output shape: %s", vlm_outputs.shape)

        torch.cuda.synchronize()
        vlm_time = time.perf_counter() - start_time
        logger.info("VLM generation time: %.3fs", vlm_time)

        # Measure peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        logger.info("Peak GPU memory: %.1f MB", peak_mem)

        # Get sequences for Expert
        if hasattr(vlm_outputs, 'sequences'):
            sequences = vlm_outputs.sequences
        else:
            sequences = vlm_outputs

        # Test Expert denoising (Flow Matching)
        logger.info("\n=== Testing Expert Denoising ===")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        expert_start = time.perf_counter()

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Build Expert inputs
                B = fused_ids.shape[0]
                num_traj_samples = 1
                n_diffusion_tokens = student.action_space.get_action_space_dims()[0]

                # Get KV cache from VLM
                prompt_cache = vlm_outputs.past_key_values if hasattr(vlm_outputs, 'past_key_values') else vlm_out.past_key_values

                # Build step function for Expert
                def step_fn(x, t):
                    bs = x.shape[0]
                    future_token_embeds = student.action_in_proj(x, t)
                    if future_token_embeds.dim() == 2:
                        future_token_embeds = future_token_embeds.view(bs, n_diffusion_tokens, -1)

                    prefill_len = prompt_cache.get_seq_length()
                    expert_out = student.expert(
                        inputs_embeds=future_token_embeds,
                        past_key_values=prompt_cache,
                        use_cache=True,
                    )
                    prompt_cache.crop(prefill_len)
                    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
                    pred = student.action_out_proj(last_hidden).view(
                        -1, *student.action_space.get_action_space_dims()
                    )
                    return pred

                # Run flow matching
                total_batch = B * num_traj_samples
                sampled_action = differentiable_flow_matching_sample(
                    student.diffusion,
                    batch_size=total_batch,
                    step_fn=step_fn,
                    device=fused_ids.device,
                    dtype=torch.bfloat16,
                    return_all_steps=False,
                )
                logger.info("Expert denoising success! Action shape: %s", sampled_action.shape)

        torch.cuda.synchronize()
        expert_time = time.perf_counter() - expert_start
        logger.info("Expert denoising time: %.3fs", expert_time)

        expert_peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        logger.info("Expert peak GPU memory: %.1f MB", expert_peak_mem)

        # Total inference time
        total_time = vlm_time + expert_time
        logger.info("\n=== Inference Summary (single run) ===")
        logger.info("VLM generation: %.3fs", vlm_time)
        logger.info("Expert denoising: %.3fs", expert_time)
        logger.info("Total inference: %.3fs", total_time)
        logger.info("Peak GPU memory: %.1f MB", max(peak_mem, expert_peak_mem))

        # Multi-run benchmark
        logger.info("\n=== Running Multi-Run Benchmark (%d runs) ===", args.num_runs)
        vlm_times = []
        expert_times = []
        total_times = []

        for run_idx in range(args.num_runs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            run_start = time.perf_counter()

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # VLM generate
                    vlm_start = time.perf_counter()
                    vlm_outputs = student.vlm.generate(
                        input_ids=fused_ids,
                        generation_config=generation_config,
                        stopping_criteria=stopping_criteria,
                        **tokenized_data,
                    )
                    torch.cuda.synchronize()
                    vlm_t = time.perf_counter() - vlm_start

                    # Get sequences
                    if hasattr(vlm_outputs, 'sequences'):
                        seq = vlm_outputs.sequences
                        pkv = vlm_outputs.past_key_values
                    else:
                        seq = vlm_outputs
                        pkv = vlm_out.past_key_values

                    # Expert denoising
                    expert_start = time.perf_counter()

                    def step_fn_run(x, t):
                        bs = x.shape[0]
                        future_token_embeds = student.action_in_proj(x, t)
                        if future_token_embeds.dim() == 2:
                            future_token_embeds = future_token_embeds.view(bs, n_diffusion_tokens, -1)
                        prefill_len = pkv.get_seq_length()
                        expert_out = student.expert(
                            inputs_embeds=future_token_embeds,
                            past_key_values=pkv,
                            use_cache=True,
                        )
                        pkv.crop(prefill_len)
                        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
                        return student.action_out_proj(last_hidden).view(-1, *student.action_space.get_action_space_dims())

                    sampled_action = differentiable_flow_matching_sample(
                        student.diffusion,
                        batch_size=total_batch,
                        step_fn=step_fn_run,
                        device=fused_ids.device,
                        dtype=torch.bfloat16,
                        return_all_steps=False,
                    )
                    torch.cuda.synchronize()
                    expert_t = time.perf_counter() - expert_start

            total_t = time.perf_counter() - run_start
            vlm_times.append(vlm_t)
            expert_times.append(expert_t)
            total_times.append(total_t)

            logger.info("  Run %d: VLM=%.3fs Expert=%.3fs Total=%.3fs",
                        run_idx + 1, vlm_t, expert_t, total_t)

        # Print benchmark summary
        avg_vlm = sum(vlm_times) / len(vlm_times)
        avg_expert = sum(expert_times) / len(expert_times)
        avg_total = sum(total_times) / len(total_times)
        min_total = min(total_times)
        max_total = max(total_times)

        print("\n" + "=" * 60)
        print("Student Model Inference Benchmark Results")
        print("=" * 60)
        print(f"Model: Alpamayo1_5_Distilled (Qwen3-VL-2B + 2B Expert)")
        print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Flow Matching Steps: {student.diffusion.num_inference_steps}")
        print(f"Runs: {args.num_runs}")
        print("-" * 60)
        print(f"VLM Generation (avg):    {avg_vlm:.3f}s")
        print(f"Expert Denoising (avg):  {avg_expert:.3f}s")
        print(f"Total Inference (avg):   {avg_total:.3f}s")
        print(f"Total Inference (min):   {min_total:.3f}s")
        print(f"Total Inference (max):   {max_total:.3f}s")
        print(f"GPU Memory (loaded):     {mem_after_load:.1f} MB")
        print(f"GPU Memory (peak):       {max(peak_mem, expert_peak_mem):.1f} MB")
        print("=" * 60)

    except Exception as e:
        logger.error("Full inference failed: %s", e)
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("Student Model Configuration Summary")
    print("=" * 60)
    print(f"Model: Alpamayo1_5_Distilled (Qwen3-VL-2B + 2B Expert)")
    print(f"VLM Layers: {student.vlm.config.text_config.num_hidden_layers}")
    print(f"Expert Layers: {student.expert.config.num_hidden_layers}")
    print(f"Hidden Size: {student.expert.config.hidden_size}")
    print(f"Flow Matching Steps: {student.diffusion.num_inference_steps}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"GPU Memory (loaded): {mem_after_load:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
