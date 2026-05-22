# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared training utilities used by both single-GPU and pipeline training scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import huggingface_hub
import pandas as pd
import physical_ai_av
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig

from omegaconf import DictConfig


def _find_cached_file(cache_dir: Path, filename: str) -> Path | None:
    """Locate a file inside the HF cache layout by name (any snapshot)."""
    matches = [m for m in cache_dir.rglob(filename) if "snapshots" in m.parts]
    return matches[0] if matches else None


def _build_avdi(
    cache_dir: str | None, revision: str | None
) -> physical_ai_av.PhysicalAIAVDatasetInterface:
    """Build PhysicalAIAVDatasetInterface from local cache when cache_dir is set."""
    if cache_dir is None:
        return physical_ai_av.PhysicalAIAVDatasetInterface()

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface.__new__(
        physical_ai_av.PhysicalAIAVDatasetInterface
    )
    avdi.token = None
    avdi.api = huggingface_hub.HfApi(token=None)
    avdi.fs = huggingface_hub.HfFileSystem(token=None)
    avdi.repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"
    avdi.repo_type = "dataset"
    avdi.revision = revision
    avdi.repo_snapshot_info = {
        "repo_id": avdi.repo_id,
        "repo_type": avdi.repo_type,
        "revision": revision,
    }
    avdi.cache_dir = cache_dir
    avdi.local_dir = None
    avdi.confirm_download_threshold_gb = float("inf")

    features_csv = _find_cached_file(Path(cache_dir), "features.csv")
    clip_index_path = _find_cached_file(Path(cache_dir), "clip_index.parquet")
    if features_csv is None or clip_index_path is None:
        raise FileNotFoundError(
            "Metadata (features.csv / clip_index.parquet) not found in cache. "
            "Re-run download with --include-metadata."
        )
    features_df = pd.read_csv(features_csv, index_col="feature")
    features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
        json.loads, na_action="ignore"
    )
    avdi.features = physical_ai_av.dataset.Features(features_df)
    avdi.clip_index = pd.read_parquet(clip_index_path)
    if "clip_id" in avdi.clip_index.columns:
        avdi.clip_index = avdi.clip_index.set_index("clip_id")
    return avdi


def _get_cached_clip_ids(avdi: physical_ai_av.PhysicalAIAVDatasetInterface) -> list[str]:
    """Return list of clip_ids whose egomotion chunk is present in local cache."""
    valid = avdi.clip_index[avdi.clip_index["clip_is_valid"]]
    downloaded_chunks: set[int] = set()
    for chunk_idx in sorted(valid["chunk"].unique()):
        chunk_idx = int(chunk_idx)
        fname = f"labels/egomotion/egomotion.chunk_{chunk_idx:04d}.zip"
        if avdi.is_file_cached(fname):
            downloaded_chunks.add(chunk_idx)
    return valid[valid["chunk"].isin(downloaded_chunks)].index.tolist()


def build_student_config(cfg: DictConfig) -> Alpamayo1_5_DistilledConfig:
    """Build student config from Hydra config."""
    diffusion_cfg = {
        "_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching",
        "num_inference_steps": cfg.student.get("diffusion_steps", 4),
        "int_method": "euler",
    }
    action_in_proj_cfg = {
        "_target_": "alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2",
        "num_enc_layers": 4,
        "hidden_size": 1024,
        "num_fourier_feats": 20,
        "max_freq": 100.0,
    }
    return Alpamayo1_5_DistilledConfig(
        vlm_name_or_path=cfg.student.vlm_name_or_path,
        diffusion_cfg=diffusion_cfg,
        action_in_proj_cfg=action_in_proj_cfg,
        teacher_model_name=cfg.teacher.model_name,
        distill_loss_weights={
            "vlm_logits": cfg.loss.vlm_logits_weight,
            "expert_hidden": cfg.loss.expert_hidden_weight,
            "trajectory_l2": cfg.loss.trajectory_l2_weight,
        },
        attn_implementation=cfg.student.get("attn_implementation", "flash_attention_2"),
    )


def resolve_clip_ids(cfg: DictConfig, epoch: int = 0) -> list[str]:
    """Resolve clip IDs for one epoch without loading clip tensors."""
    cache_dir = cfg.data.get("cache_dir")
    clip_ids = cfg.data.get("clip_ids")
    if not clip_ids:
        if cache_dir is not None:
            avdi = _build_avdi(cache_dir, cfg.data.get("revision"))
            clip_ids = _get_cached_clip_ids(avdi)
            if not clip_ids:
                raise RuntimeError(f"No clips found in cache_dir={cache_dir}")
        else:
            clip_ids = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]

    clip_ids = list(clip_ids)
    if cfg.data.get("shuffle", True):
        seed = cfg.data.get("seed", 42)
        rng = random.Random(seed + epoch)
        rng.shuffle(clip_ids)
    return clip_ids


def build_dataloader(cfg: DictConfig, epoch: int = 0):
    """Yield clip data dicts for training, loading from local cache when available."""
    cache_dir = cfg.data.get("cache_dir")
    avdi = _build_avdi(cache_dir, cfg.data.get("revision"))
    clip_ids = resolve_clip_ids(cfg, epoch=epoch)
    maybe_stream = cache_dir is None
    for clip_id in clip_ids:
        yield load_physical_aiavdataset(
            clip_id, avdi=avdi, maybe_stream=maybe_stream, t0_us=5_100_000
        )


def prepare_model_inputs(data: dict, processor, device: str) -> dict:
    """Tokenize image/text inputs and build the model_inputs dict."""
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, device)


def repeat_visual_inputs(
    tokenized_data: dict[str, Any], batch_size: int, num_traj_samples: int,
) -> dict[str, torch.Tensor]:
    """Repeat visual tensors to match num_return_sequences batch expansion."""
    visual_kwargs: dict[str, Any] = {}
    for key in ("pixel_values", "image_grid_thw", "image_grid_thw_batch"):
        if key in tokenized_data:
            val = tokenized_data[key]
            if isinstance(val, torch.Tensor) and val.shape[0] == batch_size:
                val = val.repeat_interleave(num_traj_samples, dim=0)
            visual_kwargs[key] = val
    return visual_kwargs


def shallow_copy_data(data: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy data dict, isolating only tokenized_data for safe mutation."""
    data = dict(data)
    if "tokenized_data" in data:
        data["tokenized_data"] = dict(data["tokenized_data"])
    return data
