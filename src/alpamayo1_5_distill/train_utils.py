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
    """Return clip_ids whose required training feature chunks are present in local cache."""
    valid = avdi.clip_index[avdi.clip_index["clip_is_valid"]]
    downloaded_chunks: set[int] = set()
    required_feature_roots = [
        "labels/egomotion/egomotion",
        "camera/camera_cross_left_120fov/camera_cross_left_120fov",
        "camera/camera_front_wide_120fov/camera_front_wide_120fov",
        "camera/camera_cross_right_120fov/camera_cross_right_120fov",
        "camera/camera_front_tele_30fov/camera_front_tele_30fov",
    ]
    for chunk_idx in sorted(valid["chunk"].unique()):
        chunk_idx = int(chunk_idx)
        if all(
            avdi.is_file_cached(f"{feature_root}.chunk_{chunk_idx:04d}.zip")
            for feature_root in required_feature_roots
        ):
            downloaded_chunks.add(chunk_idx)
    return valid[valid["chunk"].isin(downloaded_chunks)].index.tolist()


def _get_training_camera_features(avdi: physical_ai_av.PhysicalAIAVDatasetInterface) -> list[str]:
    return [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]


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


def resolve_clip_ids(
    cfg: DictConfig,
    epoch: int = 0,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
) -> list[str]:
    """Resolve clip IDs without loading clip tensors."""
    cache_dir = cfg.data.get("cache_dir")
    clip_ids = cfg.data.get("clip_ids")
    if not clip_ids:
        if cache_dir is not None:
            if avdi is None:
                avdi = _build_avdi(cache_dir, cfg.data.get("revision"))
            clip_ids = _get_cached_clip_ids(avdi)
            if not clip_ids:
                raise RuntimeError(f"No clips found in cache_dir={cache_dir}")
        else:
            clip_ids = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]
    return list(clip_ids)


def _ceil_to_grid_us(value_us: int, grid_us: int) -> int:
    return ((value_us + grid_us - 1) // grid_us) * grid_us


def _sample_t0s_from_time_range(
    t_min_us: int,
    t_max_us: int,
    history_us: int,
    future_us: int,
    step_us: int,
    grid_us: int = 100_000,
) -> list[int]:
    """Sample valid t0 timestamps from a clip time range."""
    min_t0_us = history_us + 2 * grid_us
    t0_start = _ceil_to_grid_us(max(t_min_us + history_us, min_t0_us), grid_us)
    t0_end = t_max_us - future_us
    if t0_start > t0_end:
        return []
    return list(range(t0_start, t0_end + 1, step_us))


def _get_feature_time_range(feature: Any) -> tuple[int, int]:
    if hasattr(feature, "time_range"):
        t_min_us, t_max_us = feature.time_range
        return int(t_min_us), int(t_max_us)

    timestamps = getattr(feature, "timestamps", None)
    if timestamps is None:
        raise AttributeError(f"{type(feature).__name__} must expose time_range or timestamps")
    return int(timestamps.min()), int(timestamps.max())


def _resolve_sample_time_range(
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface,
    clip_id: str,
    maybe_stream: bool,
    history_us: int,
    future_us: int,
    camera_history_us: int = 300_000,
) -> tuple[int, int]:
    egomotion = avdi.get_clip_feature(
        clip_id,
        avdi.features.LABELS.EGOMOTION,
        maybe_stream=maybe_stream,
    )
    t_min_us, t_max_us = _get_feature_time_range(egomotion)

    for camera_feature in _get_training_camera_features(avdi):
        camera = avdi.get_clip_feature(clip_id, camera_feature, maybe_stream=maybe_stream)
        camera_min_us, camera_max_us = _get_feature_time_range(camera)
        t_min_us = max(t_min_us, camera_min_us + camera_history_us - history_us)
        t_max_us = min(t_max_us, camera_max_us + future_us)

    return t_min_us, t_max_us


def resolve_clip_samples(
    cfg: DictConfig,
    epoch: int = 0,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
) -> list[tuple[str, int]]:
    """Resolve (clip_id, t0_us) samples for one epoch."""
    cache_dir = cfg.data.get("cache_dir")
    if avdi is None:
        avdi = _build_avdi(cache_dir, cfg.data.get("revision"))
    maybe_stream = cache_dir is None
    history_us = cfg.data.get("history_us", 1_500_000)
    future_us = cfg.data.get("future_us", 6_400_000)
    step_us = cfg.data.get("sample_step_us", 1_000_000)

    samples: list[tuple[str, int]] = []
    for clip_id in resolve_clip_ids(cfg, epoch=0, avdi=avdi):
        t_min_us, t_max_us = _resolve_sample_time_range(
            avdi, clip_id, maybe_stream, history_us, future_us
        )
        samples.extend(
            (clip_id, t0_us)
            for t0_us in _sample_t0s_from_time_range(
                t_min_us, t_max_us, history_us, future_us, step_us
            )
        )

    if cfg.data.get("shuffle", True):
        seed = cfg.data.get("seed", 42)
        rng = random.Random(seed + epoch)
        rng.shuffle(samples)
    return samples


def load_clip_sample(
    cfg: DictConfig,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface,
    clip_id: str,
    t0_us: int,
) -> dict[str, Any]:
    """Load one (clip_id, t0_us) sample."""
    cache_dir = cfg.data.get("cache_dir")
    return load_physical_aiavdataset(
        clip_id,
        avdi=avdi,
        maybe_stream=cache_dir is None,
        t0_us=t0_us,
    )


def build_dataloader(cfg: DictConfig, epoch: int = 0):
    """Yield clip data dicts for training, loading from local cache when available."""
    cache_dir = cfg.data.get("cache_dir")
    avdi = _build_avdi(cache_dir, cfg.data.get("revision"))
    for clip_id, t0_us in resolve_clip_samples(cfg, epoch=epoch, avdi=avdi):
        yield load_clip_sample(cfg, avdi, clip_id, t0_us)


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
