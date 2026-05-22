# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that downloaded data can be loaded and (optionally) run through inference.

Stage 1 (default, no GPU required):
  - Read clip_index from local cache
  - Sample a few clip_ids and call load_physical_aiavdataset with maybe_stream=False
  - Validate tensor shapes, dtypes, and value ranges

Stage 2 (optional, requires GPU + model weights):
  - Load teacher model, tokenize inputs, run one forward pass
  - Print a minADE score

Usage:
    # Data-only validation (fast, no GPU)
    python scripts/test_downloaded_data.py --cache-dir /path/to/cache --num-clips 5

    # Full inference validation (slow, needs GPU)
    python scripts/test_downloaded_data.py --cache-dir /path/to/cache --num-clips 1 --inference
用法

  # 最快路径（推荐当网络不通时）：跳过在线 init
  python scripts/test_downloaded_data.py --cache-dir ./.cache/ --num-clips 5 --no-online-init

  # 默认路径：先尝试在线 init，失败自动 fallback 到 cache-only
  python scripts/test_downloaded_data.py --cache-dir ./.cache/ --num-clips 5

  # 测试指定 clip
  python scripts/test_downloaded_data.py --cache-dir ./.cache/ \
    --clip-ids e13eaabc-6287-46ba-964c-548b7d5615b8 --no-online-init

  # 完整推理验证（需 GPU + 模型权重）
  python scripts/test_downloaded_data.py --cache-dir ./.cache/ \
    --num-clips 1 --inference --no-online-init

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _find_cached_file(cache_dir: Path, filename: str) -> Path | None:
    """Locate a file inside the HF cache layout by name (any snapshot)."""
    # HF cache layout: <cache>/datasets--<repo-id-with-slash-replaced>/snapshots/<sha>/<filename>
    matches = list(cache_dir.rglob(filename))
    matches = [m for m in matches if "snapshots" in m.parts]
    return matches[0] if matches else None


def validate_clip_data(data: dict, clip_id: str) -> bool:
    """Check shapes, dtypes, and basic value sanity. Returns True if all pass."""
    passed = True

    # image_frames: (N_cameras, num_frames, 3, H, W) uint8
    img = data["image_frames"]
    if img.ndim != 5 or img.shape[0] != 4 or img.shape[1] != 4 or img.shape[2] != 3:
        _fail(f"image_frames shape {img.shape}, expected (4, 4, 3, H, W)")
        passed = False
    elif img.dtype != torch.uint8:
        _fail(f"image_frames dtype {img.dtype}, expected torch.uint8")
        passed = False
    else:
        _ok(f"image_frames shape={tuple(img.shape)} dtype={img.dtype}")

    # camera_indices: (4,) int64, sorted
    ci = data["camera_indices"]
    expected_cams = torch.tensor([0, 1, 2, 6], dtype=torch.int64)
    if not torch.equal(ci, expected_cams):
        _fail(f"camera_indices {ci.tolist()}, expected {expected_cams.tolist()}")
        passed = False
    else:
        _ok(f"camera_indices={ci.tolist()}")

    # ego history/future
    for key, t_steps in [("ego_history_xyz", 16), ("ego_history_rot", 16),
                         ("ego_future_xyz", 64), ("ego_future_rot", 64)]:
        t = data[key]
        if t.shape[0] != 1 or t.shape[1] != 1 or t.shape[2] != t_steps:
            _fail(f"{key} shape {tuple(t.shape)}, expected (1,1,{t_steps},...)")
            passed = False
        elif torch.isnan(t).any():
            _fail(f"{key} contains NaN")
            passed = False
        else:
            _ok(f"{key} shape={tuple(t.shape)} no-NaN")

    # History xyz at t0 (last step) should be ~0 (ego-frame origin)
    hist_xyz = data["ego_history_xyz"][0, 0]  # (16, 3)
    t0_xyz = hist_xyz[-1]
    if torch.norm(t0_xyz) > 1e-3:
        _fail(f"ego_history_xyz at t0 is {t0_xyz.tolist()}, expected ~0 (ego frame)")
        passed = False
    else:
        _ok(f"ego_history_xyz t0 origin norm={torch.norm(t0_xyz):.2e}")

    # Future xyz should not be all zeros (vehicle actually moved)
    fut_xyz = data["ego_future_xyz"][0, 0]
    if torch.norm(fut_xyz) < 1e-6:
        _fail("ego_future_xyz is all zeros — vehicle did not move?")
        passed = False
    else:
        _ok(f"ego_future_xyz norm={torch.norm(fut_xyz):.3f} m")

    # Timestamps
    ts = data["relative_timestamps"]
    if ts.ndim != 2 or ts.shape[0] != 4:
        _fail(f"relative_timestamps shape {tuple(ts.shape)}")
        passed = False
    elif (ts < 0).any():
        _fail("relative_timestamps has negative values")
        passed = False
    else:
        _ok(f"relative_timestamps range [{ts.min():.3f}, {ts.max():.3f}] s")

    # Rotations should be valid rotation matrices (det ~1, R^T R ~ I)
    hist_rot = data["ego_history_rot"][0, 0]  # (16, 3, 3)
    det = torch.det(hist_rot.float())
    if (det - 1.0).abs().max() > 0.01:
        _fail(f"ego_history_rot det range [{det.min():.4f}, {det.max():.4f}], expected ~1")
        passed = False
    else:
        _ok(f"ego_history_rot det range [{det.min():.6f}, {det.max():.6f}]")

    return passed


def run_inference(data: dict) -> None:
    """Load teacher model and run one forward pass."""
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    print("\n  Loading teacher model nvidia/Alpamayo-1.5-10B ...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)

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
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    print(f"  CoT: {extra['cot'][0][:200]}...")
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print(f"  minADE: {min_ade:.3f} m")
    if min_ade < 1.0:
        _ok(f"minADE {min_ade:.3f}m < 1.0m")
    else:
        _fail(f"minADE {min_ade:.3f}m >= 1.0m (stochastic, may re-run)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="HuggingFace cache dir used during download_dataset_1tb.py.",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=3,
        help="Number of clips to validate (default 3).",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Also run teacher inference (requires GPU + model weights).",
    )
    parser.add_argument(
        "--clip-ids",
        nargs="+",
        default=None,
        help="Specific clip IDs to test. If omitted, samples from downloaded chunks.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (set HF_HUB_OFFLINE=1). No network requests at all.",
    )
    parser.add_argument(
        "--no-online-init",
        action="store_true",
        help="Skip trying online PhysicalAIAVDatasetInterface init; go straight to cache-only build.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="b719eea7f0a63619ef51ec7f54178af0937ef050",
        help="Pinned dataset revision (must match what download_dataset_1tb.py used).",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import pandas as pd
    import huggingface_hub
    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    print(f"Cache dir: {args.cache_dir}")
    print()

    # --- Build avdi from local cache ---
    # PhysicalAIAVDatasetInterface needs network on init (downloads features.csv,
    # checks feature_presence.parquet). When the network is unavailable, manually
    # construct a minimal avdi with just enough state for local-only reads.
    revision = args.revision
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None
    avdi = None
    if not args.no_online_init:
        try:
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
                revision=revision,
                cache_dir=str(args.cache_dir),
            )
            clip_index = avdi.clip_index
            print("Initialised PhysicalAIAVDatasetInterface (online).")
        except Exception as e:  # noqa: BLE001
            print(f"Could not initialise PhysicalAIAVDatasetInterface: {type(e).__name__}")
            avdi = None

    if avdi is None:
        print("Building offline avdi from cache files...")
        import json

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface.__new__(
            physical_ai_av.PhysicalAIAVDatasetInterface
        )
        # Set base class (HfRepoInterface) fields needed for open_file / is_file_cached
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
        avdi.cache_dir = str(args.cache_dir)
        avdi.local_dir = None
        avdi.confirm_download_threshold_gb = float("inf")

        # Load metadata from cache
        clip_index_path = _find_cached_file(args.cache_dir, "clip_index.parquet")
        if clip_index_path is None:
            _fail("clip_index.parquet not found in cache. Re-run download with --include-metadata.")
            return 1
        clip_index = pd.read_parquet(clip_index_path)
        if "clip_id" in clip_index.columns:
            clip_index = clip_index.set_index("clip_id")

        features_csv_path = _find_cached_file(args.cache_dir, "features.csv")
        if features_csv_path is not None:
            features_df = pd.read_csv(features_csv_path, index_col="feature")
            features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
                json.loads, na_action="ignore"
            )
            avdi.features = physical_ai_av.dataset.Features(features_df)
        else:
            _fail("features.csv not found in cache. Re-run download with --include-metadata.")
            return 1

        avdi.clip_index = clip_index
    valid_clips_df = clip_index[clip_index["clip_is_valid"]]
    print(f"clip_index loaded: {len(valid_clips_df)} valid clips in index (full dataset)")

    # --- Filter to clips whose egomotion chunk is actually cached ---
    # egomotion is the smallest required feature; if its chunk is present,
    # the matching camera chunks should also be present (download is chunk-major).
    downloaded_chunks = set()
    for chunk_idx in sorted(valid_clips_df["chunk"].unique()):
        chunk_idx = int(chunk_idx)
        fname = f"labels/egomotion/egomotion.chunk_{chunk_idx:04d}.zip"
        if avdi.is_file_cached(fname):
            downloaded_chunks.add(chunk_idx)
    print(f"Detected {len(downloaded_chunks)} downloaded chunks in cache.")

    valid_clips_df = valid_clips_df[valid_clips_df["chunk"].isin(downloaded_chunks)]
    valid_clips = valid_clips_df.index.tolist()
    print(f"After filtering to downloaded chunks: {len(valid_clips)} clips available for testing")

    if not valid_clips:
        _fail("No clips available — either nothing downloaded yet, or metadata is missing.")
        return 1

    # --- Select clips to test ---
    if args.clip_ids:
        test_ids = args.clip_ids
    else:
        rng = np.random.default_rng(42)
        test_ids = rng.choice(valid_clips, size=min(args.num_clips, len(valid_clips)), replace=False).tolist()
    print(f"Testing {len(test_ids)} clips: {test_ids[:5]}{'...' if len(test_ids) > 5 else ''}")
    print()

    # --- Stage 1: data loading + shape validation ---
    all_pass = True
    for i, clip_id in enumerate(test_ids):
        print(f"--- clip {i + 1}/{len(test_ids)}: {clip_id} ---")
        try:
            data = load_physical_aiavdataset(clip_id, avdi=avdi, maybe_stream=False, t0_us=5_100_000)
        except FileNotFoundError as e:
            _fail(f"File not found (not in cache?): {e}")
            all_pass = False
            continue
        except Exception as e:  # noqa: BLE001
            _fail(f"Load error: {e}")
            all_pass = False
            continue

        ok = validate_clip_data(data, clip_id)
        if not ok:
            all_pass = False

        # --- Stage 2: optional inference ---
        if args.inference and i == 0:
            try:
                run_inference(data)
            except Exception as e:  # noqa: BLE001
                _fail(f"Inference error: {e}")
                all_pass = False
        print()

    # --- Summary ---
    print("=" * 50)
    if all_pass:
        print(f"ALL PASSED ({len(test_ids)} clips validated)")
    else:
        print(f"SOME CHECKS FAILED — see above for details")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
