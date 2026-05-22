# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download a size-capped subset of nvidia/PhysicalAI-Autonomous-Vehicles.

Downloads only the features used by `load_physical_aiavdataset`
(4 cameras + egomotion), going chunk-by-chunk until a user-specified
disk-budget (default 1 TB) is reached. Each chunk is one zip per feature
covering ~100 clips, so chunks form a natural download unit.

Usage:
    python scripts/download_dataset_1tb.py \
        --cache-dir /path/to/cache \
        --max-bytes 1e12

scripts/download_dataset_1tb.py — 按 chunk 顺序下载，达到 1TB 上限自动停止。

  用法

  # 1TB 下载到指定 cache 目录
  /home/winterwang/Alpamayo1_5_distillation/a1_5_venv/bin/python \
    scripts/download_dataset_1tb.py \
    --cache-dir ./.cache/ \
    --max-bytes 9e9 \
    --include-metadata

  # 先 dry-run 看看会下哪些
  ... --dry-run

  # 自定义下载量（例如 500 GB）
  ... --max-bytes 5e11

  # 从某个 chunk 续传
  ... --start-chunk 50


The HF cache layout under `--cache-dir` is the standard
`huggingface_hub` layout, so `physical_ai_av.PhysicalAIAVDatasetInterface(
cache_dir=...)` can read it back with `maybe_stream=False`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"

# Features actually consumed by src/alpamayo1_5/load_physical_aiavdataset.py
DEFAULT_FEATURES = [
    "camera/camera_cross_left_120fov",
    "camera/camera_front_wide_120fov",
    "camera/camera_cross_right_120fov",
    "camera/camera_front_tele_30fov",
    "labels/egomotion",
]


def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def list_feature_chunks(api: HfApi, revision: str, feature: str) -> list[tuple[str, int]]:
    """Return [(path, size), ...] for a feature, sorted by chunk index."""
    entries = api.list_repo_tree(
        repo_id=REPO_ID, repo_type=REPO_TYPE, revision=revision, path_in_repo=feature
    )
    files = []
    for entry in entries:
        size = getattr(entry, "size", None)
        if size is None:
            continue
        files.append((entry.path, size))
    files.sort(key=lambda x: x[0])
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="HuggingFace cache directory to write into (passed to hf_hub_download).",
    )
    parser.add_argument(
        "--max-bytes",
        type=float,
        default=1e12,
        help="Disk-budget in bytes. Stops before exceeding this. Default 1 TB (1e12).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="b719eea7f0a63619ef51ec7f54178af0937ef050",
        help="Dataset revision (commit SHA) to pin against.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature dirs to download (paths within the repo).",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=0,
        help="First chunk index to download (inclusive). Default 0.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Also download top-level metadata (clip_index.parquet, features.csv, README, LICENSE).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded; do not download.",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    # Build a chunk-major plan: chunk_idx -> [(file_path, size), ...]
    # We want to download all features for chunk 0, then all for chunk 1, etc.,
    # so a partial download leaves whole chunks usable.
    print(f"Listing {len(args.features)} features on revision {args.revision[:8]}...")
    per_feature: dict[str, list[tuple[str, int]]] = {}
    for feat in args.features:
        files = list_feature_chunks(api, args.revision, feat)
        per_feature[feat] = files
        total = sum(s for _, s in files)
        print(f"  {feat}: {len(files)} chunks, {_human(total)}")

    num_chunks = max(len(v) for v in per_feature.values())
    print(f"\nTotal chunks per feature: {num_chunks}")
    print(f"Budget: {_human(args.max_bytes)}, starting at chunk {args.start_chunk}")
    print()

    metadata_files: list[tuple[str, int]] = []
    if args.include_metadata:
        root_entries = api.list_repo_tree(
            repo_id=REPO_ID, repo_type=REPO_TYPE, revision=args.revision, path_in_repo=""
        )
        for entry in root_entries:
            size = getattr(entry, "size", None)
            if size is None:
                continue
            metadata_files.append((entry.path, size))

    downloaded_bytes = 0
    downloaded_files = 0
    skipped_files = 0

    # Metadata first.
    for path, size in metadata_files:
        if downloaded_bytes + size > args.max_bytes:
            print(f"Budget exhausted before metadata file {path}; stopping.")
            return 0
        print(f"[meta] {path} ({_human(size)})")
        if not args.dry_run:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=args.revision,
                filename=path,
                cache_dir=str(args.cache_dir),
            )
        downloaded_bytes += size
        downloaded_files += 1

    # Iterate chunks in order; within a chunk, all features.
    for chunk_idx in range(args.start_chunk, num_chunks):
        chunk_files: list[tuple[str, int]] = []
        for feat in args.features:
            files = per_feature[feat]
            if chunk_idx >= len(files):
                continue
            chunk_files.append(files[chunk_idx])
        chunk_size = sum(s for _, s in chunk_files)

        if downloaded_bytes + chunk_size > args.max_bytes:
            remaining = args.max_bytes - downloaded_bytes
            print(
                f"\nStopping: chunk {chunk_idx} is {_human(chunk_size)} "
                f"but only {_human(remaining)} of budget remains."
            )
            break

        print(
            f"\n=== chunk {chunk_idx}/{num_chunks - 1} "
            f"({_human(chunk_size)}, cumulative {_human(downloaded_bytes + chunk_size)}) ==="
        )
        for path, size in chunk_files:
            print(f"  -> {path} ({_human(size)})")
            if not args.dry_run:
                try:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type=REPO_TYPE,
                        revision=args.revision,
                        filename=path,
                        cache_dir=str(args.cache_dir),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  !! failed: {e}", file=sys.stderr)
                    skipped_files += 1
                    continue
            downloaded_bytes += size
            downloaded_files += 1

    print()
    print(f"Done. Downloaded {downloaded_files} files ({_human(downloaded_bytes)}).")
    if skipped_files:
        print(f"Skipped {skipped_files} files due to errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
