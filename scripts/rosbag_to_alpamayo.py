#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROS1 Bag to Alpamayo format converter.

Converts ROS1 bag files containing 4 cameras and egomotion data
to the Alpamayo training format.

Usage:
    python scripts/rosbag_to_alpamayo.py --input /path/to/rosbag --output /path/to/output
    python scripts/rosbag_to_alpamayo.py --input /path/to/rosbag/*.bag --output /path/to/output
"""

import argparse
import logging
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Camera configuration (matching Alpamayo format)
CAMERA_CONFIG = {
    "front": {"index": 1, "topic": "/camera/front/compressed"},
    "frontleft": {"index": 0, "topic": "/camera/frontleft/compressed"},
    "frontright": {"index": 2, "topic": "/camera/frontright/compressed"},
    "back": {"index": 6, "topic": "/camera/back/compressed"},
}


def extract_egomotion_from_bag(bag_path: Path) -> pd.DataFrame:
    """Extract egomotion data from ROS bag.

    Args:
        bag_path: Path to .bag file

    Returns:
        DataFrame with columns: timestamp, x, y, z, qx, qy, qz, qw
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)

    records = []
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == "/LocalPose":
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                # Convert from cm to meters
                x = msg.dr_x / 100.0
                y = msg.dr_y / 100.0
                z = msg.dr_z / 100.0
                # Convert heading from 0.01 degree to radians
                heading = np.radians(msg.dr_heading / 100.0)
                # Create quaternion from heading (rotation around z-axis)
                qx, qy, qz, qw = Rotation.from_euler("z", heading).as_quat()

                records.append({
                    "timestamp": timestamp,  # nanoseconds
                    "x": x,
                    "y": y,
                    "z": z,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "qw": qw,
                })

    df = pd.DataFrame(records)
    # Convert timestamp from nanoseconds to microseconds
    df["timestamp"] = df["timestamp"] // 1000
    return df


def extract_camera_frames_from_bag(
    bag_path: Path, camera_name: str
) -> tuple[list[np.ndarray], list[int]]:
    """Extract camera frames from ROS bag.

    Args:
        bag_path: Path to .bag file
        camera_name: Camera name (front, frontleft, frontright, back)

    Returns:
        Tuple of (frames, timestamps_us)
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    topic = CAMERA_CONFIG[camera_name]["topic"]

    frames = []
    timestamps_us = []

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                # Decode JPEG image
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(img)
                    timestamps_us.append(timestamp // 1000)  # ns to us

    return frames, timestamps_us


def interpolate_trajectory(
    egomotion_df: pd.DataFrame,
    t0_us: int,
    history_us: int = 1_500_000,
    future_us: int = 6_400_000,
    history_points: int = 16,
    future_points: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate trajectory to standard Alpamayo grid.

    Args:
        egomotion_df: DataFrame with egomotion data
        t0_us: Current timestamp in microseconds
        history_us: History duration in microseconds
        future_us: Future duration in microseconds
        history_points: Number of history points
        future_points: Number of future points

    Returns:
        Tuple of (ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot)
    """
    # Convert timestamps to seconds relative to t0
    timestamps_s = (egomotion_df["timestamp"].values - t0_us) / 1e6
    x = egomotion_df["x"].values
    y = egomotion_df["y"].values
    z = egomotion_df["z"].values
    qx = egomotion_df["qx"].values
    qy = egomotion_df["qy"].values
    qz = egomotion_df["qz"].values
    qw = egomotion_df["qw"].values

    # Create time grids
    history_times = np.linspace(-history_us / 1e6, 0, history_points)
    future_times = np.linspace(0.1, future_us / 1e6, future_points)

    # Interpolate positions
    x_interp = np.interp(history_times, timestamps_s, x)
    y_interp = np.interp(history_times, timestamps_s, y)
    z_interp = np.interp(history_times, timestamps_s, z)

    # Interpolate quaternions (using SLERP)
    from scipy.spatial.transform import Slerp

    rotations = Rotation.from_quat(np.column_stack([qx, qy, qz, qw]))
    slerp = Slerp(timestamps_s, rotations)

    # History
    history_rotations = slerp(history_times)
    history_xyz = np.column_stack([x_interp, y_interp, z_interp])

    # Future
    x_future = np.interp(future_times, timestamps_s, x)
    y_future = np.interp(future_times, timestamps_s, y)
    z_future = np.interp(future_times, timestamps_s, z)
    future_rotations = slerp(future_times)
    future_xyz = np.column_stack([x_future, y_future, z_future])

    # Convert to rotation matrices
    history_rot = history_rotations.as_matrix()  # (16, 3, 3)
    future_rot = future_rotations.as_matrix()  # (64, 3, 3)

    # Transform to local t0 frame
    t0_idx = np.argmin(np.abs(timestamps_s))
    t0_xyz = np.array([x[t0_idx], y[t0_idx], z[t0_idx]])
    t0_rot = Rotation.from_quat([qx[t0_idx], qy[t0_idx], qz[t0_idx], qw[t0_idx]])
    t0_rot_inv = t0_rot.inv()

    # Transform positions
    history_xyz_local = t0_rot_inv.apply(history_xyz - t0_xyz)
    future_xyz_local = t0_rot_inv.apply(future_xyz - t0_xyz)

    # Transform rotations
    history_rot_local = np.matmul(t0_rot_inv.as_matrix()[np.newaxis, :, :], history_rot)
    future_rot_local = np.matmul(t0_rot_inv.as_matrix()[np.newaxis, :, :], future_rot)

    # Reshape to match Alpamayo format
    ego_history_xyz = history_xyz_local.reshape(1, 1, 16, 3)
    ego_history_rot = history_rot_local.reshape(1, 1, 16, 3, 3)
    ego_future_xyz = future_xyz_local.reshape(1, 1, 64, 3)
    ego_future_rot = future_rot_local.reshape(1, 1, 64, 3, 3)

    return ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot


def get_camera_frames_at_t0(
    frames: list[np.ndarray],
    timestamps_us: list[int],
    t0_us: int,
    num_frames: int = 4,
    fps: int = 10,
) -> np.ndarray:
    """Get camera frames around t0.

    Args:
        frames: List of camera frames
        timestamps_us: List of timestamps in microseconds
        t0_us: Current timestamp in microseconds
        num_frames: Number of frames to return
        fps: Camera FPS

    Returns:
        Array of shape (num_frames, H, W, 3)
    """
    frame_interval_us = 1_000_000 // fps

    # Get frames at t0-0.3s, t0-0.2s, t0-0.1s, t0
    target_times = [t0_us - i * frame_interval_us for i in range(num_frames - 1, -1, -1)]

    selected_frames = []
    timestamps_arr = np.array(timestamps_us)

    for target_time in target_times:
        # Find closest frame
        idx = np.argmin(np.abs(timestamps_arr - target_time))
        selected_frames.append(frames[idx])

    return np.array(selected_frames)


def process_single_bag(
    bag_path: Path,
    output_dir: Path,
    sample_interval_us: int = 1_000_000,
) -> list[dict[str, Any]]:
    """Process a single ROS bag file.

    Args:
        bag_path: Path to .bag file
        output_dir: Output directory
        sample_interval_us: Sample interval in microseconds

    Returns:
        List of sample metadata
    """
    logger.info("Processing: %s", bag_path.name)

    # Generate clip ID
    clip_id = str(uuid.uuid4())
    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Extract egomotion
    logger.info("  Extracting egomotion...")
    egomotion_df = extract_egomotion_from_bag(bag_path)

    if len(egomotion_df) == 0:
        logger.warning("  No egomotion data found, skipping")
        return []

    # Save egomotion
    egomotion_df.to_parquet(clip_dir / "egomotion.parquet", index=False)

    # Extract camera frames
    camera_data = {}
    for camera_name in CAMERA_CONFIG:
        logger.info("  Extracting camera: %s", camera_name)
        frames, timestamps_us = extract_camera_frames_from_bag(bag_path, camera_name)
        if len(frames) > 0:
            camera_data[camera_name] = {
                "frames": frames,
                "timestamps_us": timestamps_us,
            }
            # Save camera timestamps
            pd.DataFrame({"timestamp": timestamps_us}).to_parquet(
                clip_dir / f"{camera_name}_timestamps.parquet", index=False
            )

    if not camera_data:
        logger.warning("  No camera data found, skipping")
        return []

    # Get valid time range
    t_min_us = egomotion_df["timestamp"].min()
    t_max_us = egomotion_df["timestamp"].max()

    # Generate samples
    samples = []
    t0_us = t_min_us + 1_500_000  # Start after history_us

    while t0_us <= t_max_us - 6_400_000:  # Ensure enough future
        # Interpolate trajectory
        try:
            ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot = (
                interpolate_trajectory(egomotion_df, t0_us)
            )
        except Exception as e:
            logger.warning("  Failed to interpolate at t0_us=%d: %s", t0_us, e)
            t0_us += sample_interval_us
            continue

        # Get camera frames
        image_frames = []
        for camera_name in ["frontleft", "front", "frontright", "back"]:
            if camera_name in camera_data:
                frames = get_camera_frames_at_t0(
                    camera_data[camera_name]["frames"],
                    camera_data[camera_name]["timestamps_us"],
                    t0_us,
                )
                image_frames.append(frames)
            else:
                # Fill with zeros if camera missing
                image_frames.append(np.zeros((4, 1080, 1920, 3), dtype=np.uint8))

        image_frames = np.array(image_frames)  # (4, 4, H, W, 3)

        # Save sample
        sample = {
            "clip_id": clip_id,
            "t0_us": t0_us,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
            "image_frames": image_frames,
            "camera_indices": np.array([0, 1, 2, 6]),
        }

        # Save as numpy file
        sample_file = clip_dir / f"sample_{t0_us}.npz"
        np.savez_compressed(sample_file, **sample)

        samples.append({
            "clip_id": clip_id,
            "t0_us": t0_us,
            "sample_file": str(sample_file),
        })

        t0_us += sample_interval_us

    logger.info("  Generated %d samples", len(samples))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert ROS bag to Alpamayo format")
    parser.add_argument("--input", type=str, required=True, help="Input bag file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--sample-interval", type=int, default=1_000_000,
                        help="Sample interval in microseconds (default: 1s)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find bag files
    if input_path.is_file():
        bag_files = [input_path]
    else:
        bag_files = sorted(input_path.glob("*.bag"))

    logger.info("Found %d bag files", len(bag_files))

    # Process each bag
    all_samples = []
    for bag_file in bag_files:
        samples = process_single_bag(bag_file, output_dir, args.sample_interval)
        all_samples.extend(samples)

    # Save metadata
    if all_samples:
        metadata_df = pd.DataFrame(all_samples)
        metadata_df.to_csv(output_dir / "samples.csv", index=False)
        logger.info("Total samples: %d", len(all_samples))
        logger.info("Metadata saved to: %s", output_dir / "samples.csv")

    logger.info("Done!")


if __name__ == "__main__":
    main()
