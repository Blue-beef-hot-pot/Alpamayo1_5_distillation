# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from omegaconf import OmegaConf

from alpamayo1_5.models.base_model import ReasoningVLAConfig
from alpamayo1_5_distill import train_utils
from alpamayo1_5_distill.train_utils import (
    _sample_t0s_from_time_range,
    build_student_config,
    repeat_visual_inputs,
    resolve_clip_samples,
)


class _FakeEgomotion:
    def __init__(self, t_min: int, t_max: int) -> None:
        self.time_range = (t_min, t_max)


class _FakeCamera:
    def __init__(self, t_min: int, t_max: int) -> None:
        self.timestamps = np.array([t_min, t_max], dtype=np.int64)


class _FakeFeatures:
    class LABELS:
        EGOMOTION = "egomotion"

    class CAMERA:
        CAMERA_CROSS_LEFT_120FOV = "camera_cross_left_120fov"
        CAMERA_FRONT_WIDE_120FOV = "camera_front_wide_120fov"
        CAMERA_CROSS_RIGHT_120FOV = "camera_cross_right_120fov"
        CAMERA_FRONT_TELE_30FOV = "camera_front_tele_30fov"


class _FakeAvdi:
    features = _FakeFeatures()

    def __init__(self) -> None:
        self.ranges = {
            "clip-a": (0, 10_000_000),
            "clip-b": (0, 9_000_000),
        }
        self.camera_ranges = {
            clip_id: {
                self.features.CAMERA.CAMERA_CROSS_LEFT_120FOV: time_range,
                self.features.CAMERA.CAMERA_FRONT_WIDE_120FOV: time_range,
                self.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV: time_range,
                self.features.CAMERA.CAMERA_FRONT_TELE_30FOV: time_range,
            }
            for clip_id, time_range in self.ranges.items()
        }

    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        if feature == self.features.LABELS.EGOMOTION:
            return _FakeEgomotion(*self.ranges[clip_id])
        return _FakeCamera(*self.camera_ranges[clip_id][feature])


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "cache_dir": "./.cache/",
                "revision": "rev",
                "clip_ids": ["clip-a", "clip-b"],
                "shuffle": True,
                "seed": 42,
                "sample_step_us": 1_000_000,
                "history_us": 1_500_000,
                "future_us": 6_400_000,
            }
        }
    )


def test_build_student_config_sets_required_fields(monkeypatch) -> None:
    monkeypatch.setattr(ReasoningVLAConfig, "_initialize_vlm_config", lambda self: None)
    cfg = OmegaConf.create(
        {
            "student": {"vlm_name_or_path": "Qwen/Qwen3-VL-2B-Instruct"},
            "teacher": {"model_name": "nvidia/Alpamayo-1.5-10B"},
            "loss": {
                "vlm_logits_weight": 1.0,
                "expert_hidden_weight": 0.5,
                "trajectory_l2_weight": 1.0,
            },
        }
    )

    student_cfg = build_student_config(cfg)

    assert student_cfg.action_space_cfg is not None
    assert student_cfg.action_out_proj_cfg is not None
    assert student_cfg.traj_tokenizer_cfg is not None
    assert student_cfg.hist_traj_tokenizer_cfg is not None
    assert (
        student_cfg.traj_tokenizer_cfg["_target_"]
        == "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer"
    )
    assert (
        student_cfg.hist_traj_tokenizer_cfg["_target_"]
        == "alpamayo1_5.models.delta_tokenizer.DeltaTrajectoryTokenizer"
    )
    assert student_cfg.traj_tokenizer_cfg["num_bins"] == student_cfg.traj_vocab_size
    assert student_cfg.hist_traj_tokenizer_cfg["num_bins"] == student_cfg.traj_vocab_size


def test_sample_t0s_from_time_range_returns_inclusive_1s_steps() -> None:
    assert _sample_t0s_from_time_range(
        t_min_us=0,
        t_max_us=10_000_000,
        history_us=1_500_000,
        future_us=6_400_000,
        step_us=1_000_000,
    ) == [1_700_000, 2_700_000]


def test_sample_t0s_from_time_range_aligns_start_to_100ms_grid() -> None:
    assert _sample_t0s_from_time_range(
        t_min_us=123_456,
        t_max_us=10_000_000,
        history_us=1_500_000,
        future_us=6_400_000,
        step_us=1_000_000,
    ) == [1_700_000, 2_700_000]


def test_sample_t0s_from_time_range_keeps_exact_boundary() -> None:
    assert _sample_t0s_from_time_range(
        t_min_us=0,
        t_max_us=8_100_000,
        history_us=1_500_000,
        future_us=6_400_000,
        step_us=1_000_000,
    ) == [1_700_000]


def test_sample_t0s_from_time_range_skips_too_short_clip() -> None:
    assert _sample_t0s_from_time_range(
        t_min_us=0,
        t_max_us=7_800_000,
        history_us=1_500_000,
        future_us=6_400_000,
        step_us=1_000_000,
    ) == []


def test_repeat_visual_inputs_repeats_flattened_qwen_pixel_values_by_grid_patch_counts() -> None:
    image_grid_thw = torch.tensor([[1, 2, 3], [2, 1, 2]])
    pixel_values = torch.arange(10).view(10, 1)

    out = repeat_visual_inputs(
        {"image_grid_thw": image_grid_thw, "pixel_values": pixel_values},
        batch_size=1,
        num_traj_samples=3,
    )

    assert out["image_grid_thw"].tolist() == [[1, 2, 3]] * 3 + [[2, 1, 2]] * 3
    expected = torch.cat(
        [
            pixel_values[:6],
            pixel_values[:6],
            pixel_values[:6],
            pixel_values[6:],
            pixel_values[6:],
            pixel_values[6:],
        ],
        dim=0,
    )
    assert torch.equal(out["pixel_values"], expected)


def test_repeat_visual_inputs_repeats_list_pixel_values_preserving_per_image_order() -> None:
    image_grid_thw = torch.tensor([[1, 1, 2], [1, 1, 3]])
    pixels = [torch.tensor([[1], [2]]), torch.tensor([[3], [4], [5]])]

    out = repeat_visual_inputs(
        {"image_grid_thw": image_grid_thw, "pixel_values": pixels},
        batch_size=1,
        num_traj_samples=2,
    )

    assert out["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 2], [1, 1, 3], [1, 1, 3]]
    assert [p.tolist() for p in out["pixel_values"]] == [
        pixels[0].tolist(),
        pixels[0].tolist(),
        pixels[1].tolist(),
        pixels[1].tolist(),
    ]


def test_resolve_clip_samples_shuffles_samples_by_epoch(monkeypatch) -> None:
    cfg = _cfg()
    monkeypatch.setattr(train_utils, "_build_avdi", lambda cache_dir, revision: _FakeAvdi())

    epoch0 = resolve_clip_samples(cfg, epoch=0)
    epoch0_again = resolve_clip_samples(cfg, epoch=0)
    epoch1 = resolve_clip_samples(cfg, epoch=1)

    assert sorted(epoch0) == sorted(
        [
            ("clip-a", 1_700_000),
            ("clip-a", 2_700_000),
            ("clip-b", 1_700_000),
        ]
    )
    assert epoch0 == epoch0_again
    assert epoch0 != epoch1


def test_resolve_clip_samples_clamps_to_camera_max_range(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": "./.cache/",
                "revision": "rev",
                "clip_ids": ["clip-a"],
                "shuffle": False,
                "seed": 42,
                "sample_step_us": 1_000_000,
                "history_us": 0,
                "future_us": 0,
            }
        }
    )
    avdi = _FakeAvdi()
    avdi.ranges["clip-a"] = (0, 30_000_000)
    for camera_feature in avdi.camera_ranges["clip-a"]:
        avdi.camera_ranges["clip-a"][camera_feature] = (0, 20_000_000)
    monkeypatch.setattr(train_utils, "_build_avdi", lambda cache_dir, revision: avdi)

    samples = resolve_clip_samples(cfg, epoch=0)

    assert samples
    assert max(t0_us for _, t0_us in samples) <= 20_000_000


def test_resolve_clip_samples_accounts_for_camera_image_history(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "cache_dir": "./.cache/",
                "revision": "rev",
                "clip_ids": ["clip-a"],
                "shuffle": False,
                "seed": 42,
                "sample_step_us": 100_000,
                "history_us": 0,
                "future_us": 0,
            }
        }
    )
    avdi = _FakeAvdi()
    avdi.ranges["clip-a"] = (0, 5_000_000)
    for camera_feature in avdi.camera_ranges["clip-a"]:
        avdi.camera_ranges["clip-a"][camera_feature] = (1_000_000, 5_000_000)
    monkeypatch.setattr(train_utils, "_build_avdi", lambda cache_dir, revision: avdi)

    samples = resolve_clip_samples(cfg, epoch=0)

    assert samples[0] == ("clip-a", 1_300_000)
