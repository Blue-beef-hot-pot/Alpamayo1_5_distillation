# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
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

student_forward_module = importlib.import_module("alpamayo1_5_distill.student_forward")
distributed_module = importlib.import_module("alpamayo1_5_distill.distributed")
_pipeline_spec = importlib.util.spec_from_file_location(
    "train_distill_pipeline", Path(__file__).parents[1] / "scripts" / "train_distill_pipeline.py"
)
train_distill_pipeline = importlib.util.module_from_spec(_pipeline_spec)
assert _pipeline_spec.loader is not None
_pipeline_spec.loader.exec_module(train_distill_pipeline)


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


class _FakeVisualStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm = torch.nn.Module()
        self.vlm.model = torch.nn.Module()
        self.vlm.model.visual = torch.nn.Linear(2, 2)
        self.vlm.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


def test_freeze_student_visual_tower_only_freezes_visual_params() -> None:
    student = _FakeVisualStudent()

    distributed_module.freeze_student_visual_tower(student)

    assert not any(p.requires_grad for p in student.vlm.model.visual.parameters())
    assert all(p.requires_grad for p in student.vlm.model.layers.parameters())


def test_pipeline_config_uses_minimal_stable_losses() -> None:
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "distill_pipeline.yaml")

    assert cfg.teacher.num_traj_samples == 1
    assert cfg.loss.vlm_logits_weight == 0.0
    assert cfg.loss.vlm_hidden_weight == 0.0
    assert cfg.loss.expert_hidden_weight > 0
    assert cfg.loss.trajectory_l2_weight > 0


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
    assert student_cfg.traj_vocab_size > 3000
    assert student_cfg.expert_cfg["_attn_implementation"] == "sdpa"


def test_build_student_config_allows_expert_attention_override(monkeypatch) -> None:
    monkeypatch.setattr(ReasoningVLAConfig, "_initialize_vlm_config", lambda self: None)
    cfg = OmegaConf.create(
        {
            "student": {
                "vlm_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
                "expert_attn_implementation": "eager",
            },
            "teacher": {"model_name": "nvidia/Alpamayo-1.5-10B"},
            "loss": {
                "vlm_logits_weight": 1.0,
                "expert_hidden_weight": 0.5,
                "trajectory_l2_weight": 1.0,
            },
        }
    )

    student_cfg = build_student_config(cfg)

    assert student_cfg.expert_cfg["_attn_implementation"] == "eager"


def test_build_student_config_adds_full_alpamayo_special_tokens(monkeypatch) -> None:
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

    assert student_cfg.add_special_tokens is True


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


class _FakeTokenizer:
    def __init__(self, id_to_token: dict[int, str], token_to_id: dict[str, int]) -> None:
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id[token]


class _FakeEmbeddings:
    num_embeddings = 100


class _FakeVlm:
    def get_input_embeddings(self):
        return _FakeEmbeddings()


class _FakeStudent:
    tokenizer = _FakeTokenizer(
        id_to_token={},
        token_to_id={"<i0>": 50, "<|traj_future_start|>": 80},
    )
    vlm = _FakeVlm()


class _FakeTeacher:
    tokenizer = _FakeTokenizer(
        id_to_token={120: "<i0>", 130: "<|traj_future_start|>"},
        token_to_id={},
    )


def test_align_teacher_sequences_remaps_teacher_added_tokens_to_student_ids() -> None:
    sequences = torch.tensor([[10, 120, 130]])

    aligned = student_forward_module._align_teacher_sequences_to_student(
        sequences, _FakeStudent(), _FakeTeacher()
    )

    assert aligned.tolist() == [[10, 50, 80]]


class _FakeLargeStudent:
    tokenizer = _FakeTokenizer(
        id_to_token={},
        token_to_id={"<i3000>": 90},
    )
    vlm = _FakeVlm()


class _FakeLargeTeacher:
    tokenizer = _FakeTokenizer(
        id_to_token={154669: "<i3000>"},
        token_to_id={},
    )


def test_align_teacher_sequences_remaps_large_teacher_trajectory_token() -> None:
    sequences = torch.tensor([[10, 154669]])

    aligned = student_forward_module._align_teacher_sequences_to_student(
        sequences, _FakeLargeStudent(), _FakeLargeTeacher()
    )

    assert aligned.tolist() == [[10, 90]]


def test_align_teacher_output_sequences_for_student_before_dispatch() -> None:
    teacher_dict = {"sequences": torch.tensor([[10, 120, 130]])}

    train_distill_pipeline._align_teacher_output_for_student(
        teacher_dict, _FakeStudent(), _FakeTeacher()
    )

    assert teacher_dict["sequences"].tolist() == [[10, 50, 80]]


def test_validate_qwen_visual_inputs_rejects_mismatched_patch_counts() -> None:
    with pytest.raises(ValueError, match="pixel_values has 5 patches but image_grid_thw describes 6"):
        student_forward_module._validate_qwen_visual_inputs(
            {
                "pixel_values": torch.zeros(5, 3, 2, 14, 14),
                "image_grid_thw": torch.tensor([[1, 2, 3]]),
            }
        )


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
