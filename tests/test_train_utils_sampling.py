# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf

from alpamayo1_5_distill import train_utils
from alpamayo1_5_distill.train_utils import _sample_t0s_from_time_range, resolve_clip_samples


class _FakeEgomotion:
    def __init__(self, t_min: int, t_max: int) -> None:
        self.time_range = (t_min, t_max)


class _FakeFeatures:
    class LABELS:
        EGOMOTION = "egomotion"


class _FakeAvdi:
    features = _FakeFeatures()

    def __init__(self) -> None:
        self.ranges = {
            "clip-a": (0, 10_000_000),
            "clip-b": (0, 9_000_000),
        }

    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        assert feature == "egomotion"
        return _FakeEgomotion(*self.ranges[clip_id])


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
