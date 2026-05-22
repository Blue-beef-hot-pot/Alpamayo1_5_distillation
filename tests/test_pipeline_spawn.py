# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.train_distill_pipeline import infer_pipeline_process_count, infer_student_ranks


def test_infer_student_ranks_uses_all_non_teacher_ranks() -> None:
    assert infer_student_ranks(world_size=4, teacher_rank=0) == [1, 2, 3]
    assert infer_student_ranks(world_size=4, teacher_rank=2) == [0, 1, 3]


def test_infer_student_ranks_rejects_invalid_world_size() -> None:
    with pytest.raises(ValueError, match="at least 2 processes"):
        infer_student_ranks(world_size=1, teacher_rank=0)


def test_infer_student_ranks_rejects_invalid_teacher_rank() -> None:
    with pytest.raises(ValueError, match="teacher_rank"):
        infer_student_ranks(world_size=4, teacher_rank=4)


def test_infer_pipeline_process_count_defaults_to_cuda_device_count() -> None:
    assert infer_pipeline_process_count(configured=None, cuda_device_count=8) == 8
    assert infer_pipeline_process_count(configured=3, cuda_device_count=8) == 3


def test_infer_pipeline_process_count_requires_two_processes() -> None:
    with pytest.raises(ValueError, match="at least 2 processes"):
        infer_pipeline_process_count(configured=None, cuda_device_count=1)
    with pytest.raises(ValueError, match="at least 2 processes"):
        infer_pipeline_process_count(configured=1, cuda_device_count=8)
