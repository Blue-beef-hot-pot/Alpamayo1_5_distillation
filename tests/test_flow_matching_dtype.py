# SPDX-License-Identifier: Apache-2.0

import torch

from alpamayo1_5.diffusion.flow_matching import FlowMatching


def test_sample_passes_requested_dtype_to_x_and_t() -> None:
    diffusion = FlowMatching(x_dims=(2, 3), num_inference_steps=1)
    seen = {}

    def step_fn(*, x, t):
        seen["x_dtype"] = x.dtype
        seen["t_dtype"] = t.dtype
        return torch.zeros_like(x)

    out = diffusion.sample(
        batch_size=4,
        step_fn=step_fn,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert seen == {"x_dtype": torch.bfloat16, "t_dtype": torch.bfloat16}
    assert out.dtype is torch.bfloat16


def test_sample_defaults_to_float32() -> None:
    diffusion = FlowMatching(x_dims=(2,), num_inference_steps=1)

    def step_fn(*, x, t):
        assert x.dtype is torch.float32
        assert t.dtype is torch.float32
        return torch.zeros_like(x)

    out = diffusion.sample(batch_size=2, step_fn=step_fn)

    assert out.dtype is torch.float32


def test_sample_return_all_steps_returns_requested_dtype_time_steps() -> None:
    diffusion = FlowMatching(x_dims=(2,), num_inference_steps=2)

    def step_fn(*, x, t):
        return torch.zeros_like(x)

    all_steps, time_steps = diffusion.sample(
        batch_size=1,
        step_fn=step_fn,
        dtype=torch.bfloat16,
        return_all_steps=True,
    )

    assert all_steps.dtype is torch.bfloat16
    assert time_steps.dtype is torch.bfloat16
