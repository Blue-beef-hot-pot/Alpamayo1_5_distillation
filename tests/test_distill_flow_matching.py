# SPDX-License-Identifier: Apache-2.0

import importlib

import torch

from alpamayo1_5.diffusion.flow_matching import FlowMatching

student_forward_module = importlib.import_module("alpamayo1_5_distill.student_forward")


def test_original_flow_matching_sample_remains_no_grad() -> None:
    diffusion = FlowMatching(x_dims=(2,), num_inference_steps=1)
    scale = torch.nn.Parameter(torch.tensor(2.0))

    def step_fn(*, x, t):
        return x * scale + t

    out = diffusion.sample(batch_size=3, step_fn=step_fn, device=torch.device("cpu"))

    assert not out.requires_grad


def test_differentiable_flow_matching_sample_preserves_step_fn_gradients() -> None:
    diffusion = FlowMatching(x_dims=(2,), num_inference_steps=1)
    scale = torch.nn.Parameter(torch.tensor(2.0))

    def step_fn(*, x, t):
        return x * scale + t

    out = student_forward_module.differentiable_flow_matching_sample(
        diffusion,
        batch_size=3,
        step_fn=step_fn,
        device=torch.device("cpu"),
    )
    loss = out.sum()

    assert loss.requires_grad
    loss.backward()
    assert scale.grad is not None
