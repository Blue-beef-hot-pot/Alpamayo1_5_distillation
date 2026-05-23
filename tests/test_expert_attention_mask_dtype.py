# SPDX-License-Identifier: Apache-2.0

import torch

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def _mask_kwargs() -> dict:
    return {
        "offset": torch.tensor([2], dtype=torch.long),
        "rope_deltas": torch.zeros((1, 1), dtype=torch.long),
        "kv_cache_seq_len": 3,
        "n_diffusion_tokens": 2,
        "b_star": 1,
        "device": torch.device("cpu"),
        "prefix_mask": torch.tensor([[1, 0, 1]], dtype=torch.long),
    }


def test_expert_attention_mask_defaults_to_float32() -> None:
    _, attention_mask = Alpamayo1_5._build_expert_pos_ids_and_attn_mask(**_mask_kwargs())

    assert attention_mask.dtype is torch.float32
    assert attention_mask.min() == torch.finfo(torch.float32).min


def test_expert_attention_mask_accepts_bfloat16() -> None:
    _, attention_mask = Alpamayo1_5._build_expert_pos_ids_and_attn_mask(
        **_mask_kwargs(), dtype=torch.bfloat16
    )

    assert attention_mask.dtype is torch.bfloat16
    assert attention_mask.min() == torch.finfo(torch.bfloat16).min
