# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distilled Alpamayo 1.5 model with Qwen3-VL-2B backbone.

This model inherits all inference logic from Alpamayo1_5. The only differences
are in configuration:
- VLM: Qwen3-VL-2B (24 layers, hidden_size=1536) instead of 8B
- Expert: derived from 2B VLM text_config (same 24 layers, hidden=1536)
- Flow Matching: 4 inference steps instead of 10

No method overrides are needed because Alpamayo1_5 is fully dimension-agnostic:
- Expert is created from copy.deepcopy(self.vlm.config.text_config)
- action_in_proj out_dim and action_out_proj in_features are set dynamically
  from expert_config.hidden_size at construction time
- KV Cache dimensions match naturally (same VLM and Expert architecture)
"""

from transformers import AutoConfig, AutoModel

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5_distill.config import Alpamayo1_5_DistilledConfig


class Alpamayo1_5_Distilled(Alpamayo1_5):
    """Distilled Alpamayo 1.5 with 2B VLM and 4-step flow matching."""

    config_class = Alpamayo1_5_DistilledConfig


AutoConfig.register("alpamayo1_5_distilled", Alpamayo1_5_DistilledConfig)
AutoModel.register(Alpamayo1_5_DistilledConfig, Alpamayo1_5_Distilled)
