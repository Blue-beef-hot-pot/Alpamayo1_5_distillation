# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a distilled Alpamayo 1.5 student model.

Computes trajectory prediction metrics (minADE, minFDE, miss rate) on
test clips and optionally compares against the teacher model.

Usage:
    python scripts/eval_student.py --config-name=eval
    python scripts/eval_student.py --config-name=eval model.checkpoint_path=outputs/distilled/best
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5_distill.model import Alpamayo1_5_Distilled

logger = logging.getLogger(__name__)


def compute_metrics(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
) -> dict[str, float]:
    """Compute trajectory prediction metrics.

    Args:
        pred_xyz: Predicted xyz [B, n_sets, n_samples, T, 3].
        gt_xyz: Ground truth xyz [B, 1, 1, T, 3].

    Returns:
        Dict of metric name to value.
    """
    pred_xy = pred_xyz[..., :2].cpu().numpy()
    gt_xy = gt_xyz[..., :2].cpu().numpy()

    B, n_sets, n_samples, T, _ = pred_xy.shape
    pred_xy = pred_xy.reshape(B, n_sets * n_samples, T, 2)
    gt_xy = gt_xy.reshape(B, 1, T, 2)

    # ADE per sample: mean L2 over time
    ade = np.linalg.norm(pred_xy - gt_xy, axis=-1).mean(axis=-1)  # [B, N]
    min_ade = ade.min(axis=-1)  # [B]

    # FDE per sample: L2 at last timestep
    fde = np.linalg.norm(pred_xy[..., -1, :] - gt_xy[..., -1, :], axis=-1)  # [B, N]
    min_fde = fde.min(axis=-1)  # [B]

    miss_rate = (min_fde > 2.0).mean()

    return {
        "minADE": float(min_ade.mean()),
        "minFDE": float(min_fde.mean()),
        "miss_rate": float(miss_rate),
    }


@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run student model evaluation."""
    device = "cuda"

    # Load student model
    logger.info("Loading student model from: %s", cfg.model.checkpoint_path)
    student = Alpamayo1_5_Distilled.from_pretrained(
        cfg.model.checkpoint_path,
        dtype=getattr(torch, cfg.model.dtype),
    ).to(device)
    student.eval()

    processor = helper.get_processor(student.tokenizer)

    # Test clips — use the default example clip for now
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    logger.info("Evaluating on clip: %s", clip_id)

    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=getattr(torch, cfg.model.dtype)):
        with torch.no_grad():
            pred_xyz, pred_rot, extra = student.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=cfg.model.top_p,
                temperature=cfg.model.temperature,
                num_traj_samples=cfg.model.num_traj_samples,
                max_generation_length=cfg.model.max_generation_length,
                return_extra=True,
            )

    gt_xyz = data["ego_future_xyz"].cpu()
    gt_xyz = gt_xyz.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T, 3]

    metrics = compute_metrics(pred_xyz.cpu(), gt_xyz)

    logger.info("=" * 50)
    logger.info("Evaluation Results (clip: %s)", clip_id)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)
    logger.info("=" * 50)

    if "cot" in extra:
        logger.info("Chain-of-Causation (first sample):\n%s", extra["cot"][0][0])


if __name__ == "__main__":
    main()
