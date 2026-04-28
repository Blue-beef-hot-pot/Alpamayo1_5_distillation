# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and cache teacher soft labels for offline distillation.

Runs the teacher model on the dataset and saves VLM logits, Expert hidden
states, and sampled trajectories to disk. The cached data can then be loaded
directly during training, avoiding the cost of running the teacher every epoch.

Usage:
    python scripts/generate_teacher_data.py --config-name=distill

Output:
    outputs/teacher_data/<clip_id>.pt — tensors for each clip
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5_distill.teacher import load_teacher, teacher_forward

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="distill", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Generate teacher soft labels for all clips."""
    device = "cuda"
    output_dir = Path("outputs/teacher_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher
    teacher = load_teacher(
        model_name=cfg.teacher.model_name,
        device=device,
        dtype=getattr(torch, cfg.teacher.dtype),
    )
    processor = helper.get_processor(teacher.tokenizer)

    # For now, process the example clip
    # In production, iterate over all clips in the dataset
    clip_ids = ["030c760c-ae38-49aa-9ad8-f5650a545d26"]

    for clip_id in clip_ids:
        out_path = output_dir / f"{clip_id}.pt"
        if out_path.exists():
            logger.info("Skipping existing: %s", out_path)
            continue

        logger.info("Processing clip: %s", clip_id)
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

        teacher_out = teacher_forward(
            teacher,
            model_inputs,
            top_p=cfg.teacher.top_p,
            temperature=cfg.teacher.temperature,
            num_traj_samples=cfg.teacher.num_traj_samples,
            max_generation_length=cfg.teacher.max_generation_length,
            collect_expert_hiddens=cfg.teacher.collect_expert_hiddens,
        )

        # Save to disk
        save_dict = {
            "pred_xyz": teacher_out.pred_xyz.cpu(),
            "pred_rot": teacher_out.pred_rot.cpu(),
            "sampled_traj": teacher_out.sampled_traj.cpu() if teacher_out.sampled_traj is not None else None,
            "num_expert_layers": teacher_out.num_expert_layers,
        }
        if teacher_out.vlm_logits is not None:
            save_dict["vlm_logits"] = teacher_out.vlm_logits.cpu()
        if teacher_out.expert_hiddens:
            save_dict["expert_hiddens"] = [h.cpu() for h in teacher_out.expert_hiddens]
        if teacher_out.cot is not None:
            save_dict["cot"] = teacher_out.cot

        torch.save(save_dict, out_path)
        logger.info("Saved: %s", out_path)

    logger.info("Teacher data generation complete.")


if __name__ == "__main__":
    main()
