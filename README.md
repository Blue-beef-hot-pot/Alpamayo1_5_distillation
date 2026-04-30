# Alpamayo 1.5 Distillation

This repository contains a lightweight distillation workflow for Alpamayo 1.5.
It keeps the original Alpamayo 1.5 inference implementation under `src/alpamayo1_5`
and adds a student model plus teacher/student distillation utilities under
`src/alpamayo1_5_distill`.

## What Is Included

- Original Alpamayo 1.5 model code for VLM rollout, Expert denoising, action
  spaces, diffusion, geometry, and data loading.
- A distilled student model that reuses the original inference pipeline while
  replacing the VLM backbone with `Qwen/Qwen3-VL-2B-Instruct`.
- Teacher wrappers that run `nvidia/Alpamayo-1.5-10B` and collect soft labels.
- Student teacher-forcing forward logic for differentiable VLM distillation.
- Distillation losses for VLM logits, Expert hidden states, and action
  trajectories.
- Hydra configs and scripts for training, teacher-data generation, and
  evaluation.

## Repository Layout

```text
.
├── configs/
│   ├── distill.yaml              # Distillation training config
│   └── eval.yaml                 # Student evaluation config
├── notebooks/                    # Original Alpamayo 1.5 example notebooks
├── scripts/
│   ├── train_distill.py          # Online teacher-student distillation
│   ├── generate_teacher_data.py  # Prototype teacher soft-label cache script
│   └── eval_student.py           # Prototype student trajectory evaluation
├── src/
│   ├── alpamayo1_5/              # Original Alpamayo 1.5 implementation
│   └── alpamayo1_5_distill/      # Distilled student and distillation logic
├── pyproject.toml
└── uv.lock
```

## Student Model Design

The student model is `Alpamayo1_5_Distilled`, a subclass of the original
`Alpamayo1_5` model. It does not rewrite the inference pipeline; it changes the
configuration so the existing pipeline instantiates smaller components.

Current lightweight choices:

- VLM backbone: `Qwen/Qwen3-VL-2B-Instruct` instead of the 10B Alpamayo teacher.
- Expert model: derived from the student VLM `text_config`, so hidden size and
  layer count follow the smaller VLM.
- Flow Matching: `num_inference_steps=4`.
- Action input projection: `PerWaypointActionInProjV2` with 4 encoder layers and
  hidden size 1024.

This is structural downsizing, not pruning, quantization, or LoRA.

## Distillation Signals

Training uses an online teacher-student path:

1. Load the teacher model `nvidia/Alpamayo-1.5-10B` in eval mode.
2. Load the student model from `Alpamayo1_5_DistilledConfig`.
3. Build image/text inputs and fuse trajectory history tokens.
4. Run the teacher autoregressively to collect generated sequences, VLM logits,
   Expert hidden states, and sampled action trajectories.
5. Feed the teacher-generated token sequences through the student VLM with
   gradients enabled.
6. Run the student Expert diffusion denoising path using the resulting KV cache.
7. Optimize the weighted sum of:
   - VLM logits KL distillation
   - Expert hidden-state MSE, with layer mapping and optional hidden projection
   - sampled action trajectory L2 loss

## Setup

Requirements:

- Python 3.12
- NVIDIA GPU with CUDA
- CUDA Toolkit 12.x if building `flash-attn`
- Hugging Face access to the gated Alpamayo model and PhysicalAI AV dataset

Install with uv:

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```

Authenticate with Hugging Face before running model or dataset code:

```bash
hf auth login
```

## Training

Run online distillation:

```bash
python scripts/train_distill.py --config-name=distill
```

Useful overrides:

```bash
python scripts/train_distill.py --config-name=distill \
  training.num_epochs=20 \
  training.gradient_accumulation_steps=4 \
  teacher.num_traj_samples=6
```

By default, `configs/distill.yaml` points at the gated teacher model and uses a
development clip fallback when `data.clip_ids` is unset.

## Teacher Data Cache

The cache script is currently a prototype and processes a hard-coded example
clip:

```bash
python scripts/generate_teacher_data.py --config-name=distill
```

It writes tensors under `outputs/teacher_data/`.

## Evaluation

Evaluate a saved student checkpoint:

```bash
python scripts/eval_student.py --config-name=eval \
  model.checkpoint_path=outputs/distilled/final
```

The current evaluation script also uses a single example clip. Treat reported
metrics as a smoke/prototype signal until dataset iteration is implemented.

## Known Implementation Notes

- VLM logits KD is skipped when teacher and student vocab dimensions differ.
  This should be made explicit before long training runs.
- The hidden projection inside `DistillationLoss` is trainable but is not saved
  by `student.save_pretrained()`. Checkpointing should include this module plus
  optimizer and scheduler state.
- Gradient accumulation currently needs careful handling when the final partial
  accumulation window is not full.
- `data.clip_ids: null` currently falls back to a single development clip, not
  the full dataset.

## License

The original Alpamayo 1.5 source files retain their NVIDIA copyright headers and
Apache-2.0 license text. Model weights and datasets may have separate gated or
non-commercial terms on Hugging Face; review those terms before use.
