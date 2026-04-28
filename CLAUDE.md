# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **distillation project** for Alpamayo 1.5, NVIDIA's Vision-Language-Action (VLA) model for autonomous driving. The goal is to distill the 10B teacher (Qwen3-VL-8B VLM + 8B Expert) into a ~4B student (Qwen3-VL-2B VLM + 2B Expert, 4-step flow matching).

## Project Structure

```
src/alpamayo1_5/           # Original inference code (teacher model, untouched)
src/alpamayo1_5_distill/   # Distillation package (student model, losses, teacher wrapper)
configs/                   # Hydra configs (distill.yaml, eval.yaml)
scripts/                   # Training/eval/data generation scripts
```

## Build & Run Commands

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active

# Flash-attn fallback:
uv sync --active --no-install-package flash-attn

# Distillation training
python scripts/train_distill.py --config-name=distill

# Evaluate student
python scripts/eval_student.py --config-name=eval

# Generate teacher soft labels for offline training
python scripts/generate_teacher_data.py --config-name=distill

# Test teacher inference (original code)
python src/alpamayo1_5/test_inference.py

pre-commit run --all-files
pytest
```

## Architecture

### Teacher (Alpamayo1_5, ~16B effective)
- VLM: Qwen3-VL-8B (36 layers, hidden=4096) → chain-of-causation reasoning
- Expert: derived from VLM text_config (same 36 layers, hidden=4096) → flow matching trajectory denoising (10 steps)
- KV Cache flows from VLM to Expert directly

### Student (Alpamayo1_5_Distilled, ~4B)
- VLM: Qwen3-VL-2B (24 layers, hidden=1536) → same reasoning pipeline
- Expert: derived from 2B VLM text_config (same 24 layers, hidden=1536) → 4-step flow matching
- KV Cache dimensions match naturally (Expert inherits from VLM text_config)

### Key: No logic overrides needed
`Alpamayo1_5_Distilled` inherits from `Alpamayo1_5` without overriding any methods because the parent class is fully dimension-agnostic — Expert is created via `copy.deepcopy(self.vlm.config.text_config)`, so switching to 2B VLM automatically makes the Expert 2B.

### Distillation losses
1. **VLM Logits KD** — KL divergence between teacher/student VLM output distributions
2. **Expert Hidden KD** — MSE between teacher/student Expert per-layer hidden states (layer mapping: 36→24 via uniform sampling)
3. **Trajectory L2** — MSE between predicted trajectories

## Key Files

**Teacher (original, do not modify):**
- `src/alpamayo1_5/models/alpamayo1_5.py` — Alpamayo1_5 model class
- `src/alpamayo1_5/models/base_model.py` — ReasoningVLA base class
- `src/alpamayo1_5/config.py` — Alpamayo1_5Config
- `src/alpamayo1_5/helper.py` — Input processing (BASE_PROCESSOR_NAME already points to Qwen3-VL-2B)
- `src/alpamayo1_5/diffusion/flow_matching.py` — FlowMatching (num_inference_steps is configurable)

**Student (distillation package):**
- `src/alpamayo1_5_distill/config.py` — Alpamayo1_5_DistilledConfig (2B defaults, 4-step FM)
- `src/alpamayo1_5_distill/model.py` — Alpamayo1_5_Distilled (subclass, AutoModel registered)
- `src/alpamayo1_5_distill/teacher.py` — load_teacher(), teacher_forward() with hidden state extraction
- `src/alpamayo1_5_distill/distill_loss.py` — DistillationLoss (VLM KD + Expert hidden KD + Traj L2)

**Configs:**
- `configs/distill.yaml` — Training config (teacher, student, loss weights, optimizer, scheduler)
- `configs/eval.yaml` — Evaluation config

## Conventions

- `ruff` formatting (line-length: 100)
- SPDX license headers on all source files
- Config sub-dicts use Hydra `_target_` convention for `hydra.utils.instantiate`
- Original `alpamayo1_5` package is read-only — all distillation code goes in `alpamayo1_5_distill`

## Flash Attention Fallback

```python
model = Alpamayo1_5_Distilled.from_pretrained("...", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
```
