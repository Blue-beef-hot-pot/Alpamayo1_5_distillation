# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **distillation project** for Alpamayo 1.5, NVIDIA's Vision-Language-Action (VLA) model for autonomous driving. The goal is to distill the 10B teacher (Qwen3-VL-8B VLM + 8B Expert) into a ~4B student (Qwen3-VL-2B VLM + 2B Expert, 4-step flow matching).

## Project Structure

```
src/alpamayo1_5/           # Original inference code (teacher model, untouched)
src/alpamayo1_5_distill/   # Distillation package (student model, losses, teacher wrapper)
configs/                   # Hydra configs (distill.yaml, distill_pipeline.yaml, eval.yaml)
scripts/                   # Training/eval/data generation scripts
```

## Build & Run Commands

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active

# Flash-attn fallback:
uv sync --active --no-install-package flash-attn

# Download local PhysicalAI AV cache (4 cameras + egomotion, size-capped)
python scripts/download_dataset_1tb.py --cache-dir ./.cache/ --max-bytes 1e12 --include-metadata

# Validate local data cache
python scripts/test_downloaded_data.py --cache-dir ./.cache/ --num-clips 5 --no-online-init

# Distillation training (single GPU, local cache)
python scripts/train_distill.py --config-name=distill data.cache_dir=./.cache/

# Pipeline-parallel distillation (spawn one process per GPU by default, local cache)
python scripts/train_distill_pipeline.py --config-name=distill_pipeline data.cache_dir=./.cache/

# Resume training from a full checkpoint
python scripts/train_distill.py --config-name=distill data.cache_dir=./.cache/ training.resume_from_checkpoint=outputs/distilled/epoch_5
python scripts/train_distill_pipeline.py --config-name=distill_pipeline data.cache_dir=./.cache/ training.resume_from_checkpoint=outputs/distilled_pipeline/epoch_5

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
- Action space: same 64-waypoint unicycle acceleration/curvature space, with
  `PerWaypointActionInProjV2` input projection and linear output projection
- Trajectory tokenizer: `DeltaTrajectoryTokenizer` with `num_bins` matched to
  `traj_vocab_size` for history/future trajectory token fusion
- KV Cache dimensions match naturally (Expert inherits from VLM text_config)

### Key: No logic overrides needed
`Alpamayo1_5_Distilled` inherits from `Alpamayo1_5` without overriding any methods because the parent class is fully dimension-agnostic — Expert is created via `copy.deepcopy(self.vlm.config.text_config)`, so switching to 2B VLM automatically makes the Expert 2B.

### Distillation losses
1. **VLM Logits KD** — KL divergence between teacher/student VLM output distributions
2. **Expert Hidden KD** — MSE between teacher/student Expert hidden states across all diffusion steps (layer mapping: 36→24 via uniform sampling; step mapping: 10→4 via uniform sampling). Uses grouped learnable projections (3 groups: shallow/mid/deep) with margin ReLU (Heo et al., ICCV 2019) to suppress weak teacher activations.
3. **VLM Hidden KD** — MSE between teacher/student VLM per-layer hidden states. Same grouped projection + margin ReLU scheme as Expert Hidden KD.
4. **Trajectory L2** — MSE between predicted trajectories

## Key Files

**Teacher (original, do not modify):**
- `src/alpamayo1_5/models/alpamayo1_5.py` — Alpamayo1_5 model class
- `src/alpamayo1_5/models/base_model.py` — ReasoningVLA base class
- `src/alpamayo1_5/config.py` — Alpamayo1_5Config
- `src/alpamayo1_5/helper.py` — Input processing (BASE_PROCESSOR_NAME already points to Qwen3-VL-2B)
- `src/alpamayo1_5/diffusion/flow_matching.py` — FlowMatching (num_inference_steps is configurable)

**Student (distillation package):**
- `src/alpamayo1_5_distill/config.py` — Alpamayo1_5_DistilledConfig (2B defaults, 4-step FM)
- `src/alpamayo1_5_distill/checkpoint.py` — training_state.pt save/load helpers for complete resume
- `src/alpamayo1_5_distill/model.py` — Alpamayo1_5_Distilled (subclass, AutoModel registered)
- `src/alpamayo1_5_distill/teacher.py` — load_teacher(), teacher_forward() with VLM hidden states + Expert hidden states across all diffusion steps
- `src/alpamayo1_5_distill/student_forward.py` — student_forward() with teacher-forcing (differentiable VLM), inference modes, teacher/student token ID alignment, and Qwen visual patch-count validation
- `src/alpamayo1_5_distill/distill_loss.py` — DistillationLoss (VLM Logits KD + Expert Hidden KD + VLM Hidden KD + Traj L2), grouped projections with margin ReLU, `_uniform_index_mapping`
- `src/alpamayo1_5_distill/train_utils.py` — Shared utilities: build_student_config (including full Alpamayo special tokens), resolve_clip_ids/resolve_clip_samples, build_dataloader (local cache + sample-level shuffle), prepare_model_inputs, repeat_visual_inputs, shallow_copy_data
- `src/alpamayo1_5_distill/comm.py` — Cross-GPU serialization for pipeline parallelism (NCCL send/recv)
- `src/alpamayo1_5_distill/distributed.py` — DDP setup, StudentWithLoss wrapper, process group management

**Configs:**
- `configs/distill.yaml` — Single-GPU training config (teacher, student, loss weights, optimizer, scheduler)
- `configs/distill_pipeline.yaml` — Pipeline-parallel training config (spawned ranks, round-robin teacher dispatch)
- `configs/eval.yaml` — Evaluation config

## Conventions

- `ruff` formatting (line-length: 100)
- SPDX license headers on all source files
- Config sub-dicts use Hydra `_target_` convention for `hydra.utils.instantiate`
- Original `alpamayo1_5` package is read-only — all distillation code goes in `alpamayo1_5_distill`
- After every code change or design discussion, update CLAUDE.md and README.md to reflect the current state

## Pipeline Parallelism

`train_distill_pipeline.py` uses `torch.multiprocessing.spawn` instead of `torchrun`. `pipeline.num_processes: null` means use `torch.cuda.device_count()`; students are all ranks except `pipeline.teacher_rank`. The teacher rank remaps teacher-added trajectory/special token IDs to the student tokenizer before dispatching sequences to student ranks.

## Data Loading

`build_dataloader(cfg, epoch=...)` supports two modes:
- `data.cache_dir: null` — stream from HuggingFace Hub with `maybe_stream=True` and the legacy single-clip fallback when `clip_ids` is null.
- `data.cache_dir: ./path` — read only from local HF cache with `maybe_stream=False`; when `clip_ids` is null it auto-detects all downloaded chunks and uses all cached clips.

Training samples are `(clip_id, t0_us)` pairs. Each clip is sampled every `data.sample_step_us` microseconds (default 1s) inside the common valid timestamp range of egomotion and the four required training cameras, aligned to the 0.1s data grid, leaving `data.history_us` (1.5s) before `t0_us`, `data.future_us` (6.4s) after it, and enough camera history for the 4-frame image window ending at `t0_us`. Shuffle uses `data.seed + epoch` and applies at sample level.

## Resume Training

Both `train_distill.py` and `train_distill_pipeline.py` support `training.resume_from_checkpoint`. Full checkpoints contain student weights, tokenizer, `training_state.pt`, `training_progress.json`, distill loss state, optimizer, scheduler, saved epoch, global step, and best loss. Pipeline teacher ranks read `training_progress.json` to resume epoch iteration without loading optimizer state. Resume starts at the next epoch after the saved checkpoint; mid-epoch batches are not resumed.

## Flash Attention Fallback

```python
model = Alpamayo1_5_Distilled.from_pretrained("...", dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
```
