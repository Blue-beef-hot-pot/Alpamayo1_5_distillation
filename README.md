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
- Teacher wrappers that run `nvidia/Alpamayo-1.5-10B` and collect soft labels
  (VLM logits, VLM hidden states, Expert hidden states across all diffusion
  steps, and sampled action trajectories).
- Student teacher-forcing forward logic for differentiable VLM distillation.
- Distillation losses: VLM Logits KD, Expert Hidden KD (full diffusion
  trajectory), VLM Hidden KD, and Trajectory L2 — all with grouped learnable
  projections and margin ReLU.
- Pipeline-parallel training: GPU 0 runs teacher inference, round-robin
  dispatches results to GPU 1-3 for student DDP training (4×A100).
- Hydra configs and scripts for single-GPU and pipeline-parallel training,
  teacher-data generation, and evaluation.

## Repository Layout

```text
.
├── configs/
│   ├── distill.yaml              # Distillation training config (single GPU)
│   ├── distill_pipeline.yaml     # Pipeline-parallel training config (spawned multi-GPU)
│   └── eval.yaml                 # Student evaluation config
├── notebooks/                    # Original Alpamayo 1.5 example notebooks
├── scripts/
│   ├── train_distill.py          # Online teacher-student distillation (single GPU)
│   ├── train_distill_pipeline.py # Pipeline-parallel distillation (spawned multi-GPU)
│   ├── submit_pipeline.sh        # SLURM launch script for pipeline training
│   ├── download_dataset_1tb.py   # Size-capped PhysicalAI AV cache downloader
│   ├── test_downloaded_data.py   # Local dataset loading / inference smoke test
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
- Full Alpamayo special-token set and a 4096-token trajectory vocabulary are
  added to the student tokenizer so teacher sequences can be teacher-forced after
  ID alignment.
- Student training freezes the Qwen visual tower; teacher-forcing still trains
  text/expert components while avoiding repeated visual-tower gradient memory.

This is structural downsizing, not pruning, quantization, or LoRA.

## Distillation Signals

Training uses an online teacher-student path:

1. Load the teacher model `nvidia/Alpamayo-1.5-10B` in eval mode.
2. Load the student model from `Alpamayo1_5_DistilledConfig`.
3. Build image/text inputs and fuse trajectory history tokens.
4. Run the teacher autoregressively to collect generated sequences, Expert
   hidden states across all diffusion steps, and sampled action trajectories.
5. Align teacher-added trajectory/special token IDs to the student tokenizer and
   validate Qwen visual patch counts before student teacher-forcing.
6. Feed the teacher-generated token sequence through the student VLM with
   gradients enabled to build the Expert KV cache.
7. Run the student Expert diffusion denoising path using the resulting KV cache,
   collecting Expert hidden states at every diffusion step.
8. Optimize the weighted sum of:
   - Expert hidden-state MSE across all diffusion steps, with layer mapping
     (36→24) and step mapping (10→4), grouped learnable projections and
     margin ReLU
   - sampled action trajectory L2 loss

Pipeline-parallel training defaults to this minimal stable Expert/trajectory
loss set (`teacher.num_traj_samples=1`, VLM logits/hidden KD disabled) to keep
Qwen3-VL teacher-forcing within GPU memory. Single-GPU configs can still enable
VLM logits KD and VLM hidden KD for smaller experiments.

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

## Local Dataset Cache

Download a size-capped subset of the PhysicalAI AV dataset containing the
features used by training (4 cameras + egomotion):

```bash
python scripts/download_dataset_1tb.py --cache-dir ./.cache/ --max-bytes 1e12 --include-metadata
```

Validate local data loading without network access:

```bash
python scripts/test_downloaded_data.py --cache-dir ./.cache/ --num-clips 5 --no-online-init
```

Run a full teacher inference smoke test on one downloaded clip:

```bash
python scripts/test_downloaded_data.py --cache-dir ./.cache/ --num-clips 1 --inference --no-online-init
```

## Training

Run online distillation (single GPU):

```bash
# From local cache (recommended):
python scripts/train_distill.py --config-name=distill data.cache_dir=./.cache/

# Or stream from HF Hub (no local cache):
python scripts/train_distill.py --config-name=distill
```

Run pipeline-parallel distillation (spawns one process per GPU by default):

```bash
python scripts/train_distill_pipeline.py --config-name=distill_pipeline data.cache_dir=./.cache/

# Or override the process count explicitly:
python scripts/train_distill_pipeline.py --config-name=distill_pipeline \
  data.cache_dir=./.cache/ pipeline.num_processes=4
```

Useful overrides:

```bash
python scripts/train_distill.py --config-name=distill \
  data.cache_dir=./.cache/ \
  training.num_epochs=20 \
  training.gradient_accumulation_steps=4 \
  teacher.num_traj_samples=6
```

Resume from an epoch/best/final checkpoint:

```bash
python scripts/train_distill.py --config-name=distill \
  data.cache_dir=./.cache/ \
  training.resume_from_checkpoint=outputs/distilled/epoch_5

python scripts/train_distill_pipeline.py --config-name=distill_pipeline \
  data.cache_dir=./.cache/ \
  training.resume_from_checkpoint=outputs/distilled_pipeline/epoch_5
```

Resume restores student weights, distillation loss state, optimizer, scheduler,
epoch, global step, and best loss. Checkpoints also write `training_progress.json`
so pipeline teacher ranks can resume epoch iteration without loading optimizer
state. Resume starts from the next epoch after the saved checkpoint; mid-epoch
batches are not resumed.

When `data.cache_dir` is set, the dataloader reads from the local HF cache
(using `download_dataset_1tb.py` output) with `maybe_stream=False` and
auto-detects all downloaded clips. Training samples are `(clip_id, t0_us)` pairs:
each selected clip is sampled every 1 second by default (`data.sample_step_us=1000000`)
inside the common valid timestamp range of egomotion and the four required cameras,
aligned to the 0.1s data grid, leaving 1.5s for trajectory history, 6.4s for trajectory future,
and enough camera history for the 4-frame image window ending at `t0_us`.
When cache is unset, it falls back to streaming from HF Hub. Shuffling is enabled
by default (`data.shuffle=true`, `data.seed=42`) and applies at sample level.

### Pipeline Parallelism

The pipeline-parallel mode uses `torch.multiprocessing.spawn`: one configured
teacher rank runs teacher inference and round-robin dispatches results to all
other ranks, which run student training with DDP gradient synchronization. Before
dispatch, teacher-added trajectory/special token IDs are remapped to the student
tokenizer so teacher-forcing cannot feed out-of-range IDs to the student VLM. The
default pipeline config uses one trajectory sample and disables VLM logits/hidden
KD so the run prioritizes stable Expert Hidden KD + Trajectory L2 training. By
default `pipeline.num_processes=null` uses `torch.cuda.device_count()`, so the
number of students is `num_processes - 1`.

```bash
# Via SLURM
sbatch scripts/submit_pipeline.sh

# Or directly
python scripts/train_distill_pipeline.py --config-name=distill_pipeline \
    data.cache_dir=./.cache/
```

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
- The grouped projections and margin parameters inside `DistillationLoss` are
  trainable. The pipeline-parallel script saves them alongside optimizer and
  scheduler state in `training_state.pt`; the single-GPU script does not yet
  save them (needs a similar checkpoint update).
- Gradient accumulation currently needs careful handling when the final partial
  accumulation window is not full.
- `data.clip_ids: null` currently falls back to a single development clip, not
  the full dataset.
- Teacher VLM hidden states require a separate forward pass (`use_cache=False`)
  after generation, which increases peak GPU memory.
- The pipeline-parallel script loads all clips into memory before dispatching.
  For large datasets, streaming would reduce memory pressure on rank 0.

## License

The original Alpamayo 1.5 source files retain their NVIDIA copyright headers and
Apache-2.0 license text. Model weights and datasets may have separate gated or
non-commercial terms on Hugging Face; review those terms before use.
