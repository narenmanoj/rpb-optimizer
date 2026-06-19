# RPB optimizer

A small research repo for a **row-norm preconditioned bound (RPB)** optimizer for
attention, derived in [`smoothness_bound_and_update_rule.md`](smoothness_bound_and_update_rule.md),
alongside two Newton–Muon GPT-2 speedrun baselines to compare against.

| file | what it is |
|---|---|
| [`train_gpt_newton_muon_1.py`](train_gpt_newton_muon_1.py) | Single-GPU GPT-2 (124M); AdamW (head) + Muon w/ activation preconditioner (blocks). |
| [`train_gpt_newton_muon_2.py`](train_gpt_newton_muon_2.py) | Multi-GPU (DDP/`torchrun`) modded-nanogpt speedrun; DistAdam + Muon, FP8, FlexAttention. |
| [`train_gpt_rpb_1.py`](train_gpt_rpb_1.py) | Single-GPU; the **RPB** optimizer drives the attention QKV weights (Muon/AdamW elsewhere). |
| [`diagnostics.py`](diagnostics.py) | Shared TensorBoard + (opt-in) Weights & Biases logging, norm helpers, checkpointing. |
| [`smoothness_bound_and_update_rule.md`](smoothness_bound_and_update_rule.md) | The derivation behind the RPB update. |

All scripts require an NVIDIA GPU (CUDA) and log to `logs/<run_id>/`.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Download the dataset shards

The scripts train on the GPT-2-tokenized FineWeb-10B shards. Download them with:

```bash
# Download N train chunks (~100M tokens each) + the validation chunk into data/fineweb10B/
python data/cached_fineweb10B.py 8     # ~8 chunks: enough for short runs / experiments
python data/cached_fineweb10B.py        # all 103 chunks (full FineWeb-10B, ~10B tokens)
```

Rough guidance on how many chunks each full run consumes (the loader cycles shards if it
runs short, so fewer still works — it just repeats data):

- `train_gpt_newton_muon_1.py` / `train_gpt_rpb_1.py`: ~33 chunks for a no-repeat full run.
- `train_gpt_newton_muon_2.py`: ~8 chunks.

Data lands in `data/fineweb10B/` regardless of the current directory. To point at a prewritten
copy, file 2 also honors `DATA_PATH=/path/to/parent_of_data`.

## 2. Reproduce the Newton–Muon results

**Single GPU (file 1):**

```bash
python train_gpt_newton_muon_1.py
```

**Multi-GPU speedrun (file 2)** — launch with `torchrun` (`world_size` must divide 8):

```bash
# 8 GPUs
torchrun --standalone --nproc_per_node=8 train_gpt_newton_muon_2.py

# fewer GPUs also work (grad-accum compensates); e.g. 2 GPUs
torchrun --standalone --nproc_per_node=2 train_gpt_newton_muon_2.py

# disable FP8 (e.g. pre-Hopper GPUs)
DISABLE_FP8=1 torchrun --standalone --nproc_per_node=8 train_gpt_newton_muon_2.py
```

Hyperparameters for the Newton–Muon scripts live in their `Hyperparameters` dataclass and the
optimizer constructors near the bottom of each file (e.g. `learning_rate`, `num_iterations`,
the `Muon(...)` / `DistAdam(...)` LRs and momentum). Edit those to change a run.

## 3. Run RPB and vary its hyperparameters

```bash
python train_gpt_rpb_1.py
```

`train_gpt_rpb_1.py` exposes the RPB knobs (and a few common ones) as **environment-variable
overrides**, so you can sweep without editing the file:

| env var | default | meaning |
|---|---|---|
| `RPB_ETA` | `0.025` | damping η in `T = -η r* rsgn(G)` (acts as the RPB learning rate) |
| `RPB_MOMENTUM` | `0.95` | Muon-style momentum on the pulled-back numerator (0 disables) |
| `RPB_HSIGMA` | `8.0` | softmax-Hessian constant `h_sigma` in the curvature bound `C_t(r)` |
| `RPB_RIDGE_MULT` | `0.2` | Gram-inverse ridge, relative to the mean Gram diagonal |
| `RPB_RMAX` | `0` | trust-region cap on `r*` (0 ⇒ uncapped) |
| `RPB_PRECOND_REFRESH` | `32` | steps between Gram-inverse refreshes (mirrors Muon's cadence; first refresh at step `RPB_PRECOND_REFRESH - 1`, identity preconditioner before it) |
| `RPB_PRECOND_EWMA` | `0.95` | EWMA decay of the Gram covariance across refreshes (higher ⇒ smoother/slower) |
| `RPB_PRECOND_INIT_DIAG` | `0.001` | initial covariance diagonal before the first refresh seeds it |
| `LEARNING_RATE` | `0.004` | base LR for the AdamW (head) / Muon (other blocks) optimizers |
| `NUM_ITERATIONS` | `6200` | training steps |
| `SAVE_EVERY` | `1000` | checkpoint cadence in steps (0 disables) |
| `DIAG_EVERY` | `100` | cadence for grad/update/weight-norm diagnostics |

Examples:

```bash
# Larger trust-region damping, no momentum (closest to the "faithful" note update)
RPB_ETA=0.5 RPB_MOMENTUM=0 python train_gpt_rpb_1.py

# Tighter curvature bound (smaller h_sigma => larger r*) + a trust-region cap
RPB_HSIGMA=2.0 RPB_RMAX=1.0 python train_gpt_rpb_1.py

# Heavier Gram regularization and more frequent diagnostics
RPB_RIDGE_MULT=1.0 DIAG_EVERY=25 python train_gpt_rpb_1.py

# Sweep the Gram-preconditioner refresh: re-invert every 16 steps with a heavier EWMA
RPB_PRECOND_REFRESH=16 RPB_PRECOND_EWMA=0.99 python train_gpt_rpb_1.py

# Recover the original behavior: refresh the Gram inverse every step
RPB_PRECOND_REFRESH=1 python train_gpt_rpb_1.py

# Quick smoke run
NUM_ITERATIONS=200 SAVE_EVERY=0 python train_gpt_rpb_1.py
```

The Gram preconditioner mirrors the Muon right-preconditioner: the damped Gram inverse is
recomputed only every `RPB_PRECOND_REFRESH` steps from an EWMA (`RPB_PRECOND_EWMA`) of the
per-step input Gram, and the cached inverse is reused in between. `train_gpt_rpb_1.py`
logs whether each step refreshed via `rpb/precond_refresh`.

The remaining RPB knobs (`rpb_nesterov`, `bisect_iters`, `eps_gram`) are in the
`Hyperparameters` dataclass / `RPB(...)` constructor in the file.

## Logging and checkpoints

Every script logs **train loss, validation loss, gradient norms, update norms, and
weight-matrix norms** (per optimizer group + global), plus learning rates; `train_gpt_rpb_1.py`
additionally logs RPB internals (`rpb/r_star_mean`, `rpb/r_star_max`, `rpb/S_G_mean`,
`rpb/precond_refresh`).

**TensorBoard** is always on (local):

```bash
tensorboard --logdir logs/
```

**Weights & Biases** is opt-in and offline-safe — it stays off unless configured, and never
breaks a run if unavailable:

```bash
# online (requires `wandb login` or WANDB_API_KEY)
WANDB=1 WANDB_PROJECT=rpb-optimizer python train_gpt_rpb_1.py

# local-only, no account needed
WANDB=1 WANDB_MODE=offline python train_gpt_rpb_1.py
```

**Checkpoints** (model + all optimizer states + step + embedded source) are written to
`logs/<run_id>/state_step<NNNNNN>.pt` every `save_every` steps (kept, not rotated). For the
DDP script, periodic saving is opt-in via `save_every`; the final-step save uses its existing
`save_checkpoint` flag.
