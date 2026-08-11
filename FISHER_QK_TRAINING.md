# Joint Fisher-QK training branch

This branch turns the validated local-model primitive into a trainable optimizer.

## Parameter assignment

- `W_Q, W_K`: joint pulled-back categorical-Fisher CG.
- `W_V`: Newton-Muon using the cached input Gram inverse, momentum, and matrix sign.
- `W_O` and MLP matrices: the repository's Newton-Muon matrix optimizer.
- embeddings, norms, and scalar/vector parameters: AdamW.

The Q/K optimizer solves, approximately,

```text
(J* C J + lambda I) [D_Q, D_K] = -[G_Q, G_K]
```

where the right-hand side is the full accumulated minibatch gradient. A small sampled curvature minibatch supplies `X`, current post-RoPE `Q/K`, attention probabilities, and optional projected coefficient shape.

The first experiments use:

```text
FISHER_POST_TRANSFORM=none
FISHER_BETA=0
FISHER_MOMENTUM_MODE=none
FISHER_CURV_BATCH=1
FISHER_CURV_REFRESH=1
```

so the new direction remains a direct Fisher-CG proposal rather than a spectral or row-adaptive hybrid.

## Scale modes

### `native`

Applies the model-produced displacement directly:

```text
update_QK = schedule * FISHER_OUTER_SCALE * D_Fisher
```

This is closest to the local quadratic derivation, but the audit showed that its magnitude varies greatly across layers.

### `rms`

Preserves the joint Q/K Fisher direction and normalizes its joint element RMS:

```text
update_QK = schedule * FISHER_QK_LR * D_Fisher / rms(D_Fisher)
```

This decouples direction from the uneven native scale. It does not apply matrix sign.

### `nm_match`

Matches the pair norm to a current Newton-Muon Q/K reference, then applies `FISHER_QK_LR`. This mode is diagnostic; the first sweep uses `native` and `rms`.

## Coefficients

- `unit`: `c_ai = 1`.
- `projected`: current projected functional coefficient, normalized by the positive median and floored by `FISHER_COEFF_FLOOR`.

`FISHER_BETA=0` means the first sweep does not activate the trace-majorized cross-head downstream term.

## Curvature capture

The normal compiled training pass produces the full gradient. After gradient accumulation, a separate small uncompiled forward uses the first `FISHER_CURV_BATCH` sequences from the first microbatch of the same optimizer step. `torch.autograd.grad` computes only the gradients of the attention outputs needed by projected coefficients and does not overwrite parameter `.grad` fields.

This is a stochastic curvature preconditioner:

```text
full minibatch gradient + sampled attention curvature
```

`FISHER_CURV_PRECISION=bf16` is the practical default. Curvature operators and CG algebra use FP32 tensors.

## Momentum

- `none`: current full gradient enters the Fisher solve.
- `rhs`: momentum enters the right-hand side before geometry.
- `direction`: momentum smooths solved directions.

Only `none` is used in the first sweep.

## Optional future levers

- `FISHER_POST_TRANSFORM=matrix_sign`: norm-matched matrix-sign postprocessing; disabled initially.
- `FISHER_SCORE_OSC_CAP>0`: halves an update until its exact sampled Q/K score change respects the cap.
- `FISHER_BETA>0`: activates the trace-majorized head-coupling coefficient.
- `FISHER_CURV_REFRESH>1`: reuses stale curvature samples.
- arbitrary `FISHER_CG_ITERS` and `FISHER_CURV_BATCH`.

## Key diagnostics

TensorBoard records:

```text
fisher_qk/cg_final_residual_mean
fisher_qk/cg_final_residual_max
fisher_qk/damping_mean
fisher_qk/rayleigh_mean
fisher_qk/qk_update_rms_mean
fisher_qk/v_update_rms_mean
fisher_qk/current_gradient_dot_mean
fisher_qk/descent_fallback_layers
fisher_qk/curvature_capture_seconds
fisher_qk/optimizer_seconds
fisher_qk/score_osc_max_mean
fisher_qk/bilinear_ratio_mean
```

A nonzero `descent_fallback_layers` means truncated CG or optional momentum produced a non-descent direction for the current full gradient; that layer fell back to the raw negative Q/K gradient.

## First short sweep

The fixed 12-run table contains:

- two Newton-Muon controls;
- projected/unit coefficients;
- CG1 and CG3;
- relative damping `0.1` and `1.0`;
- native and RMS-normalized scales;
- RMS targets `0.00011`, `0.00022`, and `0.00033`.

All runs preserve the ordinary 6200-step schedule but stop after three hours. Rank them at the latest validation step reached by every successful run.

## Caveats

This production candidate deliberately departs from the clean affine-block theorem in several ways:

- it updates Q and K jointly;
- it samples curvature from a small subset of sequences;
- it solves with a fixed CG budget;
- it updates other model blocks simultaneously;
- it does not yet use actual-loss trust acceptance.

The first sweep therefore tests an optimizer proposal informed by the derived geometry, not a literal full-network descent theorem.
