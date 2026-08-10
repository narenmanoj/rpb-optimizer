# Attention Geometry Local-Model Audit v2

This branch implements the first diagnostic stage for the attention-specific
optimizers derived in *Three Geometry-Aware Optimizers for Transformer
Attention* and *Descent Guarantees for Geometry-Aware Attention Optimizers*.
It does **not** yet replace the training optimizer with Fisher-CG, mirror
descent, or oscillation descent. Instead, it trains a common Newton-Muon
baseline to a chosen step, freezes the model, and compares candidate one-step
updates on one selected attention layer.

## Why the v2 audit uses two scale protocols

The v1 audit Frobenius-matched every candidate to Newton-Muon before applying a
common outer learning rate. That is a useful direction comparison, but it does
not test the magnitude selected by Fisher, mirror, q2, Feasible-RPB, or RPB
itself. The v2 audit therefore evaluates every candidate under two protocols:

1. **Native protocol.** Fisher, q2, mirror, and RPB retain the displacement
   magnitude produced by their own local model or radius solve. Scale factors
   multiply that native proposal directly. Newton-Muon and the raw-gradient
   control still require an optimizer learning rate, so their native protocol
   uses `ATTN_AUDIT_BASE_LR` before applying the grid factor.
2. **Matched protocol.** Q/K candidates are Frobenius-matched to the current
   Q/K Newton-Muon direction; Q/K/V candidates are matched to Q/K/V
   Newton-Muon. The common base learning rate and grid factor are then applied.

The native protocol asks whether the optimizer's own local scale is sensible.
The matched protocol asks whether its direction is better at a comparable
update norm. These are intentionally separate questions.

## Candidate directions

Batch A generates:

1. `newton_muon_qk`: current-batch Newton-Muon on Q and K only.
2. `newton_muon_qkv`: current-batch Newton-Muon on Q, K, and V.
3. `faithful_rpb_qkv`: original RPB headwise radius and pullback.
4. `feasible_rpb_qkv`: projection-aware RPB with the radius recomputed after
   the unit row-sign target is pulled back through the current input matrix.
5. `raw_gradient_qk`: negative Q/K weight gradient.
6. `fisher_cg1_unit_*` and `fisher_cg3_unit_*`: joint Q/K pulled-back
   categorical-Fisher systems with unit row coefficients.
7. `fisher_cg1_projected_*` and `fisher_cg3_projected_*`: the same systems with
   current projected functional coefficients. Coefficient shape is normalized
   independently of outer scale.
8. `oscillation_q2_unit_cg3_*`: the quadratic `q2` surrogate for the fixed
   oscillation geometry.
9. `mirror_joint_n2_unit` and `mirror_joint_n2_projected`: two Newton outer
   steps on the convex joint tangent log-partition mirror subproblem.

The exact alternating mirror method and exact active-extrema oscillation QP are
deferred until these local primitives pass the direction, scale, transfer, and
cost audit.

## Faithful geometric details

- The score Jacobian and adjoint include the repository's RoPE convention.
- The causal mask appears in every score, Fisher, mirror, and oscillation
  computation.
- Fisher and q2 energies are averaged over batch-query rows, matching the
  averaged language-model loss convention.
- No matrix-sign postprocessing is applied to Fisher, mirror, or oscillation
  candidates. No Nor-style row adaptation is applied.
- The projected coefficient uses the current output gradient and effective
  values induced by the output projection. `ATTN_AUDIT_BETA` controls the
  trace-majorized cross-head term. The first audit uses beta zero to isolate the
  current functional coefficient.
- Mirror predicted reduction includes the complete local objective:
  linear gain, log-partition Bregman term, and parameter damping.

## Native local-model sanity

Before any protocol-specific rescaling, v2 evaluates each Fisher, q2, or mirror
candidate in its own local model. It records:

- linear gain;
- geometry or Bregman energy;
- parameter regularizer energy;
- predicted reduction;
- model objective relative to the zero step;
- a scale-relative pass/fail tolerance.

A successful PCG iterate or mirror subproblem step should have a nonpositive
model objective, equivalently nonnegative predicted reduction, up to numerical
tolerance. This check detects solver or normalization inconsistencies before
long training experiments.

## Two-batch protocol

Batch A constructs the gradient, coefficients, and candidate directions. Batch
B is an independent batch from the same audit loader. For every candidate,
protocol, and scale, the audit records:

- current-batch and held-out-batch actual loss reduction;
- current and held-out gradient alignment;
- predicted reduction and current-batch trust ratio when a local model exists;
- update norm, Fisher or q2 energy, score oscillation, and Q/K bilinear
  remainder;
- cosine with the Q/K Newton-Muon direction;
- CG residual history, damping, solve time, total audit time, and peak memory.

Batch B is diagnostic only. It never selects a scale or accepts a step.

## Summary views

Every run writes three separate rankings:

1. best actual batch-A reduction for each candidate/protocol, with batch-B
   transfer reported at that same scale;
2. held-out batch-B transfer at the batch-A-selected scale;
3. best batch-A trial whose local model predicts positive reduction.

The summary also reports whether the selected scale lies on a grid boundary.
A boundary winner means that the grid still does not bracket the useful region.

## Main environment variables

- `ATTN_AUDIT_STEP`
- `ATTN_AUDIT_LAYER`
- `ATTN_AUDIT_BATCH_SIZE`
- `ATTN_AUDIT_SEQUENCE_LENGTH`
- `ATTN_AUDIT_BASE_LR`
- `ATTN_AUDIT_PROTOCOLS` (`native,matched` by default)
- `ATTN_AUDIT_NATIVE_SCALE_GRID`
- `ATTN_AUDIT_MATCHED_SCALE_GRID`
- `ATTN_AUDIT_MODEL_SANITY_TOL`
- `ATTN_AUDIT_DAMPING_RELS`
- `ATTN_AUDIT_BETA`
- `ATTN_AUDIT_COEFF_NORMALIZE` (`none`, `mean`, `median`)
- `ATTN_AUDIT_COEFF_FLOOR`
- `ATTN_AUDIT_MIRROR_NEWTON_ITERS`
- `ATTN_AUDIT_MIRROR_CG_ITERS`
- `ATTN_AUDIT_RIDGE_MULT`
- `ATTN_AUDIT_OUTPUT_DIR`

`ATTN_AUDIT_SCALE_GRID` remains a backward-compatible legacy option. When it
is present, it initializes both v2 grids unless a protocol-specific variable
overrides it.

The model state used for the audit follows the normal 6200-step learning-rate
schedule. `NUM_ITERATIONS` remains 6200; the script exits after the requested
audit.

## Outputs

Each run writes:

- `audit_step<step>_layer<layer>_seed<seed>.json`
- `audit_step<step>_layer<layer>_seed<seed>.tsv`
- `audit_step<step>_layer<layer>_seed<seed>_summary.txt`

Use `summarize_attention_geometry_audits.py` to combine layers and checkpoints.
The summarizer groups candidate name and protocol separately.

## Current scope and limitations

- The audit changes one layer at a time.
- Fisher, mirror, and q2 candidates update Q/K only; RPB and the Q/K/V
  Newton-Muon control update Q/K/V. Every output labels the scope.
- Projected coefficients are current-point quantities, not full-path
  certificates.
- The joint mirror candidate uses the feasible tangent score map. Actual Q/K
  application includes the measured bilinear remainder.
- The audit uses small independent batches because full attention probability
  tensors are expensive. Long-run training should begin only after the
  operator, native-scale, trust, transfer, and cost diagnostics are favorable.
