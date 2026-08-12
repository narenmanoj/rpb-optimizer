# Focused Fisher-QK momentum sweep

This experiment follows the first trainable Fisher-QK sweep.

The first sweep established four practical facts:

1. Three Fisher-CG iterations consistently outperform the one-iteration, raw-gradient-like control at the same Q/K update RMS.
2. Native Fisher scaling makes Q/K updates far too small; RMS-normalized scaling is the viable production branch.
3. Unit coefficients are as good as projected coefficients in training while running faster.
4. Fisher-QK starts strongly but loses to Newton-Muon later, making temporal smoothing the most important missing lever.

## Fixed optimizer choices

All Fisher runs use:

- unit coefficients;
- three CG iterations unless explicitly marked CG1;
- relative damping `0.1`;
- RMS-normalized Q/K updates;
- Q/K target RMS in `{0.00022, 0.00033}`;
- one curvature sequence, refreshed every step;
- no cross-head beta term;
- no matrix-sign post-transform on the Fisher direction;
- Newton-Muon on V, the attention output matrix, and MLP matrices;
- AdamW on vectors and scalars.

The Fisher momentum implementation uses the Muon-style Nesterov convention already present in `train_gpt_fisher_qk.py`:

```text
buffer <- momentum * buffer + input
output <- input + momentum * buffer
```

Because RMS normalization happens after the momentum operation, momentum changes temporal orientation and coherence without changing the configured Q/K RMS target.

## Momentum modes

### `none`

Solve the current Fisher system from the current full-batch Q/K gradient.

### `rhs`

Apply momentum to the Q/K gradient first, then use that momentum-smoothed gradient as the Fisher-CG right-hand side.

### `direction`

Solve the current Fisher system from the current gradient, then apply momentum to the resulting Fisher direction before RMS normalization.

## Scientific controls

The CG1 + RHS-momentum runs are the key controls. With zero initialization, one unpreconditioned CG iteration is raw-gradient-like. Therefore, comparing CG3 and CG1 at the same momentum mode, momentum coefficient, and Q/K RMS tests whether Fisher geometry still adds value after temporal smoothing.

## Configuration count

The sweep contains 12 runs:

- 2 Newton-Muon controls;
- 2 Fisher-CG3 runs without Q/K momentum;
- 4 Fisher-CG3 runs with RHS momentum (`0.90`, `0.95`);
- 2 Fisher-CG3 runs with direction momentum (`0.95`);
- 2 Fisher-CG1 RHS-momentum controls (`0.95`).

All experiments use seed 0 and retain the 6200-step schedule, with a 180-minute internal cap.
