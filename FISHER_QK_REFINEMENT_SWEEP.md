# Fisher-QK momentum and scale refinement sweep

This focused experiment follows the first Fisher-QK momentum sweep. It answers two remaining questions:

1. Does three-step Fisher geometry still improve over the CG1/raw-gradient-like endpoint once RHS momentum is tuned fairly?
2. Is the best RMS scale centered near `0.00022` at RHS momentum `0.90`?

The ten configurations contain two same-harness Newton–Muon controls, matched CG1 and CG3 runs at momentum values `0.85`, `0.90`, and `0.925` with Q/K RMS `0.00022`, and two local RMS refinements (`0.00018`, `0.00026`) for CG3 at momentum `0.90`.

All Fisher runs use unit coefficients, three or one CG iterations as tagged, damping-relative scale `0.1`, RMS normalization, curvature batch one, curvature refresh every step, no explicit cross-head beta term, and no post-transform. V/O and other matrix parameters retain the existing Newton–Muon baseline.

## Deferred experiment: two-stage routing optimizer

The previous sweeps show a consistent early advantage for Fisher-QK that later gives way to Newton–Muon. A later experiment should test scheduled optimizer transitions, for example:

- Fisher-QK during an initial routing warmup, then Newton–Muon for Q/K;
- Fisher-QK during an initial routing warmup, then Muon for Q/K;
- a smooth interpolation rather than a hard switch.

This experiment is deliberately deferred until the matched CG1-versus-CG3 comparison and the local momentum/RMS optimum are resolved.

## Relation to the mirror and oscillation derivations

The trainable method in this sweep is the **Fisher-CG primitive**. It is the local quadratic approximation to pulled-back mirror descent and can also be viewed as steepest descent in the current damped Fisher metric. It is **not** the fixed oscillation-norm steepest-descent method from the derivation.

The oscillation branch was tested only in the local audit through the quadratic `q2` surrogate. Its native scale was extremely conservative, and it did not outperform Fisher-CG3 after norm matching. The exact active-extrema oscillation method and higher-`p` smooth surrogates remain open later variants. The exact alternating nonlinear mirror method also remains an oracle rather than the current production optimizer.
