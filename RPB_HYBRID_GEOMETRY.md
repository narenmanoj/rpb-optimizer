# RPB hybrid-geometry experiment

This experimental branch keeps the existing RPB activation-space radius and softened token-row direction, then exposes three independent post-target controls.

## Update order

1. Build the activation-gradient direction with `RPB_ROWSIGN_POWER`.
2. Apply the per-head radius `r_star`.
3. Pull the target back through the right input geometry.
4. Optionally shape the singular spectrum of each Q/K/V block.
5. Optionally adapt output-neuron rows with a NorMuon-style second moment.
6. Apply the scheduled scalar step size.

## Controls

### `RPB_SPECTRAL_BLEND`

- `0`: ordinary RPB pullback.
- `1`: approximate matrix sign of each pulled-back Q/K/V block.
- Intermediate values linearly blend the two directions.

The matrix-sign endpoint and the final blend are Frobenius-matched to the ordinary RPB block. This isolates direction from update scale. The implementation uses the existing five-step Newton-Schulz kernel (`RPB_SPECTRAL_STEPS`).

This is a computationally cheap interpolation toward spectral flattening. It is not exactly a fractional singular-value power map.

### `RPB_PRECOND_BLEND`

- `1`: full cached Gram-inverse direction, exactly matching the existing RPB pullback.
- `0`: identity right geometry.
- Intermediate values blend the directions.

The identity endpoint and the final blend are matched blockwise to the full-preconditioned Frobenius norm. This avoids conflating geometry with a trivial scale change.

### `RPB_NOR_ENABLE`

When enabled, the optimizer maintains an EMA of mean squared update entries for every output-neuron row, divides each row by the square root of that statistic, and restores the original global Frobenius norm. `RPB_NOR_BETA2` controls the EMA.

## Baseline recovery

The current softened-RPB baseline is recovered exactly by:

```bash
RPB_ROWSIGN_POWER=0.9
RPB_SPECTRAL_BLEND=0
RPB_PRECOND_BLEND=1
RPB_NOR_ENABLE=0
```

The original exact row-sign RPB baseline additionally sets `RPB_ROWSIGN_POWER=1`.

Even the endpoint `RPB_SPECTRAL_BLEND=1` is not identical to Newton-Muon: RPB still uses its activation-row direction, per-head radii, and existing momentum ordering. The endpoint tests whether the singular-mode equalization missing from RPB helps after its own pullback.

## Recommended experiment order

1. Sweep `RPB_SPECTRAL_BLEND` while fixing full Gram geometry and disabling neuron-row adaptation.
2. At the spectral winner, sweep `RPB_PRECOND_BLEND`.
3. At the joint winner, enable Nor-style adaptation and sweep `RPB_NOR_BETA2`.

This staged design prevents a positive or negative interaction from hiding which mechanism caused the result.

## Full confirmation

After the three timeout-capped stages select a joint candidate, use
`rpb_hybrid_full_confirmation_h200.sh`. Tasks 0–2 run that candidate at seeds 0, 1,
and 2. Task 3 reruns the established `p=0.9, eta=25, spectral=0, precond=1,
Nor=off` control through the same training file. The script requests ten hours so the
extra spectral work cannot recreate the six-hour timeout seen in earlier full runs.

## Weight-row diagnostics for the later Muown decision

The training file also records the maximum, mean, maximum-to-mean ratio, and
layer-averaged coefficient of variation of the QKV weight-row norms. Growth is
reported relative to initialization. These diagnostics are cheap and let the Muown
branch wait for direct evidence of row-scale drift rather than mixing a new
parameterization into the first hybrid sweep.
