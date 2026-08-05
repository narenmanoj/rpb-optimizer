# RPB-to-Newton–Muon Bridge Experiment

## Purpose

The broad joint sweep showed that spectral shaping closes a large fraction of the gap between RPB and Newton–Muon, but the best hybrid still trails the exact same-harness Newton–Muon control. This experiment isolates the remaining difference rather than expanding the old hybrid grid.

The central question is:

> Does any part of the RPB activation-space geometry improve an otherwise exact Newton–Muon QKV update?

## Exact common pipeline

Every bridge configuration uses the same downstream pipeline:

1. apply the full cached input-Gram inverse;
2. apply the inverse **before** momentum;
3. use the same Nesterov momentum rule;
4. apply separate Q/K/V Newton–Schulz matrix-sign maps;
5. use the same `-lr * sqrt(d)` matrix-update scale.

This removes the scale, ordering, and spectral-map differences that separated the previous hybrid from the exact control.

## Two candidate directions

The backward pass produces both:

- the raw QKV weight gradient, `G`;
- an RPB activation-space candidate, `A`, built from softened token-row gradients, the attention smoothness radius, and headwise normalization.

Each Q/K/V block of `A` is Frobenius-matched to the corresponding block of `G`. The optimizer then forms

```text
G_mix = (1 - alpha) * G + alpha * A
```

and Frobenius-matches each mixed block back to `G` before the common Newton–Muon pipeline.

`RPB_NM_BRIDGE_BLEND=alpha` therefore has literal endpoints:

- `alpha=0`: exact Newton–Muon;
- `alpha=1`: only the RPB activation direction, but under Newton–Muon ordering, matrix-sign shaping, and update scale.

A negative best-bridge-minus-control validation gap would show that the RPB activation construction adds value beyond exact Newton–Muon.

## RPB geometry controls

### Token-row power

`RPB_ROWSIGN_POWER=p` controls

```text
g_i / ||g_i||^p
```

before the activation numerator is pulled into weight space.

The sweep tests:

```text
p = 0, 0.25, 0.50, 0.70, 0.85
```

### Radius-pattern blend

`RPB_RADIUS_BLEND=rho` controls the headwise smoothness-radius pattern:

- `rho=1`: retain the current per-head `r*` values;
- `rho=0`: replace them by one uniform radius within the layer;
- intermediate values interpolate the pattern.

The final block norm is matched later, so this primarily changes direction rather than global scale.

### Head-normalization blend

`RPB_HEADNORM_BLEND=nu` controls the per-(Q/K/V, head) max-row normalization:

- `nu=1`: current RPB normalization;
- `nu=0`: no headwise max-row normalization;
- intermediate values interpolate the scale pattern.

At `p=0`, `rho=0`, and `nu=0`, the RPB candidate is collinear with the raw gradient. With `alpha=1`, this provides an internal Newton–Muon recovery check.

## Sweep

The 64-task table contains:

- 56 bridge configurations from explicit anchors plus a deterministic Sobol design;
- 8 exact Newton–Muon controls.

The bridge search spans:

```text
QKV learning rate:   0.00032, 0.00036, 0.00040, 0.00044, 0.00048
QKV momentum:        0.90, 0.95, 0.97
bridge alpha:        0.125, 0.25, 0.50, 0.75, 1.00
row power:           0, 0.25, 0.50, 0.70, 0.85
radius blend:        0, 0.50, 1.00
headnorm blend:      0, 0.50, 1.00
```

The slower settings remain fixed:

```text
refresh interval = 32
Gram EWMA        = 0.95
ridge            = 0.20
h_sigma          = 8
NS steps         = 5
seed             = 0
```

Nor adaptation is disabled in this experiment because its measured effect was small and the present goal is to isolate the activation geometry.

## Required recovery checks

Before trusting the sweep, the smoke test compares:

1. exact `newton_muon` mode;
2. bridge mode with `alpha=0`;
3. bridge mode with `alpha=1, p=0, radius=0, headnorm=0`;
4. a nontrivial bridge candidate.

At a common validation step, the first three should be very close. A material discrepancy means the bridge does not yet recover the intended control and the full sweep should not launch.
