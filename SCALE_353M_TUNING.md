# Bounded 353M Newton-Muon and spectral-transition tuning

This sidecar runs after the 124M MLOPT suite is frozen. It does not change the primary 124M claims.

## Scientific purpose

The frozen Cycle-C transfer result showed final parity and a large early compute-quality advantage at 353M without scale-specific tuning. This experiment asks whether a small, symmetric tuning budget changes either conclusion.

## Tuning protocol

- Model: 24 layers, 16 heads, width 1024, 353,453,056 parameters.
- Horizon: 7500 updates, batch 512, sequence length 1024, about 3.93B tokens.
- Hardware: one H200 per run.
- Tuning seed: 3.
- Budget: 8 Newton-Muon configurations and 8 spectral-transition configurations.
- Selection metric: validation loss at step 7500 only.
- Held-out confirmation: seeds 4, 5, and 6 for the selected configuration from each family.

The tuning budget is deliberately bounded. It is not a new broad search and it must not change the frozen 124M finalist.

## Newton-Muon grid

The eight configurations reuse the bounded family tested at 124M:

- historical split: O/MLP 0.00040 / 0.95, QKV 0.00044 / 0.97;
- uniform LR 0.00036, 0.00040, 0.00044, or 0.00048 at momentum 0.95;
- uniform LR 0.00036, 0.00040, or 0.00044 at momentum 0.97.

## Spectral-transition grid

The frozen center uses Fisher Q/K RMS 0.00022, spectral blend 0.75, direction blend 500-1000, and scale ramp 1000-1500. Seven local variations test:

- Q/K RMS 0.00018 and 0.00026;
- spectral blend 0.50 and 1.00;
- an earlier 400-900 handoff;
- a later 700-1400 handoff;
- a uniform Newton-Muon 0.00040 / 0.95 late endpoint.

## Interpretation

The 353M study is supporting evidence. The primary paper table remains the five-seed 124M H100 comparison. The 353M result can strengthen the compute-frontier claim only if the held-out confirmation preserves the early advantage and final parity after both families receive the same tuning budget.
