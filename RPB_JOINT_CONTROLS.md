# RPB Joint Geometry Sweep and Exact Controls

`train_gpt_rpb_joint_controls.py` extends the corrected hybrid RPB script with
three QKV optimizer modes selected by `QKV_OPT_MODE`:

- `hybrid`: softened activation-space RPB, optional Gram/identity blend,
  spectral shaping, and optional Nor-style output-row adaptation.
- `newton_muon`: raw QKV weight gradient, right input-Gram preconditioning
  before momentum, followed by separate Q/K/V Newton–Schulz matrix-sign maps.
- `muon`: raw QKV weight gradient, momentum, and separate Q/K/V matrix-sign
  maps. The non-QKV matrix optimizer also disables its right preconditioner in
  this mode, yielding a full-model Muon control.

The exact-control paths use the standard differentiable `c_attn` linear layer,
so QKV weights receive ordinary gradients. The hybrid path retains the custom
activation-gradient capture used by RPB.

## Sweep files

- `generate_rpb_joint_sweep_configs.py` deterministically creates 64 tasks:
  56 hybrid configurations, 4 Newton–Muon controls, and 4 Muon controls.
- `rpb_joint_sweep_configs.tsv` is the generated fixed configuration table.
- `rank_rpb_joint_sweep.py` ranks all methods at the latest validation step
  shared by every successful run and reports the best configuration by mode.

The first joint sweep fixes the slower geometry knobs at refresh 32, EWMA 0.95,
ridge 0.2, `h_sigma=8`, and five Newton–Schulz steps. It jointly searches eta,
row-sign power, spectral blend, preconditioner blend, RPB momentum, and Nor
adaptation.
