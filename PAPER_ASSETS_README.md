# MLOPT paper asset generator

This package generates all frozen figures, tables, CSVs, statistical summaries, and ready-to-paste LaTeX prose from the raw Cycle A-C and bounded 353M run directories. It performs no training.

Required run directories under `~/project_pi_das227/kp875/LLM_optimizer`:

- `sweep_runs_cycle_a_baselines_h100_v1`
- `sweep_runs_cycle_a_companion_h100_v1`
- `sweep_runs_cycle_a_v_h100_v1`
- `sweep_runs_cycle_b_h100_main_v1`
- `sweep_runs_cycle_b_long10b_h200_v1`
- `sweep_runs_cycle_c_h100_main_v1`
- `sweep_runs_cycle_c_long10b_spectral_h200_v1`
- `sweep_runs_scale353m_tuning_v1`
- `scale353m_tuning_selection_v1`
- `sweep_runs_scale353m_confirmation_v1`

Optional but recommended for the full ablation table:

- `sweep_runs_fisher_corrected_nm_v1`

The generator validates the headline statistics against the frozen Cycle C and held-out 353M summaries. It fails rather than silently emitting inconsistent paper numbers.
