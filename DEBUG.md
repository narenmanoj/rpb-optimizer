# Debugging the Newton-Muon "immediate NaN"

**Symptom:** training `train_gpt_newton_muon_1.py` prints a sane `train_loss` on
the first step, then `train_loss` is `nan` from the next step onward.

**Hypothesis we are testing:** the forward pass is stock PyTorch and produces a
clean loss; the NaN is introduced by the **first optimizer step**, specifically
the Muon Newton-Schulz orthogonalization, which is the only place the Triton
kernels (`XXT`, `ba_plus_cAA`) run before the failure. We want to localize it to
one of:

1. the Triton kernels are numerically wrong, or
2. the kernels are fine but break when called as raw Triton inside `@torch.compile`, or
3. the bug is elsewhere (forward/backward producing NaN grads).

All diagnostics are **opt-in** via environment variables / a standalone script,
so default behavior is unchanged.

## Requirements

- A CUDA GPU (the trainer asserts `torch.cuda.is_available()`).
- The Python env from `requirements.txt` (PyTorch + Triton).
- For the **training-loop** diagnostics only: the FineWeb data bins under
  `data/fineweb10B/` (same as a normal run). The **kernel unit test** below
  needs no data.

Please capture and send back the **full stdout** of each step you run.

---

## Diagnostic 1 — kernel unit test (no data needed, run this first)

Directly checks whether `XXT` and `ba_plus_cAA` compute the right thing, and
compares all four Newton-Schulz backends against a pure-PyTorch reference.

```bash
python3 test_ns_kernels.py ; echo "exit=$?"
```

- `exit=0` → all kernels and backends match the reference. The kernel **math is
  correct**; the bug is in how they are *invoked* (go to Diagnostic 2/3) or
  elsewhere.
- `exit=1` → something printed `FAIL`. Send the output; the failing line names
  the kernel (`XXT` / `ba_plus_cAA`) and which backend diverged. In particular,
  if `triton_compile` FAILs but `triton_eager` PASSes, that is the
  raw-Triton-under-`torch.compile` problem.

---

## Diagnostic 2 — swap the Newton-Schulz backend (training run)

The trainer now selects the orthogonalization backend via `NS_BACKEND`:

| `NS_BACKEND`     | what runs                                         |
|------------------|---------------------------------------------------|
| `triton_compile` | **default** — Triton kernels under `@torch.compile` |
| `triton_eager`   | same Triton kernels, **not** compiled             |
| `torch_compile`  | pure-PyTorch Newton-Schulz, compiled              |
| `torch_eager`    | pure-PyTorch Newton-Schulz, not compiled          |

Run the trainer for just a handful of steps under each backend and note whether
`train_loss` goes `nan`. You can `Ctrl-C` once you have ~5 step lines.

```bash
# baseline (reproduce the bug)
NS_BACKEND=triton_compile python3 train_gpt_newton_muon_1.py 2>&1 | tee run_triton_compile.log

# same kernels, no compile
NS_BACKEND=triton_eager   python3 train_gpt_newton_muon_1.py 2>&1 | tee run_triton_eager.log

# no Triton at all (pure torch), compiled
NS_BACKEND=torch_compile  python3 train_gpt_newton_muon_1.py 2>&1 | tee run_torch_compile.log

# no Triton at all, not compiled
NS_BACKEND=torch_eager    python3 train_gpt_newton_muon_1.py 2>&1 | tee run_torch_eager.log
```

Each log starts with a line like `[debug] NS_BACKEND=...` confirming the backend.

**Interpretation:**

- NaN under `triton_compile` but **not** under `torch_*` → the Triton kernels (or
  their use) are the cause.
- NaN under `triton_compile` but **not** `triton_eager` → it's the
  raw-Triton-under-`torch.compile` interaction, **not** the kernel math.
- NaN under **all** backends (including `torch_eager`) → the kernels are *not*
  the cause; the bug is elsewhere (LR/init/data/backward) → run Diagnostic 3.

---

## Diagnostic 3 — NaN guard (pinpoint backward vs optimizer)

Set `NAN_GUARD=1` to scan grads after each backward and params after each
optimizer step. On the first non-finite value it prints a banner saying whether
**grads** were already NaN (forward/backward bug) or only **params** went NaN
after the step (optimizer / Newton-Schulz bug), then exits cleanly.

Combine with any backend, e.g. the default:

```bash
NAN_GUARD=1 NS_BACKEND=triton_compile python3 train_gpt_newton_muon_1.py 2>&1 | tee run_nanguard.log
```

Look for the `[NAN_GUARD]` banner. The two lines that matter:

```
[NAN_GUARD]   first non-finite GRAD  (pre-step):  ...
[NAN_GUARD]   first non-finite PARAM (post-step): ...
```

- GRAD `None`, PARAM not `None` → grads were finite, params went NaN **after the
  step** ⇒ the optimizer update (Muon / Newton-Schulz) is the culprit.
- GRAD not `None` → grads were already non-finite **before** the step ⇒ the
  forward/backward is the culprit, not the optimizer.

---

## What to send back

For each command: the `[debug] NS_BACKEND=...` line, the first ~5 `train_loss`
lines, and any `[NAN_GUARD]` banner. Plus the full `test_ns_kernels.py` output
and its `exit=` code. That set is enough to localize the bug.
