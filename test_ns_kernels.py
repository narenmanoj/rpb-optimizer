"""
Standalone correctness test for the Triton kernels used by the Newton-Schulz /
Muon path, plus an end-to-end comparison of the three Newton-Schulz backends.

This file is self-contained on purpose: it imports ONLY triton_kernels.py, never
the training script (importing the trainer would kick off a full training run).

Run on the GPU box:

    python test_ns_kernels.py

It exits non-zero if any check fails. See DEBUG.md for interpretation.
"""

import sys
import torch

from triton_kernels import XXT, ba_plus_cAA

# Newton-Schulz coefficients, identical to the trainer.
A_COEF, B_COEF, C_COEF = (3.4445, -4.7750, 2.0315)

# Shapes that show up in the GPT-2 124M model (n_embd=768, 4*n_embd=3072), plus a
# couple of non-block-aligned sizes to stress the masking / mirror-store paths.
SHAPES = [(768, 768), (768, 3072), (3072, 768), (768, 2304), (100, 257), (256, 256)]


def _check(name, got, ref, atol):
    got_f = got.float()
    ref_f = ref.float()
    nan = bool(torch.isnan(got_f).any())
    max_err = (got_f - ref_f).abs().max().item()
    # Relative error guards against scale; bf16 accumulation needs a loose tol.
    denom = ref_f.abs().max().clamp_min(1e-6).item()
    rel = max_err / denom
    ok = (not nan) and (rel <= atol)
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name:28s} max_err={max_err:.4e} rel={rel:.4e} nan={nan}")
    return ok


def test_xxt():
    print("XXT  (C = A @ A.T):")
    ok = True
    for (m, k) in SHAPES:
        X = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, m, device="cuda", dtype=torch.bfloat16)
        XXT(X, out=out)
        ref = X.float() @ X.float().T
        ok &= _check(f"m={m},k={k}", out, ref, atol=2e-2)
    return ok


def test_ba_plus_caa():
    # ba_plus_cAA(A, beta=b, alpha=c) = c * A @ A.T + b * A, called by NS on a
    # SYMMETRIC A (= X @ X.T), which is the only regime it must be correct in.
    print("ba_plus_cAA (c*A@A.T + b*A, symmetric A):")
    ok = True
    for (m, _) in SHAPES:
        Xs = torch.randn(m, m, device="cuda", dtype=torch.float32)
        A = (Xs @ Xs.T).bfloat16()          # symmetric
        out = torch.empty(m, m, device="cuda", dtype=torch.bfloat16)
        ba_plus_cAA(A, beta=B_COEF, alpha=C_COEF, out=out)
        Af = A.float()
        ref = C_COEF * (Af @ Af.T) + B_COEF * Af
        ok &= _check(f"m={m}", out, ref, atol=2e-2)
    return ok


# --- the three Newton-Schulz backends, copied to avoid importing the trainer ---

def ns_triton(G, steps=5, eps=1e-7):
    X = G.bfloat16() / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X.contiguous()
    m = X.size(0)
    A = torch.empty((m, m), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    for _ in range(steps):
        XXT(X, out=A)
        ba_plus_cAA(A, beta=B_COEF, alpha=C_COEF, out=B)
        torch.mm(B, X, out=C)
        C.add_(X, alpha=A_COEF)
        X, C = C, X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def ns_torch(G, steps=5, eps=1e-7):
    X = G.bfloat16() / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X.contiguous()
    for _ in range(steps):
        A = X @ X.T
        B = B_COEF * A + C_COEF * (A @ A)
        X = A_COEF * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def test_newton_schulz():
    print("Newton-Schulz backends vs pure-torch reference:")
    ok = True
    ns_triton_c = torch.compile(ns_triton)
    ns_torch_c = torch.compile(ns_torch)
    for (m, k) in SHAPES:
        G = torch.randn(m, k, device="cuda", dtype=torch.float32)
        ref = ns_torch(G).float()                      # eager torch = ground truth
        for name, fn in [
            ("torch_eager", ns_torch),
            ("torch_compile", ns_torch_c),
            ("triton_eager", ns_triton),
            ("triton_compile", ns_triton_c),
        ]:
            out = fn(G).float()
            ok &= _check(f"m={m},k={k} {name}", out, ref, atol=5e-2)
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this test must run on the GPU box.")
        sys.exit(2)
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}\n")

    results = []
    results.append(("XXT", test_xxt()));            print()
    results.append(("ba_plus_cAA", test_ba_plus_caa())); print()
    results.append(("newton_schulz", test_newton_schulz())); print()

    print("=" * 60)
    all_ok = True
    for name, ok in results:
        print(f"  {name:16s} {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
