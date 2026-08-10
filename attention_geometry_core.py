"""Matrix-free attention-geometry primitives used by the local-model audit.

The module intentionally keeps the faithful geometry separate from the training
harness.  Tensor conventions:

* x:       [B, T, d_model]
* q, k:    [B, H, T, d_head] after RoPE
* A, B:    [H, d_head, d_model] weight perturbations for W_Q and W_K
* score U: [B, H, T, T], with zero entries above the causal diagonal

The Fisher and q2 operators use a mean over query rows by default.  This makes
their scale compatible with an averaged language-model loss and keeps outer
step scale separate from coefficient shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import time

import torch
from torch import Tensor

Pair = Tuple[Tensor, Tensor]


def pair_zeros_like(x: Pair) -> Pair:
    return torch.zeros_like(x[0]), torch.zeros_like(x[1])


def pair_clone(x: Pair) -> Pair:
    return x[0].clone(), x[1].clone()


def pair_dot(x: Pair, y: Pair) -> Tensor:
    return (x[0].float() * y[0].float()).sum() + (x[1].float() * y[1].float()).sum()


def pair_norm(x: Pair) -> Tensor:
    return pair_dot(x, x).clamp_min(0.0).sqrt()


def pair_add(x: Pair, y: Pair, alpha: float = 1.0) -> Pair:
    return x[0] + float(alpha) * y[0], x[1] + float(alpha) * y[1]


def pair_scale(x: Pair, alpha: Tensor | float) -> Pair:
    return x[0] * alpha, x[1] * alpha


def pair_sub(x: Pair, y: Pair) -> Pair:
    return x[0] - y[0], x[1] - y[1]


def pair_to(x: Pair, *, dtype: torch.dtype) -> Pair:
    return x[0].to(dtype), x[1].to(dtype)


def causal_mask(T: int, device: torch.device) -> Tensor:
    return torch.ones((T, T), device=device, dtype=torch.bool).tril_()


def apply_rope_bthd(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply the repository's RoPE convention to [B,T,H,d_h]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def inverse_rope_bthd(y: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Inverse of apply_rope_bthd for the same cached cos/sin tensors."""
    half = y.shape[-1] // 2
    y1, y2 = y[..., :half], y[..., half:]
    return torch.cat((y1 * cos - y2 * sin, y1 * sin + y2 * cos), dim=-1)


@dataclass
class AttentionGeometry:
    x: Tensor                  # [B,T,D]
    q: Tensor                  # [B,H,T,dh], post-RoPE
    k: Tensor                  # [B,H,T,dh], post-RoPE
    cos: Tensor                # [1,T,1,dh/2]
    sin: Tensor                # [1,T,1,dh/2]
    scores: Tensor             # [B,H,T,T], invalid entries -inf
    p: Tensor                  # [B,H,T,T], invalid entries zero
    mask: Tensor               # [T,T] bool

    @property
    def B(self) -> int:
        return int(self.x.shape[0])

    @property
    def T(self) -> int:
        return int(self.x.shape[1])

    @property
    def D(self) -> int:
        return int(self.x.shape[2])

    @property
    def H(self) -> int:
        return int(self.q.shape[1])

    @property
    def dh(self) -> int:
        return int(self.q.shape[-1])

    @property
    def n_rows(self) -> int:
        return self.B * self.T


def build_geometry(x: Tensor, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> AttentionGeometry:
    x = x.float()
    q = q.float()
    k = k.float()
    B, H, T, dh = q.shape
    mask = causal_mask(T, q.device)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(dh))
    scores = scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.where(mask.view(1, 1, T, T), p, torch.zeros_like(p))
    return AttentionGeometry(x=x, q=q, k=k, cos=cos.float(), sin=sin.float(), scores=scores, p=p, mask=mask)


def joint_jvp_parts(direction: Pair, geom: AttentionGeometry) -> Tuple[Tensor, Tensor, Tensor]:
    """Return linear score tangent U and its query/key activation components."""
    A, B = direction
    A = A.float()
    B = B.float()
    dq_pre = torch.einsum("btd,hkd->bthk", geom.x, A)
    dk_pre = torch.einsum("btd,hkd->bthk", geom.x, B)
    dq = apply_rope_bthd(dq_pre, geom.cos, geom.sin).transpose(1, 2).contiguous()
    dk = apply_rope_bthd(dk_pre, geom.cos, geom.sin).transpose(1, 2).contiguous()
    U = (
        torch.matmul(dq, geom.k.transpose(-2, -1))
        + torch.matmul(geom.q, dk.transpose(-2, -1))
    ) / math.sqrt(float(geom.dh))
    U = U * geom.mask.view(1, 1, geom.T, geom.T)
    return U, dq, dk


def joint_jvp(direction: Pair, geom: AttentionGeometry) -> Tensor:
    return joint_jvp_parts(direction, geom)[0]


def joint_bilinear_remainder(direction: Pair, geom: AttentionGeometry) -> Tensor:
    """Exact quadratic Q/K score remainder for a unit outer scale."""
    _, dq, dk = joint_jvp_parts(direction, geom)
    R = torch.matmul(dq, dk.transpose(-2, -1)) / math.sqrt(float(geom.dh))
    return R * geom.mask.view(1, 1, geom.T, geom.T)


def joint_vjp(U: Tensor, geom: AttentionGeometry) -> Pair:
    U = U.float() * geom.mask.view(1, 1, geom.T, geom.T)
    scale = 1.0 / math.sqrt(float(geom.dh))
    dq = torch.matmul(U, geom.k) * scale             # [B,H,T,dh]
    dk = torch.matmul(U.transpose(-2, -1), geom.q) * scale
    dq_pre = inverse_rope_bthd(dq.transpose(1, 2), geom.cos, geom.sin)
    dk_pre = inverse_rope_bthd(dk.transpose(1, 2), geom.cos, geom.sin)
    A = torch.einsum("bthk,btd->hkd", dq_pre, geom.x)
    B = torch.einsum("bthk,btd->hkd", dk_pre, geom.x)
    return A, B


def fisher_score_apply(U: Tensor, p: Tensor, c: Tensor, *, reduction: str = "mean") -> Tensor:
    U = U.float()
    p = p.float()
    c = c.float()
    centered = U - (p * U).sum(dim=-1, keepdim=True)
    V = c.unsqueeze(-1) * p * centered
    if reduction == "mean":
        V = V / float(U.shape[0] * U.shape[2])
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")
    return V


def fisher_energy(U: Tensor, p: Tensor, c: Tensor, *, reduction: str = "mean") -> Tensor:
    return (U.float() * fisher_score_apply(U, p, c, reduction=reduction)).sum()


def q2_score_apply(U: Tensor, mask: Tensor, *, reduction: str = "mean") -> Tensor:
    """Apply the faithful q2 quadratic score operator rowwise.

    q2(u)^2 = m/2 ||Pi u||_2^2 on a row with m visible entries, hence
    the quadratic operator is (m/2) Pi.
    """
    U = U.float()
    T = U.shape[-1]
    m = torch.arange(1, T + 1, device=U.device, dtype=U.dtype).view(1, 1, T, 1)
    msk = mask.view(1, 1, T, T).to(U.dtype)
    mean = (U * msk).sum(dim=-1, keepdim=True) / m
    V = 0.5 * m * (U - mean) * msk
    if reduction == "mean":
        V = V / float(U.shape[0] * U.shape[2])
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")
    return V


def q2_energy(U: Tensor, mask: Tensor, *, reduction: str = "mean") -> Tensor:
    return (U.float() * q2_score_apply(U, mask, reduction=reduction)).sum()


def make_quadratic_operator(
    geom: AttentionGeometry,
    *,
    kind: str,
    c: Optional[Tensor],
    damping: float,
    reduction: str = "mean",
) -> Callable[[Pair], Pair]:
    kind = str(kind).lower()

    def apply(direction: Pair) -> Pair:
        U = joint_jvp(direction, geom)
        if kind == "fisher":
            if c is None:
                raise ValueError("Fisher operator requires coefficients c")
            V = fisher_score_apply(U, geom.p, c, reduction=reduction)
        elif kind == "q2":
            V = q2_score_apply(U, geom.mask, reduction=reduction)
        else:
            raise ValueError(f"Unknown operator kind={kind!r}")
        out = joint_vjp(V, geom)
        return out[0] + float(damping) * direction[0], out[1] + float(damping) * direction[1]

    return apply


def estimate_relative_damping(
    geom: AttentionGeometry,
    gradient: Pair,
    *,
    kind: str,
    c: Optional[Tensor],
    damping_rel: float,
    damping_floor: float = 1e-8,
    reduction: str = "mean",
) -> Tuple[float, float]:
    base = make_quadratic_operator(geom, kind=kind, c=c, damping=0.0, reduction=reduction)
    Hg = base(gradient)
    denom = float(pair_dot(gradient, gradient).clamp_min(1e-30))
    rayleigh = max(float(pair_dot(gradient, Hg)) / denom, 0.0)
    damping = max(float(damping_floor), float(damping_rel) * max(rayleigh, float(damping_floor)))
    return damping, rayleigh


def pcg_solve(
    apply: Callable[[Pair], Pair],
    rhs: Pair,
    *,
    iterations: int,
    tol: float = 0.0,
    preconditioner: Optional[Callable[[Pair], Pair]] = None,
) -> Tuple[Pair, Dict[str, object]]:
    """Small deterministic PCG implementation for a pair of Q/K matrices."""
    x = pair_zeros_like(rhs)
    r = pair_clone(rhs)
    z = pair_clone(r) if preconditioner is None else preconditioner(r)
    p = pair_clone(z)
    rz = pair_dot(r, z)
    rhs_norm = float(pair_norm(rhs).clamp_min(1e-30))
    residuals: List[float] = [float(pair_norm(r)) / rhs_norm]
    alphas: List[float] = []
    breakdown = False

    for _ in range(int(iterations)):
        Ap = apply(p)
        denom = pair_dot(p, Ap)
        if not torch.isfinite(denom) or float(denom) <= 0.0:
            breakdown = True
            break
        alpha = rz / denom
        x = pair_add(x, p, float(alpha))
        r = pair_add(r, Ap, -float(alpha))
        rel = float(pair_norm(r)) / rhs_norm
        residuals.append(rel)
        alphas.append(float(alpha))
        if tol > 0.0 and rel <= tol:
            break
        z = pair_clone(r) if preconditioner is None else preconditioner(r)
        rz_new = pair_dot(r, z)
        if not torch.isfinite(rz_new):
            breakdown = True
            break
        beta = rz_new / rz.clamp_min(1e-30)
        p = pair_add(z, p, float(beta))
        rz = rz_new

    return x, {
        "iterations": len(alphas),
        "relative_residuals": residuals,
        "alphas": alphas,
        "breakdown": breakdown,
    }


def masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    T = scores.shape[-1]
    s = scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.where(mask.view(1, 1, T, T), p, torch.zeros_like(p))


def mirror_bregman(base_scores: Tensor, U: Tensor, p: Tensor, c: Tensor, mask: Tensor, *, reduction: str = "mean") -> Tensor:
    T = base_scores.shape[-1]
    new_scores = base_scores + U
    old_lse = torch.logsumexp(base_scores, dim=-1)
    new_lse = torch.logsumexp(new_scores.masked_fill(~mask.view(1, 1, T, T), float("-inf")), dim=-1)
    row = new_lse - old_lse - (p * U).sum(dim=-1)
    value = (c * row).sum()
    if reduction == "mean":
        value = value / float(U.shape[0] * U.shape[2])
    elif reduction != "sum":
        raise ValueError(f"Unknown reduction={reduction!r}")
    return value


def solve_joint_mirror_newton(
    geom: AttentionGeometry,
    gradient: Pair,
    c: Tensor,
    *,
    damping: float,
    eta: float = 1.0,
    newton_iters: int = 2,
    cg_iters: int = 3,
    cg_tol: float = 0.0,
    line_search_steps: int = 8,
    reduction: str = "mean",
) -> Tuple[Pair, Dict[str, object]]:
    """Solve the convex joint tangent mirror subproblem by damped Newton-CG.

    One Newton iteration from zero is the Fisher quadratic solve.  Two iterations
    test whether the nonlinear log-partition geometry changes the direction.
    """
    d = pair_zeros_like(gradient)
    history: List[Dict[str, float]] = []

    def objective(direction: Pair) -> Tensor:
        U = joint_jvp(direction, geom)
        linear = pair_dot(gradient, direction)
        breg = mirror_bregman(geom.scores, U, geom.p, c, geom.mask, reduction=reduction)
        reg = 0.5 * float(damping) * pair_dot(direction, direction)
        return linear + breg / float(eta) + reg

    for outer in range(int(newton_iters)):
        U = joint_jvp(d, geom)
        p_new = masked_softmax(geom.scores + U, geom.mask)
        delta_p = c.unsqueeze(-1) * (p_new - geom.p)
        if reduction == "mean":
            delta_p = delta_p / float(geom.n_rows)
        residual_geo = joint_vjp(delta_p, geom)
        residual = (
            gradient[0] + residual_geo[0] / float(eta) + float(damping) * d[0],
            gradient[1] + residual_geo[1] / float(eta) + float(damping) * d[1],
        )

        def hess(v: Pair) -> Pair:
            Jv = joint_jvp(v, geom)
            V = fisher_score_apply(Jv, p_new, c, reduction=reduction)
            Hv = joint_vjp(V, geom)
            return (
                Hv[0] / float(eta) + float(damping) * v[0],
                Hv[1] / float(eta) + float(damping) * v[1],
            )

        delta, cg_info = pcg_solve(
            hess,
            pair_scale(residual, -1.0),
            iterations=int(cg_iters),
            tol=float(cg_tol),
        )
        before = float(objective(d))
        step_scale = 1.0
        accepted = False
        after = before
        for _ in range(int(line_search_steps)):
            trial = pair_add(d, delta, step_scale)
            after = float(objective(trial))
            if math.isfinite(after) and after <= before:
                d = trial
                accepted = True
                break
            step_scale *= 0.5
        history.append({
            "outer": float(outer),
            "objective_before": before,
            "objective_after": after,
            "newton_scale": step_scale,
            "residual_norm": float(pair_norm(residual)),
            "cg_iterations": float(cg_info["iterations"]),
            "cg_final_residual": float(cg_info["relative_residuals"][-1]),
            "accepted": float(accepted),
        })
        if not accepted:
            break

    return d, {"history": history, "objective": float(objective(d))}


def unit_coefficients(geom: AttentionGeometry) -> Tensor:
    return torch.ones((geom.B, geom.H, geom.T), device=geom.x.device, dtype=torch.float32)


def projected_coefficients(
    *,
    geom: AttentionGeometry,
    v_pre: Tensor,          # [B,T,H,dh]
    g_out: Tensor,          # [B,T,D], grad w.r.t. output-projected attention output
    w_o: Tensor,            # [D,D]
    beta: float = 0.0,
    normalize: str = "median",
    floor: float = 1e-3,
) -> Tuple[Tensor, Dict[str, Tensor | float]]:
    """Current-point projected coefficient with a trace-majorized head term.

    rho uses the certified center bound max_j ||z_aj-c_a|| with the token mean as
    center.  This is shared over query rows, avoiding a quadratic value-distance
    tensor.  The coefficient shape is optionally normalized so its absolute scale
    remains a separate outer/damping lever.
    """
    B, T, H, dh = v_pre.shape
    mask = geom.mask.view(1, T, T)
    b_list: List[Tensor] = []
    rho_list: List[Tensor] = []
    w_o = w_o.float()
    v_pre = v_pre.float()
    g_out = g_out.float()

    for h in range(H):
        block = w_o[:, h * dh:(h + 1) * dh]                  # [D,dh]
        gh = torch.matmul(g_out, block)                      # [B,T,dh]
        vh = v_pre[:, :, h, :]                               # [B,T,dh]
        proj = torch.matmul(gh, vh.transpose(-2, -1))        # [B,T,T]
        ph = geom.p[:, h]
        mean = (ph * proj).sum(dim=-1, keepdim=True)
        centered = proj - mean
        centered = centered.masked_fill(~mask, float("-inf"))
        bh = centered.amax(dim=-1).clamp_min(0.0)            # [B,T]
        b_list.append(bh)

        # rho <= max_j ||W_O,h (v_j - token_mean)||.
        vc = vh - vh.mean(dim=1, keepdim=True)
        gram = torch.matmul(block.transpose(0, 1), block)     # [dh,dh]
        n2 = torch.einsum("bti,ij,btj->bt", vc, gram, vc).clamp_min(0.0)
        rho_list.append(n2.sqrt().amax(dim=1))               # [B]

    b = torch.stack(b_list, dim=1)                            # [B,H,T]
    rho = torch.stack(rho_list, dim=1)                        # [B,H]
    coupling = float(beta) * rho * rho.sum(dim=1, keepdim=True)
    raw = b + coupling.unsqueeze(-1)

    normalize = str(normalize).lower()
    if normalize == "none":
        scale = torch.ones((), device=raw.device, dtype=raw.dtype)
    elif normalize == "mean":
        scale = raw.mean().clamp_min(1e-12)
    elif normalize == "median":
        positive = raw[raw > 0]
        scale = positive.median().clamp_min(1e-12) if positive.numel() else raw.new_tensor(1.0)
    else:
        raise ValueError(f"Unknown coefficient normalization {normalize!r}")
    c = (raw / scale).clamp_min(float(floor))
    return c, {
        "b": b,
        "rho": rho,
        "raw": raw,
        "normalization_scale": float(scale),
        "raw_mean": float(raw.mean()),
        "raw_median": float(raw.median()),
        "c_mean": float(c.mean()),
        "c_min": float(c.min()),
        "c_max": float(c.max()),
    }


def score_oscillation(U: Tensor, mask: Tensor) -> Tensor:
    T = U.shape[-1]
    m = mask.view(1, 1, T, T)
    hi = U.masked_fill(~m, float("-inf")).amax(dim=-1)
    lo = U.masked_fill(~m, float("inf")).amin(dim=-1)
    return hi - lo


def direction_diagnostics(
    direction: Pair,
    gradient: Pair,
    geom: AttentionGeometry,
    *,
    c: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    rho: Optional[Tensor] = None,
    beta: float = 0.0,
    reduction: str = "mean",
) -> Dict[str, float]:
    U = joint_jvp(direction, geom)
    R = joint_bilinear_remainder(direction, geom)
    linear_norm = float(U.norm())
    fisher_unit = float(fisher_energy(U, geom.p, unit_coefficients(geom), reduction=reduction))
    out: Dict[str, float] = {
        "gradient_dot": float(pair_dot(gradient, direction)),
        "direction_norm": float(pair_norm(direction)),
        "score_linear_norm": linear_norm,
        "score_osc_max": float(score_oscillation(U, geom.mask).max()),
        "score_osc_mean": float(score_oscillation(U, geom.mask).mean()),
        "fisher_energy_unit": fisher_unit,
        "q2_energy": float(q2_energy(U, geom.mask, reduction=reduction)),
        "bilinear_norm": float(R.norm()),
        "bilinear_ratio": float(R.norm() / U.norm().clamp_min(1e-30)),
    }
    if c is not None:
        out["fisher_energy_shaped"] = float(fisher_energy(U, geom.p, c, reduction=reduction))
    if b is not None and rho is not None:
        p = geom.p
        centered = U - (p * U).sum(dim=-1, keepdim=True)
        t = (p * centered.square()).sum(dim=-1).clamp_min(0.0).sqrt()  # [B,H,T]
        rho_bht = rho.unsqueeze(-1)
        cross = float(beta) * (rho_bht * t).sum(dim=1).square()        # [B,T]
        diag = (b * t.square()).sum(dim=1)
        frac = cross / (cross + diag).clamp_min(1e-30)
        out["cross_head_fraction_mean"] = float(frac.mean())
        out["cross_head_fraction_max"] = float(frac.max())
    return out


def match_pair_norm(candidate: Pair, reference: Pair, eps: float = 1e-30) -> Pair:
    cn = pair_norm(candidate)
    rn = pair_norm(reference)
    if float(cn) <= eps:
        return pair_clone(reference)
    return pair_scale(candidate, rn / cn)


def pair_cosine(x: Pair, y: Pair, eps: float = 1e-30) -> float:
    return float(pair_dot(x, y) / (pair_norm(x) * pair_norm(y)).clamp_min(eps))


def solve_rstar_scalar(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sg: Tensor,
    gy: Tensor,
    *,
    d_h: int,
    h_sigma: float = 8.0,
    r_max: Optional[float] = None,
    iterations: int = 60,
) -> Tensor:
    """Vectorized original RPB radius solve over heads."""
    sqrt_d = math.sqrt(float(d_h))

    def C(r: Tensor) -> Tensor:
        s = q + k + 2.0 * r
        return float(h_sigma) * (v + r) * s.square() / float(d_h) + 2.0 * (v + r) / sqrt_d + 2.0 * s / sqrt_d

    def Cp(r: Tensor) -> Tensor:
        s = q + k + 2.0 * r
        return float(h_sigma) * (s.square() + 4.0 * (v + r) * s) / float(d_h) + 6.0 / sqrt_d

    active = (sg > 0) & (gy > 0)
    sg_eff = torch.where(active, sg, torch.zeros_like(sg))
    gy_eff = torch.where(active, gy, torch.ones_like(gy))

    def derivative(r: Tensor) -> Tensor:
        return -sg_eff + gy_eff * (r * C(r) + 0.5 * r.square() * Cp(r))

    lo = torch.zeros_like(sg)
    if r_max is not None and r_max > 0:
        hi = torch.full_like(sg, float(r_max))
    else:
        hi = torch.ones_like(sg)
        for _ in range(64):
            need = derivative(hi) < 0
            if not bool(need.any()):
                break
            hi = torch.where(need, hi * 2.0, hi)
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        neg = derivative(mid) < 0
        lo = torch.where(neg, mid, lo)
        hi = torch.where(neg, hi, mid)
    return torch.where(active, 0.5 * (lo + hi), torch.zeros_like(sg))


def rpb_audit_directions(
    *,
    x: Tensor,                  # [B,T,D]
    qkv_pre: Tensor,            # [B,T,3D]
    g_qkv: Tensor,              # [B,T,3D]
    g_y: Tensor,                # [B,T,D] grad before output projection
    n_head: int,
    ridge_mult: float = 0.2,
    h_sigma: float = 8.0,
    r_max: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Return faithful and projection-aware QKV RPB directions for one layer.

    Both outputs are fused [3D,D] descent increments with no outer learning-rate
    multiplier.  The feasible variant recomputes one scalar per head after the
    unit row-sign target has been projected through the current input matrix.
    """
    x = x.float()
    qkv_pre = qkv_pre.float()
    g_qkv = g_qkv.float()
    g_y = g_y.float()
    B, T, threeD = qkv_pre.shape
    D = threeD // 3
    dh = D // int(n_head)
    N = B * T
    xf = x.reshape(N, D)
    af = qkv_pre.reshape(N, 3, n_head, dh)
    gf = g_qkv.reshape(N, 3, n_head, dh)
    gyf = g_y.reshape(N, n_head, dh)

    cov = xf.transpose(0, 1) @ xf / float(N)
    ridge = cov.diagonal().mean() * float(ridge_mult) + 1e-8
    K = cov.clone()
    K.diagonal().add_(ridge)
    L, info = torch.linalg.cholesky_ex(K, check_errors=False)
    inv = torch.eye(D, device=x.device, dtype=torch.float32) if int(info.item()) else torch.cholesky_inverse(L)

    rownorm = af.norm(dim=-1).amax(dim=0)                       # [3,H]
    q, k, v = rownorm[0], rownorm[1], rownorm[2]
    sg = gf.norm(dim=-1).sum(dim=(0, 1))                        # [H], sum Q/K/V and tokens
    gy = gyf.norm(dim=-1).sum(dim=0)                            # [H]
    rstar = solve_rstar_scalar(q, k, v, sg, gy, d_h=dh, h_sigma=h_sigma, r_max=r_max)

    faithful = torch.zeros((3, n_head, dh, D), device=x.device, dtype=torch.float32)
    feasible = torch.zeros_like(faithful)
    ascent_heads = 0
    feasible_alpha: List[float] = []
    realized_radii: List[float] = []

    sqrt_d = math.sqrt(float(dh))

    def C_head(r: Tensor, qh: Tensor, kh: Tensor, vh: Tensor) -> Tensor:
        s = qh + kh + 2.0 * r
        return float(h_sigma) * (vh + r) * s.square() / float(dh) + 2.0 * (vh + r) / sqrt_d + 2.0 * s / sqrt_d

    def Cp_head(r: Tensor, qh: Tensor, kh: Tensor, vh: Tensor) -> Tensor:
        s = qh + kh + 2.0 * r
        return float(h_sigma) * (s.square() + 4.0 * (vh + r) * s) / float(dh) + 6.0 / sqrt_d

    for h in range(n_head):
        D0_blocks: List[Tensor] = []
        E_blocks: List[Tensor] = []
        for block in range(3):
            g = gf[:, block, h, :]
            direction = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            D0 = -(direction.transpose(0, 1) @ xf / float(N)) @ inv
            E = xf @ D0.transpose(0, 1)
            D0_blocks.append(D0)
            E_blocks.append(E)
            faithful[block, h] = rstar[h] * D0

        a = -sum(float((gf[:, block, h, :] * E_blocks[block]).sum()) for block in range(3))
        R = max(float(E.norm(dim=-1).max()) for E in E_blocks)
        realized_radii.append(R)
        if a <= 0.0 or R <= 0.0 or float(gy[h]) <= 0.0:
            ascent_heads += 1
            feasible_alpha.append(0.0)
            continue

        qh, kh, vh = q[h], k[h], v[h]
        gyh = gy[h]

        def dpsi(alpha: Tensor) -> Tensor:
            rr = alpha * R
            return -a + alpha * (R * R) * gyh * C_head(rr, qh, kh, vh) + 0.5 * alpha.square() * (R ** 3) * gyh * Cp_head(rr, qh, kh, vh)

        lo = torch.zeros((), device=x.device)
        hi = torch.ones((), device=x.device)
        for _ in range(64):
            if float(dpsi(hi)) >= 0.0:
                break
            hi = hi * 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if float(dpsi(mid)) < 0.0:
                lo = mid
            else:
                hi = mid
        alpha = float(0.5 * (lo + hi))
        feasible_alpha.append(alpha)
        for block in range(3):
            feasible[block, h] = alpha * D0_blocks[block]

    return faithful.reshape(3 * D, D), feasible.reshape(3 * D, D), {
        "rstar_mean": float(rstar.mean()),
        "rstar_max": float(rstar.max()),
        "sg_mean": float(sg.mean()),
        "gy_mean": float(gy.mean()),
        "feasible_alpha_mean": sum(feasible_alpha) / max(len(feasible_alpha), 1),
        "feasible_alpha_max": max(feasible_alpha) if feasible_alpha else 0.0,
        "realized_unit_radius_mean": sum(realized_radii) / max(len(realized_radii), 1),
        "projection_ascent_heads": float(ascent_heads),
    }
