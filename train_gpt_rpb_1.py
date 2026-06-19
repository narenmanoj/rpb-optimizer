import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from triton_kernels import XXT, ba_plus_cAA
import diagnostics as diag_mod

# -----------------------------------------------------------------------------
# Custom operators: activation XtX accumulation (for the Muon right-preconditioner)

def _dummy_scalar_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(())

# compile once at module scope (do not define @torch.compile inside the custom op call path)
@torch.compile
def _accum_xtx_impl(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    A = x_2d.transpose(0, 1)
    XXT(A, out=tmp)
    tmp.mul_(1.0 / x_2d.size(0))
    accum.add_(tmp)
    count.add_(1.0)
    return _dummy_scalar_like(accum)

@torch.compile
def _accum_xtx_blocks4_impl(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    N, fourD = x_2d.shape
    assert fourD % 4 == 0
    D = fourD // 4
    A = x_2d.view(N, 4, D).permute(1, 2, 0)  # [4, D, N]
    XXT(A, out=tmp)
    tmp.mul_(1.0 / N)
    accum.add_(tmp)
    count.add_(1.0)
    return _dummy_scalar_like(accum)

@torch.library.custom_op("nanogpt::accum_xtx", mutates_args=("accum", "count", "tmp"))
@torch.no_grad()
def accum_xtx_op(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    return _accum_xtx_impl(x_2d, accum, count, tmp)

@accum_xtx_op.register_fake
def accum_xtx_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())

@torch.library.custom_op("nanogpt::accum_xtx_blocks4", mutates_args=("accum", "count", "tmp"))
@torch.no_grad()
def accum_xtx_blocks4_op(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    return _accum_xtx_blocks4_impl(x_2d, accum, count, tmp)

@accum_xtx_blocks4_op.register_fake
def accum_xtx_blocks4_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())

# -----------------------------------------------------------------------------
# RPB (row-norm preconditioned bound) optimizer for the attention QKV map.
#
# Implements the update derived in smoothness_bound_and_update_rule.md, restricted
# to the c_attn (QKV) weight of each transformer block. The note studies the single
# head F(Q,K,V) = softmax(QK^T / sqrt(d_h)) V in the activation-space row norm
# ||A||_{inf,2} = max_i ||A_{i:}||_2 and produces, per head, a radius-dependent local
# smoothness constant
#
#   C_t(r) = h_sigma (v+r)(q+k+2r)^2 / d_h + 2(v+r)/sqrt(d_h) + 2(q+k+2r)/sqrt(d_h),
#
# with q,k,v the current per-head activation row norms ||Q||,||K||,||V||_{inf,2}.
# The step radius r* per head is the unique nonnegative root of
#
#   Phi_t'(r) = -S_G + r C_t(r) + 0.5 r^2 C_t'(r) = 0,
#
# where S_G = ||G_Q||_{1,2} + ||G_K||_{1,2} + ||G_V||_{1,2} is the dual (1,2) norm of
# the activation gradients of that head. The row-normalized activation targets are
# T_{Q,K,V} = -eta r* rsgn(G_{Q,K,V}); they are pulled back to the shared weight via
# the minimum-Frobenius solution ΔW = T^T Z (Z^T Z + eps I)^{-1}, where Z is the
# (rmsnorm'd) attention input. See the note's caveat: for N >> d_model this is the
# least-squares projection of the activation target, not an exact realization.
#
# Per the model surgery below, each CausalSelfAttention captures, every training step
# and accumulated across grad-accumulation microbatches:
#   rpb_M     [3d, d]   sum_i rsgn(g_i) z_i^T            (unscaled pullback numerator)
#   rpb_gram  [d, d]    sum_i z_i z_i^T                  (input Gram Z^T Z)
#   rpb_sg    [3, H]    sum_i ||g_i||_2 per (Q/K/V, head)  (-> S_G per head)
#   rpb_rownorm [3, H]  max_i ||a_i||_2 per (Q/K/V, head)  (-> q,k,v per head)
#   rpb_count  scalar   total token count N
# rsgn and the row norms are computed per head over the d_h slice; rotary is norm-
# preserving so the pre-rotary capture point gives exact q,k row norms.

@torch.library.custom_op("nanogpt::rpb_rownorm", mutates_args=("rownorm_max",))
@torch.no_grad()
def rpb_rownorm_op(qkv_2d: Tensor, rownorm_max: Tensor, n_head: int, d_h: int) -> Tensor:
    N = qkv_2d.size(0)
    a = qkv_2d.float().view(N, 3, n_head, d_h)
    cur = a.norm(dim=-1).amax(dim=0)          # [3, n_head]
    torch.maximum(rownorm_max, cur, out=rownorm_max)
    return _dummy_scalar_like(rownorm_max)

@rpb_rownorm_op.register_fake
def rpb_rownorm_fake(qkv_2d: Tensor, rownorm_max: Tensor, n_head: int, d_h: int):
    return rownorm_max.new_empty(())

@torch.library.custom_op("nanogpt::rpb_accum", mutates_args=("M_accum", "gram_accum", "sg_accum", "count"))
@torch.no_grad()
def rpb_accum_op(g_2d: Tensor, z_2d: Tensor, M_accum: Tensor, gram_accum: Tensor,
                 sg_accum: Tensor, count: Tensor, n_head: int, d_h: int) -> Tensor:
    N = g_2d.size(0)
    g = g_2d.float()
    z = z_2d.float()
    gv = g.view(N, 3, n_head, d_h)
    rn = gv.norm(dim=-1, keepdim=True)                       # [N, 3, n_head, 1]
    rsgn = (gv / rn.clamp_min(1e-12)).view(N, -1)            # [N, 3d]
    M_accum.add_(rsgn.transpose(0, 1) @ z)                   # [3d, d]
    gram_accum.add_(z.transpose(0, 1) @ z)                   # [d, d]
    sg_accum.add_(rn.squeeze(-1).sum(dim=0))                 # [3, n_head]
    count.add_(float(N))
    return _dummy_scalar_like(M_accum)

@rpb_accum_op.register_fake
def rpb_accum_fake(g_2d: Tensor, z_2d: Tensor, M_accum: Tensor, gram_accum: Tensor,
                   sg_accum: Tensor, count: Tensor, n_head: int, d_h: int):
    return M_accum.new_empty(())


class _QKVCapture(torch.autograd.Function):
    """qkv = x2d @ W^T, capturing the RPB statistics in the backward pass.

    The weight receives no gradient (returns None): the RPB optimizer drives it from
    the captured buffers, not from W.grad. The input gradient flows normally so the
    rest of the network trains as usual.
    """
    @staticmethod
    def forward(ctx, x2d: Tensor, weight: Tensor, ref: dict, capture: bool):
        qkv = x2d @ weight.to(x2d.dtype).t()
        ctx.save_for_backward(x2d, weight)
        ctx.ref = ref
        ctx.capture = bool(capture)
        if ctx.capture:
            torch.ops.nanogpt.rpb_rownorm(qkv.detach(), ref["rownorm"], ref["n_head"], ref["d_h"])
        return qkv

    @staticmethod
    def backward(ctx, g: Tensor):
        x2d, weight = ctx.saved_tensors
        grad_x = g @ weight.to(g.dtype)
        if ctx.capture:
            ref = ctx.ref
            torch.ops.nanogpt.rpb_accum(
                g.detach().reshape(-1, g.size(-1)), x2d.detach(),
                ref["M"], ref["gram"], ref["sg"], ref["count"], ref["n_head"], ref["d_h"],
            )
        return grad_x, None, None, None


class RPB(torch.optim.Optimizer):
    """Row-norm preconditioned bound update for attention QKV weights.

    lr plays the role of the damping eta in T = -eta r* rsgn(G); it is scheduled
    exactly like the other optimizers' learning rates. eps_gram/ridge_mult damp the
    Gram inverse (Z^T Z + ridge I)^{-1}; because the captured Gram is averaged over
    tokens, the ridge is taken relative to its mean diagonal.

    Momentum follows the Muon convention: a buffer buf <- momentum*buf + M is kept on
    the pulled-back numerator M = rsgn(G)^T Z (the gradient-like quantity, before the
    per-head r* scaling and Gram inverse), and with nesterov the smoothed direction is
    M <- M + momentum*buf. As with Muon, momentum is applied before the optimizer's
    own transform. Note that, unlike Muon (whose Newton-Schulz step renormalizes the
    direction), the RPB transform does not renormalize M, so momentum carries the usual
    ~1/(1-momentum) steady-state gain; scale eta down accordingly when enabling it.

    Gram preconditioner refresh -- mirrors the Muon right-preconditioner exactly. The
    preconditioner is the damped Gram inverse (E[Z^T Z] + ridge I)^{-1}. As in Muon:
    a per-layer EWMA (decay precond_ewma, matching Muon's 0.950) of the per-step input
    Gram is held; the damped Cholesky inverse is recomputed only every
    precond_refresh_period steps (first refresh at step precond_refresh_period-1, like
    Muon's t%32==0 schedule); and the cached inverse -- identity until that first
    refresh -- is reused on the steps in between. The per-step numerator M (with
    momentum) and the per-head r* are computed every step; only the inversion is throttled.

    One deliberate deviation, forced by the (different) update rule: the EWMA covariance
    is SEEDED with the first observed Gram instead of lerping up from precond_init_diag*I.
    Muon can lerp from a near-zero init because its Newton-Schulz step renormalizes the
    update, making it invariant to the preconditioner's absolute scale; the RPB update is
    NOT renormalized, so an under-warmed (too-small) covariance would inflate the inverse
    and blow up the step. Seeding keeps the inverse correctly scaled from the first refresh.
    """
    def __init__(self, params, lr=0.5, momentum=0.95, nesterov=True, h_sigma=8.0,
                 r_max=None, ridge_mult=0.2, eps_gram=1e-8, bisect_iters=60,
                 precond_refresh_period=32, precond_ewma=0.950, precond_init_diag=0.001):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.h_sigma = float(h_sigma)
        self.r_max = None if r_max is None else float(r_max)
        self.ridge_mult = float(ridge_mult)
        self.eps_gram = float(eps_gram)
        self.bisect_iters = int(bisect_iters)
        # Gram preconditioner refresh schedule (mirrors Muon).
        self.precond_refresh_period = int(precond_refresh_period)
        self.precond_ewma = float(precond_ewma)
        self.precond_init_diag = float(precond_init_diag)
        self.global_step = 0
        self._regime_step = 0   # refresh schedule origin (mirrors Muon; never reset here)
        self.last_diag = {}   # populated each step() with aggregate r*/S_G diagnostics

    def _solve_rstar(self, q, k, v, d_h, S_G):
        """Vectorized bisection for r* over a tensor of heads. q,k,v,S_G are [H]."""
        sqrt_d = float(d_h) ** 0.5
        hs = self.h_sigma

        def C(r):
            s = q + k + 2.0 * r
            return hs * (v + r) * (s * s) / d_h + 2.0 * (v + r) / sqrt_d + 2.0 * s / sqrt_d

        def Cp(r):
            s = q + k + 2.0 * r
            return hs * (s * s + 4.0 * (v + r) * s) / d_h + 6.0 / sqrt_d

        def phip(r):
            return -S_G + r * C(r) + 0.5 * r * r * Cp(r)

        lo = torch.zeros_like(S_G)
        if self.r_max is not None:
            hi = torch.full_like(S_G, self.r_max)
        else:
            hi = torch.ones_like(S_G)
            for _ in range(64):
                need = phip(hi) < 0.0
                if not bool(need.any()):
                    break
                hi = torch.where(need, hi * 2.0, hi)

        for _ in range(self.bisect_iters):
            mid = 0.5 * (lo + hi)
            neg = phip(mid) < 0.0
            lo = torch.where(neg, mid, lo)
            hi = torch.where(neg, hi, mid)
        rstar = 0.5 * (lo + hi)
        return torch.where(S_G > 0.0, rstar, torch.zeros_like(rstar))

    @torch.no_grad()
    def step(self):
        # Refresh the Gram preconditioner only every precond_refresh_period steps,
        # using Muon's exact schedule: t = since+1, refresh when t % period == 0 (so the
        # first refresh is at step period-1). global_step is set from the training loop
        # for resume-safety; it also advances internally so the schedule holds if unset.
        since = max(0, int(self.global_step) - int(self._regime_step))
        do_refresh = (((since + 1) % self.precond_refresh_period) == 0)
        did_refresh = False
        rstar_sum = 0.0; rstar_max = 0.0; sg_sum = 0.0; nheads = 0; nlayers = 0
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                ref = getattr(p, "_rpb_ref", None)
                if ref is None:
                    continue
                cnt = float(ref["count"].item())
                if cnt <= 0.0:
                    continue

                d = ref["d"]
                n_head = ref["n_head"]
                d_h = ref["d_h"]
                state = self.state[p]

                M = ref["M"] / cnt                             # [3d, d] averaged numerator

                # Momentum on the gradient-like numerator (Muon convention), applied
                # before the per-head r* scaling and Gram inverse below.
                if momentum != 0.0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(M)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(M)
                    if nesterov:
                        M = M.add(buf, alpha=momentum)
                    else:
                        M = buf

                # Gram preconditioner (mirrors Muon): EWMA covariance + cached inverse,
                # re-inverted only on refresh steps. precond_cov starts at init_diag*I and
                # precond_inv at identity (so steps before the first refresh apply I, as in
                # Muon). On refresh: EWMA the covariance toward this step's Gram and
                # recompute the damped Cholesky inverse. NOTE: the covariance is seeded with
                # the first observed Gram instead of lerping from init_diag*I -- see the
                # class docstring for why this single deviation is required for RPB.
                if "precond_cov" not in state:
                    cov = torch.zeros(d, d, device=M.device, dtype=torch.float32)
                    cov.diagonal().fill_(self.precond_init_diag)
                    state["precond_cov"] = cov
                    state["precond_inv"] = torch.eye(d, device=M.device, dtype=torch.float32)
                    state["precond_cov_seeded"] = False

                if do_refresh:
                    gram = ref["gram"] / cnt                   # [d, d] averaged Z^T Z (this step)
                    cov = state["precond_cov"]
                    if state["precond_cov_seeded"]:
                        cov.lerp_(gram, 1.0 - self.precond_ewma)   # EWMA toward refresh-step Gram
                    else:
                        cov.copy_(gram)                            # seed on first refresh
                        state["precond_cov_seeded"] = True

                    # Damped Gram inverse via Cholesky, ridge relative to mean diagonal.
                    diag = cov.diagonal()
                    ridge = (diag.mean() * self.ridge_mult + self.eps_gram).clamp_min(self.eps_gram)
                    K = cov.clone()
                    K.diagonal().add_(ridge)
                    L, info = torch.linalg.cholesky_ex(K, upper=False, check_errors=False)
                    if int(info.item()) != 0:
                        state["precond_inv"].copy_(torch.eye(d, device=cov.device, dtype=cov.dtype))
                    else:
                        state["precond_inv"].copy_(torch.cholesky_inverse(L, upper=False))
                    did_refresh = True

                inv = state["precond_inv"]

                # Per-head geometry and radius.
                rn = ref["rownorm"]                            # [3, n_head]
                q, k, v = rn[0], rn[1], rn[2]
                S_G = ref["sg"].sum(dim=0)                      # [n_head]
                rstar = self._solve_rstar(q, k, v, d_h, S_G)   # [n_head]

                rstar_sum += float(rstar.sum()); rstar_max = max(rstar_max, float(rstar.max()))
                sg_sum += float(S_G.sum()); nheads += rstar.numel(); nlayers += 1

                # T^T Z = -eta r* rsgn(G)^T Z, with r* applied per head-slice of the
                # 3d output rows (same r* across that head's Q,K,V blocks).
                scale_block = rstar.repeat_interleave(d_h)     # [d]
                scale = torch.cat([scale_block, scale_block, scale_block])  # [3d]
                Mp = scale.unsqueeze(1) * M                    # [3d, d]
                dW = (Mp @ inv).mul_(-lr)                      # [3d, d]
                p.data.add_(dW.to(p.dtype))

                # Reset accumulators for the next step.
                ref["M"].zero_()
                ref["gram"].zero_()
                ref["sg"].zero_()
                ref["count"].zero_()
                ref["rownorm"].zero_()

        self.global_step += 1
        self.last_diag = {
            "rpb/r_star_mean": (rstar_sum / nheads) if nheads else 0.0,
            "rpb/r_star_max": rstar_max,
            "rpb/S_G_mean": (sg_sum / nheads) if nheads else 0.0,
            "rpb/layers_updated": float(nlayers),
            "rpb/precond_refresh": float(did_refresh),
        }

# -----------------------------------------------------------------------------
# Muon optimizer (drives all non-QKV transformer-block weights)

# @torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)

    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True

    X = X.contiguous()

    m = X.size(0)
    A = torch.empty((m, m), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    for _ in range(steps):
        XXT(X, out=A)
        ba_plus_cAA(A, beta=b, alpha=c, out=B)
        torch.mm(B, X, out=C)
        C.add_(X, alpha=a)
        X, C = C, X

    if transposed:
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    + Right-preconditioner (EWMA second moment of activations), refresh logic:
        do_refresh = (t%32==0)
        precond_ewma = 0.950
      On refresh steps: update EWMA and compute batched Cholesky inverse.
    + Applies the inverse as a right-preconditioner to gradients BEFORE momentum+NS.

    In this RPB variant Muon no longer owns the attention QKV weights (those are driven
    by the RPB optimizer); it preconditions the attention output proj, c_fc and c_proj.
    """
    def __init__(
        self, params, lr=3e-4, momentum=0.95, nesterov=True, backend_steps=5,
        precond_init_diag: float = 0.001, precond_ridge_mult: float = 0.2, precond_eps: float = 1e-8,
        lr_mult_max: float = 1.0, lr_mult_ramp_steps: int = 32,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps)
        super().__init__(params, defaults)

        self.precond_init_diag = float(precond_init_diag)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_eps = float(precond_eps)
        self.lr_mult_max = float(lr_mult_max)
        self.lr_mult_ramp_steps = int(lr_mult_ramp_steps)

        self.global_step = 0
        self._regime_step = 0
        self._precond_attached = False
        self._precond_ready = False
        self._precond_d = None
        self._refresh_map = []
        self._refresh_K = None

        self._apply_plan = None

    def _regime_schedule_(self, step: int) -> tuple[bool, float, float]:
        since = max(0, int(step) - int(self._regime_step))
        t = since + 1
        do_refresh = (t % 32 == 0)
        precond_ewma = 0.950

        ramp = float(self.lr_mult_ramp_steps)
        if ramp <= 1.0:
            lr_mult = self.lr_mult_max
        else:
            frac = min(float(since), ramp) / ramp
            lr_mult = 1.0 + (self.lr_mult_max - 1.0) * frac

        return bool(do_refresh), float(precond_ewma), float(lr_mult)

    def precond_flag_for_step(self, step: int) -> bool:
        do_refresh, _, _ = self._regime_schedule_(int(step))
        return self._precond_attached and do_refresh

    def attach_preconditioner(self):
        self._precond_attached = True
        self._finalize_precond_buffers_()

    def _iter_params_with_stats_(self):
        for group in self.param_groups:
            for p in group['params']:
                stref = getattr(p, "_stats_ref", None)
                if stref is not None:
                    yield p, stref

    def _init_precond_state_for_param_(self, p: Tensor, stref: dict) -> None:
        st = self.state[p]
        if "precond_kind" in st:
            return

        kind = stref["kind"]
        d = int(stref["d"])
        st["precond_kind"] = kind
        st["precond_d"] = d

        if self._precond_d is None:
            self._precond_d = d
        else:
            assert self._precond_d == d, f"Expected one d; got {self._precond_d} vs {d}"

        def _fp32_mat():
            t = torch.empty((d, d), device=p.device, dtype=torch.float32)
            t.zero_()
            t.diagonal().fill_(self.precond_init_diag)
            return t

        if kind in ("o", "c_fc"):
            st["precond_cov"] = _fp32_mat()
        elif kind == "c_proj":
            cov = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
            cov.zero_()
            cov.diagonal(dim1=-2, dim2=-1).fill_(self.precond_init_diag)
            st["precond_cov"] = cov

    @torch.no_grad()
    def _apply_precond_all_grads_batched_(self):
        if (not self._precond_attached) or (not self._precond_ready):
            return
        plan = self._apply_plan
        if plan is None:
            return
        d = plan["d"]

        if plan["g_o"] is not None:
            G = plan["g_o"]
            for i, p in enumerate(plan["o_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            torch.bmm(G, plan["inv_o"], out=G)
            for i, p in enumerate(plan["o_params"]):
                if p.grad is not None:
                    p.grad.copy_(G[i], non_blocking=True)

        if plan["g_fc"] is not None:
            G = plan["g_fc"]
            for i, p in enumerate(plan["fc_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            torch.bmm(G, plan["inv_fc"], out=G)
            for i, p in enumerate(plan["fc_params"]):
                if p.grad is not None:
                    p.grad.copy_(G[i], non_blocking=True)

        if plan["g_proj"] is not None:
            Gp = plan["g_proj"]
            for i, p in enumerate(plan["proj_params"]):
                if p.grad is None:
                    Gp[i].zero_()
                else:
                    Gp[i].copy_(p.grad, non_blocking=True)

            n = Gp.size(0)

            dst_in = plan["tmp_blocks_in"].view(n, 4, d, d)
            src_in = Gp.view(n, d, 4, d).permute(0, 2, 1, 3)  # [n,4,d,d] (strided)
            dst_in.copy_(src_in)

            B = plan["inv_proj4"].view(n * 4, d, d)
            torch.bmm(plan["tmp_blocks_in"], B, out=plan["tmp_proj_blocks"])

            src_out = plan["tmp_proj_blocks"].view(n, 4, d, d).permute(0, 2, 1, 3)  # [n,d,4,d]
            Gp.view(n, d, 4, d).copy_(src_out)

            for i, p in enumerate(plan["proj_params"]):
                if p.grad is not None:
                    p.grad.copy_(Gp[i], non_blocking=True)

    @torch.no_grad()
    def _finalize_precond_buffers_(self):
        if self._precond_ready:
            return

        refresh_map = []
        o_params, fc_params, proj_params = [], [], []

        for p, stref in self._iter_params_with_stats_():
            kind = stref["kind"]
            self._init_precond_state_for_param_(p, stref)

            if kind in ("o", "c_fc"):
                refresh_map.append((p, kind, -1))
            elif kind == "c_proj":
                for j in range(4):
                    refresh_map.append((p, kind, j))

            if kind == "o":
                o_params.append(p)
            elif kind == "c_fc":
                fc_params.append(p)
            elif kind == "c_proj":
                proj_params.append(p)

        self._refresh_map = refresh_map
        d = int(self._precond_d) if self._precond_d is not None else 0
        self._refresh_K = None if not refresh_map else torch.empty(
            (len(refresh_map), d, d),
            device=refresh_map[0][0].device,
            dtype=torch.float32
        )

        dev = refresh_map[0][0].device if refresh_map else torch.device("cuda")

        def alloc_grad_buf(params, out_mult):
            n = len(params)
            if n == 0:
                return None
            return torch.empty((n, out_mult * d, d), device=dev, dtype=torch.float32)

        plan = {
            "d": d,
            "o_params": o_params,
            "fc_params": fc_params,
            "proj_params": proj_params,

            "g_o":   alloc_grad_buf(o_params,   1),
            "g_fc":  alloc_grad_buf(fc_params,  4),

            "inv_o":   torch.empty((len(o_params),   d, d), device=dev, dtype=torch.float32) if o_params else None,
            "inv_fc":  torch.empty((len(fc_params),  d, d), device=dev, dtype=torch.float32) if fc_params else None,

            "g_proj": torch.empty((len(proj_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_params else None,
            "inv_proj4": torch.empty((len(proj_params), 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
            "tmp_proj_blocks": torch.empty((len(proj_params) * 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
            "tmp_blocks_in":   torch.empty((len(proj_params) * 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
        }
        self._apply_plan = plan

        if plan["inv_o"] is not None:
            plan["inv_o"].zero_()
            plan["inv_o"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(o_params):
                self.state[p]["precond_inv_apply"] = plan["inv_o"][i]

        if plan["inv_fc"] is not None:
            plan["inv_fc"].zero_()
            plan["inv_fc"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(fc_params):
                self.state[p]["precond_inv_apply"] = plan["inv_fc"][i]

        if plan["inv_proj4"] is not None:
            plan["inv_proj4"].zero_()
            plan["inv_proj4"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(proj_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj4"][i]

        self._precond_ready = True

    @torch.no_grad()
    def _refresh_precond_all_batched_(self, do_inverse: bool, precond_ewma: float):
        if (not self._precond_attached) or (not self._precond_ready):
            return

        one_minus = 1.0 - float(precond_ewma)

        for p, stref in self._iter_params_with_stats_():
            st = self.state[p]
            kind = st["precond_kind"]

            cnt = stref["count"]
            w = (cnt > 0) * one_minus

            if kind in ("o", "c_fc"):
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)
            elif kind == "c_proj":
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)

        if not do_inverse:
            return
        if self._refresh_K is None or not self._refresh_map:
            return

        K = self._refresh_K
        d = int(self._precond_d)

        for i, (p, kind, sub) in enumerate(self._refresh_map):
            st = self.state[p]
            if kind in ("o", "c_fc"):
                K[i].copy_(st["precond_cov"])
            else:
                K[i].copy_(st["precond_cov"][sub])

        diag = K.diagonal(dim1=-2, dim2=-1)
        ridge = (diag.sum(dim=-1) / float(d)) * self.precond_ridge_mult + self.precond_eps
        diag.add_(ridge.unsqueeze(-1))

        L, info = torch.linalg.cholesky_ex(K, upper=False, check_errors=False)
        torch.cholesky_inverse(L, upper=False, out=K)

        if info.numel() == K.size(0):
            bad = info != 0
            if bad.any():
                K[bad].zero_()
                K[bad].diagonal(dim1=-2, dim2=-1).fill_(1.0)

        for i, (p, kind, sub) in enumerate(self._refresh_map):
            st = self.state[p]
            inv_i = K[i]
            if kind in ("o", "c_fc"):
                st["precond_inv_apply"].copy_(inv_i)
            else:
                st["precond_inv_apply"][sub].copy_(inv_i)

    def step(self):
        do_refresh, precond_ewma, lr_mult = self._regime_schedule_(self.global_step)
        do_inverse = bool(self._precond_attached and do_refresh)

        if self._precond_attached and do_refresh:
            self._finalize_precond_buffers_()
            self._refresh_precond_all_batched_(do_inverse=do_inverse, precond_ewma=precond_ewma)
            for _, stref in self._iter_params_with_stats_():
                stref["accum"].zero_()
                stref["count"].zero_()

        self._apply_precond_all_grads_batched_()

        for group in self.param_groups:
            lr = group['lr'] * lr_mult
            momentum = group['momentum']
            steps = group['backend_steps']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                if g.size(0) == 3 * g.size(1):
                    g = torch.cat([zeropower_via_newtonschulz5(g1, steps=steps) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_via_newtonschulz5(g, steps=steps)
                    scale = max(g.size(0), g.size(1))**0.5
                p.data.add_(g, alpha=-lr * scale)

        self.global_step += 1

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

        d = self.n_embd
        H = self.n_head

        # Muon preconditioner stats for the attention output projection only.
        self.o_xtx_accum   = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.xtx_tmp       = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.o_xtx_count   = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self.c_proj.weight._stats_ref = {"kind": "o", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}

        # RPB capture buffers for the QKV map (driven by the RPB optimizer).
        self.rpb_M       = nn.Buffer(torch.zeros(3 * d, d, dtype=torch.float32), persistent=False)
        self.rpb_gram    = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.rpb_sg      = nn.Buffer(torch.zeros(3, H, dtype=torch.float32), persistent=False)
        self.rpb_rownorm = nn.Buffer(torch.zeros(3, H, dtype=torch.float32), persistent=False)
        self.rpb_count   = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self._attach_rpb_ref()

    def _attach_rpb_ref(self):
        self.c_attn.weight._rpb_ref = {
            "M": self.rpb_M, "gram": self.rpb_gram, "sg": self.rpb_sg,
            "rownorm": self.rpb_rownorm, "count": self.rpb_count,
            "d": self.n_embd, "n_head": self.n_head, "d_h": self.head_dim,
        }

    def forward(self, x, precond_flag: bool = False):
        B, T, C = x.size()

        capture = torch.is_grad_enabled()
        x2d = x.reshape(-1, C)
        qkv = _QKVCapture.apply(x2d, self.c_attn.weight, self.c_attn.weight._rpb_ref, capture)
        qkv = qkv.view(B, T, 3 * C)

        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if precond_flag:
            y2d = y.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(y2d, self.o_xtx_accum, self.o_xtx_count, self.xtx_tmp)

        y = self.c_proj(y)
        return y

    def _apply(self, fn):
        super()._apply(fn)
        d = self.n_embd
        self.c_proj.weight._stats_ref = {"kind": "o", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}
        self._attach_rpb_ref()
        return self

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        d = config.n_embd
        self.fc_xtx_accum  = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_tmp    = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_count  = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
        self.proj_xtx_tmp   = nn.Buffer(torch.empty(4, d, d, dtype=torch.float32), persistent=False)
        self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc",   "d": d, "accum": self.fc_xtx_accum,   "count": self.fc_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "c_proj","d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}

    def forward(self, x, precond_flag: bool = False):
        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(x2d, self.fc_xtx_accum, self.fc_xtx_count, self.fc_xtx_tmp)

        x = self.c_fc(x)
        x = F.gelu(x)

        if precond_flag:
            z2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_blocks4(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)

        x = self.c_proj(x)
        return x

    def _apply(self, fn):
        super()._apply(fn)
        d = self.c_fc.weight.size(1)
        self.c_fc.weight._stats_ref = {"kind": "c_fc",   "d": d, "accum": self.fc_xtx_accum,   "count": self.fc_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "c_proj","d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        return self

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x, precond_flag: bool = False):
        x = x + self.attn_scale * self.attn(rmsnorm(x), precond_flag)
        x = x + self.mlp(rmsnorm(x), precond_flag)
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True, precond_flag: bool = False):
        precond_flag = bool(precond_flag) and self.training

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, precond_flag)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return int(header[2])

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size : int = 8*64
    device_batch_size : int = 64
    sequence_length : int = 1024
    num_iterations : int = 6200
    learning_rate : float = 0.0040
    rpb_eta : float = 0.025   # damping; momentum=0.95 adds ~1/(1-m)=20x steady-state gain
    warmup_iters : int = 0
    warmdown_iters : int = 1800
    weight_decay : float = 0
    val_loss_every : int = 100
    val_tokens : int = 10485760
    save_every : int = 1000      # periodic checkpoint cadence (0 disables)
    diag_every : int = 100       # cadence for grad/update/weight-norm diagnostics
    # RPB optimizer knobs (also overridable via env vars, see README)
    rpb_momentum : float = 0.95
    rpb_nesterov : bool = True
    rpb_h_sigma : float = 8.0    # softmax-Hessian constant in the curvature bound
    rpb_ridge_mult : float = 0.2 # Gram-inverse ridge, relative to mean diagonal
    rpb_r_max : float = 0.0      # trust-region cap on r* (0 => no cap)
    # RPB Gram-preconditioner refresh (mirrors the Muon right-preconditioner)
    rpb_precond_refresh_period : int = 32   # steps between Gram-inverse refreshes
    rpb_precond_ewma : float = 0.95         # EWMA decay of the Gram covariance
    rpb_precond_init_diag : float = 0.001   # initial covariance diagonal (pre-seed)
args = Hyperparameters()

# Optional env overrides for quick sweeps without editing the file (see README).
def _env_f(name, default):
    v = os.environ.get(name); return float(v) if v is not None else default
def _env_i(name, default):
    v = os.environ.get(name); return int(v) if v is not None else default
args.learning_rate = _env_f("LEARNING_RATE", args.learning_rate)
args.num_iterations = _env_i("NUM_ITERATIONS", args.num_iterations)
args.save_every = _env_i("SAVE_EVERY", args.save_every)
args.diag_every = _env_i("DIAG_EVERY", args.diag_every)
args.rpb_eta = _env_f("RPB_ETA", args.rpb_eta)
args.rpb_momentum = _env_f("RPB_MOMENTUM", args.rpb_momentum)
args.rpb_h_sigma = _env_f("RPB_HSIGMA", args.rpb_h_sigma)
args.rpb_ridge_mult = _env_f("RPB_RIDGE_MULT", args.rpb_ridge_mult)
args.rpb_r_max = _env_f("RPB_RMAX", args.rpb_r_max)
args.rpb_precond_refresh_period = _env_i("RPB_PRECOND_REFRESH", args.rpb_precond_refresh_period)
args.rpb_precond_ewma = _env_f("RPB_PRECOND_EWMA", args.rpb_precond_ewma)
args.rpb_precond_init_diag = _env_f("RPB_PRECOND_INIT_DIAG", args.rpb_precond_init_diag)

assert torch.cuda.is_available()
ddp_rank = 0
ddp_world_size = 1
device = 'cuda:0'
torch.cuda.set_device(0)
print(f"using device: {device}")
master_process = True

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

num_vocab = 50257
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768)).cuda()
model = torch.compile(model)
raw_model = model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# Parameter split:
#   AdamW -> lm_head (tied wte)
#   RPB   -> attention QKV weights (c_attn) of every block  [the note's update]
#   Muon  -> all other transformer-block weights (attn out proj, c_fc, c_proj)
qkv_weights = [blk.attn.c_attn.weight for blk in raw_model.transformer.h]
qkv_ids = {id(w) for w in qkv_weights}
muon_params = [p for p in raw_model.transformer.h.parameters() if id(p) not in qkv_ids]

optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(muon_params, lr=0.1*args.learning_rate, momentum=0.95)
optimizer2.attach_preconditioner()
optimizer3 = RPB(qkv_weights, lr=args.rpb_eta, momentum=args.rpb_momentum,
                 nesterov=args.rpb_nesterov, h_sigma=args.rpb_h_sigma,
                 ridge_mult=args.rpb_ridge_mult,
                 r_max=(args.rpb_r_max if args.rpb_r_max > 0 else None),
                 precond_refresh_period=args.rpb_precond_refresh_period,
                 precond_ewma=args.rpb_precond_ewma,
                 precond_init_diag=args.rpb_precond_init_diag)
optimizers = [optimizer1, optimizer2, optimizer3]

# Diagnostics: named param groups for grad/update/weight norms, plus TB/wandb sink.
# Note: RPB drives qkv_weights without a .grad (the optimizer reads captured buffers),
# so grad_norm/rpb is ~0 by construction; r*/S_G are logged separately via last_diag.
named_params = {
    "adamw": [p for p in optimizer1.param_groups[0]["params"]],
    "muon":  muon_params,
    "rpb":   qkv_weights,
}
opt_names = ["adamw", "muon", "rpb"]

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.warmdown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

if master_process:
    run_id = str(uuid.uuid4())
    os.makedirs('logs/%s/' % run_id, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

diag = diag_mod.Diagnostics(run_id, config=vars(args), enabled=master_process)
ckpt = diag_mod.CheckpointManager(run_id, enabled=master_process)

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 32:
        torch.cuda.synchronize()
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 33 else (step - 32) + 1

    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                _, loss = model(x_val, y_val, return_logits=False, precond_flag=False)
                val_loss += loss
        val_loss /= val_steps

        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            diag.log({"val/loss": float(val_loss), "time/train_ms": training_time_ms}, step)

        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        ckpt.save(step, raw_model, optimizers, extra=dict(code=code))
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    optimizer2.global_step = step
    optimizer3.global_step = step
    precond_flag = optimizer2.precond_flag_for_step(step)

    for _ in range(train_accumulation_steps):
        with ctx:
            _, loss = model(x, y, return_logits=False, precond_flag=precond_flag)
            train_loss = loss.detach()
            loss = loss / train_accumulation_steps
        x, y = train_loader.next_batch()
        loss.backward()

    # diagnostics: grad norms (grads live now), snapshot for update norms
    diag_step = master_process and (args.diag_every > 0 and step % args.diag_every == 0)
    diag_metrics = {}
    if diag_step:
        diag_metrics.update(diag_mod.grad_norms(named_params))
        snaps = diag_mod.snapshot_named(named_params)

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()

    if diag_step:
        diag_metrics.update(diag_mod.update_norms(named_params, snaps))
        diag_metrics.update(diag_mod.weight_norms(named_params))
        for name, opt in zip(opt_names, optimizers):
            diag_metrics[f"lr/{name}"] = opt.param_groups[0]["lr"]
        diag_metrics.update(optimizer3.last_diag)   # r*/S_G aggregates from RPB
        diag.log(diag_metrics, step)

    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        diag.log({"train/loss": train_loss.item()}, step)

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    diag.close()
