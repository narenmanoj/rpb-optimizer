import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import math
import json
import csv
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from triton_kernels import XXT, ba_plus_cAA
import diagnostics as diag_mod
from attention_geometry_core import (
    AttentionGeometry,
    build_geometry,
    direction_diagnostics,
    estimate_relative_damping,
    fisher_energy,
    joint_bilinear_remainder,
    joint_jvp,
    make_quadratic_operator,
    match_pair_norm,
    mirror_bregman,
    pair_add,
    pair_cosine,
    pair_dot,
    pair_norm,
    pair_scale,
    pcg_solve,
    projected_coefficients,
    q2_energy,
    rpb_audit_directions,
    score_oscillation,
    solve_joint_mirror_newton,
    unit_coefficients,
)

# Experimental RPB geometry controls in this variant:
#   RPB_ROWSIGN_POWER   : token-row gradient exponent (1=row-sign, 0=raw-gradient-like)
#   RPB_SPECTRAL_BLEND : 0=current RPB pullback, 1=Frobenius-matched matrix-sign pullback
#   RPB_PRECOND_BLEND  : 0=identity right geometry, 1=full cached Gram inverse
#   RPB_NOR_ENABLE     : post-update neuron-row second-moment normalization (NorMuon style)
# All blends preserve each Q/K/V block's Frobenius norm, so they primarily change direction.
#
# QKV_OPT_MODE selects the QKV optimizer path:
#   hybrid       : activation-space RPB construction + optional spectral/Nor stages
#   bridge       : exact Newton-Muon direction blended with an RPB activation direction
#   newton_muon  : raw QKV weight gradient -> Gram inverse -> momentum -> matrix sign
#   muon         : raw QKV weight gradient -> momentum -> matrix sign
#   fisher_qk     : joint pulled-back Fisher-CG on Q/K; Newton-Muon on V
# In newton_muon mode, the right preconditioner is applied before momentum, matching
# the Newton-Muon algorithm. In muon mode, all transformer matrices use ordinary Muon.

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
# This variant adds a geometry-preserving row-sign power via RPB_ROWSIGN_POWER.
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
#   rpb_sg    [3, H]    softened linear-decrease numerator per (Q/K/V, head)
#   rpb_gradmax [3, H]   max_i ||g_i||_2, used to preserve unit max-row direction
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

@torch.library.custom_op(
    "nanogpt::rpb_accum",
    mutates_args=("M_accum", "gram_accum", "sg_accum", "gradmax_accum", "count"),
)
@torch.no_grad()
def rpb_accum_op(g_2d: Tensor, z_2d: Tensor, M_accum: Tensor, gram_accum: Tensor,
                 sg_accum: Tensor, gradmax_accum: Tensor, count: Tensor,
                 n_head: int, d_h: int, rowsign_power: float) -> Tensor:
    """Accumulate a geometry-preserving softened row-sign direction.

    For p=rowsign_power in [0,1], first form d_i=g_i/||g_i||^p.  At optimizer.step(),
    each (Q/K/V, head) block is divided by c=max_i ||g_i||^(1-p), so the maximum
    row norm remains at most one and the smoothness radius r* retains its original
    row-norm meaning.  p=1 exactly recovers the original row-sign direction.
    """
    N = g_2d.size(0)
    g = g_2d.float()
    z = z_2d.float()
    gv = g.view(N, 3, n_head, d_h)
    rn = gv.norm(dim=-1, keepdim=True)                       # [N, 3, n_head, 1]

    denom = rn.clamp_min(1e-12).pow(float(rowsign_power))
    direction = (gv / denom).view(N, -1)                    # [N, 3d]
    M_accum.add_(direction.transpose(0, 1) @ z)              # [3d, d]
    gram_accum.add_(z.transpose(0, 1) @ z)                  # [d, d]

    # Before division by c, <g_i,d_i>=||g_i||^(2-p).
    sg_accum.add_(rn.squeeze(-1).pow(2.0 - float(rowsign_power)).sum(dim=0))
    cur_max = rn.squeeze(-1).amax(dim=0)                    # [3, n_head]
    torch.maximum(gradmax_accum, cur_max, out=gradmax_accum)

    count.add_(float(N))
    return _dummy_scalar_like(M_accum)

@rpb_accum_op.register_fake
def rpb_accum_fake(g_2d: Tensor, z_2d: Tensor, M_accum: Tensor, gram_accum: Tensor,
                   sg_accum: Tensor, gradmax_accum: Tensor, count: Tensor,
                   n_head: int, d_h: int, rowsign_power: float):
    return M_accum.new_empty(())

@torch.library.custom_op("nanogpt::rpb_accum_gy", mutates_args=("gy_accum",))
@torch.no_grad()
def rpb_accum_gy_op(gy_2d: Tensor, gy_accum: Tensor, n_head: int, d_h: int) -> Tensor:
    # gy_2d = grad of the loss w.r.t. the head output Y = P V, flattened to [N, n_head*d_h].
    # Accumulate g_Y = ||grad_Y L||_{1,2} = sum_i ||(grad_Y L)_{i:}||_2 per head (a SUM
    # over tokens, matching how S_G is accumulated, so the ratio S_G/g_Y is scale-free).
    N = gy_2d.size(0)
    gy = gy_2d.float().view(N, n_head, d_h)
    gy_accum.add_(gy.norm(dim=-1).sum(dim=0))                # [n_head]
    return _dummy_scalar_like(gy_accum)

@rpb_accum_gy_op.register_fake
def rpb_accum_gy_fake(gy_2d: Tensor, gy_accum: Tensor, n_head: int, d_h: int):
    return gy_accum.new_empty(())


class _QKVCapture(torch.autograd.Function):
    """qkv = x2d @ W^T, capturing activation-space statistics in backward.

    The ordinary hybrid path suppresses W.grad and updates QKV from the captured RPB
    statistics.  The bridge path additionally asks this function to return the raw
    weight gradient.  That lets one optimizer construct both the exact Newton-Muon
    direction and an RPB activation-space direction from the same forward/backward
    pass and interpolate between them without changing the model or data stream.
    """
    @staticmethod
    def forward(
        ctx, x2d: Tensor, weight: Tensor, ref: dict, capture: bool,
        rowsign_power: float, return_weight_grad: bool,
    ):
        qkv = x2d @ weight.to(x2d.dtype).t()
        ctx.save_for_backward(x2d, weight)
        ctx.ref = ref
        ctx.capture = bool(capture)
        ctx.rowsign_power = float(rowsign_power)
        ctx.return_weight_grad = bool(return_weight_grad)
        if ctx.capture:
            torch.ops.nanogpt.rpb_rownorm(qkv.detach(), ref["rownorm"], ref["n_head"], ref["d_h"])
        return qkv

    @staticmethod
    def backward(ctx, g: Tensor):
        x2d, weight = ctx.saved_tensors
        grad_x = g @ weight.to(g.dtype)
        grad_weight = None
        if ctx.capture:
            ref = ctx.ref
            g2d = g.detach().reshape(-1, g.size(-1))
            torch.ops.nanogpt.rpb_accum(
                g2d, x2d.detach(),
                ref["M"], ref["gram"], ref["sg"], ref["gradmax"], ref["count"],
                ref["n_head"], ref["d_h"], ctx.rowsign_power,
            )
            if ctx.return_weight_grad:
                # Match the standard Linear backward in float32.  Any global scale
                # difference from the averaged captured numerator disappears under
                # the blockwise matrix-sign map, but returning the true raw gradient
                # makes bridge_blend=0 an exact Newton-Muon recovery control.
                grad_weight = g2d.float().transpose(0, 1) @ x2d.detach().float()
                grad_weight = grad_weight.to(weight.dtype)
        return grad_x, grad_weight, None, None, None, None


class _AttnOutCapture(torch.autograd.Function):
    """Identity on the attention output Y = P V, capturing the output dual gradient
    norm g_Y = ||grad_Y L||_{1,2} per head in the backward pass.

    g_Y converts the head-map curvature C_t into a loss curvature in the radius solve
    (see smoothness_bound_and_update_rule.md, sec. 7); the radius depends on S_G/g_Y.
    Forward is the identity, so downstream computation (c_proj) is unaffected.
    """
    @staticmethod
    def forward(ctx, y: Tensor, ref: dict, capture: bool):
        ctx.ref = ref
        ctx.capture = bool(capture)
        return y

    @staticmethod
    def backward(ctx, g: Tensor):
        if ctx.capture:
            ref = ctx.ref
            torch.ops.nanogpt.rpb_accum_gy(
                g.detach().reshape(-1, g.size(-1)), ref["gy"], ref["n_head"], ref["d_h"],
            )
        return g, None, None


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

    Softened row-sign direction. With rowsign_power p in [0,1], each gradient row
    initially uses g/||g||^p. The optimizer then normalizes each (Q/K/V, head) block
    by max_i ||g_i||^(1-p), preserving a unit max-row norm and therefore preserving
    the meaning of the smoothness radius r*. p=1 is the original row-sign update;
    smaller p retains progressively more within-row gradient magnitude information.

    Optional spectral shaping. After the activation target is pulled back through the
    right preconditioner, each d-by-d Q/K/V block can be blended toward its approximate
    matrix sign. Both the matrix-sign endpoint and the final blend are Frobenius-matched
    to the unshaped block. Thus spectral_blend=0 exactly recovers the ordinary RPB
    direction, while spectral_blend=1 equalizes active singular values without changing
    the block update norm. Intermediate values partially flatten the singular spectrum.

    Optional right-geometry blend. precond_blend interpolates between the identity
    geometry and the cached Gram-inverse geometry. The identity endpoint is Frobenius-
    matched blockwise to the fully preconditioned update, and the final blend is matched
    again. Therefore precond_blend=1 exactly recovers the existing RPB pullback, while
    precond_blend=0 removes the directional effect of the Gram inverse without creating
    a trivial update-scale change.

    Optional neuron-row adaptation. After spectral shaping, a NorMuon-style EMA tracks
    the mean squared entries of every output-neuron row. Rows are divided by the square
    root of this statistic, then the whole QKV update is rescaled back to its original
    Frobenius norm. This changes row allocation while preserving global update scale.

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
                 precond_refresh_period=32, precond_ewma=0.950, precond_init_diag=0.001,
                 rowsign_power=1.0, spectral_blend=0.0, spectral_steps=5,
                 precond_blend=1.0, nor_enable=False, nor_beta2=0.95, nor_eps=1e-8):
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
        self.rowsign_power = float(rowsign_power)
        self.spectral_blend = float(spectral_blend)
        self.spectral_steps = int(spectral_steps)
        self.precond_blend = float(precond_blend)
        self.nor_enable = bool(nor_enable)
        self.nor_beta2 = float(nor_beta2)
        self.nor_eps = float(nor_eps)

        if not (0.0 <= self.rowsign_power <= 1.0):
            raise ValueError(f"rowsign_power must lie in [0,1], got {self.rowsign_power}")
        if not (0.0 <= self.spectral_blend <= 1.0):
            raise ValueError(f"spectral_blend must lie in [0,1], got {self.spectral_blend}")
        if self.spectral_steps < 1:
            raise ValueError(f"spectral_steps must be >=1, got {self.spectral_steps}")
        if not (0.0 <= self.precond_blend <= 1.0):
            raise ValueError(f"precond_blend must lie in [0,1], got {self.precond_blend}")
        if not (0.0 <= self.nor_beta2 < 1.0):
            raise ValueError(f"nor_beta2 must lie in [0,1), got {self.nor_beta2}")
        if self.nor_eps <= 0.0:
            raise ValueError(f"nor_eps must be positive, got {self.nor_eps}")

        self.global_step = 0
        self._regime_step = 0   # refresh schedule origin (mirrors Muon; never reset here)
        self.last_diag = {}   # populated each step() with aggregate r*/S_G diagnostics

    def _solve_rstar(self, q, k, v, d_h, S_G, g_Y):
        """Vectorized bisection for r* over a tensor of heads. q,k,v,S_G,g_Y are [H].

        Solves the units-corrected stationarity condition (see the .md, sec. 7-8)
            -S_G + g_Y * [ r C(r) + 0.5 r^2 C'(r) ] = 0,
        i.e. the head-map curvature C_t is weighted by the output dual gradient norm
        g_Y = ||grad_Y L||_{1,2} to become a loss curvature. g_Y == 1 recovers the old
        (mis-scaled) equation.
        """
        sqrt_d = float(d_h) ** 0.5
        hs = self.h_sigma

        def C(r):
            s = q + k + 2.0 * r
            return hs * (v + r) * (s * s) / d_h + 2.0 * (v + r) / sqrt_d + 2.0 * s / sqrt_d

        def Cp(r):
            s = q + k + 2.0 * r
            return hs * (s * s + 4.0 * (v + r) * s) / d_h + 6.0 / sqrt_d

        # Inactive heads (no gradient signal) get r* = 0; use g_eff = 1 there so the
        # bracket/bisection below stays finite (it is masked out at the end anyway).
        active = (S_G > 0.0) & (g_Y > 0.0)
        S_eff = torch.where(active, S_G, torch.zeros_like(S_G))
        g_eff = torch.where(active, g_Y, torch.ones_like(g_Y))

        def phip(r):
            return -S_eff + g_eff * (r * C(r) + 0.5 * r * r * Cp(r))

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
        return torch.where(active, rstar, torch.zeros_like(rstar))

    @staticmethod
    def _fro_match(candidate: Tensor, reference: Tensor, eps: float = 1e-12) -> Tensor:
        """Match candidate's Frobenius norm to reference without changing direction."""
        ref_norm = reference.norm()
        cand_norm = candidate.norm()
        matched = candidate * (ref_norm / cand_norm.clamp_min(eps))
        return torch.where(cand_norm > eps, matched, reference)

    def _apply_precond_blend(self, Mp: Tensor, inv: Tensor, d: int):
        """Blend identity and full right-preconditioned directions blockwise.

        lambda=1 exactly returns Mp@inv. lambda=0 returns the direction Mp, rescaled
        per Q/K/V block to the corresponding full-preconditioned Frobenius norm.
        The final blend is also norm matched, isolating directional geometry.
        """
        full = Mp @ inv
        lam = self.precond_blend
        if lam >= 1.0:
            return full, 0.0

        full_blocks = full.view(3, d, d)
        identity_blocks = Mp.view(3, d, d)
        full_norm = torch.linalg.vector_norm(full_blocks, dim=(1, 2), keepdim=True)
        identity_norm = torch.linalg.vector_norm(identity_blocks, dim=(1, 2), keepdim=True)

        identity_scaled = identity_blocks * (
            full_norm / identity_norm.clamp_min(1e-12)
        )
        cosine = (full_blocks * identity_scaled).sum(dim=(1, 2)) / (
            full_norm.flatten() *
            torch.linalg.vector_norm(identity_scaled, dim=(1, 2)).clamp_min(1e-12)
        ).clamp_min(1e-12)

        mixed = lam * full_blocks + (1.0 - lam) * identity_scaled
        mixed_norm = torch.linalg.vector_norm(mixed, dim=(1, 2), keepdim=True)
        mixed = mixed * (full_norm / mixed_norm.clamp_min(1e-12))
        mixed = torch.where(full_norm > 1e-12, mixed, full_blocks)

        return mixed.reshape(3 * d, d), float(cosine.mean())

    def _apply_spectral_blend(self, update: Tensor, d: int):
        """Blend each Q/K/V block toward its approximate matrix sign.

        The sign endpoint and the final mixture are Frobenius-matched to the original
        block. alpha=0 is exactly the input update; alpha=1 is a norm-matched matrix sign.
        """
        alpha = self.spectral_blend
        if alpha <= 0.0:
            return update, 0.0

        blocks = update.view(3, d, d)
        shaped_blocks = []

        for block in blocks:
            sign_block = zeropower_via_newtonschulz5(block, steps=self.spectral_steps)
            sign_block = self._fro_match(sign_block, block)
            mixed = (1.0 - alpha) * block + alpha * sign_block
            shaped_blocks.append(self._fro_match(mixed, block))

        shaped = torch.stack(shaped_blocks)
        block_norm = torch.linalg.vector_norm(blocks, dim=(1, 2)).clamp_min(1e-12)
        change = torch.linalg.vector_norm(shaped - blocks, dim=(1, 2)) / block_norm
        return shaped.reshape(3 * d, d), float(change.mean())

    @staticmethod
    def _qkv_row_cv(update: Tensor, d: int, eps: float = 1e-12) -> float:
        """Mean coefficient of variation of output-row norms over Q, K, and V blocks."""
        blocks = update.view(3, d, d)
        row_norms = torch.linalg.vector_norm(blocks, dim=2)  # [3, d]
        means = row_norms.mean(dim=1)                         # [3]
        cvs = row_norms.std(dim=1, unbiased=False) / means.clamp_min(eps)
        cvs = torch.where(means > eps, cvs, torch.zeros_like(cvs))
        return float(cvs.mean())

    def _apply_nor_adaptation(self, update: Tensor, state: dict, d: int):
        """Scale-free NorMuon-style row adaptation, applied separately to Q/K/V.

        The hybrid RPB update has already been radius-scaled and Frobenius-matched, so its
        raw mean-square entries can be many orders of magnitude below NorMuon's natural
        orthogonalized-update scale. We therefore normalize each Q/K/V block's row mean
        squares to unit mean before updating the EMA. This preserves only the relative
        row-allocation information that Nor adaptation is intended to use. Each adapted
        block is then Frobenius-matched back to its own pre-adaptation block.
        """
        cv_before = self._qkv_row_cv(update, d)
        if not self.nor_enable:
            return update, cv_before, cv_before, 0.0, 0.0, 0.0, 0.0

        blocks = update.view(3, d, d)
        row_ms = blocks.float().square().mean(dim=2)  # [3, d]

        # Remove the radius/global-scale dependence. The mean statistic in every Q/K/V
        # block is one, while relative row differences are retained.
        row_ms_unit = row_ms / row_ms.mean(dim=1, keepdim=True).clamp_min(1e-30)

        if "nor_v" not in state or state["nor_v"].shape != row_ms_unit.shape:
            state["nor_v"] = torch.zeros_like(row_ms_unit)
        v_state = state["nor_v"]
        v_state.mul_(self.nor_beta2).add_(row_ms_unit, alpha=1.0 - self.nor_beta2)

        # NorMuon uses sqrt(v) + eps, not sqrt(v + eps). The distinction matters when
        # the update itself has a small absolute scale.
        sqrt_v = torch.sqrt(v_state)
        adapted = blocks.float() / (sqrt_v.unsqueeze(-1) + self.nor_eps)

        # Preserve the original Q, K, and V block norms separately.
        ref_norm = torch.linalg.vector_norm(blocks.float(), dim=(1, 2), keepdim=True)
        adapted_norm = torch.linalg.vector_norm(adapted, dim=(1, 2), keepdim=True)
        adapted = adapted * (ref_norm / adapted_norm.clamp_min(1e-12))
        adapted = torch.where(ref_norm > 1e-12, adapted, blocks.float())

        rel_change = (
            torch.linalg.vector_norm(adapted - blocks.float(), dim=(1, 2))
            / torch.linalg.vector_norm(blocks.float(), dim=(1, 2)).clamp_min(1e-12)
        ).mean()
        eps_ratio = (self.nor_eps / sqrt_v.clamp_min(1e-30)).mean()

        adapted = adapted.to(update.dtype).reshape(3 * d, d)
        return (
            adapted,
            cv_before,
            self._qkv_row_cv(adapted, d),
            float(rel_change),
            float(v_state.min()),
            float(v_state.max()),
            float(eps_ratio),
        )

    @torch.no_grad()
    def step(self):
        # Refresh the Gram preconditioner only every precond_refresh_period steps,
        # using Muon's exact schedule: t = since+1, refresh when t % period == 0 (so the
        # first refresh is at step period-1). global_step is set from the training loop
        # for resume-safety; it also advances internally so the schedule holds if unset.
        since = max(0, int(self.global_step) - int(self._regime_step))
        do_refresh = (((since + 1) % self.precond_refresh_period) == 0)
        did_refresh = False
        rstar_sum = 0.0; rstar_max = 0.0; sg_sum = 0.0; gy_sum = 0.0; nheads = 0; nlayers = 0
        spectral_change_sum = 0.0; precond_cos_sum = 0.0
        nor_cv_before_sum = 0.0; nor_cv_after_sum = 0.0
        nor_relative_change_sum = 0.0; nor_eps_ratio_sum = 0.0
        nor_v_min = float("inf"); nor_v_max = 0.0
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

                # The custom op accumulated g/||g||^p. Divide each Q/K/V-head block
                # by c=max_i ||g_i||^(1-p), so the maximum direction-row norm remains one.
                gradmax = ref["gradmax"]                       # [3, n_head]
                active_grad = gradmax > 0.0
                direction_scale = torch.where(
                    active_grad,
                    gradmax.clamp_min(1e-12).pow(1.0 - self.rowsign_power),
                    torch.ones_like(gradmax),
                )
                inv_direction_scale = direction_scale.reciprocal()
                row_scale = inv_direction_scale.unsqueeze(-1).expand(3, n_head, d_h).reshape(3 * d)
                M = (ref["M"] / cnt) * row_scale.unsqueeze(1)  # [3d, d]

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
                # The matching linear decrease is sum_i ||g_i||^(2-p)/c.
                S_G = (ref["sg"] * inv_direction_scale).sum(dim=0)  # [n_head]
                g_Y = ref["gy"]                                 # [n_head] ||grad_Y L||_{1,2}
                rstar = self._solve_rstar(q, k, v, d_h, S_G, g_Y)  # [n_head]

                rstar_sum += float(rstar.sum()); rstar_max = max(rstar_max, float(rstar.max()))
                sg_sum += float(S_G.sum()); gy_sum += float(g_Y.sum())
                nheads += rstar.numel(); nlayers += 1

                # T^T Z = -eta r* rsgn(G)^T Z, with r* applied per head-slice of the
                # 3d output rows (same r* across that head's Q,K,V blocks).
                scale_block = rstar.repeat_interleave(d_h)     # [d]
                scale = torch.cat([scale_block, scale_block, scale_block])  # [3d]
                Mp = scale.unsqueeze(1) * M                    # [3d, d]

                # 1) Blend identity and Gram-inverse right geometries while matching
                #    each Q/K/V block's Frobenius norm to the full-preconditioned update.
                update, precond_cos = self._apply_precond_blend(Mp, inv, d)

                # 2) Partially flatten singular values, again preserving each block norm.
                update, spectral_change = self._apply_spectral_blend(update, d)

                # 3) Optionally redistribute update mass across output-neuron rows.
                (
                    update,
                    nor_cv_before,
                    nor_cv_after,
                    nor_relative_change,
                    layer_nor_v_min,
                    layer_nor_v_max,
                    nor_eps_ratio,
                ) = self._apply_nor_adaptation(update, state, d)

                dW = update.mul(-lr)                           # [3d, d]
                p.data.add_(dW.to(p.dtype))

                spectral_change_sum += spectral_change
                precond_cos_sum += precond_cos
                nor_cv_before_sum += nor_cv_before
                nor_cv_after_sum += nor_cv_after
                nor_relative_change_sum += nor_relative_change
                nor_eps_ratio_sum += nor_eps_ratio
                if self.nor_enable:
                    nor_v_min = min(nor_v_min, layer_nor_v_min)
                    nor_v_max = max(nor_v_max, layer_nor_v_max)

                # Reset accumulators for the next step.
                ref["M"].zero_()
                ref["gram"].zero_()
                ref["sg"].zero_()
                ref["gradmax"].zero_()
                ref["count"].zero_()
                ref["rownorm"].zero_()
                ref["gy"].zero_()

        self.global_step += 1
        self.last_diag = {
            "rpb/r_star_mean": (rstar_sum / nheads) if nheads else 0.0,
            "rpb/r_star_max": rstar_max,
            "rpb/S_G_mean": (sg_sum / nheads) if nheads else 0.0,
            "rpb/g_Y_mean": (gy_sum / nheads) if nheads else 0.0,
            "rpb/rowsign_power": self.rowsign_power,
            "rpb/spectral_blend": self.spectral_blend,
            "rpb/spectral_steps": float(self.spectral_steps),
            "rpb/spectral_relative_change_mean": (spectral_change_sum / nlayers) if nlayers else 0.0,
            "rpb/precond_blend": self.precond_blend,
            "rpb/precond_full_identity_cosine_mean": (precond_cos_sum / nlayers) if nlayers else 0.0,
            "rpb/nor_enabled": float(self.nor_enable),
            "rpb/nor_row_cv_before_mean": (nor_cv_before_sum / nlayers) if nlayers else 0.0,
            "rpb/nor_row_cv_after_mean": (nor_cv_after_sum / nlayers) if nlayers else 0.0,
            "rpb/nor_relative_change_mean": (nor_relative_change_sum / nlayers) if nlayers else 0.0,
            "rpb/nor_v_min": nor_v_min if self.nor_enable and nor_v_min != float("inf") else 0.0,
            "rpb/nor_v_max": nor_v_max if self.nor_enable else 0.0,
            "rpb/nor_eps_over_sqrt_v_mean": (nor_eps_ratio_sum / nlayers) if nlayers else 0.0,
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


class QKVBridge(torch.optim.Optimizer):
    """Interpolate exactly between Newton-Muon and RPB activation geometry.

    The optimizer receives two descriptions of the same QKV gradient from one
    backward pass:

      * p.grad: the raw stacked QKV weight gradient G;
      * captured RPB statistics: a token-row-weighted activation numerator,
        headwise smoothness radii, and the current input Gram matrix.

    It first constructs an RPB activation-space candidate A and Frobenius-matches
    each Q/K/V block of A to the corresponding raw-gradient block.  It then forms

        G_mix = (1-alpha) G + alpha A,

    matches each mixed block back to the raw-gradient block norm, applies the full
    cached Gram inverse, applies momentum, and finally applies a separate matrix-sign
    map to Q, K, and V.  The parameter update uses the exact Muon/Newton-Muon scale
    -lr*sqrt(d).

    Consequently:

      alpha=0
        is the same raw-gradient -> Gram inverse -> momentum -> matrix-sign update
        as QKVMatrixControl(mode="newton_muon").

      alpha>0
        tests whether the RPB activation-space direction improves that exact
        Newton-Muon baseline while holding the right geometry, momentum ordering,
        spectral map, and update scale fixed.

    radius_blend interpolates the headwise r* pattern toward one uniform radius.
    headnorm_blend interpolates the RPB per-(Q/K/V,head) max-row normalization
    toward no headwise normalization.  At rowsign_power=0, radius_blend=0, and
    headnorm_blend=0, the RPB candidate itself is collinear with the raw gradient;
    this supplies an internal recovery check even when alpha=1.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend_steps: int = 5,
        bridge_blend: float = 0.0,
        rowsign_power: float = 0.85,
        radius_blend: float = 1.0,
        headnorm_blend: float = 1.0,
        h_sigma: float = 8.0,
        r_max=None,
        bisect_iters: int = 60,
        precond_refresh_period: int = 32,
        precond_ewma: float = 0.95,
        precond_init_diag: float = 0.001,
        precond_ridge_mult: float = 0.2,
        precond_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            backend_steps=int(backend_steps),
        )
        super().__init__(params, defaults)

        self.bridge_blend = float(bridge_blend)
        self.rowsign_power = float(rowsign_power)
        self.radius_blend = float(radius_blend)
        self.headnorm_blend = float(headnorm_blend)
        self.h_sigma = float(h_sigma)
        self.r_max = None if r_max is None else float(r_max)
        self.bisect_iters = int(bisect_iters)
        self.precond_refresh_period = int(precond_refresh_period)
        self.precond_ewma = float(precond_ewma)
        self.precond_init_diag = float(precond_init_diag)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_eps = float(precond_eps)

        for name, value in (
            ("bridge_blend", self.bridge_blend),
            ("rowsign_power", self.rowsign_power),
            ("radius_blend", self.radius_blend),
            ("headnorm_blend", self.headnorm_blend),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0,1], got {value}")
        if self.precond_refresh_period < 1:
            raise ValueError("precond_refresh_period must be >=1")
        if not (0.0 <= self.precond_ewma < 1.0):
            raise ValueError("precond_ewma must lie in [0,1)")
        if self.bisect_iters < 1:
            raise ValueError("bisect_iters must be >=1")

        self.global_step = 0
        self.last_diag = {}

    def precond_flag_for_step(self, step: int) -> bool:
        return ((int(step) + 1) % self.precond_refresh_period) == 0

    def _solve_rstar(self, q, k, v, d_h, S_G, g_Y):
        sqrt_d = float(d_h) ** 0.5
        hs = self.h_sigma

        def C(r):
            s = q + k + 2.0 * r
            return (
                hs * (v + r) * (s * s) / d_h
                + 2.0 * (v + r) / sqrt_d
                + 2.0 * s / sqrt_d
            )

        def Cp(r):
            s = q + k + 2.0 * r
            return (
                hs * (s * s + 4.0 * (v + r) * s) / d_h
                + 6.0 / sqrt_d
            )

        active = (S_G > 0.0) & (g_Y > 0.0)
        S_eff = torch.where(active, S_G, torch.zeros_like(S_G))
        g_eff = torch.where(active, g_Y, torch.ones_like(g_Y))

        def phip(r):
            return -S_eff + g_eff * (
                r * C(r) + 0.5 * r * r * Cp(r)
            )

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
        return torch.where(active, rstar, torch.zeros_like(rstar))

    @staticmethod
    def _match_blocks(candidate: Tensor, reference: Tensor, d: int, eps: float = 1e-12):
        cand = candidate.view(3, d, d)
        ref = reference.view(3, d, d)
        cand_norm = torch.linalg.vector_norm(cand, dim=(1, 2), keepdim=True)
        ref_norm = torch.linalg.vector_norm(ref, dim=(1, 2), keepdim=True)
        matched = cand * (ref_norm / cand_norm.clamp_min(eps))
        matched = torch.where(cand_norm > eps, matched, ref)
        return matched.reshape(3 * d, d)

    @staticmethod
    def _block_cosine(a: Tensor, b: Tensor, d: int, eps: float = 1e-12) -> float:
        aa = a.view(3, d, d)
        bb = b.view(3, d, d)
        dot = (aa * bb).sum(dim=(1, 2))
        an = torch.linalg.vector_norm(aa, dim=(1, 2))
        bn = torch.linalg.vector_norm(bb, dim=(1, 2))
        cos = dot / (an * bn).clamp_min(eps)
        return float(cos.mean())

    def _init_precond_state(self, p: Tensor, d: int) -> dict:
        state = self.state[p]
        if "precond_cov" not in state:
            cov = torch.zeros(d, d, device=p.device, dtype=torch.float32)
            cov.diagonal().fill_(self.precond_init_diag)
            state["precond_cov"] = cov
            state["precond_inv"] = torch.eye(
                d, device=p.device, dtype=torch.float32
            )
        return state

    @torch.no_grad()
    def step(self):
        do_refresh = self.precond_flag_for_step(self.global_step)
        did_refresh = False
        n_layers = 0
        n_heads = 0
        inv_norm_sum = 0.0
        raw_rpb_cos_sum = 0.0
        raw_mix_cos_sum = 0.0
        rstar_sum = 0.0
        rstar_max = 0.0
        radius_cv_sum = 0.0

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            backend_steps = int(group["backend_steps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2 or p.shape[0] != 3 * p.shape[1]:
                    raise ValueError(
                        "QKVBridge expects stacked QKV weights of shape [3d,d], "
                        f"got {tuple(p.shape)}"
                    )

                ref = getattr(p, "_rpb_ref", None)
                if ref is None:
                    raise RuntimeError("Missing _rpb_ref on bridge QKV weight")
                cnt = float(ref["count"].item())
                if cnt <= 0.0:
                    raise RuntimeError("Bridge step has no captured activation statistics")

                d = int(ref["d"])
                n_head = int(ref["n_head"])
                d_h = int(ref["d_h"])
                state = self._init_precond_state(p, d)

                raw = p.grad.detach().float()
                soft = ref["M"] / cnt

                gradmax = ref["gradmax"]
                active_grad = gradmax > 0.0
                current_inv_scale = torch.where(
                    active_grad,
                    gradmax.clamp_min(1e-12).pow(self.rowsign_power - 1.0),
                    torch.ones_like(gradmax),
                )
                headnorm_scale = torch.lerp(
                    torch.ones_like(current_inv_scale),
                    current_inv_scale,
                    self.headnorm_blend,
                )

                # Solve the original units-corrected RPB radius.  radius_blend only
                # changes the headwise pattern subsequently; the alpha=0 endpoint does
                # not depend on this solve at all.
                rn = ref["rownorm"]
                q, k, v = rn[0], rn[1], rn[2]
                S_G = (ref["sg"] * current_inv_scale).sum(dim=0)
                g_Y = ref["gy"]
                rstar = self._solve_rstar(q, k, v, d_h, S_G, g_Y)

                active_heads = rstar > 0.0
                denom = active_heads.sum().clamp_min(1)
                uniform_radius = rstar.sum() / denom
                uniform = torch.ones_like(rstar) * uniform_radius
                radius = torch.lerp(uniform, rstar, self.radius_blend)
                radius = torch.where(active_heads, radius, torch.zeros_like(radius))

                # Current RPB scaling has one radius per head and one max-row scale
                # per (Q/K/V, head).  Both axes can be relaxed toward uniformity.
                scale_3h = headnorm_scale * radius.unsqueeze(0)
                row_scale = scale_3h.unsqueeze(-1).expand(
                    3, n_head, d_h
                ).reshape(3 * d)
                rpb_candidate = soft * row_scale.unsqueeze(1)
                rpb_candidate = self._match_blocks(rpb_candidate, raw, d)

                raw_rpb_cos_sum += self._block_cosine(raw, rpb_candidate, d)

                mixed = torch.lerp(raw, rpb_candidate, self.bridge_blend)
                mixed = self._match_blocks(mixed, raw, d)
                raw_mix_cos_sum += self._block_cosine(raw, mixed, d)

                if do_refresh:
                    gram = ref["gram"] / cnt
                    cov = state["precond_cov"]
                    # Match the exact QKVMatrixControl/Newton-Muon estimator: update
                    # the running covariance only on refresh steps.
                    cov.lerp_(gram, 1.0 - self.precond_ewma)
                    ridge = (
                        cov.diagonal().mean() * self.precond_ridge_mult
                        + self.precond_eps
                    ).clamp_min(self.precond_eps)
                    K = cov.clone()
                    K.diagonal().add_(ridge)
                    L, info = torch.linalg.cholesky_ex(
                        K, upper=False, check_errors=False
                    )
                    if int(info.item()) != 0:
                        state["precond_inv"].copy_(
                            torch.eye(d, device=p.device, dtype=torch.float32)
                        )
                    else:
                        state["precond_inv"].copy_(
                            torch.cholesky_inverse(L, upper=False)
                        )
                    did_refresh = True

                inv = state["precond_inv"]
                g = mixed @ inv
                inv_norm_sum += float(inv.norm())

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                shaped = torch.cat(
                    [
                        zeropower_via_newtonschulz5(block, steps=backend_steps)
                        for block in g.split(d)
                    ],
                    dim=0,
                )
                p.data.add_(shaped.to(p.dtype), alpha=-lr * (d ** 0.5))

                rstar_sum += float(rstar.sum())
                rstar_max = max(rstar_max, float(rstar.max()))
                n_heads += rstar.numel()
                if bool(active_heads.any()):
                    vals = radius[active_heads]
                    radius_cv_sum += float(
                        vals.std(unbiased=False) / vals.mean().clamp_min(1e-12)
                    )
                n_layers += 1

                ref["M"].zero_()
                ref["gram"].zero_()
                ref["sg"].zero_()
                ref["gradmax"].zero_()
                ref["count"].zero_()
                ref["rownorm"].zero_()
                ref["gy"].zero_()

        self.global_step += 1
        self.last_diag = {
            "bridge/blend": self.bridge_blend,
            "bridge/rowsign_power": self.rowsign_power,
            "bridge/radius_blend": self.radius_blend,
            "bridge/headnorm_blend": self.headnorm_blend,
            "bridge/raw_rpb_cosine_mean": (
                raw_rpb_cos_sum / n_layers if n_layers else 0.0
            ),
            "bridge/raw_mixed_cosine_mean": (
                raw_mix_cos_sum / n_layers if n_layers else 0.0
            ),
            "bridge/r_star_mean": rstar_sum / n_heads if n_heads else 0.0,
            "bridge/r_star_max": rstar_max,
            "bridge/radius_pattern_cv_mean": (
                radius_cv_sum / n_layers if n_layers else 0.0
            ),
            "bridge/precond_refresh": float(did_refresh),
            "bridge/precond_inv_norm_mean": (
                inv_norm_sum / n_layers if n_layers else 0.0
            ),
            "bridge/layers_updated": float(n_layers),
        }


class QKVMatrixControl(torch.optim.Optimizer):
    """Exact QKV Muon / Newton-Muon control for the joint experiment harness.

    mode="newton_muon":
        raw QKV weight gradient -> cached right Gram inverse -> momentum ->
        separate Q/K/V Newton-Schulz matrix-sign maps.

    mode="muon":
        raw QKV weight gradient -> momentum -> separate Q/K/V matrix-sign maps.

    The right preconditioner is applied before momentum, matching the ordering in
    Newton-Muon. The QKV matrix is split into three d-by-d blocks before the matrix
    sign, matching the existing single-GPU Newton-Muon baseline in this repository.
    """

    def __init__(
        self,
        params,
        *,
        mode: str,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend_steps: int = 5,
        precond_refresh_period: int = 32,
        precond_ewma: float = 0.95,
        precond_init_diag: float = 0.001,
        precond_ridge_mult: float = 0.2,
        precond_eps: float = 1e-8,
    ):
        mode = str(mode).strip().lower()
        if mode not in {"newton_muon", "muon"}:
            raise ValueError(f"Unsupported QKV control mode: {mode!r}")
        if precond_refresh_period < 1:
            raise ValueError("precond_refresh_period must be >= 1")
        if not (0.0 <= precond_ewma < 1.0):
            raise ValueError("precond_ewma must lie in [0, 1)")
        if backend_steps < 1:
            raise ValueError("backend_steps must be >= 1")

        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            nesterov=bool(nesterov),
            backend_steps=int(backend_steps),
        )
        super().__init__(params, defaults)
        self.mode = mode
        self.precond_refresh_period = int(precond_refresh_period)
        self.precond_ewma = float(precond_ewma)
        self.precond_init_diag = float(precond_init_diag)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_eps = float(precond_eps)
        self.global_step = 0
        self.last_diag = {}

    def precond_flag_for_step(self, step: int) -> bool:
        return (
            self.mode == "newton_muon"
            and ((int(step) + 1) % self.precond_refresh_period == 0)
        )

    def _init_precond_state(self, p: Tensor, d: int) -> dict:
        state = self.state[p]
        if "precond_cov" not in state:
            cov = torch.zeros(d, d, device=p.device, dtype=torch.float32)
            cov.diagonal().fill_(self.precond_init_diag)
            state["precond_cov"] = cov
            state["precond_inv"] = torch.eye(
                d, device=p.device, dtype=torch.float32
            )
        return state

    @torch.no_grad()
    def step(self):
        do_refresh = self.precond_flag_for_step(self.global_step)
        did_refresh = False
        inv_norm_sum = 0.0
        n_layers = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            backend_steps = int(group["backend_steps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2 or p.shape[0] != 3 * p.shape[1]:
                    raise ValueError(
                        "QKVMatrixControl expects stacked QKV weights of shape [3d, d], "
                        f"got {tuple(p.shape)}"
                    )

                d = int(p.shape[1])
                state = self.state[p]
                g = p.grad.detach().float()

                if self.mode == "newton_muon":
                    state = self._init_precond_state(p, d)
                    ref = getattr(p, "_qkv_stats_ref", None)
                    if ref is None:
                        raise RuntimeError("Missing _qkv_stats_ref on QKV weight")

                    if do_refresh:
                        cnt = float(ref["count"].item())
                        if cnt <= 0.0:
                            raise RuntimeError(
                                "Newton-Muon QKV refresh was requested, but no input Gram "
                                "statistics were captured on this step."
                            )
                        gram = ref["accum"] / cnt
                        cov = state["precond_cov"]
                        cov.lerp_(gram, 1.0 - self.precond_ewma)

                        ridge = (
                            cov.diagonal().mean() * self.precond_ridge_mult
                            + self.precond_eps
                        ).clamp_min(self.precond_eps)
                        K = cov.clone()
                        K.diagonal().add_(ridge)
                        L, info = torch.linalg.cholesky_ex(
                            K, upper=False, check_errors=False
                        )
                        if int(info.item()) != 0:
                            state["precond_inv"].copy_(
                                torch.eye(d, device=p.device, dtype=torch.float32)
                            )
                        else:
                            state["precond_inv"].copy_(
                                torch.cholesky_inverse(L, upper=False)
                            )
                        ref["accum"].zero_()
                        ref["count"].zero_()
                        did_refresh = True

                    inv = state["precond_inv"]
                    g = g @ inv
                    inv_norm_sum += float(inv.norm())

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                shaped = torch.cat(
                    [
                        zeropower_via_newtonschulz5(block, steps=backend_steps)
                        for block in g.split(d)
                    ],
                    dim=0,
                )
                p.data.add_(shaped.to(p.dtype), alpha=-lr * (d ** 0.5))
                n_layers += 1

        self.global_step += 1
        self.last_diag = {
            "qkv_control/mode_newton_muon": float(self.mode == "newton_muon"),
            "qkv_control/mode_muon": float(self.mode == "muon"),
            "qkv_control/precond_refresh": float(did_refresh),
            "qkv_control/precond_inv_norm_mean": (
                inv_norm_sum / n_layers
                if self.mode == "newton_muon" and n_layers
                else 0.0
            ),
            "qkv_control/layers_updated": float(n_layers),
        }


class FisherQK(torch.optim.Optimizer):
    """Production joint pulled-back Fisher optimizer for Q/K with Newton-Muon V.

    The full accumulated Q/K parameter gradient is the right-hand side.  A small
    sampled curvature minibatch supplies X, current Q/K, attention probabilities,
    and optionally the projected coefficient shape.  The matrix-free operator is

        H(D_Q,D_K) = J^* C J(D_Q,D_K) + lambda (D_Q,D_K).

    The Q/K displacement is obtained with a fixed number of CG iterations.  The V
    block uses a configurable AdamW, Muon, or Newton-Muon companion path.
    """

    def __init__(
        self,
        params,
        *,
        coeff_mode: str = "projected",
        coeff_normalize: str = "median",
        coeff_floor: float = 1e-3,
        beta: float = 0.0,
        cg_iters: int = 3,
        cg_tol: float = 0.0,
        damp_rel: float = 0.1,
        damp_floor: float = 1e-8,
        scale_mode: str = "native",
        outer_scale: float = 0.5,
        qk_lr: float = 1.1e-4,
        curv_refresh: int = 1,
        momentum_mode: str = "none",
        momentum: float = 0.0,
        nesterov: bool = True,
        post_transform: str = "none",
        score_osc_cap: float = 0.0,
        v_lr: float = 4e-4,
        v_momentum: float = 0.95,
        v_nesterov: bool = True,
        v_mode: str = "newton_muon",
        v_beta1: float = 0.9,
        v_beta2: float = 0.95,
        v_weight_decay: float = 0.0,
        v_eps: float = 1e-8,
        backend_steps: int = 5,
        precond_refresh_period: int = 32,
        precond_ewma: float = 0.95,
        precond_init_diag: float = 0.001,
        precond_ridge_mult: float = 0.2,
        precond_eps: float = 1e-8,
        diag_every: int = 50,
    ):
        coeff_mode = str(coeff_mode).lower()
        scale_mode = str(scale_mode).lower()
        momentum_mode = str(momentum_mode).lower()
        post_transform = str(post_transform).lower()
        v_mode = str(v_mode).strip().lower()
        if v_mode not in {"adamw", "muon", "newton_muon"}:
            raise ValueError(f"Unknown Fisher V mode {v_mode!r}")
        if coeff_mode not in {"unit", "projected"}:
            raise ValueError(f"Unknown Fisher coefficient mode {coeff_mode!r}")
        if scale_mode not in {"native", "rms", "nm_match"}:
            raise ValueError(f"Unknown Fisher scale mode {scale_mode!r}")
        if momentum_mode not in {"none", "rhs", "direction"}:
            raise ValueError(f"Unknown Fisher momentum mode {momentum_mode!r}")
        if post_transform not in {"none", "matrix_sign"}:
            raise ValueError(f"Unknown Fisher post transform {post_transform!r}")
        if int(cg_iters) < 1:
            raise ValueError("FISHER_CG_ITERS must be >= 1")
        if int(curv_refresh) < 1:
            raise ValueError("FISHER_CURV_REFRESH must be >= 1")
        if int(precond_refresh_period) < 1:
            raise ValueError("precond_refresh_period must be >= 1")

        defaults = dict(lr=1.0)  # scheduler multiplier shared by Q/K and V
        super().__init__(params, defaults)
        self.coeff_mode = coeff_mode
        self.coeff_normalize = str(coeff_normalize).lower()
        self.coeff_floor = float(coeff_floor)
        self.beta = float(beta)
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)
        self.damp_rel = float(damp_rel)
        self.damp_floor = float(damp_floor)
        self.scale_mode = scale_mode
        self.outer_scale = float(outer_scale)
        self.qk_lr = float(qk_lr)
        self.curv_refresh = int(curv_refresh)
        self.momentum_mode = momentum_mode
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.post_transform = post_transform
        self.score_osc_cap = float(score_osc_cap)
        self.v_lr = float(v_lr)
        self.v_momentum = float(v_momentum)
        self.v_nesterov = bool(v_nesterov)
        self.v_mode = v_mode
        self.v_beta1 = float(v_beta1)
        self.v_beta2 = float(v_beta2)
        self.v_weight_decay = float(v_weight_decay)
        self.v_eps = float(v_eps)
        self.backend_steps = int(backend_steps)
        self.precond_refresh_period = int(precond_refresh_period)
        self.precond_ewma = float(precond_ewma)
        self.precond_init_diag = float(precond_init_diag)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_eps = float(precond_eps)
        self.diag_every = int(diag_every)
        self.global_step = 0
        self.curvature_samples = None
        self.curvature_capture_seconds = 0.0
        self.curvature_age = 10**9
        self.last_diag = {}

    def precond_flag_for_step(self, step: int) -> bool:
        if self.v_mode != "newton_muon":
            return False
        return ((int(step) + 1) % self.precond_refresh_period) == 0

    def needs_curvature_capture(self, step: int) -> bool:
        return self.curvature_samples is None or (int(step) % self.curv_refresh) == 0

    def should_time(self, step: int) -> bool:
        return self.diag_every > 0 and (int(step) % self.diag_every) == 0

    def set_curvature_samples(self, samples, *, capture_seconds: float = 0.0) -> None:
        n_params = sum(len(group["params"]) for group in self.param_groups)
        if len(samples) != n_params:
            raise ValueError(
                f"Expected {n_params} Fisher curvature samples, received {len(samples)}"
            )
        self.curvature_samples = samples
        self.curvature_capture_seconds = float(capture_seconds)
        self.curvature_age = 0

    def _init_precond_state(self, p: Tensor, d: int) -> dict:
        state = self.state[p]
        if "precond_cov" not in state:
            cov = torch.zeros(d, d, device=p.device, dtype=torch.float32)
            cov.diagonal().fill_(self.precond_init_diag)
            state["precond_cov"] = cov
            state["precond_inv"] = torch.eye(d, device=p.device, dtype=torch.float32)
        return state

    def _update_preconditioner(self, p: Tensor, state: dict, d: int, do_refresh: bool) -> bool:
        if not do_refresh:
            return False
        ref = getattr(p, "_qkv_stats_ref", None)
        if ref is None:
            raise RuntimeError("Missing _qkv_stats_ref on Fisher QKV weight")
        cnt = float(ref["count"].item())
        if cnt <= 0.0:
            raise RuntimeError(
                "Fisher-QK V preconditioner refresh requested without captured input Gram"
            )
        gram = ref["accum"] / cnt
        cov = state["precond_cov"]
        cov.lerp_(gram, 1.0 - self.precond_ewma)
        ridge = (
            cov.diagonal().mean() * self.precond_ridge_mult + self.precond_eps
        ).clamp_min(self.precond_eps)
        K = cov.clone()
        K.diagonal().add_(ridge)
        L, info = torch.linalg.cholesky_ex(K, upper=False, check_errors=False)
        if int(info.item()) != 0:
            state["precond_inv"].copy_(
                torch.eye(d, device=p.device, dtype=torch.float32)
            )
        else:
            state["precond_inv"].copy_(torch.cholesky_inverse(L, upper=False))
        ref["accum"].zero_()
        ref["count"].zero_()
        return True

    def _momentum_pair(self, state: dict, gradient, *, prefix: str):
        if self.momentum_mode == "none" or self.momentum <= 0.0:
            return gradient
        key_q = f"{prefix}_q"
        key_k = f"{prefix}_k"
        if key_q not in state:
            state[key_q] = torch.zeros_like(gradient[0])
            state[key_k] = torch.zeros_like(gradient[1])
        bq, bk = state[key_q], state[key_k]
        bq.mul_(self.momentum).add_(gradient[0])
        bk.mul_(self.momentum).add_(gradient[1])
        if self.nesterov:
            return (
                gradient[0] + self.momentum * bq,
                gradient[1] + self.momentum * bk,
            )
        return bq, bk

    def _postprocess(self, direction, d: int):
        if self.post_transform == "none":
            return direction
        q = direction[0].reshape(d, d)
        k = direction[1].reshape(d, d)
        shaped = (
            zeropower_via_newtonschulz5(q, steps=self.backend_steps).reshape_as(direction[0]),
            zeropower_via_newtonschulz5(k, steps=self.backend_steps).reshape_as(direction[1]),
        )
        return match_pair_norm(shaped, direction)

    def _scale_direction(self, direction, gradient, inv: Tensor, d: int, schedule: float):
        if self.scale_mode == "native":
            return pair_scale(direction, schedule * self.outer_scale)
        if self.scale_mode == "rms":
            rms = pair_norm(direction) / math.sqrt(float(2 * d * d))
            if float(rms) <= 1e-30:
                return pair_scale(direction, 0.0)
            return pair_scale(direction, schedule * self.qk_lr / rms)

        # Diagnostic match to the current Newton-Muon Q/K direction.
        gq = gradient[0].reshape(d, d) @ inv
        gk = gradient[1].reshape(d, d) @ inv
        ref = (
            (-math.sqrt(float(d)) * zeropower_via_newtonschulz5(
                gq, steps=self.backend_steps
            )).reshape_as(direction[0]),
            (-math.sqrt(float(d)) * zeropower_via_newtonschulz5(
                gk, steps=self.backend_steps
            )).reshape_as(direction[1]),
        )
        matched = match_pair_norm(direction, ref)
        return pair_scale(matched, schedule * self.qk_lr)

    @torch.no_grad()
    def _apply_v_update(
        self,
        p_v: Tensor,
        state: dict,
        g_raw: Tensor,
        inv: Tensor,
        schedule: float,
        d: int,
    ) -> Tensor:
        """Apply the selected V optimizer and return the actual V delta."""
        lr = float(schedule) * self.v_lr
        if self.v_mode in {"muon", "newton_muon"}:
            g = g_raw @ inv if self.v_mode == "newton_muon" else g_raw
            key = f"v_{self.v_mode}_momentum_buffer"
            if key not in state:
                state[key] = torch.zeros_like(g)
            buf = state[key]
            buf.mul_(self.v_momentum).add_(g)
            g_eff = g + self.v_momentum * buf if self.v_nesterov else buf
            shaped = zeropower_via_newtonschulz5(
                g_eff, steps=self.backend_steps
            )
            delta = -lr * math.sqrt(float(d)) * shaped
            p_v.add_(delta.to(p_v.dtype))
            return delta

        # Decoupled AdamW on the V block of the fused QKV parameter.
        step_key = "v_adamw_step"
        state[step_key] = int(state.get(step_key, 0)) + 1
        if "v_adamw_exp_avg" not in state:
            state["v_adamw_exp_avg"] = torch.zeros_like(g_raw)
            state["v_adamw_exp_avg_sq"] = torch.zeros_like(g_raw)
        exp_avg = state["v_adamw_exp_avg"]
        exp_avg_sq = state["v_adamw_exp_avg_sq"]
        exp_avg.mul_(self.v_beta1).add_(g_raw, alpha=1.0 - self.v_beta1)
        exp_avg_sq.mul_(self.v_beta2).addcmul_(
            g_raw, g_raw, value=1.0 - self.v_beta2
        )
        t = int(state[step_key])
        bc1 = 1.0 - self.v_beta1 ** t
        bc2 = 1.0 - self.v_beta2 ** t
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(self.v_eps)
        delta = -lr * self.v_weight_decay * p_v.float()
        delta = delta.addcdiv(exp_avg, denom, value=-(lr / bc1))
        p_v.add_(delta.to(p_v.dtype))
        return delta

    @torch.no_grad()
    def step(self):
        if self.curvature_samples is None:
            raise RuntimeError("FisherQK.step() called before curvature samples were supplied")

        do_refresh = self.precond_flag_for_step(self.global_step)
        do_diag = self.diag_every > 0 and (self.global_step % self.diag_every == 0)
        start_event = end_event = None
        if do_diag:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        n_layers = 0
        did_refresh = False
        fallback_count = 0
        cap_scale_sum = 0.0
        damping_sum = rayleigh_sum = final_residual_sum = 0.0
        final_residual_max = 0.0
        direction_norm_sum = update_norm_sum = current_dot_sum = 0.0
        qk_update_rms_sum = v_update_rms_sum = 0.0
        score_osc_sum = bilinear_ratio_sum = 0.0
        c_mean_sum = c_min_sum = c_max_sum = 0.0
        inv_norm_sum = 0.0

        sample_index = 0
        for group in self.param_groups:
            schedule = float(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    sample_index += 1
                    continue
                if p.ndim != 2 or p.shape[0] != 3 * p.shape[1]:
                    raise ValueError(
                        f"FisherQK expects fused [3d,d] weights, got {tuple(p.shape)}"
                    )
                sample = self.curvature_samples[sample_index]
                sample_index += 1
                d = int(p.shape[1])
                H = int(sample["q"].shape[1])
                dh = d // H
                state = self._init_precond_state(p, d)
                did_refresh = self._update_preconditioner(p, state, d, do_refresh) or did_refresh
                inv = state["precond_inv"]
                inv_norm_sum += float(inv.norm())

                g_blocks = p.grad.detach().float().view(3, d, d)
                current_gradient = (
                    g_blocks[0].reshape(H, dh, d),
                    g_blocks[1].reshape(H, dh, d),
                )
                solve_gradient = current_gradient
                if self.momentum_mode == "rhs":
                    solve_gradient = self._momentum_pair(
                        state, current_gradient, prefix="fisher_rhs_momentum"
                    )

                geom = build_geometry(
                    sample["x"], sample["q"], sample["k"], sample["cos"], sample["sin"]
                )
                if self.coeff_mode == "unit":
                    c = unit_coefficients(geom)
                    coeff_info = {"c_mean": 1.0, "c_min": 1.0, "c_max": 1.0}
                else:
                    c, coeff_info = projected_coefficients(
                        geom=geom,
                        v_pre=sample["v_pre"],
                        g_out=sample["g_out"],
                        w_o=sample["w_o"],
                        beta=self.beta,
                        normalize=self.coeff_normalize,
                        floor=self.coeff_floor,
                    )

                damping, rayleigh = estimate_relative_damping(
                    geom,
                    solve_gradient,
                    kind="fisher",
                    c=c,
                    damping_rel=self.damp_rel,
                    damping_floor=self.damp_floor,
                    reduction="mean",
                )
                apply = make_quadratic_operator(
                    geom,
                    kind="fisher",
                    c=c,
                    damping=damping,
                    reduction="mean",
                )
                direction, cg_info = pcg_solve(
                    apply,
                    pair_scale(solve_gradient, -1.0),
                    iterations=self.cg_iters,
                    tol=self.cg_tol,
                )
                if self.momentum_mode == "direction":
                    direction = self._momentum_pair(
                        state, direction, prefix="fisher_direction_momentum"
                    )

                current_dot = float(pair_dot(current_gradient, direction))
                if (
                    (not math.isfinite(current_dot))
                    or current_dot >= 0.0
                    or bool(cg_info.get("breakdown", False))
                ):
                    direction = pair_scale(current_gradient, -1.0)
                    current_dot = -float(pair_dot(current_gradient, current_gradient))
                    fallback_count += 1

                direction = self._postprocess(direction, d)
                update = self._scale_direction(
                    direction, current_gradient, inv, d, schedule
                )

                cap_scale = 1.0
                if self.score_osc_cap > 0.0:
                    for _ in range(16):
                        score_change = joint_jvp(update, geom) + joint_bilinear_remainder(update, geom)
                        osc = float(score_oscillation(score_change, geom.mask).max())
                        if osc <= self.score_osc_cap:
                            break
                        update = pair_scale(update, 0.5)
                        cap_scale *= 0.5
                cap_scale_sum += cap_scale

                p_blocks = p.data.view(3, d, d)
                p_blocks[0].add_(update[0].reshape(d, d).to(p.dtype))
                p_blocks[1].add_(update[1].reshape(d, d).to(p.dtype))

                v_update = self._apply_v_update(
                    p_blocks[2], state, g_blocks[2], inv, schedule, d
                )

                n_layers += 1
                damping_sum += float(damping)
                rayleigh_sum += float(rayleigh)
                final_res = float(cg_info["relative_residuals"][-1])
                final_residual_sum += final_res
                final_residual_max = max(final_residual_max, final_res)
                direction_norm_sum += float(pair_norm(direction))
                update_norm_sum += float(pair_norm(update))
                current_dot_sum += current_dot
                qk_update_rms_sum += float(
                    pair_norm(update) / math.sqrt(float(2 * d * d))
                )
                v_update_rms_sum += float(v_update.float().square().mean().sqrt())
                c_mean_sum += float(coeff_info["c_mean"])
                c_min_sum += float(coeff_info["c_min"])
                c_max_sum += float(coeff_info["c_max"])

                if do_diag:
                    U = joint_jvp(update, geom)
                    R = joint_bilinear_remainder(update, geom)
                    score_osc_sum += float(score_oscillation(U + R, geom.mask).max())
                    bilinear_ratio_sum += float(R.norm() / U.norm().clamp_min(1e-30))

                # Release the large score geometry before the next layer.
                del geom, c, apply, sample

        solve_seconds = 0.0
        if do_diag and start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            solve_seconds = start_event.elapsed_time(end_event) / 1000.0

        denom = max(n_layers, 1)
        self.last_diag = {
            "fisher_qk/coeff_projected": float(self.coeff_mode == "projected"),
            "fisher_qk/cg_iters": float(self.cg_iters),
            "fisher_qk/damp_rel": float(self.damp_rel),
            "fisher_qk/scale_native": float(self.scale_mode == "native"),
            "fisher_qk/scale_rms": float(self.scale_mode == "rms"),
            "fisher_qk/outer_scale": float(self.outer_scale),
            "fisher_qk/qk_lr": float(self.qk_lr),
            "fisher_qk/effective_qk_scale": float(
                self.outer_scale if self.scale_mode == "native" else self.qk_lr
            ) * float(self.param_groups[0]["lr"]),
            "fisher_qk/effective_v_lr": float(self.v_lr) * float(self.param_groups[0]["lr"]),
            "fisher_qk/v_mode_code": float({"adamw": 0, "muon": 1, "newton_muon": 2}[self.v_mode]),
            "fisher_qk/damping_mean": damping_sum / denom,
            "fisher_qk/rayleigh_mean": rayleigh_sum / denom,
            "fisher_qk/cg_final_residual_mean": final_residual_sum / denom,
            "fisher_qk/cg_final_residual_max": final_residual_max,
            "fisher_qk/direction_norm_mean": direction_norm_sum / denom,
            "fisher_qk/update_norm_mean": update_norm_sum / denom,
            "fisher_qk/current_gradient_dot_mean": current_dot_sum / denom,
            "fisher_qk/qk_update_rms_mean": qk_update_rms_sum / denom,
            "fisher_qk/v_update_rms_mean": v_update_rms_sum / denom,
            "fisher_qk/coeff_mean": c_mean_sum / denom,
            "fisher_qk/coeff_min_mean": c_min_sum / denom,
            "fisher_qk/coeff_max_mean": c_max_sum / denom,
            "fisher_qk/precond_inv_norm_mean": inv_norm_sum / denom,
            "fisher_qk/precond_refresh": float(did_refresh),
            "fisher_qk/descent_fallback_layers": float(fallback_count),
            "fisher_qk/score_cap_scale_mean": cap_scale_sum / denom,
            "fisher_qk/score_osc_max_mean": score_osc_sum / denom if do_diag else 0.0,
            "fisher_qk/bilinear_ratio_mean": bilinear_ratio_sum / denom if do_diag else 0.0,
            "fisher_qk/curvature_age": float(self.curvature_age),
            "fisher_qk/curvature_capture_seconds": float(self.curvature_capture_seconds),
            "fisher_qk/optimizer_seconds": float(solve_seconds),
            "fisher_qk/layers_updated": float(n_layers),
        }
        self.global_step += 1
        self.curvature_age += 1

class AdvancedFisherQK(FisherQK):
    """Extended Fisher-QK optimizer for large structural and hybrid sweeps.

    The class preserves the validated Fisher-CG implementation and adds five
    independent experimental controls:

    1. a time-varying Q/K RMS target;
    2. a time-varying curvature refresh interval;
    3. continuous matrix-sign shaping of the Fisher direction;
    4. a fixed or scheduled blend with a shadow Newton-Muon Q/K direction;
    5. layer-selective Fisher/Newton-Muon assignment.

    Every directional blend is Frobenius-norm matched before the final RMS
    normalization.  Consequently, the blend parameters primarily change the
    direction, while the scale policy remains explicit.
    """

    def __init__(
        self,
        params,
        *,
        qk_lr_end: float = 0.0,
        qk_lr_schedule: str = "constant",
        qk_lr_schedule_start: int = 0,
        qk_lr_schedule_end: int = 0,
        curv_refresh_late: int = 0,
        curv_refresh_switch_step: int = -1,
        spectral_blend: float = 0.0,
        nm_blend_start: float = 0.0,
        nm_blend_end: float = 0.0,
        nm_blend_schedule: str = "constant",
        nm_blend_schedule_start: int = 0,
        nm_blend_schedule_end: int = 0,
        nm_lr: float = 4.4e-4,
        nm_momentum: float = 0.97,
        nm_nesterov: bool = True,
        nm_shadow: bool = False,
        blend_scale_mode: str = "fisher",
        layer_policy: str = "all_fisher",
        layer_mask: str = "",
        **kwargs,
    ):
        params = list(params)
        super().__init__(params, **kwargs)

        self.qk_lr_start = float(self.qk_lr)
        self.qk_lr_end = (
            float(qk_lr_end) if float(qk_lr_end) > 0.0 else self.qk_lr_start
        )
        self.qk_lr_schedule = self._validate_schedule(qk_lr_schedule)
        self.qk_lr_schedule_start = int(qk_lr_schedule_start)
        self.qk_lr_schedule_end = int(qk_lr_schedule_end)

        self.curv_refresh_late = (
            int(curv_refresh_late)
            if int(curv_refresh_late) > 0
            else int(self.curv_refresh)
        )
        self.curv_refresh_switch_step = int(curv_refresh_switch_step)

        self.spectral_blend = float(spectral_blend)
        if self.post_transform == "matrix_sign" and self.spectral_blend == 0.0:
            self.spectral_blend = 1.0
        if not (0.0 <= self.spectral_blend <= 1.0):
            raise ValueError("FISHER_SPECTRAL_BLEND must lie in [0,1]")

        self.nm_blend_start = float(nm_blend_start)
        self.nm_blend_end = float(nm_blend_end)
        if not (0.0 <= self.nm_blend_start <= 1.0):
            raise ValueError("FISHER_NM_BLEND_START must lie in [0,1]")
        if not (0.0 <= self.nm_blend_end <= 1.0):
            raise ValueError("FISHER_NM_BLEND_END must lie in [0,1]")
        self.nm_blend_schedule = self._validate_schedule(nm_blend_schedule)
        self.nm_blend_schedule_start = int(nm_blend_schedule_start)
        self.nm_blend_schedule_end = int(nm_blend_schedule_end)
        self.nm_lr = float(nm_lr)
        self.nm_momentum = float(nm_momentum)
        self.nm_nesterov = bool(nm_nesterov)
        self.nm_shadow = bool(nm_shadow)

        self.blend_scale_mode = str(blend_scale_mode).strip().lower()
        if self.blend_scale_mode not in {"fisher", "interpolate", "nm"}:
            raise ValueError(
                "FISHER_BLEND_SCALE_MODE must be fisher, interpolate, or nm"
            )

        self.layer_policy = str(layer_policy).strip().lower()
        allowed_policies = {
            "all_fisher",
            "all_nm",
            "outer4",
            "outer3",
            "early6",
            "late6",
            "even_fisher",
            "odd_fisher",
            "custom",
        }
        if self.layer_policy not in allowed_policies:
            raise ValueError(
                f"Unknown FISHER_LAYER_POLICY={self.layer_policy!r}; "
                f"expected one of {sorted(allowed_policies)}"
            )
        self.layer_mask = {
            int(piece.strip())
            for piece in str(layer_mask).split(",")
            if piece.strip()
        }
        self.total_layers = sum(len(group["params"]) for group in self.param_groups)
        if self.layer_policy == "custom" and not self.layer_mask:
            raise ValueError("FISHER_LAYER_POLICY=custom requires FISHER_LAYER_MASK")

    @staticmethod
    def _validate_schedule(schedule: str) -> str:
        schedule = str(schedule).strip().lower()
        if schedule not in {"constant", "step", "linear", "cosine"}:
            raise ValueError(
                f"Unknown schedule {schedule!r}; expected constant, step, linear, or cosine"
            )
        return schedule

    @staticmethod
    def _scheduled_value(
        step: int,
        start_value: float,
        end_value: float,
        start_step: int,
        end_step: int,
        schedule: str,
    ) -> float:
        step = int(step)
        if schedule == "constant" or start_value == end_value:
            return float(start_value)
        if schedule == "step":
            return float(end_value if step >= start_step else start_value)
        if end_step <= start_step:
            return float(end_value if step >= start_step else start_value)
        u = min(max((step - start_step) / float(end_step - start_step), 0.0), 1.0)
        if schedule == "cosine":
            u = 0.5 - 0.5 * math.cos(math.pi * u)
        return float(start_value + u * (end_value - start_value))

    def qk_lr_for_step(self, step: int) -> float:
        return self._scheduled_value(
            step,
            self.qk_lr_start,
            self.qk_lr_end,
            self.qk_lr_schedule_start,
            self.qk_lr_schedule_end,
            self.qk_lr_schedule,
        )

    def nm_blend_for_step(self, step: int) -> float:
        return self._scheduled_value(
            step,
            self.nm_blend_start,
            self.nm_blend_end,
            self.nm_blend_schedule_start,
            self.nm_blend_schedule_end,
            self.nm_blend_schedule,
        )

    def current_curv_refresh(self, step: int) -> int:
        if (
            self.curv_refresh_switch_step >= 0
            and int(step) >= self.curv_refresh_switch_step
        ):
            return int(self.curv_refresh_late)
        return int(self.curv_refresh)

    def _layer_is_fisher(self, layer_index: int) -> bool:
        n = int(self.total_layers)
        i = int(layer_index)
        if self.layer_policy == "all_fisher":
            return True
        if self.layer_policy == "all_nm":
            return False
        if self.layer_policy == "outer4":
            return i < 4 or i >= max(n - 4, 0)
        if self.layer_policy == "outer3":
            return i < 3 or i >= max(n - 3, 0)
        if self.layer_policy == "early6":
            return i < min(6, n)
        if self.layer_policy == "late6":
            return i >= max(n - 6, 0)
        if self.layer_policy == "even_fisher":
            return (i % 2) == 0
        if self.layer_policy == "odd_fisher":
            return (i % 2) == 1
        return i in self.layer_mask

    def layer_nm_alpha(self, layer_index: int, step: int) -> float:
        if not self._layer_is_fisher(layer_index):
            return 1.0
        return self.nm_blend_for_step(step)

    def uses_any_fisher(self, step: int) -> bool:
        global_alpha = self.nm_blend_for_step(step)
        if global_alpha >= 1.0 - 1e-12:
            return False
        return any(self._layer_is_fisher(i) for i in range(self.total_layers))

    def precond_flag_for_step(self, step: int) -> bool:
        needs_nm = (
            self.v_mode == "newton_muon"
            or self.nm_blend_start > 0.0
            or self.nm_blend_end > 0.0
            or self.layer_policy != "all_fisher"
        )
        if not needs_nm:
            return False
        return ((int(step) + 1) % self.precond_refresh_period) == 0

    def needs_curvature_capture(self, step: int) -> bool:
        if not self.uses_any_fisher(step):
            return False
        refresh = self.current_curv_refresh(step)
        return self.curvature_samples is None or (int(step) % refresh) == 0

    @staticmethod
    def _pair_lerp(x, y, alpha: float):
        alpha = float(alpha)
        return (
            x[0] * (1.0 - alpha) + y[0] * alpha,
            x[1] * (1.0 - alpha) + y[1] * alpha,
        )

    def _spectral_direction(self, direction, d: int):
        alpha = float(self.spectral_blend)
        if alpha <= 0.0:
            return direction
        shaped = (
            zeropower_via_newtonschulz5(
                direction[0].reshape(d, d), steps=self.backend_steps
            ).reshape_as(direction[0]),
            zeropower_via_newtonschulz5(
                direction[1].reshape(d, d), steps=self.backend_steps
            ).reshape_as(direction[1]),
        )
        shaped = match_pair_norm(shaped, direction)
        mixed = self._pair_lerp(direction, shaped, alpha)
        return match_pair_norm(mixed, direction)

    def _nm_direction(self, state: dict, current_gradient, inv: Tensor, d: int):
        outputs = []
        for block_index, name in enumerate(("q", "k")):
            g_pre = current_gradient[block_index].reshape(d, d) @ inv
            key = f"advanced_nm_{name}_momentum_buffer"
            if key not in state:
                state[key] = torch.zeros_like(g_pre)
            buf = state[key]
            buf.mul_(self.nm_momentum).add_(g_pre)
            if self.nm_nesterov:
                g_eff = g_pre + self.nm_momentum * buf
            else:
                g_eff = buf
            shaped = -math.sqrt(float(d)) * zeropower_via_newtonschulz5(
                g_eff, steps=self.backend_steps
            )
            outputs.append(shaped.reshape_as(current_gradient[block_index]))
        return outputs[0], outputs[1]

    def _target_qk_rms(self, step: int, nm_alpha: float) -> float:
        fisher_lr = self.qk_lr_for_step(step)
        if self.blend_scale_mode == "fisher":
            return fisher_lr
        if self.blend_scale_mode == "nm":
            return self.nm_lr
        return (1.0 - float(nm_alpha)) * fisher_lr + float(nm_alpha) * self.nm_lr

    def _scale_advanced_direction(
        self,
        direction,
        gradient,
        inv: Tensor,
        d: int,
        schedule: float,
        step: int,
        nm_alpha: float,
    ):
        if self.scale_mode == "rms":
            rms = pair_norm(direction) / math.sqrt(float(2 * d * d))
            if float(rms) <= 1e-30:
                return pair_scale(direction, 0.0), 0.0
            target = self._target_qk_rms(step, nm_alpha)
            return pair_scale(direction, schedule * target / rms), target
        update = self._scale_direction(direction, gradient, inv, d, schedule)
        target = float(
            pair_norm(update) / math.sqrt(float(2 * d * d))
        )
        return update, target

    @torch.no_grad()
    def step(self):
        step = int(self.global_step)
        if self.uses_any_fisher(step) and self.curvature_samples is None:
            raise RuntimeError(
                "AdvancedFisherQK.step() called without required curvature samples"
            )

        do_refresh = self.precond_flag_for_step(step)
        do_diag = self.diag_every > 0 and (step % self.diag_every == 0)
        start_event = end_event = None
        if do_diag:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        n_layers = 0
        n_fisher_layers = 0
        n_diag_layers = 0
        did_refresh = False
        fallback_count = 0
        cap_scale_sum = 0.0
        damping_sum = rayleigh_sum = final_residual_sum = 0.0
        final_residual_max = 0.0
        direction_norm_sum = update_norm_sum = current_dot_sum = 0.0
        qk_update_rms_sum = v_update_rms_sum = 0.0
        score_osc_sum = bilinear_ratio_sum = 0.0
        c_mean_sum = c_min_sum = c_max_sum = 0.0
        inv_norm_sum = 0.0
        nm_alpha_sum = target_rms_sum = fisher_nm_cos_sum = 0.0
        fisher_nm_cos_count = 0

        layer_index = 0
        for group in self.param_groups:
            schedule = float(group["lr"])
            for p in group["params"]:
                this_layer = layer_index
                layer_index += 1
                if p.grad is None:
                    continue
                if p.ndim != 2 or p.shape[0] != 3 * p.shape[1]:
                    raise ValueError(
                        f"AdvancedFisherQK expects fused [3d,d] weights, got {tuple(p.shape)}"
                    )

                d = int(p.shape[1])
                state = self._init_precond_state(p, d)
                did_refresh = (
                    self._update_preconditioner(p, state, d, do_refresh) or did_refresh
                )
                inv = state["precond_inv"]
                inv_norm_sum += float(inv.norm())

                g_blocks = p.grad.detach().float().view(3, d, d)
                nm_alpha = float(self.layer_nm_alpha(this_layer, step))
                need_fisher = nm_alpha < 1.0 - 1e-12
                need_nm = self.nm_shadow or nm_alpha > 1e-12

                sample = None
                geom = None
                c = None
                apply = None
                coeff_info = {"c_mean": 0.0, "c_min": 0.0, "c_max": 0.0}
                damping = rayleigh = 0.0
                cg_info = {
                    "relative_residuals": [0.0],
                    "breakdown": False,
                }

                if need_fisher:
                    sample = self.curvature_samples[this_layer]
                    H = int(sample["q"].shape[1])
                    dh = d // H
                else:
                    # Infer the architectural head count from the QKV parameter
                    # metadata.  The old exploratory code hard-coded 12 here,
                    # which is invalid after a Fisher-to-NM transition in wider
                    # models such as 24-layer / 16-head GPT-2-medium.
                    ref = getattr(p, "_rpb_ref", None)
                    if not isinstance(ref, dict) or "n_head" not in ref:
                        raise RuntimeError(
                            "Missing QKV head metadata for AdvancedFisherQK"
                        )
                    H = int(ref["n_head"])
                    if d % H != 0:
                        raise ValueError(f"d={d} is not divisible by n_head={H}")
                    dh = d // H

                current_gradient = (
                    g_blocks[0].reshape(H, dh, d),
                    g_blocks[1].reshape(H, dh, d),
                )

                fisher_direction = None
                if need_fisher:
                    solve_gradient = current_gradient
                    if self.momentum_mode == "rhs":
                        solve_gradient = self._momentum_pair(
                            state,
                            current_gradient,
                            prefix="fisher_rhs_momentum",
                        )

                    geom = build_geometry(
                        sample["x"],
                        sample["q"],
                        sample["k"],
                        sample["cos"],
                        sample["sin"],
                    )
                    if self.coeff_mode == "unit":
                        c = unit_coefficients(geom)
                        coeff_info = {
                            "c_mean": 1.0,
                            "c_min": 1.0,
                            "c_max": 1.0,
                        }
                    else:
                        c, coeff_info = projected_coefficients(
                            geom=geom,
                            v_pre=sample["v_pre"],
                            g_out=sample["g_out"],
                            w_o=sample["w_o"],
                            beta=self.beta,
                            normalize=self.coeff_normalize,
                            floor=self.coeff_floor,
                        )

                    damping, rayleigh = estimate_relative_damping(
                        geom,
                        solve_gradient,
                        kind="fisher",
                        c=c,
                        damping_rel=self.damp_rel,
                        damping_floor=self.damp_floor,
                        reduction="mean",
                    )
                    apply = make_quadratic_operator(
                        geom,
                        kind="fisher",
                        c=c,
                        damping=damping,
                        reduction="mean",
                    )
                    fisher_direction, cg_info = pcg_solve(
                        apply,
                        pair_scale(solve_gradient, -1.0),
                        iterations=self.cg_iters,
                        tol=self.cg_tol,
                    )
                    if self.momentum_mode == "direction":
                        fisher_direction = self._momentum_pair(
                            state,
                            fisher_direction,
                            prefix="fisher_direction_momentum",
                        )

                    fisher_dot = float(pair_dot(current_gradient, fisher_direction))
                    if (
                        (not math.isfinite(fisher_dot))
                        or fisher_dot >= 0.0
                        or bool(cg_info.get("breakdown", False))
                    ):
                        fisher_direction = pair_scale(current_gradient, -1.0)
                        fallback_count += 1
                    fisher_direction = self._spectral_direction(fisher_direction, d)
                    n_fisher_layers += 1

                nm_direction = None
                if need_nm:
                    nm_direction = self._nm_direction(
                        state, current_gradient, inv, d
                    )

                if fisher_direction is None:
                    direction = nm_direction
                elif nm_direction is None or nm_alpha <= 1e-12:
                    direction = fisher_direction
                else:
                    nm_matched = match_pair_norm(nm_direction, fisher_direction)
                    direction = self._pair_lerp(
                        fisher_direction, nm_matched, nm_alpha
                    )
                    direction = match_pair_norm(direction, fisher_direction)
                    if do_diag:
                        fisher_nm_cos_sum += float(
                            pair_cosine(fisher_direction, nm_direction)
                        )
                        fisher_nm_cos_count += 1

                current_dot = float(pair_dot(current_gradient, direction))
                if (not math.isfinite(current_dot)) or current_dot >= 0.0:
                    fisher_dot = (
                        float(pair_dot(current_gradient, fisher_direction))
                        if fisher_direction is not None
                        else float("inf")
                    )
                    nm_dot = (
                        float(pair_dot(current_gradient, nm_direction))
                        if nm_direction is not None
                        else float("inf")
                    )
                    if math.isfinite(fisher_dot) and fisher_dot < 0.0:
                        direction = fisher_direction
                        current_dot = fisher_dot
                    elif math.isfinite(nm_dot) and nm_dot < 0.0:
                        direction = nm_direction
                        current_dot = nm_dot
                    else:
                        direction = pair_scale(current_gradient, -1.0)
                        current_dot = -float(
                            pair_dot(current_gradient, current_gradient)
                        )
                    fallback_count += 1

                update, target_rms = self._scale_advanced_direction(
                    direction,
                    current_gradient,
                    inv,
                    d,
                    schedule,
                    step,
                    nm_alpha,
                )

                cap_scale = 1.0
                if self.score_osc_cap > 0.0 and geom is not None:
                    for _ in range(16):
                        score_change = joint_jvp(
                            update, geom
                        ) + joint_bilinear_remainder(update, geom)
                        osc = float(
                            score_oscillation(score_change, geom.mask).max()
                        )
                        if osc <= self.score_osc_cap:
                            break
                        update = pair_scale(update, 0.5)
                        cap_scale *= 0.5
                cap_scale_sum += cap_scale

                p_blocks = p.data.view(3, d, d)
                p_blocks[0].add_(update[0].reshape(d, d).to(p.dtype))
                p_blocks[1].add_(update[1].reshape(d, d).to(p.dtype))

                v_update = self._apply_v_update(
                    p_blocks[2], state, g_blocks[2], inv, schedule, d
                )

                n_layers += 1
                damping_sum += float(damping)
                rayleigh_sum += float(rayleigh)
                final_res = float(cg_info["relative_residuals"][-1])
                final_residual_sum += final_res
                final_residual_max = max(final_residual_max, final_res)
                direction_norm_sum += float(pair_norm(direction))
                update_norm_sum += float(pair_norm(update))
                current_dot_sum += current_dot
                qk_update_rms_sum += float(
                    pair_norm(update) / math.sqrt(float(2 * d * d))
                )
                v_update_rms_sum += float(
                    v_update.float().square().mean().sqrt()
                )
                c_mean_sum += float(coeff_info["c_mean"])
                c_min_sum += float(coeff_info["c_min"])
                c_max_sum += float(coeff_info["c_max"])
                nm_alpha_sum += nm_alpha
                target_rms_sum += float(target_rms)

                if do_diag and geom is not None:
                    U = joint_jvp(update, geom)
                    R = joint_bilinear_remainder(update, geom)
                    score_osc_sum += float(
                        score_oscillation(U + R, geom.mask).max()
                    )
                    bilinear_ratio_sum += float(
                        R.norm() / U.norm().clamp_min(1e-30)
                    )
                    n_diag_layers += 1

                del geom, c, apply, sample

        solve_seconds = 0.0
        if do_diag and start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            solve_seconds = start_event.elapsed_time(end_event) / 1000.0

        denom = max(n_layers, 1)
        fisher_denom = max(n_fisher_layers, 1)
        diag_denom = max(n_diag_layers, 1)
        self.last_diag = {
            "fisher_qk/coeff_projected": float(self.coeff_mode == "projected"),
            "fisher_qk/cg_iters": float(self.cg_iters),
            "fisher_qk/damp_rel": float(self.damp_rel),
            "fisher_qk/scale_native": float(self.scale_mode == "native"),
            "fisher_qk/scale_rms": float(self.scale_mode == "rms"),
            "fisher_qk/qk_lr": float(self.qk_lr_for_step(step)),
            "fisher_qk/qk_lr_start": float(self.qk_lr_start),
            "fisher_qk/qk_lr_end": float(self.qk_lr_end),
            "fisher_qk/effective_v_lr": float(self.v_lr)
            * float(self.param_groups[0]["lr"]),
            "fisher_qk/v_mode_code": float({"adamw": 0, "muon": 1, "newton_muon": 2}[self.v_mode]),
            "fisher_qk/damping_mean": damping_sum / fisher_denom,
            "fisher_qk/rayleigh_mean": rayleigh_sum / fisher_denom,
            "fisher_qk/cg_final_residual_mean": final_residual_sum / fisher_denom,
            "fisher_qk/cg_final_residual_max": final_residual_max,
            "fisher_qk/direction_norm_mean": direction_norm_sum / denom,
            "fisher_qk/update_norm_mean": update_norm_sum / denom,
            "fisher_qk/current_gradient_dot_mean": current_dot_sum / denom,
            "fisher_qk/qk_update_rms_mean": qk_update_rms_sum / denom,
            "fisher_qk/qk_target_rms_mean": target_rms_sum / denom,
            "fisher_qk/v_update_rms_mean": v_update_rms_sum / denom,
            "fisher_qk/coeff_mean": c_mean_sum / fisher_denom,
            "fisher_qk/coeff_min_mean": c_min_sum / fisher_denom,
            "fisher_qk/coeff_max_mean": c_max_sum / fisher_denom,
            "fisher_qk/precond_inv_norm_mean": inv_norm_sum / denom,
            "fisher_qk/precond_refresh": float(did_refresh),
            "fisher_qk/descent_fallback_layers": float(fallback_count),
            "fisher_qk/score_cap_scale_mean": cap_scale_sum / denom,
            "fisher_qk/score_osc_max_mean": score_osc_sum / diag_denom
            if do_diag
            else 0.0,
            "fisher_qk/bilinear_ratio_mean": bilinear_ratio_sum / diag_denom
            if do_diag
            else 0.0,
            "fisher_qk/curvature_age": float(self.curvature_age),
            "fisher_qk/curvature_refresh_active": float(
                self.current_curv_refresh(step)
            ),
            "fisher_qk/curvature_capture_seconds": float(
                self.curvature_capture_seconds
            ),
            "fisher_qk/optimizer_seconds": float(solve_seconds),
            "fisher_qk/layers_updated": float(n_layers),
            "fisher_qk/fisher_layers": float(n_fisher_layers),
            "fisher_qk/nm_blend_mean": nm_alpha_sum / denom,
            "fisher_qk/spectral_blend": float(self.spectral_blend),
            "fisher_qk/fisher_nm_cosine_mean": (
                fisher_nm_cos_sum / max(fisher_nm_cos_count, 1)
            ),
        }
        self.global_step += 1
        self.curvature_age += 1


class FisherCorrectedNewtonMuon(FisherQK):
    """Fisher-corrected Muon/Newton-Muon hierarchy for fused QKV weights.

    Q and K support six method families:

      direct_muon
          raw Q/K gradient -> Muon momentum -> matrix sign.

      direct_nm
          Newton input inverse -> Newton-Muon momentum -> matrix sign.

      identity_nested
          exact Muon momentum pseudo-RHS -> identity-preconditioned Fisher PCG.
          At CG1 with spectral_blend=1 this recovers direct Q/K Muon.

      exact_nm_nested
          exact Newton-Muon momentum pseudo-RHS -> Newton-preconditioned Fisher
          PCG. At PCG1 with spectral_blend=1 this recovers direct Q/K
          Newton-Muon up to a positive PCG scalar removed by matrix sign.

      faithful_nm_pcg
          literal Fisher system H_F D = -G solved with the Newton input inverse
          as a PCG preconditioner. This is closer to the mirror/Fisher model but
          does not provide the exact momentum endpoint.

      plain_fisher
          the validated unpreconditioned Fisher-CG path with raw-gradient RHS
          momentum. It serves as the current Fisher control.

    V always uses the full Newton-Muon settings supplied to this optimizer. All
    methods use one joint Q/K Fisher operator, so Q/K cross-block coupling is
    retained whenever PCG depth exceeds zero.
    """

    _METHOD_CODES = {
        "direct_muon": 0,
        "direct_nm": 1,
        "identity_nested": 2,
        "exact_nm_nested": 3,
        "faithful_nm_pcg": 4,
        "plain_fisher": 5,
    }

    def __init__(
        self,
        params,
        *,
        method: str = "exact_nm_nested",
        cg_iters: int = 3,
        spectral_blend: float = 0.75,
        qk_lr: float = 4.4e-4,
        endpoint_lr: float = 4.4e-4,
        endpoint_momentum: float = 0.97,
        endpoint_nesterov: bool = True,
        plain_fisher_momentum: float = 0.90,
        curvature_refresh: int = 4,
        coeff_mode: str = "unit",
        coeff_normalize: str = "median",
        coeff_floor: float = 1e-3,
        beta: float = 0.0,
        cg_tol: float = 0.0,
        damp_rel: float = 0.1,
        damp_floor: float = 1e-8,
        v_lr: float = 4.4e-4,
        v_momentum: float = 0.97,
        v_nesterov: bool = True,
        backend_steps: int = 5,
        precond_refresh_period: int = 32,
        precond_ewma: float = 0.95,
        precond_init_diag: float = 0.001,
        precond_ridge_mult: float = 0.2,
        precond_eps: float = 1e-8,
        diag_every: int = 50,
        svd_layer: int = 5,
        n_head: int = 12,
    ):
        method = str(method).strip().lower()
        if method not in self._METHOD_CODES:
            raise ValueError(
                f"Unknown FCNM_METHOD={method!r}; expected one of "
                f"{sorted(self._METHOD_CODES)}"
            )
        if not (0.0 <= float(spectral_blend) <= 1.0):
            raise ValueError("FCNM_SPECTRAL_BLEND must lie in [0,1]")
        if int(cg_iters) < 1:
            raise ValueError("FCNM_CG_ITERS must be >= 1")
        if int(n_head) < 1:
            raise ValueError("FCNM_N_HEAD must be >= 1")

        # Initialize the validated Fisher/preconditioner state. The subclass
        # supplies its own momentum and scaling logic.
        super().__init__(
            params,
            coeff_mode=coeff_mode,
            coeff_normalize=coeff_normalize,
            coeff_floor=coeff_floor,
            beta=beta,
            cg_iters=cg_iters,
            cg_tol=cg_tol,
            damp_rel=damp_rel,
            damp_floor=damp_floor,
            scale_mode="rms",
            outer_scale=1.0,
            qk_lr=qk_lr,
            curv_refresh=curvature_refresh,
            momentum_mode="none",
            momentum=0.0,
            nesterov=True,
            post_transform="none",
            score_osc_cap=0.0,
            v_lr=v_lr,
            v_momentum=v_momentum,
            v_nesterov=v_nesterov,
            backend_steps=backend_steps,
            precond_refresh_period=precond_refresh_period,
            precond_ewma=precond_ewma,
            precond_init_diag=precond_init_diag,
            precond_ridge_mult=precond_ridge_mult,
            precond_eps=precond_eps,
            diag_every=diag_every,
        )
        self.method = method
        self.spectral_blend = float(spectral_blend)
        self.endpoint_lr = float(endpoint_lr)
        self.endpoint_momentum = float(endpoint_momentum)
        self.endpoint_nesterov = bool(endpoint_nesterov)
        self.plain_fisher_momentum = float(plain_fisher_momentum)
        self.svd_layer = int(svd_layer)
        self.n_head = int(n_head)

    def needs_curvature_capture(self, step: int) -> bool:
        if self.method in {"direct_muon", "direct_nm"}:
            return False
        return super().needs_curvature_capture(step)

    def _init_precond_state(self, p: Tensor, d: int) -> dict:
        state = super()._init_precond_state(p, d)
        if "precond_matrix" not in state:
            state["precond_matrix"] = torch.eye(
                d, device=p.device, dtype=torch.float32
            )
        return state

    def _update_preconditioner(
        self, p: Tensor, state: dict, d: int, do_refresh: bool
    ) -> bool:
        if not do_refresh:
            return False
        ref = getattr(p, "_qkv_stats_ref", None)
        if ref is None:
            raise RuntimeError("Missing _qkv_stats_ref on Fisher-corrected QKV weight")
        cnt = float(ref["count"].item())
        if cnt <= 0.0:
            raise RuntimeError(
                "Fisher-corrected preconditioner refresh requested without input Gram"
            )
        gram = ref["accum"] / cnt
        cov = state["precond_cov"]
        cov.lerp_(gram, 1.0 - self.precond_ewma)
        ridge = (
            cov.diagonal().mean() * self.precond_ridge_mult + self.precond_eps
        ).clamp_min(self.precond_eps)
        K = cov.clone()
        K.diagonal().add_(ridge)
        L, info = torch.linalg.cholesky_ex(K, upper=False, check_errors=False)
        if int(info.item()) != 0:
            eye = torch.eye(d, device=p.device, dtype=torch.float32)
            state["precond_inv"].copy_(eye)
            state["precond_matrix"].copy_(eye)
        else:
            state["precond_inv"].copy_(torch.cholesky_inverse(L, upper=False))
            state["precond_matrix"].copy_(K)
        ref["accum"].zero_()
        ref["count"].zero_()
        return True

    @staticmethod
    def _right_multiply_pair(pair, matrix: Tensor, d: int, H: int, dh: int):
        return (
            (pair[0].reshape(d, d) @ matrix).reshape(H, dh, d),
            (pair[1].reshape(d, d) @ matrix).reshape(H, dh, d),
        )

    @staticmethod
    def _pair_difference(x, y):
        return pair_add(x, y, -1.0)

    def _momentum_with_beta(
        self,
        state: dict,
        value,
        *,
        prefix: str,
        beta: float,
        nesterov: bool,
    ):
        key_q = f"{prefix}_q"
        key_k = f"{prefix}_k"
        if key_q not in state:
            state[key_q] = torch.zeros_like(value[0])
            state[key_k] = torch.zeros_like(value[1])
        bq, bk = state[key_q], state[key_k]
        bq.mul_(beta).add_(value[0])
        bk.mul_(beta).add_(value[1])
        if nesterov:
            return (
                value[0] + beta * bq,
                value[1] + beta * bk,
            )
        return bq, bk

    @staticmethod
    def _pair_lerp(x, y, alpha: float):
        alpha = float(alpha)
        return (
            x[0] * (1.0 - alpha) + y[0] * alpha,
            x[1] * (1.0 - alpha) + y[1] * alpha,
        )

    def _matrix_sign_pair(self, direction, d: int):
        return (
            zeropower_via_newtonschulz5(
                direction[0].reshape(d, d), steps=self.backend_steps
            ).reshape_as(direction[0]),
            zeropower_via_newtonschulz5(
                direction[1].reshape(d, d), steps=self.backend_steps
            ).reshape_as(direction[1]),
        )

    def _spectral_blend_pair(self, direction, d: int, alpha: float):
        alpha = float(alpha)
        if alpha <= 0.0:
            return direction
        shaped = match_pair_norm(self._matrix_sign_pair(direction, d), direction)
        if alpha >= 1.0:
            return shaped
        mixed = self._pair_lerp(direction, shaped, alpha)
        return match_pair_norm(mixed, direction)

    def _endpoint_update(self, pre_sign, *, lr: float, schedule: float, d: int):
        # Match QKVMatrixControl exactly: matrix sign followed by
        # -lr*sqrt(d). The negative sign is already contained in pre_sign.
        shaped = self._matrix_sign_pair(pre_sign, d)
        return pair_scale(shaped, float(schedule) * float(lr) * math.sqrt(float(d)))

    def _scale_to_endpoint(self, direction, endpoint_update):
        matched = match_pair_norm(direction, endpoint_update)
        if self.endpoint_lr <= 0.0:
            return pair_scale(matched, 0.0)
        return pair_scale(matched, self.qk_lr / self.endpoint_lr)

    @staticmethod
    def _norm_matched_relative_error(candidate, reference) -> float:
        candidate = match_pair_norm(candidate, reference)
        return float(
            pair_norm(pair_add(candidate, reference, -1.0))
            / pair_norm(reference).clamp_min(1e-30)
        )

    @staticmethod
    def _effective_rank_stats(direction, d: int):
        eranks = []
        stable = []
        for block in direction:
            s = torch.linalg.svdvals(block.reshape(d, d).float())
            total = s.sum().clamp_min(1e-30)
            prob = s / total
            entropy = -(prob * prob.clamp_min(1e-30).log()).sum()
            eranks.append(float(entropy.exp()))
            stable.append(
                float(s.square().sum() / s.max().square().clamp_min(1e-30))
            )
        return sum(eranks) / len(eranks), sum(stable) / len(stable)

    @torch.no_grad()
    def step(self):
        needs_fisher = self.method not in {"direct_muon", "direct_nm"}
        if needs_fisher and self.curvature_samples is None:
            raise RuntimeError(
                "FisherCorrectedNewtonMuon.step() called without curvature samples"
            )

        step = int(self.global_step)
        do_refresh = self.precond_flag_for_step(step)
        do_diag = self.diag_every > 0 and (step % self.diag_every == 0)
        start_event = end_event = None
        if do_diag:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        n_layers = 0
        n_fisher_layers = 0
        fallback_count = 0
        did_refresh = False
        damping_sum = rayleigh_sum = residual_sum = 0.0
        residual_max = 0.0
        qk_rms_sum = v_rms_sum = current_dot_sum = 0.0
        score_osc_sum = bilinear_sum = fisher_energy_sum = 0.0
        diag_geom_count = 0
        inv_norm_sum = 0.0
        cos_muon_sum = cos_nm_sum = 0.0
        rel_muon_sum = rel_nm_sum = 0.0
        rms_ratio_muon_sum = rms_ratio_nm_sum = 0.0
        spectral_k1_cos_sum = 0.0
        raw_pcg1_endpoint_cos_sum = 0.0
        raw_pcg1_endpoint_relerr_sum = 0.0
        raw_pcg1_endpoint_count = 0
        pre_erank_sum = pre_stable_rank_sum = 0.0
        pre_rank_count = 0

        layer_index = 0
        for group in self.param_groups:
            schedule = float(group["lr"])
            for p in group["params"]:
                this_layer = layer_index
                layer_index += 1
                if p.grad is None:
                    continue
                if p.ndim != 2 or p.shape[0] != 3 * p.shape[1]:
                    raise ValueError(
                        f"Fisher-corrected optimizer expects [3d,d], got {tuple(p.shape)}"
                    )

                d = int(p.shape[1])
                H = self.n_head
                if d % H != 0:
                    raise ValueError(f"d={d} is not divisible by n_head={H}")
                dh = d // H
                state = self._init_precond_state(p, d)
                did_refresh = (
                    self._update_preconditioner(p, state, d, do_refresh)
                    or did_refresh
                )
                inv = state["precond_inv"]
                Rmat = state["precond_matrix"]
                inv_norm_sum += float(inv.norm())

                g_blocks = p.grad.detach().float().view(3, d, d)
                current_gradient = (
                    g_blocks[0].reshape(H, dh, d),
                    g_blocks[1].reshape(H, dh, d),
                )
                pre_gradient = self._right_multiply_pair(
                    current_gradient, inv, d, H, dh
                )

                # Shadow direct endpoint states are advanced on every step so
                # endpoint cosines compare mature momentum trajectories.
                raw_momentum = self._momentum_with_beta(
                    state,
                    current_gradient,
                    prefix="fcnm_muon_momentum",
                    beta=self.endpoint_momentum,
                    nesterov=self.endpoint_nesterov,
                )
                nm_momentum = self._momentum_with_beta(
                    state,
                    pre_gradient,
                    prefix="fcnm_nm_momentum",
                    beta=self.endpoint_momentum,
                    nesterov=self.endpoint_nesterov,
                )
                muon_pre_sign = pair_scale(raw_momentum, -1.0)
                nm_pre_sign = pair_scale(nm_momentum, -1.0)
                muon_ref = self._endpoint_update(
                    muon_pre_sign,
                    lr=self.endpoint_lr,
                    schedule=schedule,
                    d=d,
                )
                nm_ref = self._endpoint_update(
                    nm_pre_sign,
                    lr=self.endpoint_lr,
                    schedule=schedule,
                    d=d,
                )

                sample = geom = c = apply = None
                damping = rayleigh = 0.0
                cg_info = {"relative_residuals": [0.0], "breakdown": False}
                direction_pre = None
                relevant_endpoint_pre = nm_pre_sign

                if self.method == "direct_muon":
                    direction_pre = muon_pre_sign
                    relevant_endpoint_pre = muon_pre_sign
                elif self.method == "direct_nm":
                    direction_pre = nm_pre_sign
                    relevant_endpoint_pre = nm_pre_sign
                else:
                    sample = self.curvature_samples[this_layer]
                    geom = build_geometry(
                        sample["x"],
                        sample["q"],
                        sample["k"],
                        sample["cos"],
                        sample["sin"],
                    )
                    if self.coeff_mode == "unit":
                        c = unit_coefficients(geom)
                    else:
                        c, _ = projected_coefficients(
                            geom=geom,
                            v_pre=sample["v_pre"],
                            g_out=sample["g_out"],
                            w_o=sample["w_o"],
                            beta=self.beta,
                            normalize=self.coeff_normalize,
                            floor=self.coeff_floor,
                        )

                    preconditioner = None
                    if self.method == "identity_nested":
                        solve_source = raw_momentum
                        rhs = pair_scale(raw_momentum, -1.0)
                        relevant_endpoint_pre = muon_pre_sign
                    elif self.method == "exact_nm_nested":
                        solve_source = self._right_multiply_pair(
                            nm_momentum, Rmat, d, H, dh
                        )
                        rhs = pair_scale(solve_source, -1.0)
                        relevant_endpoint_pre = nm_pre_sign

                        def preconditioner(v, inv=inv, d=d, H=H, dh=dh):
                            return self._right_multiply_pair(v, inv, d, H, dh)

                    elif self.method == "faithful_nm_pcg":
                        solve_source = current_gradient
                        rhs = pair_scale(current_gradient, -1.0)
                        relevant_endpoint_pre = nm_pre_sign

                        def preconditioner(v, inv=inv, d=d, H=H, dh=dh):
                            return self._right_multiply_pair(v, inv, d, H, dh)

                    elif self.method == "plain_fisher":
                        solve_source = self._momentum_with_beta(
                            state,
                            current_gradient,
                            prefix="fcnm_plain_fisher_rhs",
                            beta=self.plain_fisher_momentum,
                            nesterov=True,
                        )
                        rhs = pair_scale(solve_source, -1.0)
                        relevant_endpoint_pre = muon_pre_sign
                    else:
                        raise RuntimeError(f"Unhandled method {self.method!r}")

                    damping, rayleigh = estimate_relative_damping(
                        geom,
                        solve_source,
                        kind="fisher",
                        c=c,
                        damping_rel=self.damp_rel,
                        damping_floor=self.damp_floor,
                        reduction="mean",
                    )
                    apply = make_quadratic_operator(
                        geom,
                        kind="fisher",
                        c=c,
                        damping=damping,
                        reduction="mean",
                    )
                    direction_pre, cg_info = pcg_solve(
                        apply,
                        rhs,
                        iterations=self.cg_iters,
                        tol=self.cg_tol,
                        preconditioner=preconditioner,
                    )
                    n_fisher_layers += 1

                    # For the exact low-depth nesting identities, PCG1 is only a
                    # positive scalar multiple of the endpoint pre-sign direction.
                    # The repository's BF16 finite-step Newton-Schulz map is not
                    # perfectly homogeneous at the tiny native PCG1 scale because
                    # of its fixed epsilon and BF16 cast.  Record the *raw* PCG1
                    # algebraic recovery before spectral shaping, then canonicalize
                    # k=1 to the mathematically equivalent endpoint direction.  This
                    # also makes partial-spectral k=1 independent of the arbitrary
                    # positive PCG line-search scalar.  This preserves the established Muon/NM
                    # matrix-sign implementation and makes the intended endpoint
                    # identity exact without changing any k>1 experiment.
                    if (
                        self.cg_iters == 1
                        and self.method in {"identity_nested", "exact_nm_nested"}
                    ):
                        raw_pcg1_endpoint_cos_sum += float(
                            pair_cosine(direction_pre, relevant_endpoint_pre)
                        )
                        raw_pcg1_endpoint_relerr_sum += self._norm_matched_relative_error(
                            direction_pre, relevant_endpoint_pre
                        )
                        raw_pcg1_endpoint_count += 1
                        direction_pre = relevant_endpoint_pre

                direction = self._spectral_blend_pair(
                    direction_pre, d, self.spectral_blend
                )
                current_dot = float(pair_dot(current_gradient, direction))
                if (
                    (not math.isfinite(current_dot))
                    or current_dot >= 0.0
                    or bool(cg_info.get("breakdown", False))
                ):
                    direction_pre = relevant_endpoint_pre
                    direction = self._spectral_blend_pair(
                        relevant_endpoint_pre, d, self.spectral_blend
                    )
                    current_dot = float(pair_dot(current_gradient, direction))
                    fallback_count += 1

                endpoint_scale_ref = (
                    muon_ref
                    if self.method in {"direct_muon", "identity_nested", "plain_fisher"}
                    else nm_ref
                )
                update = self._scale_to_endpoint(direction, endpoint_scale_ref)

                p_blocks = p.data.view(3, d, d)
                p_blocks[0].add_(update[0].reshape(d, d).to(p.dtype))
                p_blocks[1].add_(update[1].reshape(d, d).to(p.dtype))

                # V uses the full Newton-Muon endpoint settings.
                gv = g_blocks[2] @ inv
                if "fcnm_v_momentum" not in state:
                    state["fcnm_v_momentum"] = torch.zeros_like(gv)
                vbuf = state["fcnm_v_momentum"]
                vbuf.mul_(self.v_momentum).add_(gv)
                if self.v_nesterov:
                    gv_eff = gv + self.v_momentum * vbuf
                else:
                    gv_eff = vbuf
                v_shaped = zeropower_via_newtonschulz5(
                    gv_eff, steps=self.backend_steps
                )
                v_update = (
                    -schedule
                    * self.v_lr
                    * math.sqrt(float(d))
                    * v_shaped
                )
                p_blocks[2].add_(v_update.to(p.dtype))

                n_layers += 1
                damping_sum += float(damping)
                rayleigh_sum += float(rayleigh)
                residual = float(cg_info["relative_residuals"][-1])
                residual_sum += residual
                residual_max = max(residual_max, residual)
                current_dot_sum += current_dot
                qk_rms = float(
                    pair_norm(update) / math.sqrt(float(2 * d * d))
                )
                qk_rms_sum += qk_rms
                v_rms_sum += float(v_update.float().square().mean().sqrt())

                cos_muon_sum += float(pair_cosine(update, muon_ref))
                cos_nm_sum += float(pair_cosine(update, nm_ref))
                rel_muon_sum += self._norm_matched_relative_error(update, muon_ref)
                rel_nm_sum += self._norm_matched_relative_error(update, nm_ref)
                muon_rms = float(
                    pair_norm(muon_ref) / math.sqrt(float(2 * d * d))
                )
                nm_rms = float(
                    pair_norm(nm_ref) / math.sqrt(float(2 * d * d))
                )
                rms_ratio_muon_sum += qk_rms / max(muon_rms, 1e-30)
                rms_ratio_nm_sum += qk_rms / max(nm_rms, 1e-30)

                spectral_candidate = self._spectral_blend_pair(
                    direction_pre, d, 1.0
                )
                spectral_endpoint = self._spectral_blend_pair(
                    relevant_endpoint_pre, d, 1.0
                )
                spectral_k1_cos_sum += float(
                    pair_cosine(spectral_candidate, spectral_endpoint)
                )

                if do_diag and geom is not None:
                    U = joint_jvp(update, geom)
                    R = joint_bilinear_remainder(update, geom)
                    score_osc_sum += float(
                        score_oscillation(U + R, geom.mask).max()
                    )
                    bilinear_sum += float(
                        R.norm() / U.norm().clamp_min(1e-30)
                    )
                    pre_U = joint_jvp(direction_pre, geom)
                    fisher_energy_sum += float(
                        fisher_energy(pre_U, geom.p, c, reduction="mean")
                    )
                    diag_geom_count += 1

                if do_diag and this_layer == self.svd_layer:
                    erank, stable_rank = self._effective_rank_stats(
                        direction_pre, d
                    )
                    pre_erank_sum += erank
                    pre_stable_rank_sum += stable_rank
                    pre_rank_count += 1

                del sample, geom, c, apply

        solve_seconds = 0.0
        if do_diag and start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            solve_seconds = start_event.elapsed_time(end_event) / 1000.0

        denom = max(n_layers, 1)
        fisher_denom = max(n_fisher_layers, 1)
        geom_denom = max(diag_geom_count, 1)
        rank_denom = max(pre_rank_count, 1)
        self.last_diag = {
            "fcnm/method_code": float(self._METHOD_CODES[self.method]),
            "fcnm/cg_iters": float(self.cg_iters),
            "fcnm/spectral_blend": float(self.spectral_blend),
            "fcnm/qk_lr": float(self.qk_lr),
            "fcnm/endpoint_lr": float(self.endpoint_lr),
            "fcnm/damping_mean": damping_sum / fisher_denom,
            "fcnm/rayleigh_mean": rayleigh_sum / fisher_denom,
            "fcnm/cg_final_residual_mean": residual_sum / fisher_denom,
            "fcnm/cg_final_residual_max": residual_max,
            "fcnm/qk_update_rms_mean": qk_rms_sum / denom,
            "fcnm/v_update_rms_mean": v_rms_sum / denom,
            "fcnm/current_gradient_dot_mean": current_dot_sum / denom,
            "fcnm/descent_fallback_layers": float(fallback_count),
            "fcnm/endpoint_cos_muon_mean": cos_muon_sum / denom,
            "fcnm/endpoint_cos_nm_mean": cos_nm_sum / denom,
            "fcnm/endpoint_relerr_muon_mean": rel_muon_sum / denom,
            "fcnm/endpoint_relerr_nm_mean": rel_nm_sum / denom,
            "fcnm/endpoint_rms_ratio_muon_mean": rms_ratio_muon_sum / denom,
            "fcnm/endpoint_rms_ratio_nm_mean": rms_ratio_nm_sum / denom,
            "fcnm/spectral_k1_cosine_mean": spectral_k1_cos_sum / denom,
            "fcnm/raw_pcg1_endpoint_cosine_mean": (
                raw_pcg1_endpoint_cos_sum / max(raw_pcg1_endpoint_count, 1)
            ),
            "fcnm/raw_pcg1_endpoint_relerr_mean": (
                raw_pcg1_endpoint_relerr_sum / max(raw_pcg1_endpoint_count, 1)
            ),
            "fcnm/pre_shape_effective_rank": pre_erank_sum / rank_denom,
            "fcnm/pre_shape_stable_rank": pre_stable_rank_sum / rank_denom,
            "fcnm/fisher_energy_mean": fisher_energy_sum / geom_denom,
            "fcnm/score_osc_max_mean": score_osc_sum / geom_denom,
            "fcnm/bilinear_ratio_mean": bilinear_sum / geom_denom,
            "fcnm/precond_inv_norm_mean": inv_norm_sum / denom,
            "fcnm/precond_refresh": float(did_refresh),
            "fcnm/curvature_age": float(self.curvature_age),
            "fcnm/curvature_capture_seconds": float(
                self.curvature_capture_seconds
            ),
            "fcnm/optimizer_seconds": float(solve_seconds),
            "fcnm/layers_updated": float(n_layers),
        }
        self.global_step += 1
        self.curvature_age += 1


class ScheduledQK(torch.optim.Optimizer):
    """Two-stage Q/K schedule over validated Fisher and Newton-Muon primitives.

    Phase modes:
      fisher_cgN  -- joint Fisher-QK with N truncated CG iterations; V remains
                     on the FisherQK object's Newton-Muon V path.
      newton_muon_qk  -- Newton-Muon on Q/K while V keeps the phase-one
                         Newton-Muon baseline and momentum state.
      newton_muon_qkv -- full fused Q/K/V Newton-Muon control.

    Phase 1 owns steps ``step < switch_step`` and phase 2 owns steps
    ``step >= switch_step``. A phase-2 value of ``none`` keeps phase 1 for
    the full run.

    Fisher-to-CG1 reuses the same Fisher object and therefore preserves its RHS
    momentum, curvature state, V momentum, and input preconditioner. A switch to
    Newton-Muon transfers the input covariance/inverse and the compatible V
    preconditioned-momentum block; Q/K Newton-Muon momentum starts at zero because
    Fisher RHS momentum lives in a different geometry.
    """

    def __init__(
        self,
        params,
        *,
        phase1_mode: str,
        phase2_mode: str,
        switch_step: int,
        fisher_kwargs: dict,
        nm_lr: float,
        nm_momentum: float,
        nm_nesterov: bool = True,
        nm_backend_steps: int = 5,
        precond_refresh_period: int = 32,
        precond_ewma: float = 0.95,
        precond_init_diag: float = 0.001,
        precond_ridge_mult: float = 0.2,
        precond_eps: float = 1e-8,
    ):
        params = list(params)
        if not params:
            raise ValueError("ScheduledQK requires at least one QKV parameter")

        self.phase1_mode = self._validate_phase_mode(phase1_mode, allow_none=False)
        self.phase2_mode = self._validate_phase_mode(phase2_mode, allow_none=True)
        self.switch_step = int(switch_step)
        if self.phase2_mode != "none" and self.switch_step < 0:
            raise ValueError("A nontrivial phase 2 requires QK_SWITCH_STEP >= 0")

        defaults = dict(lr=1.0)
        super().__init__(params, defaults)

        fisher_kwargs = dict(fisher_kwargs)
        fisher_kwargs["cg_iters"] = self._fisher_iters_for_mode(
            self.phase1_mode if self.phase1_mode.startswith("fisher_cg") else "fisher_cg3"
        )
        self.fisher = AdvancedFisherQK(params, **fisher_kwargs)
        self.nm_base_lr = float(nm_lr)
        self.nm = QKVMatrixControl(
            params,
            mode="newton_muon",
            lr=self.nm_base_lr,
            momentum=float(nm_momentum),
            nesterov=bool(nm_nesterov),
            backend_steps=int(nm_backend_steps),
            precond_refresh_period=int(precond_refresh_period),
            precond_ewma=float(precond_ewma),
            precond_init_diag=float(precond_init_diag),
            precond_ridge_mult=float(precond_ridge_mult),
            precond_eps=float(precond_eps),
        )

        self.global_step = 0
        self.last_diag = {}
        self._active_mode = None
        self._previous_mode = None
        self._transition_count = 0
        self._switched_this_step = False
        self._ensure_step = None

    @staticmethod
    def _validate_phase_mode(mode: str, *, allow_none: bool) -> str:
        mode = str(mode).strip().lower()
        if allow_none and mode == "none":
            return mode
        if mode in {"newton_muon_qk", "newton_muon_qkv"} or re.fullmatch(
            r"fisher_cg[1-9][0-9]*", mode
        ):
            return mode
        allowed = "newton_muon_qk, newton_muon_qkv, fisher_cgN" + (
            ", or none" if allow_none else ""
        )
        raise ValueError(f"Unsupported Q/K schedule phase {mode!r}; expected {allowed}")

    @staticmethod
    def _fisher_iters_for_mode(mode: str) -> int:
        match = re.fullmatch(r"fisher_cg([1-9][0-9]*)", str(mode))
        if match is None:
            raise ValueError(f"Not a Fisher phase mode: {mode!r}")
        return int(match.group(1))

    def mode_for_step(self, step: int) -> str:
        if (
            self.phase2_mode != "none"
            and self.switch_step >= 0
            and int(step) >= self.switch_step
        ):
            return self.phase2_mode
        return self.phase1_mode

    def _transfer_fisher_to_nm(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                d = int(p.shape[1])
                fisher_state = self.fisher.state.get(p, {})
                nm_state = self.nm.state[p]

                if "precond_cov" in fisher_state:
                    nm_state["precond_cov"] = fisher_state["precond_cov"].clone()
                    nm_state["precond_inv"] = fisher_state["precond_inv"].clone()

                momentum_buffer = torch.zeros_like(p.detach(), dtype=torch.float32)
                if "v_momentum_buffer" in fisher_state:
                    momentum_buffer[2 * d : 3 * d].copy_(
                        fisher_state["v_momentum_buffer"]
                    )
                nm_state["momentum_buffer"] = momentum_buffer

                ref = getattr(p, "_qkv_stats_ref", None)
                if ref is not None:
                    ref["accum"].zero_()
                    ref["count"].zero_()

    def _ensure_mode(self, step: int) -> str:
        step = int(step)
        mode = self.mode_for_step(step)
        if self._ensure_step != step:
            self._switched_this_step = False
            self._ensure_step = step
        if self._active_mode is None:
            self._active_mode = mode
            self._previous_mode = mode
        elif mode != self._active_mode:
            previous = self._active_mode
            if previous.startswith("fisher_cg") and mode == "newton_muon_qkv":
                self._transfer_fisher_to_nm()
            self._previous_mode = previous
            self._active_mode = mode
            self._transition_count += 1
            self._switched_this_step = True
        return mode

    def precond_flag_for_step(self, step: int) -> bool:
        mode = self._ensure_mode(step)
        if mode == "newton_muon_qkv":
            return self.nm.precond_flag_for_step(step)
        return self.fisher.precond_flag_for_step(step)

    def needs_curvature_capture(self, step: int) -> bool:
        mode = self._ensure_mode(step)
        return mode.startswith("fisher_cg") and self.fisher.needs_curvature_capture(step)

    def should_time(self, step: int) -> bool:
        mode = self._ensure_mode(step)
        return mode.startswith("fisher_cg") and self.fisher.should_time(step)

    def set_curvature_samples(self, samples, *, capture_seconds: float = 0.0) -> None:
        mode = self._ensure_mode(self.global_step)
        if not mode.startswith("fisher_cg"):
            raise RuntimeError("Curvature samples supplied while Newton-Muon phase is active")
        self.fisher.set_curvature_samples(samples, capture_seconds=capture_seconds)

    @torch.no_grad()
    def _step_qk_newton_muon(self, step: int, schedule: float) -> dict:
        """Newton-Muon on Q/K only; keep V on the continuous FisherQK V path."""
        do_refresh = self.fisher.precond_flag_for_step(step)
        did_refresh = False
        inv_norm_sum = 0.0
        qk_update_rms_sum = 0.0
        v_update_rms_sum = 0.0
        current_dot_sum = 0.0
        n_layers = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d = int(p.shape[1])
                fisher_state = self.fisher._init_precond_state(p, d)
                did_refresh = (
                    self.fisher._update_preconditioner(
                        p, fisher_state, d, do_refresh
                    )
                    or did_refresh
                )
                inv = fisher_state["precond_inv"]
                inv_norm_sum += float(inv.norm())
                g_blocks = p.grad.detach().float().view(3, d, d)
                p_blocks = p.data.view(3, d, d)
                schedule_state = self.state[p]
                qk_updates = []

                for block_index, name in [(0, "q"), (1, "k")]:
                    g_pre = g_blocks[block_index] @ inv
                    key = f"nm_qk_{name}_momentum_buffer"
                    if key not in schedule_state:
                        schedule_state[key] = torch.zeros_like(g_pre)
                    buf = schedule_state[key]
                    momentum = float(self.nm.param_groups[0]["momentum"])
                    buf.mul_(momentum).add_(g_pre)
                    if bool(self.nm.param_groups[0]["nesterov"]):
                        g_eff = g_pre + momentum * buf
                    else:
                        g_eff = buf
                    shaped = zeropower_via_newtonschulz5(
                        g_eff,
                        steps=int(self.nm.param_groups[0]["backend_steps"]),
                    )
                    update = -schedule * self.nm_base_lr * math.sqrt(float(d)) * shaped
                    p_blocks[block_index].add_(update.to(p.dtype))
                    qk_updates.append(update)

                # Preserve the Fisher-phase V baseline and its compatible state.
                gv = g_blocks[2] @ inv
                if "v_momentum_buffer" not in fisher_state:
                    fisher_state["v_momentum_buffer"] = torch.zeros_like(gv)
                vbuf = fisher_state["v_momentum_buffer"]
                vbuf.mul_(self.fisher.v_momentum).add_(gv)
                if self.fisher.v_nesterov:
                    gv_eff = gv + self.fisher.v_momentum * vbuf
                else:
                    gv_eff = vbuf
                v_shaped = zeropower_via_newtonschulz5(
                    gv_eff, steps=self.fisher.backend_steps
                )
                v_update = (
                    -schedule
                    * self.fisher.v_lr
                    * math.sqrt(float(d))
                    * v_shaped
                )
                p_blocks[2].add_(v_update.to(p.dtype))

                n_layers += 1
                qk_norm = torch.sqrt(
                    qk_updates[0].float().square().sum()
                    + qk_updates[1].float().square().sum()
                )
                qk_update_rms_sum += float(
                    qk_norm / math.sqrt(float(2 * d * d))
                )
                v_update_rms_sum += float(v_update.float().square().mean().sqrt())
                current_dot_sum += float(
                    (g_blocks[0] * qk_updates[0]).sum()
                    + (g_blocks[1] * qk_updates[1]).sum()
                )

        denom = max(n_layers, 1)
        return {
            "qkv_control/mode_newton_muon": 1.0,
            "qkv_control/qk_only": 1.0,
            "qkv_control/precond_refresh": float(did_refresh),
            "qkv_control/precond_inv_norm_mean": inv_norm_sum / denom,
            "qkv_control/layers_updated": float(n_layers),
            "fisher_qk/qk_update_rms_mean": qk_update_rms_sum / denom,
            "fisher_qk/v_update_rms_mean": v_update_rms_sum / denom,
            "fisher_qk/current_gradient_dot_mean": current_dot_sum / denom,
            "fisher_qk/descent_fallback_layers": 0.0,
        }

    @torch.no_grad()
    def step(self):
        step = int(self.global_step)
        mode = self._ensure_mode(step)
        schedule = float(self.param_groups[0]["lr"])

        if mode.startswith("fisher_cg"):
            self.fisher.cg_iters = self._fisher_iters_for_mode(mode)
            self.fisher.global_step = step
            self.fisher.param_groups[0]["lr"] = schedule
            self.fisher.step()
            child_diag = dict(self.fisher.last_diag)
        elif mode == "newton_muon_qk":
            child_diag = self._step_qk_newton_muon(step, schedule)
        else:
            self.nm.global_step = step
            self.nm.param_groups[0]["lr"] = schedule * self.nm_base_lr
            self.nm.step()
            child_diag = dict(self.nm.last_diag)

        if mode == "newton_muon_qkv":
            mode_code = 0.0
        elif mode == "newton_muon_qk":
            mode_code = -1.0
        else:
            mode_code = float(self._fisher_iters_for_mode(mode))
        child_diag.update({
            "qk_schedule/active_mode_code": mode_code,
            "qk_schedule/active_newton_muon": float(mode.startswith("newton_muon")),
            "qk_schedule/active_newton_muon_qk_only": float(mode == "newton_muon_qk"),
            "qk_schedule/active_fisher": float(mode.startswith("fisher_cg")),
            "qk_schedule/active_fisher_cg_iters": (
                float(self._fisher_iters_for_mode(mode)) if mode.startswith("fisher_cg") else 0.0
            ),
            "qk_schedule/switch_step": float(self.switch_step),
            "qk_schedule/switched_this_step": float(self._switched_this_step),
            "qk_schedule/transition_count": float(self._transition_count),
        })
        self.last_diag = child_diag
        self.global_step = step + 1

    def state_dict(self):
        result = super().state_dict()
        result["_scheduled_qk_fisher"] = self.fisher.state_dict()
        result["_scheduled_qk_nm"] = self.nm.state_dict()
        result["_scheduled_qk_meta"] = {
            "global_step": int(self.global_step),
            "active_mode": self._active_mode,
            "previous_mode": self._previous_mode,
            "transition_count": int(self._transition_count),
            "ensure_step": self._ensure_step,
        }
        return result

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        fisher_state = state_dict.pop("_scheduled_qk_fisher", None)
        nm_state = state_dict.pop("_scheduled_qk_nm", None)
        meta = state_dict.pop("_scheduled_qk_meta", {})
        super().load_state_dict(state_dict)
        if fisher_state is not None:
            self.fisher.load_state_dict(fisher_state)
        if nm_state is not None:
            self.nm.load_state_dict(nm_state)
        self.global_step = int(meta.get("global_step", self.global_step))
        self._active_mode = meta.get("active_mode", self._active_mode)
        self._previous_mode = meta.get("previous_mode", self._previous_mode)
        self._transition_count = int(
            meta.get("transition_count", self._transition_count)
        )
        self._ensure_step = meta.get("ensure_step", self._ensure_step)

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
        self.rpb_rowsign_power = float(config.rpb_rowsign_power)
        self.qkv_opt_mode = str(config.qkv_opt_mode).strip().lower()
        if self.qkv_opt_mode not in {"hybrid", "bridge", "newton_muon", "muon", "fisher_qk", "fisher_corrected"}:
            raise ValueError(f"Unsupported qkv_opt_mode={self.qkv_opt_mode!r}")
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
        self.rpb_gradmax = nn.Buffer(torch.zeros(3, H, dtype=torch.float32), persistent=False)
        self.rpb_rownorm = nn.Buffer(torch.zeros(3, H, dtype=torch.float32), persistent=False)
        self.rpb_count   = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self.rpb_gy      = nn.Buffer(torch.zeros(H, dtype=torch.float32), persistent=False)

        # Exact Newton-Muon control: QKV input second-moment statistics. These are
        # captured only on refresh steps and are not used by the hybrid RPB path.
        self.qkv_xtx_accum = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.qkv_xtx_tmp = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.qkv_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self._attach_rpb_ref()
        self._attach_qkv_stats_ref()

        # Local one-step attention-geometry audit.  The audit uses the uncompiled
        # model and enables this flag on one selected layer for two small batches.
        # Training forwards leave it disabled and incur no tensor retention.
        self.audit_capture = False
        self.audit_cache = {}

    def _attach_rpb_ref(self):
        self.c_attn.weight._rpb_ref = {
            "M": self.rpb_M, "gram": self.rpb_gram, "sg": self.rpb_sg,
            "gradmax": self.rpb_gradmax,
            "rownorm": self.rpb_rownorm, "count": self.rpb_count, "gy": self.rpb_gy,
            "d": self.n_embd, "n_head": self.n_head, "d_h": self.head_dim,
        }

    def _attach_qkv_stats_ref(self):
        self.c_attn.weight._qkv_stats_ref = {
            "kind": "qkv",
            "d": self.n_embd,
            "accum": self.qkv_xtx_accum,
            "count": self.qkv_xtx_count,
        }

    def forward(self, x, precond_flag: bool = False):
        B, T, C = x.size()
        audit = bool(self.audit_capture and torch.is_grad_enabled())

        x2d = x.reshape(-1, C)
        if self.qkv_opt_mode in {"hybrid", "bridge"}:
            capture = torch.is_grad_enabled()
            qkv = _QKVCapture.apply(
                x2d, self.c_attn.weight, self.c_attn.weight._rpb_ref, capture,
                self.rpb_rowsign_power, self.qkv_opt_mode == "bridge",
            )
            qkv = qkv.view(B, T, 3 * C)
        else:
            # Standard differentiable linear path so exact Muon/Newton-Muon controls
            # receive the raw QKV weight gradient.
            if precond_flag and self.qkv_opt_mode in {"newton_muon", "fisher_qk", "fisher_corrected"}:
                torch.ops.nanogpt.accum_xtx(
                    x2d, self.qkv_xtx_accum, self.qkv_xtx_count, self.qkv_xtx_tmp
                )
            qkv = self.c_attn(x)

        if audit:
            qkv.retain_grad()
            self.audit_cache = {"x": x.detach(), "qkv": qkv}

        q_pre, k_pre, v = qkv.split(self.n_embd, dim=2)
        k_pre = k_pre.view(B, T, self.n_head, self.head_dim)
        q_pre = q_pre.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q_pre)
        q = apply_rotary_emb(q_pre, cos, sin)
        k = apply_rotary_emb(k_pre, cos, sin)

        if audit:
            self.audit_cache.update({
                "q_pre": q_pre.detach(),
                "k_pre": k_pre.detach(),
                "v_pre": v.detach(),
                "q": q.detach().transpose(1, 2).contiguous(),
                "k": k.detach().transpose(1, 2).contiguous(),
                "cos": cos.detach(),
                "sin": sin.detach(),
            })

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if audit:
            y.retain_grad()
            self.audit_cache["y_preproj"] = y

        # Capture g_Y only for the activation-space hybrid path.
        if self.qkv_opt_mode in {"hybrid", "bridge"}:
            y = _AttnOutCapture.apply(y, self.c_attn.weight._rpb_ref, capture)

        if precond_flag:
            y2d = y.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(y2d, self.o_xtx_accum, self.o_xtx_count, self.xtx_tmp)

        out = self.c_proj(y)
        if audit:
            out.retain_grad()
            self.audit_cache["out"] = out
        return out

    def _apply(self, fn):
        super()._apply(fn)
        d = self.n_embd
        self.c_proj.weight._stats_ref = {"kind": "o", "d": d, "accum": self.o_xtx_accum, "count": self.o_xtx_count}
        self._attach_rpb_ref()
        self._attach_qkv_stats_ref()
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
    rpb_rowsign_power : float = 1.0
    qkv_opt_mode : str = "fisher_qk"

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

    # Model and data-loader dimensions.  Cycle B exposes these through
    # environment variables so the same validated code path can run the
    # 124M and GPT-2-medium-like 353M configurations.
    model_n_layer : int = 12
    model_n_head : int = 12
    model_n_embd : int = 768
    model_vocab_size : int = 50257
    diag_every : int = 100       # cadence for grad/update/weight-norm diagnostics

    # One-step attention-geometry audit. The model trains under the selected baseline
    # until attention_audit_step, audits one layer on two small independent batches,
    # writes JSON/TSV results, and optionally exits.
    attention_audit_step : int = -1
    attention_audit_layer : int = 5
    attention_audit_batch_size : int = 1
    attention_audit_sequence_length : int = 1024
    attention_audit_exit_after : int = 1
    attention_audit_base_lr : float = 0.00044
    # v2 evaluates every candidate under two protocols:
    #   native  : retain the candidate's own model-determined magnitude
    #             (or the ordinary optimizer LR for NM/raw-gradient controls)
    #   matched : Frobenius-match to the corresponding Newton-Muon block,
    #             then apply the common base LR.
    attention_audit_native_scale_grid : str = "0.03125,0.0625,0.125,0.25,0.5,1.0,2.0,4.0"
    attention_audit_matched_scale_grid : str = "0.03125,0.0625,0.125,0.25,0.5,1.0,2.0,4.0"
    attention_audit_protocols : str = "native,matched"
    attention_audit_model_sanity_tol : float = 1e-6
    attention_audit_damping_rels : str = "0.1,1.0"
    attention_audit_beta : float = 0.0
    attention_audit_coeff_normalize : str = "median"
    attention_audit_coeff_floor : float = 0.001
    attention_audit_mirror_newton_iters : int = 2
    attention_audit_mirror_cg_iters : int = 3
    attention_audit_ridge_mult : float = 0.2
    attention_audit_output_dir : str = "attention_geometry_audit"

    # QKV optimizer mode and exact-control settings.
    qkv_opt_mode : str = "fisher_qk"  # fisher_qk, fisher_corrected, newton_muon, hybrid, bridge, or muon
    block_matrix_lr_mult : float = 0.1
    block_matrix_momentum : float = 0.95
    qkv_control_lr : float = 0.0004
    qkv_control_momentum : float = 0.95
    qkv_control_steps : int = 5

    # Production joint Fisher-QK optimizer. Q/K use a matrix-free pulled-back
    # categorical-Fisher solve; V retains the Newton-Muon update.
    fisher_coeff_mode : str = "projected"       # unit or projected
    fisher_coeff_normalize : str = "median"     # none, mean, or median
    fisher_coeff_floor : float = 0.001
    fisher_beta : float = 0.0                    # trace-majorized head coupling; 0 initially
    fisher_cg_iters : int = 3
    fisher_cg_tol : float = 0.0
    fisher_damp_rel : float = 0.1
    fisher_damp_floor : float = 1e-8
    fisher_scale_mode : str = "native"          # native, rms, or nm_match
    fisher_outer_scale : float = 0.5             # native mode multiplier
    fisher_qk_lr : float = 0.00011               # target joint Q/K element RMS in rms mode
    fisher_curv_batch : int = 1                  # sampled sequences for curvature
    fisher_curv_refresh : int = 1                # reuse captured geometry for this many steps
    fisher_curv_precision : str = "bf16"         # bf16 or fp32 extra capture forward
    fisher_momentum_mode : str = "none"          # none, rhs, or direction
    fisher_momentum : float = 0.0
    fisher_nesterov : bool = True
    fisher_post_transform : str = "none"         # none or matrix_sign (later ablation)
    fisher_score_osc_cap : float = 0.0            # 0 disables exact-score cap
    fisher_v_lr : float = 0.0004
    fisher_v_momentum : float = 0.95
    fisher_v_nesterov : bool = True
    fisher_diag_every : int = 50

    # Advanced Fisher-QK structural and hybrid controls. Neutral defaults
    # exactly recover the validated FisherQK implementation.
    fisher_qk_lr_end : float = 0.0
    fisher_qk_lr_schedule : str = "constant"
    fisher_qk_lr_schedule_start : int = 0
    fisher_qk_lr_schedule_end : int = 0
    fisher_curv_refresh_late : int = 0
    fisher_curv_refresh_switch_step : int = -1
    fisher_spectral_blend : float = 0.0
    fisher_nm_blend_start : float = 0.0
    fisher_nm_blend_end : float = 0.0
    fisher_nm_blend_schedule : str = "constant"
    fisher_nm_blend_schedule_start : int = 0
    fisher_nm_blend_schedule_end : int = 0
    fisher_nm_lr : float = 0.00044
    fisher_nm_momentum : float = 0.97
    fisher_nm_nesterov : int = 1
    fisher_nm_shadow : int = 0
    fisher_blend_scale_mode : str = "fisher"
    fisher_layer_policy : str = "all_fisher"
    fisher_layer_mask : str = ""

    # Fisher-corrected Muon/Newton-Muon hierarchy.
    fcnm_method : str = "exact_nm_nested"
    fcnm_cg_iters : int = 3
    fcnm_spectral_blend : float = 0.75
    fcnm_qk_lr : float = 0.00044
    fcnm_endpoint_lr : float = 0.00044
    fcnm_endpoint_momentum : float = 0.97
    fcnm_plain_fisher_momentum : float = 0.90
    fcnm_curv_refresh : int = 4
    fcnm_coeff_mode : str = "unit"
    fcnm_coeff_normalize : str = "median"
    fcnm_coeff_floor : float = 0.001
    fcnm_beta : float = 0.0
    fcnm_damp_rel : float = 0.1
    fcnm_damp_floor : float = 1e-8
    fcnm_v_lr : float = 0.00044
    fcnm_v_momentum : float = 0.97
    fcnm_diag_every : int = 50
    fcnm_svd_layer : int = 5

    # Optional fixed two-stage Q/K schedule. The model keeps the standard
    # differentiable fisher_qk QKV path; optimizer3 changes phase.
    qk_schedule_enable : int = 0
    qk_phase1_mode : str = "fisher_cg3"
    qk_phase2_mode : str = "none"
    qk_switch_step : int = -1
    qk_nm_lr : float = 0.00044
    qk_nm_momentum : float = 0.97

    # Cycle-A whole-system assignment controls.
    # system_mode: adamw, muon, newton_muon, or fisher.
    # For fisher, backbone_mode controls O/MLP and v_mode controls V.
    cyclea_system_mode : str = "newton_muon"
    cyclea_backbone_mode : str = "newton_muon"
    cyclea_v_mode : str = "newton_muon"
    cyclea_head_lr : float = 0.004
    cyclea_adamw_lr : float = 0.001
    cyclea_adamw_beta1 : float = 0.9
    cyclea_adamw_beta2 : float = 0.95
    cyclea_adamw_weight_decay : float = 0.1
    cyclea_matrix_lr : float = 0.0004
    cyclea_matrix_momentum : float = 0.95
    cyclea_qkv_lr : float = 0.00044
    cyclea_qkv_momentum : float = 0.97
    cyclea_v_lr : float = 0.00044
    cyclea_v_momentum : float = 0.97
    cyclea_v_adamw_beta1 : float = 0.9
    cyclea_v_adamw_beta2 : float = 0.95
    cyclea_v_adamw_weight_decay : float = 0.1

    # RPB-to-Newton-Muon bridge. bridge_blend=0 is exact Newton-Muon;
    # bridge_blend=1 uses only the RPB activation direction, but with the same
    # precondition-before-momentum ordering and Muon update scale.
    rpb_nm_bridge_blend : float = 0.0
    rpb_radius_blend : float = 1.0      # 0=one uniform head radius, 1=headwise r*
    rpb_headnorm_blend : float = 1.0    # 0=no per-head max-row rescale, 1=current RPB

    # RPB optimizer knobs (also overridable via env vars, see README)
    rpb_momentum : float = 0.95
    rpb_nesterov : bool = True
    rpb_h_sigma : float = 8.0    # softmax-Hessian constant in the curvature bound
    rpb_ridge_mult : float = 0.2 # Gram-inverse ridge, relative to mean diagonal
    rpb_r_max : float = 0.0      # trust-region cap on r* (0 => no cap)
    rpb_rowsign_power : float = 1.0  # 1=row-sign; smaller retains row-magnitude information
    rpb_spectral_blend : float = 0.0 # 0=RPB update, 1=norm-matched matrix-sign update
    rpb_spectral_steps : int = 5     # Newton-Schulz iterations for the matrix-sign endpoint
    rpb_precond_blend : float = 1.0  # 0=identity direction, 1=full Gram-inverse direction
    rpb_nor_enable : int = 0         # 1 enables NorMuon-style row adaptation
    rpb_nor_beta2 : float = 0.95
    rpb_nor_eps : float = 1e-8
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
def _env_s(name, default):
    v = os.environ.get(name); return str(v) if v is not None else default
args.learning_rate = _env_f("LEARNING_RATE", args.learning_rate)
args.num_iterations = _env_i("NUM_ITERATIONS", args.num_iterations)
args.batch_size = _env_i("BATCH_SIZE", args.batch_size)
args.device_batch_size = _env_i("DEVICE_BATCH_SIZE", args.device_batch_size)
args.sequence_length = _env_i("SEQUENCE_LENGTH", args.sequence_length)
args.warmup_iters = _env_i("WARMUP_ITERS", args.warmup_iters)
args.warmdown_iters = _env_i("WARMDOWN_ITERS", args.warmdown_iters)
args.val_loss_every = _env_i("VAL_LOSS_EVERY", args.val_loss_every)
args.val_tokens = _env_i("VAL_TOKENS", args.val_tokens)
args.input_bin = _env_s("INPUT_BIN", args.input_bin)
args.input_val_bin = _env_s("INPUT_VAL_BIN", args.input_val_bin)
args.model_n_layer = _env_i("MODEL_N_LAYER", args.model_n_layer)
args.model_n_head = _env_i("MODEL_N_HEAD", args.model_n_head)
args.model_n_embd = _env_i("MODEL_N_EMBD", args.model_n_embd)
args.model_vocab_size = _env_i("MODEL_VOCAB_SIZE", args.model_vocab_size)
args.save_every = _env_i("SAVE_EVERY", args.save_every)
args.diag_every = _env_i("DIAG_EVERY", args.diag_every)
args.attention_audit_step = _env_i("ATTN_AUDIT_STEP", args.attention_audit_step)
args.attention_audit_layer = _env_i("ATTN_AUDIT_LAYER", args.attention_audit_layer)
args.attention_audit_batch_size = _env_i("ATTN_AUDIT_BATCH_SIZE", args.attention_audit_batch_size)
args.attention_audit_sequence_length = _env_i("ATTN_AUDIT_SEQUENCE_LENGTH", args.attention_audit_sequence_length)
args.attention_audit_exit_after = _env_i("ATTN_AUDIT_EXIT_AFTER", args.attention_audit_exit_after)
args.attention_audit_base_lr = _env_f("ATTN_AUDIT_BASE_LR", args.attention_audit_base_lr)
# Backward-compatible legacy grid: when supplied, use it for both protocols.
_legacy_audit_grid = os.environ.get("ATTN_AUDIT_SCALE_GRID")
if _legacy_audit_grid is not None:
    args.attention_audit_native_scale_grid = str(_legacy_audit_grid)
    args.attention_audit_matched_scale_grid = str(_legacy_audit_grid)
args.attention_audit_native_scale_grid = _env_s(
    "ATTN_AUDIT_NATIVE_SCALE_GRID", args.attention_audit_native_scale_grid
)
args.attention_audit_matched_scale_grid = _env_s(
    "ATTN_AUDIT_MATCHED_SCALE_GRID", args.attention_audit_matched_scale_grid
)
args.attention_audit_protocols = _env_s(
    "ATTN_AUDIT_PROTOCOLS", args.attention_audit_protocols
)
args.attention_audit_model_sanity_tol = _env_f(
    "ATTN_AUDIT_MODEL_SANITY_TOL", args.attention_audit_model_sanity_tol
)
args.attention_audit_damping_rels = _env_s("ATTN_AUDIT_DAMPING_RELS", args.attention_audit_damping_rels)
args.attention_audit_beta = _env_f("ATTN_AUDIT_BETA", args.attention_audit_beta)
args.attention_audit_coeff_normalize = _env_s("ATTN_AUDIT_COEFF_NORMALIZE", args.attention_audit_coeff_normalize)
args.attention_audit_coeff_floor = _env_f("ATTN_AUDIT_COEFF_FLOOR", args.attention_audit_coeff_floor)
args.attention_audit_mirror_newton_iters = _env_i("ATTN_AUDIT_MIRROR_NEWTON_ITERS", args.attention_audit_mirror_newton_iters)
args.attention_audit_mirror_cg_iters = _env_i("ATTN_AUDIT_MIRROR_CG_ITERS", args.attention_audit_mirror_cg_iters)
args.attention_audit_ridge_mult = _env_f("ATTN_AUDIT_RIDGE_MULT", args.attention_audit_ridge_mult)
args.attention_audit_output_dir = _env_s("ATTN_AUDIT_OUTPUT_DIR", args.attention_audit_output_dir)
args.qkv_opt_mode = _env_s("QKV_OPT_MODE", args.qkv_opt_mode).strip().lower()
args.block_matrix_lr_mult = _env_f("BLOCK_MATRIX_LR_MULT", args.block_matrix_lr_mult)
args.block_matrix_momentum = _env_f("BLOCK_MATRIX_MOMENTUM", args.block_matrix_momentum)
args.qkv_control_lr = _env_f("QKV_CONTROL_LR", args.qkv_control_lr)
args.qkv_control_momentum = _env_f("QKV_CONTROL_MOMENTUM", args.qkv_control_momentum)
args.qkv_control_steps = _env_i("QKV_CONTROL_STEPS", args.qkv_control_steps)
args.fisher_coeff_mode = _env_s("FISHER_COEFF_MODE", args.fisher_coeff_mode).strip().lower()
args.fisher_coeff_normalize = _env_s("FISHER_COEFF_NORMALIZE", args.fisher_coeff_normalize).strip().lower()
args.fisher_coeff_floor = _env_f("FISHER_COEFF_FLOOR", args.fisher_coeff_floor)
args.fisher_beta = _env_f("FISHER_BETA", args.fisher_beta)
args.fisher_cg_iters = _env_i("FISHER_CG_ITERS", args.fisher_cg_iters)
args.fisher_cg_tol = _env_f("FISHER_CG_TOL", args.fisher_cg_tol)
args.fisher_damp_rel = _env_f("FISHER_DAMP_REL", args.fisher_damp_rel)
args.fisher_damp_floor = _env_f("FISHER_DAMP_FLOOR", args.fisher_damp_floor)
args.fisher_scale_mode = _env_s("FISHER_SCALE_MODE", args.fisher_scale_mode).strip().lower()
args.fisher_outer_scale = _env_f("FISHER_OUTER_SCALE", args.fisher_outer_scale)
args.fisher_qk_lr = _env_f("FISHER_QK_LR", args.fisher_qk_lr)
args.fisher_curv_batch = _env_i("FISHER_CURV_BATCH", args.fisher_curv_batch)
args.fisher_curv_refresh = _env_i("FISHER_CURV_REFRESH", args.fisher_curv_refresh)
args.fisher_curv_precision = _env_s("FISHER_CURV_PRECISION", args.fisher_curv_precision).strip().lower()
args.fisher_momentum_mode = _env_s("FISHER_MOMENTUM_MODE", args.fisher_momentum_mode).strip().lower()
args.fisher_momentum = _env_f("FISHER_MOMENTUM", args.fisher_momentum)
args.fisher_post_transform = _env_s("FISHER_POST_TRANSFORM", args.fisher_post_transform).strip().lower()
args.fisher_score_osc_cap = _env_f("FISHER_SCORE_OSC_CAP", args.fisher_score_osc_cap)
args.fisher_v_lr = _env_f("FISHER_V_LR", args.fisher_v_lr)
args.fisher_v_momentum = _env_f("FISHER_V_MOMENTUM", args.fisher_v_momentum)
args.fisher_diag_every = _env_i("FISHER_DIAG_EVERY", args.fisher_diag_every)
args.fisher_qk_lr_end = _env_f("FISHER_QK_LR_END", args.fisher_qk_lr_end)
args.fisher_qk_lr_schedule = _env_s("FISHER_QK_LR_SCHEDULE", args.fisher_qk_lr_schedule).strip().lower()
args.fisher_qk_lr_schedule_start = _env_i("FISHER_QK_LR_SCHEDULE_START", args.fisher_qk_lr_schedule_start)
args.fisher_qk_lr_schedule_end = _env_i("FISHER_QK_LR_SCHEDULE_END", args.fisher_qk_lr_schedule_end)
args.fisher_curv_refresh_late = _env_i("FISHER_CURV_REFRESH_LATE", args.fisher_curv_refresh_late)
args.fisher_curv_refresh_switch_step = _env_i("FISHER_CURV_REFRESH_SWITCH_STEP", args.fisher_curv_refresh_switch_step)
args.fisher_spectral_blend = _env_f("FISHER_SPECTRAL_BLEND", args.fisher_spectral_blend)
args.fisher_nm_blend_start = _env_f("FISHER_NM_BLEND_START", args.fisher_nm_blend_start)
args.fisher_nm_blend_end = _env_f("FISHER_NM_BLEND_END", args.fisher_nm_blend_end)
args.fisher_nm_blend_schedule = _env_s("FISHER_NM_BLEND_SCHEDULE", args.fisher_nm_blend_schedule).strip().lower()
args.fisher_nm_blend_schedule_start = _env_i("FISHER_NM_BLEND_SCHEDULE_START", args.fisher_nm_blend_schedule_start)
args.fisher_nm_blend_schedule_end = _env_i("FISHER_NM_BLEND_SCHEDULE_END", args.fisher_nm_blend_schedule_end)
args.fisher_nm_lr = _env_f("FISHER_NM_LR", args.fisher_nm_lr)
args.fisher_nm_momentum = _env_f("FISHER_NM_MOMENTUM", args.fisher_nm_momentum)
args.fisher_nm_nesterov = _env_i("FISHER_NM_NESTEROV", args.fisher_nm_nesterov)
args.fisher_nm_shadow = _env_i("FISHER_NM_SHADOW", args.fisher_nm_shadow)
args.fisher_blend_scale_mode = _env_s("FISHER_BLEND_SCALE_MODE", args.fisher_blend_scale_mode).strip().lower()
args.fisher_layer_policy = _env_s("FISHER_LAYER_POLICY", args.fisher_layer_policy).strip().lower()
args.fisher_layer_mask = _env_s("FISHER_LAYER_MASK", args.fisher_layer_mask)
args.fcnm_method = _env_s("FCNM_METHOD", args.fcnm_method).strip().lower()
args.fcnm_cg_iters = _env_i("FCNM_CG_ITERS", args.fcnm_cg_iters)
args.fcnm_spectral_blend = _env_f("FCNM_SPECTRAL_BLEND", args.fcnm_spectral_blend)
args.fcnm_qk_lr = _env_f("FCNM_QK_LR", args.fcnm_qk_lr)
args.fcnm_endpoint_lr = _env_f("FCNM_ENDPOINT_LR", args.fcnm_endpoint_lr)
args.fcnm_endpoint_momentum = _env_f("FCNM_ENDPOINT_MOMENTUM", args.fcnm_endpoint_momentum)
args.fcnm_plain_fisher_momentum = _env_f("FCNM_PLAIN_FISHER_MOMENTUM", args.fcnm_plain_fisher_momentum)
args.fcnm_curv_refresh = _env_i("FCNM_CURV_REFRESH", args.fcnm_curv_refresh)
args.fcnm_coeff_mode = _env_s("FCNM_COEFF_MODE", args.fcnm_coeff_mode).strip().lower()
args.fcnm_coeff_normalize = _env_s("FCNM_COEFF_NORMALIZE", args.fcnm_coeff_normalize).strip().lower()
args.fcnm_coeff_floor = _env_f("FCNM_COEFF_FLOOR", args.fcnm_coeff_floor)
args.fcnm_beta = _env_f("FCNM_BETA", args.fcnm_beta)
args.fcnm_damp_rel = _env_f("FCNM_DAMP_REL", args.fcnm_damp_rel)
args.fcnm_damp_floor = _env_f("FCNM_DAMP_FLOOR", args.fcnm_damp_floor)
args.fcnm_v_lr = _env_f("FCNM_V_LR", args.fcnm_v_lr)
args.fcnm_v_momentum = _env_f("FCNM_V_MOMENTUM", args.fcnm_v_momentum)
args.fcnm_diag_every = _env_i("FCNM_DIAG_EVERY", args.fcnm_diag_every)
args.fcnm_svd_layer = _env_i("FCNM_SVD_LAYER", args.fcnm_svd_layer)
args.qk_schedule_enable = _env_i("QK_SCHEDULE_ENABLE", args.qk_schedule_enable)
args.qk_phase1_mode = _env_s("QK_PHASE1_MODE", args.qk_phase1_mode).strip().lower()
args.qk_phase2_mode = _env_s("QK_PHASE2_MODE", args.qk_phase2_mode).strip().lower()
args.qk_switch_step = _env_i("QK_SWITCH_STEP", args.qk_switch_step)
args.qk_nm_lr = _env_f("QK_NM_LR", args.qk_nm_lr)
args.qk_nm_momentum = _env_f("QK_NM_MOMENTUM", args.qk_nm_momentum)
args.rpb_nm_bridge_blend = _env_f("RPB_NM_BRIDGE_BLEND", args.rpb_nm_bridge_blend)
args.rpb_radius_blend = _env_f("RPB_RADIUS_BLEND", args.rpb_radius_blend)
args.rpb_headnorm_blend = _env_f("RPB_HEADNORM_BLEND", args.rpb_headnorm_blend)
args.rpb_eta = _env_f("RPB_ETA", args.rpb_eta)
args.rpb_momentum = _env_f("RPB_MOMENTUM", args.rpb_momentum)
args.rpb_h_sigma = _env_f("RPB_HSIGMA", args.rpb_h_sigma)
args.rpb_ridge_mult = _env_f("RPB_RIDGE_MULT", args.rpb_ridge_mult)
args.rpb_r_max = _env_f("RPB_RMAX", args.rpb_r_max)
args.rpb_rowsign_power = _env_f("RPB_ROWSIGN_POWER", args.rpb_rowsign_power)
args.rpb_spectral_blend = _env_f("RPB_SPECTRAL_BLEND", args.rpb_spectral_blend)
args.rpb_spectral_steps = _env_i("RPB_SPECTRAL_STEPS", args.rpb_spectral_steps)
args.rpb_precond_blend = _env_f("RPB_PRECOND_BLEND", args.rpb_precond_blend)
args.rpb_nor_enable = _env_i("RPB_NOR_ENABLE", args.rpb_nor_enable)
args.rpb_nor_beta2 = _env_f("RPB_NOR_BETA2", args.rpb_nor_beta2)
args.rpb_nor_eps = _env_f("RPB_NOR_EPS", args.rpb_nor_eps)
args.rpb_precond_refresh_period = _env_i("RPB_PRECOND_REFRESH", args.rpb_precond_refresh_period)
args.rpb_precond_ewma = _env_f("RPB_PRECOND_EWMA", args.rpb_precond_ewma)
args.rpb_precond_init_diag = _env_f("RPB_PRECOND_INIT_DIAG", args.rpb_precond_init_diag)

# Cycle-A system assignment. These overrides deliberately sit after the legacy
# QKV settings so one config defines the whole parameter-to-optimizer map.
args.cyclea_system_mode = _env_s("CYCLEA_SYSTEM_MODE", args.cyclea_system_mode).strip().lower()
args.cyclea_backbone_mode = _env_s("CYCLEA_BACKBONE_MODE", args.cyclea_backbone_mode).strip().lower()
args.cyclea_v_mode = _env_s("CYCLEA_V_MODE", args.cyclea_v_mode).strip().lower()
args.cyclea_head_lr = _env_f("CYCLEA_HEAD_LR", args.cyclea_head_lr)
args.cyclea_adamw_lr = _env_f("CYCLEA_ADAMW_LR", args.cyclea_adamw_lr)
args.cyclea_adamw_beta1 = _env_f("CYCLEA_ADAMW_BETA1", args.cyclea_adamw_beta1)
args.cyclea_adamw_beta2 = _env_f("CYCLEA_ADAMW_BETA2", args.cyclea_adamw_beta2)
args.cyclea_adamw_weight_decay = _env_f("CYCLEA_ADAMW_WEIGHT_DECAY", args.cyclea_adamw_weight_decay)
args.cyclea_matrix_lr = _env_f("CYCLEA_MATRIX_LR", args.cyclea_matrix_lr)
args.cyclea_matrix_momentum = _env_f("CYCLEA_MATRIX_MOMENTUM", args.cyclea_matrix_momentum)
args.cyclea_qkv_lr = _env_f("CYCLEA_QKV_LR", args.cyclea_qkv_lr)
args.cyclea_qkv_momentum = _env_f("CYCLEA_QKV_MOMENTUM", args.cyclea_qkv_momentum)
args.cyclea_v_lr = _env_f("CYCLEA_V_LR", args.cyclea_v_lr)
args.cyclea_v_momentum = _env_f("CYCLEA_V_MOMENTUM", args.cyclea_v_momentum)
args.cyclea_v_adamw_beta1 = _env_f("CYCLEA_V_ADAMW_BETA1", args.cyclea_v_adamw_beta1)
args.cyclea_v_adamw_beta2 = _env_f("CYCLEA_V_ADAMW_BETA2", args.cyclea_v_adamw_beta2)
args.cyclea_v_adamw_weight_decay = _env_f("CYCLEA_V_ADAMW_WEIGHT_DECAY", args.cyclea_v_adamw_weight_decay)

_allowed_systems = {"adamw", "muon", "newton_muon", "fisher"}
_allowed_backbones = {"adamw", "muon", "newton_muon"}
if args.cyclea_system_mode not in _allowed_systems:
    raise ValueError(f"Unknown CYCLEA_SYSTEM_MODE={args.cyclea_system_mode!r}")
if args.cyclea_backbone_mode not in _allowed_backbones:
    raise ValueError(f"Unknown CYCLEA_BACKBONE_MODE={args.cyclea_backbone_mode!r}")
if args.cyclea_v_mode not in _allowed_backbones:
    raise ValueError(f"Unknown CYCLEA_V_MODE={args.cyclea_v_mode!r}")

# Choose the differentiable QKV forward path required by the whole-system mode.
if args.cyclea_system_mode == "adamw":
    args.qkv_opt_mode = "newton_muon"  # ordinary differentiable linear path; no stats requested
elif args.cyclea_system_mode == "muon":
    args.qkv_opt_mode = "muon"
elif args.cyclea_system_mode == "newton_muon":
    args.qkv_opt_mode = "newton_muon"
else:
    args.qkv_opt_mode = "fisher_qk"

if args.qkv_opt_mode not in {"hybrid", "bridge", "newton_muon", "muon", "fisher_qk", "fisher_corrected"}:
    raise ValueError(
        f"QKV_OPT_MODE must be hybrid, bridge, newton_muon, muon, fisher_qk, or fisher_corrected; got {args.qkv_opt_mode!r}"
    )
print(f"[qkv] QKV_OPT_MODE={args.qkv_opt_mode}")
print(
    "[cycle_a] "
    f"system={args.cyclea_system_mode} "
    f"backbone={args.cyclea_backbone_mode} "
    f"v={args.cyclea_v_mode} "
    f"head_lr={args.cyclea_head_lr} "
    f"adamw_lr={args.cyclea_adamw_lr} "
    f"matrix_lr={args.cyclea_matrix_lr} "
    f"qkv_lr={args.cyclea_qkv_lr}"
)
if bool(args.qk_schedule_enable):
    if args.qkv_opt_mode != "fisher_qk":
        raise ValueError("QK_SCHEDULE_ENABLE requires QKV_OPT_MODE=fisher_qk")
    ScheduledQK._validate_phase_mode(args.qk_phase1_mode, allow_none=False)
    ScheduledQK._validate_phase_mode(args.qk_phase2_mode, allow_none=True)
    if args.qk_phase2_mode != "none" and args.qk_switch_step < 0:
        raise ValueError("A nontrivial QK_PHASE2_MODE requires QK_SWITCH_STEP >= 0")
    print(
        "[qk_schedule] "
        f"phase1={args.qk_phase1_mode} "
        f"phase2={args.qk_phase2_mode} "
        f"switch_step={args.qk_switch_step} "
        f"nm_lr={args.qk_nm_lr} "
        f"nm_momentum={args.qk_nm_momentum}"
    )

if args.batch_size < 1 or args.device_batch_size < 1:
    raise ValueError("BATCH_SIZE and DEVICE_BATCH_SIZE must be positive")
if args.sequence_length < 1:
    raise ValueError("SEQUENCE_LENGTH must be positive")
if args.warmup_iters < 0 or args.warmdown_iters < 0:
    raise ValueError("WARMUP_ITERS and WARMDOWN_ITERS must be nonnegative")
if args.warmup_iters + args.warmdown_iters > args.num_iterations:
    raise ValueError(
        "WARMUP_ITERS + WARMDOWN_ITERS cannot exceed NUM_ITERATIONS "
        f"({args.warmup_iters}+{args.warmdown_iters}>{args.num_iterations})"
    )

# Controlled seed for multi-run comparisons.
import random

SEED = int(os.environ.get("SEED", "0"))
args.seed = SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[seed] SEED={SEED}")

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
audit_loader = DistributedDataLoader(
    args.input_val_bin,
    args.attention_audit_batch_size,
    args.attention_audit_sequence_length,
    ddp_rank,
    ddp_world_size,
)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

num_vocab = int(args.model_vocab_size)
if args.model_n_embd % args.model_n_head != 0:
    raise ValueError(
        f"MODEL_N_EMBD={args.model_n_embd} must be divisible by "
        f"MODEL_N_HEAD={args.model_n_head}"
    )
raw_model = GPT(GPTConfig(
    vocab_size=num_vocab,
    n_layer=int(args.model_n_layer),
    n_head=int(args.model_n_head),
    n_embd=int(args.model_n_embd),
    rpb_rowsign_power=args.rpb_rowsign_power,
    qkv_opt_mode=args.qkv_opt_mode,
)).cuda()
model_params = sum(p.numel() for p in raw_model.parameters())
print(
    "[model] "
    f"layers={args.model_n_layer} heads={args.model_n_head} "
    f"embd={args.model_n_embd} vocab={num_vocab} "
    f"parameters={model_params}"
)
print(
    "[batch] "
    f"global_sequences={args.batch_size} device_sequences={args.device_batch_size} "
    f"sequence_length={args.sequence_length} "
    f"accumulation={train_accumulation_steps} "
    f"tokens_per_update={args.batch_size * args.sequence_length}"
)
print(
    "[schedule] "
    f"iterations={args.num_iterations} warmup={args.warmup_iters} "
    f"warmdown={args.warmdown_iters}"
)
model = torch.compile(raw_model)
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# Parameter split and explicit whole-system assignment.
# AdamW-class parameters are always the tied embedding/output head in the
# Muon/Newton-Muon/Fisher systems. In the vanilla AdamW system every parameter
# receives AdamW with the same hyperparameters (implemented as three disjoint
# optimizer instances, which is parameterwise equivalent to one AdamW object).
qkv_weights = [blk.attn.c_attn.weight for blk in raw_model.transformer.h]
qkv_ids = {id(w) for w in qkv_weights}
block_params = [p for p in raw_model.transformer.h.parameters() if id(p) not in qkv_ids]

system_mode = args.cyclea_system_mode
backbone_mode = (
    system_mode if system_mode in {"adamw", "muon", "newton_muon"}
    else args.cyclea_backbone_mode
)
v_mode = (
    system_mode if system_mode in {"adamw", "muon", "newton_muon"}
    else args.cyclea_v_mode
)

head_uses_backbone_adamw = (
    system_mode == "adamw" or (system_mode == "fisher" and backbone_mode == "adamw")
)
head_lr = args.cyclea_adamw_lr if system_mode == "adamw" else args.cyclea_head_lr
optimizer1 = torch.optim.AdamW(
    raw_model.lm_head.parameters(),
    lr=head_lr,
    betas=(args.cyclea_adamw_beta1, args.cyclea_adamw_beta2),
    weight_decay=(
        args.cyclea_adamw_weight_decay if head_uses_backbone_adamw else args.weight_decay
    ),
    fused=True,
)

if backbone_mode == "adamw":
    optimizer2 = torch.optim.AdamW(
        block_params,
        lr=args.cyclea_adamw_lr,
        betas=(args.cyclea_adamw_beta1, args.cyclea_adamw_beta2),
        weight_decay=args.cyclea_adamw_weight_decay,
        fused=True,
    )
    block_diag_name = "adamw_blocks"
else:
    optimizer2 = Muon(
        block_params,
        lr=args.cyclea_matrix_lr,
        momentum=args.cyclea_matrix_momentum,
    )
    if backbone_mode == "newton_muon":
        optimizer2.attach_preconditioner()
    block_diag_name = backbone_mode + "_blocks"

if system_mode == "adamw":
    optimizer3 = torch.optim.AdamW(
        qkv_weights,
        lr=args.cyclea_adamw_lr,
        betas=(args.cyclea_adamw_beta1, args.cyclea_adamw_beta2),
        weight_decay=args.cyclea_adamw_weight_decay,
        fused=True,
    )
    qkv_diag_name = "adamw_qkv"
elif system_mode in {"muon", "newton_muon"}:
    optimizer3 = QKVMatrixControl(
        qkv_weights,
        mode=system_mode,
        lr=args.cyclea_qkv_lr,
        momentum=args.cyclea_qkv_momentum,
        nesterov=True,
        backend_steps=args.qkv_control_steps,
        precond_refresh_period=args.rpb_precond_refresh_period,
        precond_ewma=args.rpb_precond_ewma,
        precond_init_diag=args.rpb_precond_init_diag,
        precond_ridge_mult=args.rpb_ridge_mult,
    )
    qkv_diag_name = system_mode + "_qkv"
else:
    fisher_kwargs = dict(
        coeff_mode=args.fisher_coeff_mode,
        coeff_normalize=args.fisher_coeff_normalize,
        coeff_floor=args.fisher_coeff_floor,
        beta=args.fisher_beta,
        cg_iters=args.fisher_cg_iters,
        cg_tol=args.fisher_cg_tol,
        damp_rel=args.fisher_damp_rel,
        damp_floor=args.fisher_damp_floor,
        scale_mode=args.fisher_scale_mode,
        outer_scale=args.fisher_outer_scale,
        qk_lr=args.fisher_qk_lr,
        curv_refresh=args.fisher_curv_refresh,
        momentum_mode=args.fisher_momentum_mode,
        momentum=args.fisher_momentum,
        nesterov=args.fisher_nesterov,
        post_transform=args.fisher_post_transform,
        score_osc_cap=args.fisher_score_osc_cap,
        v_lr=args.cyclea_v_lr,
        v_momentum=args.cyclea_v_momentum,
        v_nesterov=True,
        v_mode=v_mode,
        v_beta1=args.cyclea_v_adamw_beta1,
        v_beta2=args.cyclea_v_adamw_beta2,
        v_weight_decay=args.cyclea_v_adamw_weight_decay,
        backend_steps=args.qkv_control_steps,
        precond_refresh_period=args.rpb_precond_refresh_period,
        precond_ewma=args.rpb_precond_ewma,
        precond_init_diag=args.rpb_precond_init_diag,
        precond_ridge_mult=args.rpb_ridge_mult,
        diag_every=args.fisher_diag_every,
        qk_lr_end=args.fisher_qk_lr_end,
        qk_lr_schedule=args.fisher_qk_lr_schedule,
        qk_lr_schedule_start=args.fisher_qk_lr_schedule_start,
        qk_lr_schedule_end=args.fisher_qk_lr_schedule_end,
        curv_refresh_late=args.fisher_curv_refresh_late,
        curv_refresh_switch_step=args.fisher_curv_refresh_switch_step,
        spectral_blend=args.fisher_spectral_blend,
        nm_blend_start=args.fisher_nm_blend_start,
        nm_blend_end=args.fisher_nm_blend_end,
        nm_blend_schedule=args.fisher_nm_blend_schedule,
        nm_blend_schedule_start=args.fisher_nm_blend_schedule_start,
        nm_blend_schedule_end=args.fisher_nm_blend_schedule_end,
        nm_lr=args.fisher_nm_lr,
        nm_momentum=args.fisher_nm_momentum,
        nm_nesterov=bool(args.fisher_nm_nesterov),
        nm_shadow=bool(args.fisher_nm_shadow),
        blend_scale_mode=args.fisher_blend_scale_mode,
        layer_policy=args.fisher_layer_policy,
        layer_mask=args.fisher_layer_mask,
    )
    optimizer3 = AdvancedFisherQK(qkv_weights, **fisher_kwargs)
    qkv_diag_name = "fisher_qk"

optimizers = [optimizer1, optimizer2, optimizer3]
named_params = {
    "head": [p for p in optimizer1.param_groups[0]["params"]],
    block_diag_name: block_params,
    qkv_diag_name: qkv_weights,
}
opt_names = ["head", block_diag_name, qkv_diag_name]

print("[cycle_a_assignment]")
print(f"  Q/K: {'Fisher' if system_mode == 'fisher' else system_mode}")
print(f"  V: {v_mode}")
print(f"  O/MLP: {backbone_mode}")
print("  embedding/head: AdamW")
# Cheap Muown-motivated diagnostics for the QKV weights.  The
# maximum output-row norm is the quantity Muown identifies as the empirical
# driver of weight spectral-norm drift.  We record it without changing the
# optimizer or paying for a full SVD.
@torch.no_grad()
def rpb_weight_row_metrics(params):
    layer_means = []
    layer_maxes = []
    layer_cvs = []
    all_sum = 0.0
    all_count = 0
    global_max = 0.0

    for p in params:
        rows = p.detach().float().norm(dim=1)
        mean = rows.mean()
        layer_means.append(float(mean))
        layer_maxes.append(float(rows.max()))
        layer_cvs.append(float(rows.std(unbiased=False) / mean.clamp_min(1e-12)))
        all_sum += float(rows.sum())
        all_count += rows.numel()
        global_max = max(global_max, float(rows.max()))

    global_mean = all_sum / max(all_count, 1)
    return {
        "weight_row_norm/rpb_mean": global_mean,
        "weight_row_norm/rpb_max": global_max,
        "weight_row_norm/rpb_max_over_mean": global_max / max(global_mean, 1e-12),
        "weight_row_norm/rpb_layer_cv_mean": sum(layer_cvs) / max(len(layer_cvs), 1),
        "weight_row_norm/rpb_layer_max_mean": sum(layer_maxes) / max(len(layer_maxes), 1),
    }

rpb_initial_row_metrics = rpb_weight_row_metrics(qkv_weights)
rpb_initial_row_max = rpb_initial_row_metrics["weight_row_norm/rpb_max"]
rpb_initial_row_mean = rpb_initial_row_metrics["weight_row_norm/rpb_mean"]

def _parse_float_list(text: str) -> list[float]:
    values = [float(piece.strip()) for piece in str(text).split(",") if piece.strip()]
    if not values:
        raise ValueError(f"Expected at least one float in {text!r}")
    if any((not math.isfinite(value)) or value == 0 for value in values):
        raise ValueError(
            f"Audit scale factors must be finite and nonzero: {values}"
        )
    return values


def _parse_str_list(text: str) -> list[str]:
    values = [piece.strip().lower() for piece in str(text).split(",") if piece.strip()]
    if not values:
        raise ValueError(f"Expected at least one string in {text!r}")
    return values


def _timed_cuda(fn):
    torch.cuda.synchronize()
    start = time.time()
    value = fn()
    torch.cuda.synchronize()
    return value, time.time() - start


@torch.no_grad()
def _audit_nm_direction(weight_grad: Tensor, x: Tensor, n_head: int, steps: int, ridge_mult: float):
    """Current-batch Newton-Muon direction, split into Q/K and fused QKV forms."""
    D = weight_grad.shape[1]
    dh = D // n_head
    xf = x.detach().float().reshape(-1, D)
    cov = xf.transpose(0, 1) @ xf / float(xf.shape[0])
    ridge = cov.diagonal().mean() * float(ridge_mult) + 1e-8
    K = cov.clone()
    K.diagonal().add_(ridge)
    L, info = torch.linalg.cholesky_ex(K, check_errors=False)
    inv = torch.eye(D, device=x.device, dtype=torch.float32) if int(info.item()) else torch.cholesky_inverse(L)

    blocks = []
    for G in weight_grad.detach().float().view(3, D, D):
        pre = G @ inv
        blocks.append(-math.sqrt(float(D)) * zeropower_via_newtonschulz5(pre, steps=steps).float())
    fused = torch.stack(blocks, dim=0).reshape(3 * D, D)
    qk = (blocks[0].reshape(n_head, dh, D), blocks[1].reshape(n_head, dh, D))
    return qk, fused, {
        "ridge": float(ridge),
        "inv_norm": float(inv.norm()),
        "cov_mean_diag": float(cov.diagonal().mean()),
    }


def _capture_attention_audit_batch(xb: Tensor, yb: Tensor, layer_idx: int):
    """Run one uncompiled forward/backward and retain selected-layer statistics."""
    was_training = raw_model.training
    raw_model.eval()
    raw_model.zero_grad(set_to_none=True)
    attn = raw_model.transformer.h[layer_idx].attn
    attn.audit_capture = True
    attn.audit_cache = {}
    try:
        # Use FP32 for the local-model audit. Training outside the audit
        # remains under the normal BF16 autocast context.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            _, loss = raw_model(xb, yb, return_logits=False, precond_flag=False)
        loss.backward()
        cache = attn.audit_cache
        required = ["x", "qkv", "v_pre", "q", "k", "cos", "sin", "y_preproj", "out"]
        missing = [key for key in required if key not in cache]
        if missing:
            raise RuntimeError(f"Audit capture missing tensors: {missing}")
        if cache["qkv"].grad is None or cache["y_preproj"].grad is None or cache["out"].grad is None:
            raise RuntimeError("Audit capture gradients were not retained")
        if attn.c_attn.weight.grad is None:
            raise RuntimeError("Selected QKV weight gradient is missing")
        result = {
            "loss": float(loss.detach()),
            "x": cache["x"].detach(),
            "qkv": cache["qkv"].detach(),
            "g_qkv": cache["qkv"].grad.detach(),
            "v_pre": cache["v_pre"].detach(),
            "q": cache["q"].detach(),
            "k": cache["k"].detach(),
            "cos": cache["cos"].detach(),
            "sin": cache["sin"].detach(),
            "g_y": cache["y_preproj"].grad.detach(),
            "g_out": cache["out"].grad.detach(),
            "weight_grad": attn.c_attn.weight.grad.detach().float().clone(),
        }
    finally:
        attn.audit_capture = False
        attn.audit_cache = {}
        raw_model.zero_grad(set_to_none=True)
        if was_training:
            raw_model.train()
    return result


def _audit_eval_loss(xb: Tensor, yb: Tensor) -> float:
    was_training = raw_model.training
    raw_model.eval()
    with torch.enable_grad(), torch.amp.autocast(
        device_type="cuda", enabled=False
    ):
        _, loss = raw_model(xb, yb, return_logits=False, precond_flag=False)
    value = float(loss.detach())
    if was_training:
        raw_model.train()
    return value


def _pair_from_fused(fused: Tensor, n_head: int):
    D = fused.shape[1]
    dh = D // n_head
    blocks = fused.float().view(3, D, D)
    return blocks[0].reshape(n_head, dh, D), blocks[1].reshape(n_head, dh, D)


def _fused_from_pair(pair, D: int) -> Tensor:
    out = torch.zeros((3, D, D), device=pair[0].device, dtype=torch.float32)
    out[0].copy_(pair[0].reshape(D, D))
    out[1].copy_(pair[1].reshape(D, D))
    return out.reshape(3 * D, D)


def _candidate_native_prediction(candidate, delta_pair, geom, c_unit, c_projected, damping):
    """Evaluate the candidate's own local model on an applied Q/K displacement.

    Returns a dictionary so the audit can distinguish linear gain, geometry
    energy, parameter regularization, predicted reduction, and model objective.
    A positive ``predicted_reduction`` is equivalent to a nonpositive local
    model value relative to the zero step.
    """
    kind = candidate.get("native_kind", "none")
    g_pair = candidate["gradient_pair"]
    gdot = float(pair_dot(g_pair, delta_pair))
    linear_gain = -gdot
    reg = float(damping) * float(pair_dot(delta_pair, delta_pair))
    energy = float("nan")
    pred = float("nan")

    if kind == "fisher_unit":
        U = joint_jvp(delta_pair, geom)
        energy = float(fisher_energy(U, geom.p, c_unit))
        pred = linear_gain - 0.5 * (energy + reg)
    elif kind == "fisher_projected":
        U = joint_jvp(delta_pair, geom)
        energy = float(fisher_energy(U, geom.p, c_projected))
        pred = linear_gain - 0.5 * (energy + reg)
    elif kind == "q2":
        U = joint_jvp(delta_pair, geom)
        energy = float(q2_energy(U, geom.mask))
        pred = linear_gain - 0.5 * (energy + reg)
    elif kind == "mirror_unit":
        U = joint_jvp(delta_pair, geom)
        energy = float(mirror_bregman(geom.scores, U, geom.p, c_unit, geom.mask))
        # Mirror model: <g,D> + Dh(S+JD,S) + lambda/2 ||D||^2.
        pred = linear_gain - energy - 0.5 * reg
    elif kind == "mirror_projected":
        U = joint_jvp(delta_pair, geom)
        energy = float(mirror_bregman(geom.scores, U, geom.p, c_projected, geom.mask))
        pred = linear_gain - energy - 0.5 * reg

    return {
        "kind": kind,
        "linear_gain": linear_gain,
        "geometry_energy": energy,
        "regularizer_energy": reg,
        "predicted_reduction": pred,
        "model_objective": -pred if math.isfinite(pred) else float("nan"),
    }


def run_attention_geometry_audit(step: int) -> dict:
    """One-step local-model audit on one selected layer and two independent batches."""
    if args.qkv_opt_mode not in {"newton_muon", "muon"}:
        raise RuntimeError(
            "The attention audit needs ordinary QKV weight gradients. "
            "Launch with QKV_OPT_MODE=newton_muon (recommended) or muon."
        )
    layer_idx = int(args.attention_audit_layer)
    if not (0 <= layer_idx < len(raw_model.transformer.h)):
        raise ValueError(f"ATTN_AUDIT_LAYER={layer_idx} is out of range")

    audit_loader.reset()
    x_a, y_a = audit_loader.next_batch()
    x_b, y_b = audit_loader.next_batch()

    torch.cuda.synchronize()
    audit_t0 = time.time()
    cap_a = _capture_attention_audit_batch(x_a, y_a, layer_idx)
    cap_b = _capture_attention_audit_batch(x_b, y_b, layer_idx)
    block = raw_model.transformer.h[layer_idx]
    attn = block.attn
    D = attn.n_embd
    H = attn.n_head
    dh = attn.head_dim

    geom = build_geometry(cap_a["x"], cap_a["q"], cap_a["k"], cap_a["cos"], cap_a["sin"])
    g_blocks = cap_a["weight_grad"].view(3, D, D)
    g_pair = (g_blocks[0].reshape(H, dh, D), g_blocks[1].reshape(H, dh, D))
    gb_blocks = cap_b["weight_grad"].view(3, D, D)
    gb_pair = (gb_blocks[0].reshape(H, dh, D), gb_blocks[1].reshape(H, dh, D))

    c_unit = unit_coefficients(geom)
    c_projected, coeff_info = projected_coefficients(
        geom=geom,
        v_pre=cap_a["v_pre"],
        g_out=cap_a["g_out"],
        w_o=attn.c_proj.weight.detach(),
        beta=float(args.attention_audit_beta),
        normalize=args.attention_audit_coeff_normalize,
        floor=float(args.attention_audit_coeff_floor),
    )

    (nm_pair, nm_fused, nm_info), nm_seconds = _timed_cuda(lambda: _audit_nm_direction(
        cap_a["weight_grad"], cap_a["x"], H,
        steps=int(args.qkv_control_steps),
        ridge_mult=float(args.attention_audit_ridge_mult),
    ))
    nm_info["solve_seconds"] = nm_seconds
    nm_qk_fused = _fused_from_pair(nm_pair, D)

    (faithful_rpb, feasible_rpb, rpb_info), rpb_seconds = _timed_cuda(lambda: rpb_audit_directions(
        x=cap_a["x"],
        qkv_pre=cap_a["qkv"],
        g_qkv=cap_a["g_qkv"],
        g_y=cap_a["g_y"],
        n_head=H,
        ridge_mult=float(args.attention_audit_ridge_mult),
        h_sigma=float(args.rpb_h_sigma),
        r_max=(float(args.rpb_r_max) if args.rpb_r_max > 0 else None),
    ))
    rpb_info["solve_seconds"] = rpb_seconds

    candidates = []

    def add_pair(
        name, pair, *, native_kind="none", damping=0.0, solver=None,
        c_for_diag=None, native_scale_mode="direct",
    ):
        candidates.append({
            "name": name,
            "scope": "qk",
            "pair": (pair[0].float(), pair[1].float()),
            "fused": _fused_from_pair(pair, D),
            "native_kind": native_kind,
            "native_scale_mode": native_scale_mode,
            "damping": float(damping),
            "solver": solver or {},
            "c_for_diag": c_for_diag,
            "gradient_pair": g_pair,
        })

    def add_fused(name, fused, *, metadata=None, native_scale_mode="direct"):
        candidates.append({
            "name": name,
            "scope": "qkv",
            "pair": _pair_from_fused(fused, H),
            "fused": fused.float(),
            "native_kind": "none",
            "native_scale_mode": native_scale_mode,
            "damping": 0.0,
            "solver": metadata or {},
            "c_for_diag": None,
            "gradient_pair": g_pair,
        })

    # Newton-Muon and raw-gradient controls still need an optimizer learning
    # rate. Their native protocol therefore uses the common audit base LR.
    add_pair(
        "newton_muon_qk", nm_pair, solver=nm_info,
        native_scale_mode="base_lr",
    )
    add_fused(
        "newton_muon_qkv", nm_fused, metadata=nm_info,
        native_scale_mode="base_lr",
    )
    # RPB and the geometry solvers already return complete proposed parameter
    # displacements, so their native protocol does not multiply by base_lr.
    add_fused(
        "faithful_rpb_qkv", faithful_rpb, metadata=rpb_info,
        native_scale_mode="direct",
    )
    add_fused(
        "feasible_rpb_qkv", feasible_rpb, metadata=rpb_info,
        native_scale_mode="direct",
    )
    add_pair(
        "raw_gradient_qk", pair_scale(g_pair, -1.0),
        native_scale_mode="base_lr_control",
    )

    damping_rels = _parse_float_list(args.attention_audit_damping_rels)
    for damp_rel in damping_rels:
        label = str(damp_rel).replace(".", "p")
        for coeff_name, coeff, native_kind in (
            ("unit", c_unit, "fisher_unit"),
            ("projected", c_projected, "fisher_projected"),
        ):
            damping, rayleigh = estimate_relative_damping(
                geom, g_pair, kind="fisher", c=coeff,
                damping_rel=damp_rel, reduction="mean",
            )
            op = make_quadratic_operator(
                geom, kind="fisher", c=coeff, damping=damping, reduction="mean"
            )
            for iters in (1, 3):
                (sol, info), solve_seconds = _timed_cuda(
                    lambda: pcg_solve(op, pair_scale(g_pair, -1.0), iterations=iters)
                )
                info.update({
                    "damping": damping, "rayleigh": rayleigh,
                    "damping_rel": damp_rel, "solve_seconds": solve_seconds,
                })
                add_pair(
                    f"fisher_cg{iters}_{coeff_name}_dr{label}", sol,
                    native_kind=native_kind, damping=damping, solver=info,
                    c_for_diag=coeff,
                )

        q2_damping, q2_rayleigh = estimate_relative_damping(
            geom, g_pair, kind="q2", c=None,
            damping_rel=damp_rel, reduction="mean",
        )
        q2_op = make_quadratic_operator(
            geom, kind="q2", c=None, damping=q2_damping, reduction="mean"
        )
        (q2_sol, q2_info), q2_seconds = _timed_cuda(
            lambda: pcg_solve(q2_op, pair_scale(g_pair, -1.0), iterations=3)
        )
        q2_info.update({
            "damping": q2_damping, "rayleigh": q2_rayleigh,
            "damping_rel": damp_rel, "solve_seconds": q2_seconds,
        })
        add_pair(
            f"oscillation_q2_unit_cg3_dr{label}", q2_sol,
            native_kind="q2", damping=q2_damping, solver=q2_info,
        )

    # Two-outer-step joint tangent mirror candidates. One outer step at zero is
    # the corresponding Fisher solve; the second tests nonlinear log-partition geometry.
    mirror_rel = damping_rels[0]
    for coeff_name, coeff, native_kind in (
        ("unit", c_unit, "mirror_unit"),
        ("projected", c_projected, "mirror_projected"),
    ):
        damping, rayleigh = estimate_relative_damping(
            geom, g_pair, kind="fisher", c=coeff,
            damping_rel=mirror_rel, reduction="mean",
        )
        (sol, info), mirror_seconds = _timed_cuda(lambda: solve_joint_mirror_newton(
            geom, g_pair, coeff,
            damping=damping,
            eta=1.0,
            newton_iters=int(args.attention_audit_mirror_newton_iters),
            cg_iters=int(args.attention_audit_mirror_cg_iters),
            reduction="mean",
        ))
        info.update({
            "damping": damping, "rayleigh": rayleigh,
            "damping_rel": mirror_rel, "solve_seconds": mirror_seconds,
        })
        add_pair(
            f"mirror_joint_n{args.attention_audit_mirror_newton_iters}_{coeff_name}", sol,
            native_kind=native_kind, damping=damping, solver=info,
            c_for_diag=coeff,
        )

    # v2 evaluates two distinct scale protocols.
    #
    # native:
    #   - Fisher, q2, mirror, and RPB retain their solver/model-determined
    #     displacement magnitude;
    #   - Newton-Muon and raw-gradient controls use the ordinary audit base LR.
    # matched:
    #   - every direction is Frobenius-matched to the corresponding Q/K or
    #     Q/K/V Newton-Muon direction, then multiplied by the common base LR.
    #
    # This separation prevents a norm-matched direction comparison from being
    # mistaken for a faithful test of an optimizer's own local scale.
    ref_norms = {
        "qk": float(nm_qk_fused.norm()),
        "qkv": float(nm_fused.norm()),
    }
    native_scale_grid = _parse_float_list(args.attention_audit_native_scale_grid)
    matched_scale_grid = _parse_float_list(args.attention_audit_matched_scale_grid)
    requested_protocols = _parse_str_list(args.attention_audit_protocols)
    unknown_protocols = sorted(set(requested_protocols) - {"native", "matched"})
    if unknown_protocols:
        raise ValueError(f"Unknown audit protocols: {unknown_protocols}")
    # Preserve user order while removing accidental duplicates.
    requested_protocols = list(dict.fromkeys(requested_protocols))
    protocol_grids = {
        "native": native_scale_grid,
        "matched": matched_scale_grid,
    }

    base_lr = float(args.attention_audit_base_lr)
    weight = attn.c_attn.weight
    original = weight.detach().clone()
    baseline_a = _audit_eval_loss(x_a, y_a)
    baseline_b = _audit_eval_loss(x_b, y_b)
    rows = []
    candidate_json = []

    coeff_b = coeff_info["b"]
    coeff_rho = coeff_info["rho"]

    try:
        for cand in candidates:
            fused = cand["fused"].float()
            cand_norm = float(fused.norm())
            target_norm = ref_norms[cand["scope"]]

            # Sanity-check the candidate's own model at its unscaled native
            # displacement. For a quadratic PCG iterate or a successfully
            # decreased mirror subproblem, the local model value should not
            # exceed its value at zero, i.e. predicted reduction >= 0 up to
            # numerical tolerance.
            native_model = _candidate_native_prediction(
                cand, cand["pair"], geom, c_unit, c_projected,
                float(cand["damping"]),
            )
            if math.isfinite(native_model["predicted_reduction"]):
                sanity_scale = max(
                    1.0,
                    abs(native_model["linear_gain"]),
                    abs(native_model["geometry_energy"]),
                    abs(native_model["regularizer_energy"]),
                )
                sanity_tol = float(args.attention_audit_model_sanity_tol) * sanity_scale
                native_model["sanity_tolerance"] = sanity_tol
                native_model["sanity_pass"] = (
                    native_model["predicted_reduction"] >= -sanity_tol
                )
            else:
                native_model["sanity_tolerance"] = float("nan")
                native_model["sanity_pass"] = None

            protocol_payload = {}
            combined_trials = []

            for protocol in requested_protocols:
                grid = protocol_grids[protocol]

                if protocol == "native":
                    native_mode = cand.get("native_scale_mode", "direct")
                    if native_mode == "direct":
                        scalar_from_candidate = 1.0
                        base_scaling = "candidate_native_displacement"
                    elif native_mode in {"base_lr", "base_lr_control"}:
                        scalar_from_candidate = base_lr
                        base_scaling = (
                            "optimizer_base_lr"
                            if native_mode == "base_lr"
                            else "common_base_lr_control"
                        )
                    else:
                        raise ValueError(
                            f"Unknown native_scale_mode={native_mode!r} "
                            f"for {cand['name']}"
                        )
                    base_update = fused * scalar_from_candidate
                else:
                    if cand_norm <= 1e-30:
                        match_multiplier = 1.0
                    else:
                        match_multiplier = target_norm / cand_norm
                    scalar_from_candidate = match_multiplier * base_lr
                    base_scaling = "frobenius_match_to_newton_muon_then_base_lr"
                    base_update = fused * scalar_from_candidate

                pair_base = _pair_from_fused(base_update, H)
                diag = direction_diagnostics(
                    pair_base, g_pair, geom,
                    c=c_projected,
                    b=coeff_b,
                    rho=coeff_rho,
                    beta=float(args.attention_audit_beta),
                )
                diag.update({
                    "heldout_gradient_dot": float(pair_dot(gb_pair, pair_base)),
                    "full_gradient_dot": float((cap_a["weight_grad"] * base_update).sum()),
                    "heldout_full_gradient_dot": float((cap_b["weight_grad"] * base_update).sum()),
                    "cosine_to_nm_qk": pair_cosine(pair_base, nm_pair),
                    "candidate_fused_norm": cand_norm,
                    "protocol_base_update_norm": float(base_update.norm()),
                    "newton_muon_reference_norm": target_norm,
                    "protocol_base_to_reference_norm": (
                        float(base_update.norm()) / max(target_norm * base_lr, 1e-30)
                    ),
                    "v_update_norm": float(base_update.view(3, D, D)[2].norm()),
                })

                trials = []
                for factor in grid:
                    update = base_update * float(factor)
                    delta_pair = _pair_from_fused(update, H)
                    model = _candidate_native_prediction(
                        cand, delta_pair, geom, c_unit, c_projected,
                        float(cand["damping"]),
                    )
                    U_lin = joint_jvp(delta_pair, geom)
                    U_quad = joint_bilinear_remainder(delta_pair, geom)
                    exact_score = U_lin + U_quad

                    if bool(torch.isfinite(update).all()):
                        weight.data.copy_(original + update.to(original.dtype))
                        loss_a = _audit_eval_loss(x_a, y_a)
                        loss_b = _audit_eval_loss(x_b, y_b)
                    else:
                        loss_a = float("nan")
                        loss_b = float("nan")

                    pred = model["predicted_reduction"]
                    reduction_a = baseline_a - loss_a
                    reduction_b = baseline_b - loss_b
                    grad_dot_a = float((cap_a["weight_grad"] * update).sum())
                    grad_dot_b = float((cap_b["weight_grad"] * update).sum())
                    trial = {
                        "protocol": protocol,
                        "scale_factor": float(factor),
                        "scale_at_boundary": bool(
                            factor == grid[0] or factor == grid[-1]
                        ),
                        "base_scaling": base_scaling,
                        "scalar_from_candidate_before_grid": float(scalar_from_candidate),
                        "applied_scalar_from_candidate": float(scalar_from_candidate * factor),
                        "outer_lr": (
                            base_lr * float(factor)
                            if protocol == "matched"
                            or cand.get("native_scale_mode") in {"base_lr", "base_lr_control"}
                            else float("nan")
                        ),
                        "update_norm": float(update.norm()),
                        "loss_a": loss_a,
                        "loss_b": loss_b,
                        "reduction_a": reduction_a,
                        "reduction_b": reduction_b,
                        "full_gradient_dot_a": grad_dot_a,
                        "full_gradient_dot_b": grad_dot_b,
                        "linear_gain_a": -grad_dot_a,
                        "linear_gain_b": -grad_dot_b,
                        "predicted_reduction": pred,
                        "predicted_positive": bool(math.isfinite(pred) and pred > 0),
                        "model_objective": model["model_objective"],
                        "model_linear_gain": model["linear_gain"],
                        "native_energy": model["geometry_energy"],
                        "regularizer_energy": model["regularizer_energy"],
                        "trust_ratio_a": (
                            reduction_a / pred
                            if math.isfinite(pred) and pred > 0
                            else float("nan")
                        ),
                        "exact_score_osc_max": float(
                            score_oscillation(exact_score, geom.mask).max()
                        ),
                        "bilinear_to_linear_ratio_scaled": float(
                            U_quad.norm() / U_lin.norm().clamp_min(1e-30)
                        ),
                    }
                    trials.append(trial)
                    combined_trials.append(trial)
                    row = {
                        "step": step,
                        "layer": layer_idx,
                        "candidate": cand["name"],
                        "scope": cand["scope"],
                        "protocol": protocol,
                        **{f"dir_{k}": v for k, v in diag.items()},
                        **trial,
                    }
                    rows.append(row)
                    weight.data.copy_(original)

                protocol_payload[protocol] = {
                    "base_scaling": base_scaling,
                    "scalar_from_candidate_before_grid": float(scalar_from_candidate),
                    "base_update_norm": float(base_update.norm()),
                    "scale_grid": [float(value) for value in grid],
                    "direction": diag,
                    "trials": trials,
                }

            # Keep a root-level direction for compatibility with the v1
            # summarizer convention; matched is preferred when present.
            compatibility_protocol = (
                "matched" if "matched" in protocol_payload
                else requested_protocols[0]
            )
            candidate_json.append({
                "name": cand["name"],
                "scope": cand["scope"],
                "native_kind": cand["native_kind"],
                "native_scale_mode": cand.get("native_scale_mode", "direct"),
                "damping": cand["damping"],
                "solver": cand["solver"],
                "native_model": native_model,
                "direction": protocol_payload[compatibility_protocol]["direction"],
                "protocols": protocol_payload,
                "trials": combined_trials,
            })
    finally:
        weight.data.copy_(original)
        raw_model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.time() - audit_t0
    output_dir = Path(args.attention_audit_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"audit_step{step}_layer{layer_idx}_seed{SEED}"
    json_path = output_dir / f"{stem}.json"
    tsv_path = output_dir / f"{stem}.tsv"
    summary_path = output_dir / f"{stem}_summary.txt"

    payload = {
        "audit_version": 2,
        "config": vars(args),
        "protocols_requested": requested_protocols,
        "scale_grids": {
            "native": native_scale_grid,
            "matched": matched_scale_grid,
        },
        "step": step,
        "layer": layer_idx,
        "seed": SEED,
        "baseline_loss_a": baseline_a,
        "baseline_loss_b": baseline_b,
        "capture_loss_a": cap_a["loss"],
        "capture_loss_b": cap_b["loss"],
        "coefficient_info": {k: v for k, v in coeff_info.items() if not torch.is_tensor(v)},
        "newton_muon_info": nm_info,
        "rpb_info": rpb_info,
        "audit_elapsed_seconds": elapsed,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "candidates": candidate_json,
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True))

    fieldnames = sorted({key for row in rows for key in row})
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # v2 produces three distinct selections. Batch B is diagnostic only and
    # never selects a scale.
    actual_records = []
    trust_records = []
    for cand in candidate_json:
        for protocol, pdata in cand["protocols"].items():
            valid = [
                trial for trial in pdata["trials"]
                if math.isfinite(trial["reduction_a"])
                and math.isfinite(trial["reduction_b"])
            ]
            if not valid:
                continue
            best_actual = max(valid, key=lambda item: item["reduction_a"])
            actual_records.append({
                "candidate": cand["name"],
                "scope": cand["scope"],
                "protocol": protocol,
                "trial": best_actual,
            })

            trust_valid = [
                trial for trial in valid
                if math.isfinite(trial["predicted_reduction"])
                and trial["predicted_reduction"] > 0
            ]
            if trust_valid:
                best_trust = max(
                    trust_valid, key=lambda item: item["reduction_a"]
                )
                trust_records.append({
                    "candidate": cand["name"],
                    "scope": cand["scope"],
                    "protocol": protocol,
                    "trial": best_trust,
                })

    actual_by_a = sorted(
        actual_records,
        key=lambda item: item["trial"]["reduction_a"],
        reverse=True,
    )
    actual_by_b = sorted(
        actual_records,
        key=lambda item: item["trial"]["reduction_b"],
        reverse=True,
    )
    trust_by_a = sorted(
        trust_records,
        key=lambda item: item["trial"]["reduction_a"],
        reverse=True,
    )

    def _trial_line(rank, record):
        trial = record["trial"]
        boundary = "yes" if trial["scale_at_boundary"] else "no"
        return (
            f"{rank}\t{record['candidate']}\t{record['scope']}\t"
            f"{record['protocol']}\t{trial['scale_factor']:.6g}\t"
            f"{boundary}\t{trial['reduction_a']:.8g}\t"
            f"{trial['reduction_b']:.8g}\t"
            f"{trial['predicted_reduction']:.8g}\t"
            f"{trial['trust_ratio_a']:.8g}\t"
            f"{trial['update_norm']:.8g}"
        )

    lines = [
        f"Attention geometry audit v2: step={step} layer={layer_idx} seed={SEED}",
        f"baseline_loss_a={baseline_a:.8f} baseline_loss_b={baseline_b:.8f}",
        f"elapsed_seconds={elapsed:.3f} peak_memory_mib={payload['peak_memory_mib']:.1f}",
        f"protocols={','.join(requested_protocols)}",
        f"native_scale_grid={','.join(f'{v:g}' for v in native_scale_grid)}",
        f"matched_scale_grid={','.join(f'{v:g}' for v in matched_scale_grid)}",
        "",
        "NATIVE MODEL SANITY (unscaled candidate displacement)",
        "candidate\tkind\tpredicted_reduction\tmodel_objective\tpass\ttolerance",
    ]
    for cand in candidate_json:
        model = cand["native_model"]
        if not math.isfinite(model["predicted_reduction"]):
            continue
        lines.append(
            f"{cand['name']}\t{model['kind']}\t"
            f"{model['predicted_reduction']:.8g}\t"
            f"{model['model_objective']:.8g}\t"
            f"{model['sanity_pass']}\t"
            f"{model['sanity_tolerance']:.8g}"
        )

    table_header = (
        "rank\tcandidate\tscope\tprotocol\tscale\tboundary\t"
        "reduction_a\treduction_b\tpred\ttrust_a\tupdate_norm"
    )
    lines.extend([
        "",
        "BEST ACTUAL BATCH-A TRIAL (batch B reported at the A-selected scale)",
        table_header,
    ])
    for rank, record in enumerate(actual_by_a, 1):
        lines.append(_trial_line(rank, record))

    lines.extend([
        "",
        "HELD-OUT B TRANSFER AT THE BATCH-A-SELECTED SCALE",
        table_header,
    ])
    for rank, record in enumerate(actual_by_b, 1):
        lines.append(_trial_line(rank, record))

    lines.extend([
        "",
        "BEST TRIAL WITH POSITIVE LOCAL-MODEL PREDICTION",
        table_header,
    ])
    if trust_by_a:
        for rank, record in enumerate(trust_by_a, 1):
            lines.append(_trial_line(rank, record))
    else:
        lines.append("NO POSITIVE-PREDICTION TRIALS")

    summary = "\n".join(lines) + "\n"
    summary_path.write_text(summary)
    print(summary, end="")
    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {tsv_path}")
    print(f"[audit] wrote {summary_path}")
    return payload



def _capture_fisher_curvature_samples(
    xb: Tensor,
    yb: Tensor,
    *,
    precision: str,
    time_it: bool,
):
    """Capture a small all-layer curvature minibatch without touching .grad fields.

    The ordinary full-minibatch parameter gradients have already been accumulated.
    This extra forward plus autograd.grad computes only the downstream gradients of
    each attention output.  The resulting sample is a stochastic curvature object;
    it does not replace the full-gradient right-hand side.
    """
    precision = str(precision).lower()
    if precision not in {"bf16", "fp32"}:
        raise ValueError(f"Unknown FISHER_CURV_PRECISION={precision!r}")

    was_training = raw_model.training
    raw_model.train()
    attentions = [block.attn for block in raw_model.transformer.h]
    for attn in attentions:
        attn.audit_capture = True
        attn.audit_cache = {}

    if time_it:
        torch.cuda.synchronize()
        start = time.time()
    else:
        start = 0.0

    try:
        if precision == "fp32":
            amp_context = torch.amp.autocast(device_type="cuda", enabled=False)
        else:
            amp_context = torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16
            )
        with torch.enable_grad(), amp_context:
            _, curv_loss = raw_model(
                xb, yb, return_logits=False, precond_flag=False
            )
        outs = [attn.audit_cache["out"] for attn in attentions]
        g_outs = torch.autograd.grad(
            curv_loss,
            outs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        samples = []
        for attn, g_out in zip(attentions, g_outs):
            cache = attn.audit_cache
            required = ["x", "q", "k", "cos", "sin", "v_pre"]
            missing = [name for name in required if name not in cache]
            if missing:
                raise RuntimeError(
                    f"Fisher curvature capture missing tensors: {missing}"
                )
            samples.append({
                "x": cache["x"].detach().clone(),
                "q": cache["q"].detach().clone(),
                "k": cache["k"].detach().clone(),
                "cos": cache["cos"].detach().clone(),
                "sin": cache["sin"].detach().clone(),
                "v_pre": cache["v_pre"].detach().clone(),
                "g_out": g_out.detach().clone(),
                "w_o": attn.c_proj.weight.detach().float().clone(),
            })
    finally:
        for attn in attentions:
            attn.audit_capture = False
            attn.audit_cache = {}
        if not was_training:
            raw_model.eval()

    if time_it:
        torch.cuda.synchronize()
        capture_seconds = time.time() - start
    else:
        capture_seconds = 0.0
    return samples, float(curv_loss.detach()), capture_seconds

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

    if master_process and (args.save_every > 0 and (last_step or step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        ckpt.save(step, raw_model, optimizers, extra=dict(code=code))
        torch.cuda.synchronize()
        t0 = time.time()

    if args.attention_audit_step >= 0 and step == args.attention_audit_step:
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        run_attention_geometry_audit(step)
        torch.cuda.synchronize()
        t0 = time.time()
        if bool(args.attention_audit_exit_after):
            break

    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    optimizer2.global_step = step
    optimizer3.global_step = step
    precond_flag = False
    if hasattr(optimizer2, "precond_flag_for_step"):
        precond_flag = bool(optimizer2.precond_flag_for_step(step))
    if hasattr(optimizer3, "precond_flag_for_step"):
        precond_flag = bool(precond_flag or optimizer3.precond_flag_for_step(step))

    fisher_capture_needed = (
        system_mode == "fisher"
        and optimizer3.needs_curvature_capture(step)
    )
    fisher_x = fisher_y = None

    for micro_step in range(train_accumulation_steps):
        if fisher_capture_needed and micro_step == 0:
            if args.fisher_curv_batch < 1 or args.fisher_curv_batch > x.shape[0]:
                raise ValueError(
                    f"FISHER_CURV_BATCH={args.fisher_curv_batch} must lie in "
                    f"[1,{x.shape[0]}]"
                )
            fisher_x = x[:args.fisher_curv_batch].detach().clone()
            fisher_y = y[:args.fisher_curv_batch].detach().clone()

        with ctx:
            _, loss = model(x, y, return_logits=False, precond_flag=precond_flag)
            train_loss = loss.detach()
            loss = loss / train_accumulation_steps
        x, y = train_loader.next_batch()
        loss.backward()

    if fisher_capture_needed:
        samples, fisher_curv_loss, fisher_capture_seconds = (
            _capture_fisher_curvature_samples(
                fisher_x,
                fisher_y,
                precision=args.fisher_curv_precision,
                time_it=optimizer3.should_time(step),
            )
        )
        optimizer3.set_curvature_samples(
            samples, capture_seconds=fisher_capture_seconds
        )

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
        row_metrics = rpb_weight_row_metrics(qkv_weights)
        row_metrics["weight_row_norm/rpb_max_growth"] = (
            row_metrics["weight_row_norm/rpb_max"] / max(rpb_initial_row_max, 1e-12)
        )
        row_metrics["weight_row_norm/rpb_mean_growth"] = (
            row_metrics["weight_row_norm/rpb_mean"] / max(rpb_initial_row_mean, 1e-12)
        )
        diag_metrics.update(row_metrics)
        for name, opt in zip(opt_names, optimizers):
            diag_metrics[f"lr/{name}"] = opt.param_groups[0]["lr"]
        diag_metrics.update(getattr(optimizer3, "last_diag", {}))
        diag_metrics["cycle_a/system_code"] = float({"adamw": 0, "muon": 1, "newton_muon": 2, "fisher": 3}[system_mode])
        diag_metrics["cycle_a/backbone_code"] = float({"adamw": 0, "muon": 1, "newton_muon": 2}[backbone_mode])
        diag_metrics["cycle_a/v_code"] = float({"adamw": 0, "muon": 1, "newton_muon": 2}[v_mode])
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
