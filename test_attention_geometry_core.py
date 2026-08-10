from __future__ import annotations
import math
import torch

from attention_geometry_core import (
    build_geometry,
    joint_jvp,
    joint_vjp,
    fisher_energy,
    unit_coefficients,
    make_quadratic_operator,
    pair_dot,
    pcg_solve,
    pair_scale,
    solve_joint_mirror_newton,
    q2_energy,
)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, H, dh, D = 2, 7, 3, 4, 12
    x = torch.randn(B, T, D, device=device)
    q_pre = torch.randn(B, T, H, dh, device=device)
    k_pre = torch.randn(B, T, H, dh, device=device)
    theta = torch.randn(T, dh // 2, device=device)
    cos = theta.cos()[None, :, None, :]
    sin = theta.sin()[None, :, None, :]
    from attention_geometry_core import apply_rope_bthd
    q = apply_rope_bthd(q_pre, cos, sin).transpose(1, 2)
    k = apply_rope_bthd(k_pre, cos, sin).transpose(1, 2)
    geom = build_geometry(x, q, k, cos, sin)

    A = torch.randn(H, dh, D, device=device)
    K = torch.randn(H, dh, D, device=device)
    U = torch.randn(B, H, T, T, device=device)
    U = U * geom.mask.view(1, 1, T, T)
    JA = joint_jvp((A, K), geom)
    adj = joint_vjp(U, geom)
    lhs = (JA * U).sum()
    rhs = pair_dot((A, K), adj)
    rel = float((lhs - rhs).abs() / lhs.abs().clamp_min(1e-8))
    print('adjoint relative error:', rel)
    assert rel < 5e-5

    c = unit_coefficients(geom)
    op = make_quadratic_operator(geom, kind='fisher', c=c, damping=1.0)
    x1 = (torch.randn_like(A), torch.randn_like(K))
    x2 = (torch.randn_like(A), torch.randn_like(K))
    Hx1, Hx2 = op(x1), op(x2)
    sym = float((pair_dot(x1, Hx2) - pair_dot(Hx1, x2)).abs() / pair_dot(x1, Hx2).abs().clamp_min(1e-8))
    print('operator symmetry relative error:', sym)
    assert sym < 1e-4
    assert float(pair_dot(x1, Hx1)) > 0

    g = (torch.randn_like(A), torch.randn_like(K))
    sol, info = pcg_solve(op, pair_scale(g, -1), iterations=5)
    print('pcg residuals:', info['relative_residuals'])
    assert info['relative_residuals'][-1] < info['relative_residuals'][0]

    msol, minfo = solve_joint_mirror_newton(
        geom, g, c, damping=1.0, newton_iters=2, cg_iters=3
    )
    print('mirror objective:', minfo['objective'])
    assert math.isfinite(minfo['objective'])
    assert minfo['objective'] <= 1e-5

    u = joint_jvp(msol, geom)
    print('fisher energy:', float(fisher_energy(u, geom.p, c)))
    print('q2 energy:', float(q2_energy(u, geom.mask)))
    print('PASS')


if __name__ == '__main__':
    main()
