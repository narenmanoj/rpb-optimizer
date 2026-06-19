# Smoothness of a Single Attention Head and the Resulting Row-Norm Update

This note derives a local smoothness bound for one self-attention head in the **row norm** used in the proposed row-normalized activation-space update. It then derives the update rule and gives a correct one-dimensional procedure for choosing the radius $r^{\star}$ at every step.

The key point is that the smoothness constant is **not static**. It depends on the current activations and, because attention contains bilinear $QK^\top$ scores and a multiplication by $V$, it also depends on the radius of the step we are considering. The correct update therefore uses a time-dependent, radius-dependent constant $C_t(r)$, not a fixed $\lambda$.

---

## 1. Setup and notation

Let

$$
Z \in \mathbb{R}^{N \times d_{\mathrm{model}}}
$$

be the matrix of input token representations for the current batch and sequence positions, flattened into $N$ token rows. For one attention head of width $d_h$, define

$$
Q = ZW_Q^\top, \qquad K = ZW_K^\top, \qquad V = ZW_V^\top,
$$

where

$$
W_Q,W_K,W_V \in \mathbb{R}^{d_h \times d_{\mathrm{model}}}.
$$

The attention scores, probabilities, and output are

$$
S = \frac{QK^\top}{\sqrt{d_h}}, \qquad
P = \operatorname{softmax}_{\mathrm{row}}(S), \qquad
Y = PV.
$$

We study the attention map

$$
F(Q,K,V) = \operatorname{softmax}_{\mathrm{row}}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

For a matrix $A$, define the row norm

$$
\|A\|_{\infty,2} := \max_i \|A_{i:}\|_2.
$$

For a block perturbation $\Delta = (\Delta Q,\Delta K,\Delta V)$, define

$$
\|\Delta\|_{\mathcal A}
:=
\max\bigl\{
\|\Delta Q\|_{\infty,2},
\|\Delta K\|_{\infty,2},
\|\Delta V\|_{\infty,2}
\bigr\}.
$$

This is the activation-space norm induced by the current input $Z$. For weight perturbations, define

$$
\|\Delta W\|_{Z,\mathcal A}
:=
\max\bigl\{
\|Z\Delta W_Q^\top\|_{\infty,2},
\|Z\Delta W_K^\top\|_{\infty,2},
\|Z\Delta W_V^\top\|_{\infty,2}
\bigr\}.
$$

Let the current activation row norms be

$$
q := \|Q\|_{\infty,2}, \qquad
k := \|K\|_{\infty,2}, \qquad
v := \|V\|_{\infty,2}.
$$

For a radius $r \ge 0$, any point within the activation-space ball

$$
\|\Delta\|_{\mathcal A} \le r
$$

has row-norm bounds

$$
\|Q+\Delta Q\|_{\infty,2} \le q+r,
$$

$$
\|K+\Delta K\|_{\infty,2} \le k+r,
$$

$$
\|V+\Delta V\|_{\infty,2} \le v+r.
$$

It is useful to define

$$
q_r := q+r, \qquad k_r := k+r, \qquad v_r := v+r,
$$

and

$$
a_r := \frac{q_r+k_r}{\sqrt{d_h}}
= \frac{q+k+2r}{\sqrt{d_h}}.
$$

---

## 2. Softmax differential bounds

The row-softmax acts independently on each row of $S$. For one row, let

$$
p = \operatorname{softmax}(s) \in \mathbb{R}^N.
$$

The first derivative of softmax in direction $x$ is

$$
D\sigma_s[x]
= J_p x,
$$

where

$$
J_p = \operatorname{diag}(p)-pp^\top.
$$

Equivalently,

$$
\bigl(D\sigma_s[x]\bigr)_i
= p_i\left(x_i-\sum_j p_jx_j\right).
$$

For $\|x\|_\infty \le 1$, the random variable $x_i$ lies in an interval of length at most $2$. Therefore its mean absolute deviation under $p$ is at most $1$, giving

$$
\|D\sigma_s[x]\|_1 \le \|x\|_\infty.
$$

Thus

$$
\boxed{
\|D\sigma_s[x]\|_1 \le \|x\|_\infty.
}
$$

For the second derivative, write

$$
\mu_x := \sum_i p_ix_i, \qquad
\mu_y := \sum_i p_iy_i.
$$

Then

$$
\bigl(D^2\sigma_s[x,y]\bigr)_i
=
p_i
\left[
(x_i-\mu_x)(y_i-\mu_y)
-
\sum_j p_j(x_j-\mu_x)(y_j-\mu_y)
\right].
$$

Using

$$
|x_i-\mu_x| \le 2\|x\|_\infty,
\qquad
|y_i-\mu_y| \le 2\|y\|_\infty,
$$

we get

$$
\sum_i p_i |(x_i-\mu_x)(y_i-\mu_y)|
\le 4\|x\|_\infty\|y\|_\infty.
$$

Also,

$$
\left|
\sum_j p_j(x_j-\mu_x)(y_j-\mu_y)
\right|
\le
4\|x\|_\infty\|y\|_\infty.
$$

Therefore

$$
\boxed{
\|D^2\sigma_s[x,y]\|_1
\le
8\|x\|_\infty\|y\|_\infty.
}
$$

The constant $8$ is conservative but valid. The derivation below keeps this constant explicit as

$$
h_\sigma := 8.
$$

If one has a sharper certified softmax-Hessian constant for the same $\ell_\infty \to \ell_1$ operator norm, it can be substituted everywhere by replacing $h_\sigma$.

---

## 3. Score-map derivative bounds

Define the score map

$$
B(Q,K) := \frac{QK^\top}{\sqrt{d_h}}.
$$

At a point $(\bar Q,\bar K)$, the first differential in direction $(E_Q,E_K)$ is

$$
DB_{(\bar Q,\bar K)}[E_Q,E_K]
=
\frac{E_Q\bar K^\top + \bar Q E_K^\top}{\sqrt{d_h}}.
$$

For every score entry,

$$
\left|
\frac{(E_Q)_{i:}\bar K_{j:}^\top}{\sqrt{d_h}}
\right|
\le
\frac{\|E_Q\|_{\infty,2}\|\bar K\|_{\infty,2}}{\sqrt{d_h}},
$$

and similarly for the $\bar Q E_K^\top$ term. Hence

$$
\|DB_{(\bar Q,\bar K)}[E_Q,E_K]\|_{\infty,\infty}
\le
\frac{\|\bar K\|_{\infty,2}\|E_Q\|_{\infty,2}
+
\|\bar Q\|_{\infty,2}\|E_K\|_{\infty,2}}
{\sqrt{d_h}}.
$$

Inside the radius-$r$ ball,

$$
\|\bar Q\|_{\infty,2} \le q_r, \qquad
\|\bar K\|_{\infty,2} \le k_r.
$$

If

$$
\|E\|_{\mathcal A}
=
\max\{\|E_Q\|_{\infty,2},\|E_K\|_{\infty,2},\|E_V\|_{\infty,2}\},
$$

then

$$
\boxed{
\|DB_{(\bar Q,\bar K)}[E_Q,E_K]\|_{\infty,\infty}
\le
\frac{q_r+k_r}{\sqrt{d_h}}
\|E\|_{\mathcal A}
=
a_r\|E\|_{\mathcal A}.
}
$$

The score map is bilinear, so its second differential is

$$
D^2B[(E_Q,E_K),(H_Q,H_K)]
=
\frac{E_QH_K^\top+H_QE_K^\top}{\sqrt{d_h}}.
$$

Thus

$$
\boxed{
\|D^2B[E,H]\|_{\infty,\infty}
\le
\frac{2}{\sqrt{d_h}}
\|E\|_{\mathcal A}\|H\|_{\mathcal A}.
}
$$

---

## 4. Attention Hessian bound

Recall

$$
F(Q,K,V) = P(Q,K)V,
\qquad
P(Q,K) = \operatorname{softmax}_{\mathrm{row}}(B(Q,K)).
$$

Let $\theta=(Q,K,V)$. At an intermediate point $\bar\theta=(\bar Q,\bar K,\bar V)$ inside the radius-$r$ ball, let

$$
\bar P = \operatorname{softmax}_{\mathrm{row}}\!\left(\frac{\bar Q\bar K^\top}{\sqrt{d_h}}\right).
$$

For directions $E=(E_Q,E_K,E_V)$ and $H=(H_Q,H_K,H_V)$, the second differential of $F$ is

$$
D^2F_{\bar\theta}[E,H]
=
D^2P_{(\bar Q,\bar K)}[E,H] \bar V
+
DP_{(\bar Q,\bar K)}[D^2B[E,H]]\bar V
+
DP_{(\bar Q,\bar K)}[DB[E]]H_V
+
DP_{(\bar Q,\bar K)}[DB[H]]E_V.
$$

More explicitly, the first term comes from the softmax Hessian applied to the two first-order score perturbations. The second term comes from the bilinear second derivative of $QK^\top$. The last two terms come from differentiating the final multiplication by $V$.

We now bound each term in $\|\cdot\|_{\infty,2}$.

### 4.1. The softmax-Hessian term

For each row, the second derivative of softmax has $\ell_1$ norm at most

$$
h_\sigma
\|DB[E]\|_{\infty}
\|DB[H]\|_{\infty}.
$$

Multiplying by $\bar V$, whose rows have norm at most $v_r$, gives

$$
\|D^2P[DB[E],DB[H]]\bar V\|_{\infty,2}
\le
h_\sigma v_r
\|DB[E]\|_{\infty,\infty}
\|DB[H]\|_{\infty,\infty}.
$$

Using the score derivative bound,

$$
\boxed{
\|D^2P[DB[E],DB[H]]\bar V\|_{\infty,2}
\le
h_\sigma v_r a_r^2
\|E\|_{\mathcal A}\|H\|_{\mathcal A}.
}
$$

### 4.2. The score-Hessian term

Using the first softmax derivative bound and the row norm of $\bar V$,

$$
\|DP[D^2B[E,H]]\bar V\|_{\infty,2}
\le
v_r\|D^2B[E,H]\|_{\infty,\infty}.
$$

Therefore

$$
\boxed{
\|DP[D^2B[E,H]]\bar V\|_{\infty,2}
\le
\frac{2v_r}{\sqrt{d_h}}
\|E\|_{\mathcal A}\|H\|_{\mathcal A}.
}
$$

### 4.3. The two value-cross terms

For the first value-cross term,

$$
\|DP[DB[E]]H_V\|_{\infty,2}
\le
\|DB[E]\|_{\infty,\infty}\|H_V\|_{\infty,2}
\le
a_r\|E\|_{\mathcal A}\|H\|_{\mathcal A}.
$$

Similarly,

$$
\|DP[DB[H]]E_V\|_{\infty,2}
\le
a_r\|H\|_{\mathcal A}\|E\|_{\mathcal A}.
$$

Hence

$$
\boxed{
\|DP[DB[E]]H_V + DP[DB[H]]E_V\|_{\infty,2}
\le
2a_r\|E\|_{\mathcal A}\|H\|_{\mathcal A}.
}
$$

---

## 5. The local smoothness constant

Combining the three pieces gives the Hessian operator bound

$$
\boxed{
\|D^2F_{\bar\theta}[E,H]\|_{\infty,2}
\le
C_t(r)
\|E\|_{\mathcal A}\|H\|_{\mathcal A},
}
$$

where

$$
\boxed{
C_t(r)
=
h_\sigma v_r a_r^2
+
\frac{2v_r}{\sqrt{d_h}}
+
2a_r.
}
$$

Substituting

$$
v_r=v+r,
\qquad
 a_r = \frac{q+k+2r}{\sqrt{d_h}},
$$

we obtain the explicit expression

$$
\boxed{
C_t(r)
=
 h_\sigma (v+r)
\frac{(q+k+2r)^2}{d_h}
+
\frac{2(v+r)}{\sqrt{d_h}}
+
\frac{2(q+k+2r)}{\sqrt{d_h}}.
}
$$

With the conservative softmax-Hessian bound above, $h_\sigma=8$.

This is the main smoothness bound. It is time-dependent because $q,k,v$ are functions of the current iterate and the current batch. It is radius-dependent because a step of size $r$ changes the norms of $Q,K,V$, which changes the curvature bound.

---

## 6. Taylor remainder bound

Let $\Delta=(\Delta Q,\Delta K,\Delta V)$ satisfy

$$
\|\Delta\|_{\mathcal A} \le r.
$$

By Taylor's theorem with integral remainder,

$$
F(\theta+\Delta)
=
F(\theta)
+
DF_\theta[\Delta]
+
\int_0^1(1-s)
D^2F_{\theta+s\Delta}[\Delta,\Delta] \, ds.
$$

For every $s\in[0,1]$, the point $\theta+s\Delta$ lies inside the radius-$r$ ball. Therefore

$$
\|D^2F_{\theta+s\Delta}[\Delta,\Delta]\|_{\infty,2}
\le
C_t(r)\|\Delta\|_{\mathcal A}^2.
$$

Thus

$$
\boxed{
\|F(\theta+\Delta)-F(\theta)-DF_\theta[\Delta]\|_{\infty,2}
\le
\frac{1}{2}C_t(r)\|\Delta\|_{\mathcal A}^2.
}
$$

In weight space, using

$$
\Delta Q = Z\Delta W_Q^\top,
\qquad
\Delta K = Z\Delta W_K^\top,
\qquad
\Delta V = Z\Delta W_V^\top,
$$

this becomes

$$
\boxed{
\|F(W+\Delta W)-F(W)-DF_W[\Delta W]\|_{\infty,2}
\le
\frac{1}{2}C_t(r)\|\Delta W\|_{Z,\mathcal A}^2,
}
$$

provided

$$
\|\Delta W\|_{Z,\mathcal A}\le r.
$$

---

## 7. From the smoothness bound to an update rule

Let $\mathcal L$ be the scalar loss. At the current point, let

$$
G_Q := \nabla_Q\mathcal L,
\qquad
G_K := \nabla_K\mathcal L,
\qquad
G_V := \nabla_V\mathcal L.
$$

These are the activation-space gradients with respect to the current $Q,K,V$. They are exactly the quantities obtained by backpropagation if $Q,K,V$ are retained as differentiable intermediate tensors.

For any activation-space perturbation $\Delta=(\Delta Q,\Delta K,\Delta V)$, the first-order loss change is

$$
\langle G_Q,\Delta Q\rangle
+
\langle G_K,\Delta K\rangle
+
\langle G_V,\Delta V\rangle.
$$

The dual norm of

$$
\|\Delta\|_{\mathcal A}
=\max\{\|\Delta Q\|_{\infty,2},\|\Delta K\|_{\infty,2},\|\Delta V\|_{\infty,2}\}
$$

is

$$
\|G\|_{\mathcal A,*}
=
\|G_Q\|_{1,2}+
\|G_K\|_{1,2}+
\|G_V\|_{1,2},
$$

where

$$
\|G_Q\|_{1,2} := \sum_i \|(G_Q)_{i:}\|_2,
$$

and similarly for $G_K,G_V$.

Define

$$
S_G := \|G\|_{\mathcal A,*}
=
\|G_Q\|_{1,2}+
\|G_K\|_{1,2}+
\|G_V\|_{1,2}.
$$

Also define the row-sign operator

$$
\operatorname{rsgn}(A)_{i:}
:=
\begin{cases}
A_{i:}/\|A_{i:}\|_2, & \|A_{i:}\|_2>0,\\
0, & \|A_{i:}\|_2=0.
\end{cases}
$$

For a fixed radius $r$, the perturbation that minimizes the linear term subject to $\|\Delta\|_{\mathcal A}\le r$ is

$$
\boxed{
\Delta Q = -r\operatorname{rsgn}(G_Q),
\qquad
\Delta K = -r\operatorname{rsgn}(G_K),
\qquad
\Delta V = -r\operatorname{rsgn}(G_V).
}
$$

Indeed,

$$
\langle G_Q,-r\operatorname{rsgn}(G_Q)\rangle
=
-r\|G_Q\|_{1,2},
$$

and likewise for $K,V$, so the total linear decrease is

$$
-rS_G.
$$

The constant $C_t(r)$ derived in §5 bounds the curvature of the attention **output
map** $F$, not of the scalar loss $\mathcal L$. Before forming a descent model for
$\mathcal L$ we must convert head curvature into loss curvature; skipping this step is
a units error — the linear term $-rS_G$ has units of loss, whereas $\tfrac12 C_t(r)r^2$
has units of output activation, so the two cannot be added.

Write $\mathcal L=\ell(Y)$ with $Y=F(\theta)$. By the chain rule, the second-order term
of the loss along $\Delta$ is

$$
D^2_\theta\mathcal L[\Delta,\Delta]
=
\underbrace{\langle \nabla_Y\mathcal L,\; D^2F_{\bar\theta}[\Delta,\Delta]\rangle}_{\text{head curvature}}
+
\underbrace{D^2_Y\ell\bigl[DF[\Delta],\,DF[\Delta]\bigr]}_{\text{downstream curvature}} .
$$

**Head-curvature term.** Pair the $\|\cdot\|_{\infty,2}$ Hessian bound of §5 with the
dual $\|\cdot\|_{1,2}$ norm of the output gradient. Define the output dual gradient norm

$$
\boxed{
g_Y := \|\nabla_Y\mathcal L\|_{1,2} = \sum_i \|(\nabla_Y\mathcal L)_{i:}\|_2 ,
}
$$

the analogue of $S_G$ one space downstream (the gradient with respect to the head
output $Y=PV$). By Hölder for the $\|\cdot\|_{1,2}/\|\cdot\|_{\infty,2}$ pairing and the
bound of §5,

$$
|\langle \nabla_Y\mathcal L,\, D^2F[\Delta,\Delta]\rangle|
\le
\|\nabla_Y\mathcal L\|_{1,2}\,\|D^2F[\Delta,\Delta]\|_{\infty,2}
\le
g_Y\,C_t(r)\,\|\Delta\|_{\mathcal A}^2 .
$$

**Downstream-curvature term.** Let $J_t(r)$ bound the head Jacobian,
$\|DF_{\bar\theta}[\Delta]\|_{\infty,2}\le J_t(r)\,\|\Delta\|_{\mathcal A}$. From
$DF[\Delta]=DP[DB[\Delta]]\bar V+\bar P\,\Delta V$, the first-derivative bounds of §2–3
and the row-stochasticity of $\bar P$ (so $\|\bar P\,\Delta V\|_{\infty,2}\le\|\Delta V\|_{\infty,2}$) give

$$
\boxed{
J_t(r) = v_r a_r + 1 = (v+r)\frac{q+k+2r}{\sqrt{d_h}} + 1 .
}
$$

If the rest of the network and loss are $L_Y$-smooth at $Y$ in the same row-norm
geometry, $|D^2_Y\ell[u,u]|\le L_Y\|u\|_{\infty,2}^2$, the downstream term is at most
$L_Y\,J_t(r)^2\,\|\Delta\|_{\mathcal A}^2$.

**Loss smoothness constant.** Collecting both pieces gives the activation-space
smoothness constant of the **loss**,

$$
\boxed{
\Lambda_t(r) := g_Y\,C_t(r) + L_Y\,J_t(r)^2 ,
}
$$

which has units $[\text{loss}]/[\text{activation}]^2$, consistent with $S_G$
($[\text{loss}]/[\text{activation}]$). $L_Y$ is the only quantity not fixed by the
head-local analysis; it depends on the downstream network. Setting $L_Y=0$ gives the
**head-local model**, used by default below: $\Lambda_t(r)=g_Y\,C_t(r)$.

The smoothness model for the loss along this row-normalized direction is therefore

$$
\boxed{
\Phi_t(r)
:=
-rS_G
+
\frac{1}{2}\,\Lambda_t(r)\,r^2 ,
}
$$

and in the head-local model

$$
\boxed{
\Phi_t(r) = -rS_G + \tfrac12\, g_Y\, C_t(r)\, r^2 .
}
$$

Dividing the stationarity condition by $g_Y>0$ shows the radius depends only on the
**dimensionless ratio** $S_G/g_Y$, not on $S_G$ alone. The original derivation
implicitly set $g_Y=1$; that is the source of the units mismatch and of an $r^\star$
mis-scaled by the magnitude of the output gradient.

The proposed activation-space update is

$$
\boxed{
\Delta Q^\star = -r^\star\operatorname{rsgn}(G_Q),
\qquad
\Delta K^\star = -r^\star\operatorname{rsgn}(G_K),
\qquad
\Delta V^\star = -r^\star\operatorname{rsgn}(G_V),
}
$$

where

$$
\boxed{
 r^\star = \arg\min_{r\ge 0}\Phi_t(r).
}
$$

A damped practical version uses

$$
 r_{\mathrm{step}} = \eta r^\star,
 \qquad 0<\eta\le 1,
$$

and then applies the same formula with $r_{\mathrm{step}}$. The scalar $\eta$ is a conventional learning-rate or trust-region damping factor; the curvature-aware radius is $r^\star$.

---

## 8. Correct search for $r^\star$

Because $C_t(r)$ depends on $r$, the tempting closed form

$$
r = \frac{S_G}{g_Y\,C_t(0)}
$$

is generally too aggressive. It treats curvature as constant even though the attention curvature bound increases with the candidate step radius.

The correct scalar objective (head-local model, $L_Y=0$) is

$$
\Phi_t(r) = -S_Gr + \frac{1}{2}\,g_Y\,C_t(r)\,r^2 .
$$

Let

$$
b := q+k,
\qquad
s(r) := b+2r.
$$

Then

$$
C_t(r)
=
 h_\sigma (v+r)\frac{s(r)^2}{d_h}
+
\frac{2(v+r)}{\sqrt{d_h}}
+
\frac{2s(r)}{\sqrt{d_h}},
$$

with derivative

$$
\boxed{
C_t'(r)
=
\frac{h_\sigma}{d_h}
\left[s(r)^2+4(v+r)s(r)\right]
+
\frac{6}{\sqrt{d_h}}.
}
$$

Therefore

$$
\boxed{
\Phi_t'(r)
=
-S_G
+g_Y\!\left[rC_t(r)
+
\frac{1}{2}r^2C_t'(r)\right].
}
$$

The optimality condition is

$$
\boxed{
-S_G
+g_Y\!\left[r^\star C_t(r^\star)
+
\frac{1}{2}(r^\star)^2C_t'(r^\star)\right]
=0,
}
$$

equivalently $\;r^\star C_t(r^\star)+\tfrac12 (r^\star)^2 C_t'(r^\star) = S_G/g_Y$. This is the equation that should be solved; it is not $rC_t(r)=S_G/g_Y$, because the derivative of $C_t(r)$ contributes an additional term. (For $L_Y>0$, replace $g_Y C_t$ by $\Lambda_t$ and $g_Y C_t'$ by $\Lambda_t'$ throughout.)

### 8.1. Existence and uniqueness

If $S_G=0$ or $g_Y=0$, then $r^\star=0$ (with $g_Y=0$ the loss does not depend on this head's output to first order, so no step is taken).

Assume $S_G>0$ and $g_Y>0$. We have

$$
\Phi_t'(0)=-S_G<0.
$$

Also, since $C_t(r)$ grows at least linearly and in fact cubically in $r$,

$$
\Phi_t'(r)\to +\infty
\qquad\text{as}\qquad r\to\infty.
$$

Moreover, $C_t(r)$, $C_t'(r)$, and $C_t''(r)$ are nonnegative for $r\ge 0$ and $g_Y>0$, so

$$
\Phi_t''(r)
=
g_Y\!\left[C_t(r)+2rC_t'(r)+\frac{1}{2}r^2C_t''(r)\right]>0.
$$

Thus $\Phi_t$ is strictly convex on $[0,\infty)$, and the root of $\Phi_t'(r)=0$ is unique.

### 8.2. Robust bisection search

Because the root is unique, bisection gives a globally correct search.

```python
def attention_radius_star(q, k, v, d_h, S_G, g_Y=1.0, h_sigma=8.0, r_max=None, tol=1e-8, max_iter=100):
    """
    Compute r* for the row-norm attention update by minimizing
        Phi(r) = -S_G*r + 0.5*g_Y*C(r)*r**2     (head-local model, L_Y = 0).

    q, k, v: current row norms ||Q||_{inf,2}, ||K||_{inf,2}, ||V||_{inf,2}
    d_h: head dimension
    S_G: dual gradient norm ||G_Q||_{1,2}+||G_K||_{1,2}+||G_V||_{1,2}
    g_Y: output dual gradient norm ||grad_Y L||_{1,2} (gradient w.r.t. the head output
         Y = P V). Converts the head-map curvature C(r) into a loss curvature; the
         radius depends on the ratio S_G/g_Y. g_Y=1 recovers the original (mis-scaled)
         formula.
    h_sigma: certified softmax Hessian constant; conservative default is 8
    r_max: optional trust-region cap
    """
    import math

    if S_G <= 0 or g_Y <= 0:
        return 0.0

    sqrt_d = math.sqrt(d_h)

    def C(r):
        s = q + k + 2.0*r
        return (
            h_sigma * (v + r) * (s * s) / d_h
            + 2.0 * (v + r) / sqrt_d
            + 2.0 * s / sqrt_d
        )

    def C_prime(r):
        s = q + k + 2.0*r
        return (
            h_sigma * (s * s + 4.0 * (v + r) * s) / d_h
            + 6.0 / sqrt_d
        )

    def phi_prime(r):
        return -S_G + g_Y * (r * C(r) + 0.5 * r * r * C_prime(r))

    lo = 0.0

    if r_max is not None:
        hi = float(r_max)
        # If Phi is still decreasing at r_max, the constrained minimizer is r_max.
        if phi_prime(hi) <= 0.0:
            return hi
    else:
        hi = 1.0
        while phi_prime(hi) < 0.0:
            hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = phi_prime(mid)
        if abs(val) <= tol:
            return mid
        if val < 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)
```

This computes the exact minimizer of the certified scalar upper model up to the requested numerical tolerance.

### 8.3. Safeguarded Newton variant

For speed, one can use Newton's method with bisection fallback. The second derivative is

$$
\Phi_t''(r)
=
g_Y\!\left[C_t(r)+2rC_t'(r)+\frac{1}{2}r^2C_t''(r)\right],
$$

where

$$
\boxed{
C_t''(r)
=
\frac{8h_\sigma}{d_h}\bigl(s(r)+v+r\bigr).
}
$$

A Newton proposal is

$$
r_{\mathrm{new}}
=
r-
\frac{\Phi_t'(r)}{\Phi_t''(r)}.
$$

If this proposal leaves the current bracket, use the bisection midpoint instead. This preserves the correctness of the search while often converging faster.

---

## 9. Pulling the activation update back to weights

The activation update specifies target changes

$$
T_Q := -r^\star\operatorname{rsgn}(G_Q),
\qquad
T_K := -r^\star\operatorname{rsgn}(G_K),
\qquad
T_V := -r^\star\operatorname{rsgn}(G_V).
$$

We need weight perturbations satisfying

$$
Z\Delta W_Q^\top = T_Q,
\qquad
Z\Delta W_K^\top = T_K,
\qquad
Z\Delta W_V^\top = T_V.
$$

For one block, write

$$
Z\Delta W^\top = T.
$$

The minimum-Frobenius-norm solution is

$$
\boxed{
\Delta W^\top = Z^+T,
}
$$

where $Z^+$ is the Moore-Penrose pseudoinverse of $Z$. Equivalently,

$$
\boxed{
\Delta W = T^\top (Z^+)^\top.
}
$$

When $Z$ has full column rank, which is the common large-batch-token regime $N\gg d_{\mathrm{model}}$,

$$
Z^+ = (Z^\top Z)^{-1}Z^\top,
$$

so

$$
\boxed{
\Delta W = T^\top Z (Z^\top Z)^{-1}.
}
$$

In practice, use damping for numerical stability:

$$
\boxed{
\Delta W = T^\top Z (Z^\top Z + \varepsilon I)^{-1}.
}
$$

Applying this to the three attention blocks gives

$$
\boxed{
\Delta W_Q
=
-r^\star\operatorname{rsgn}(G_Q)^\top Z (Z^\top Z+\varepsilon I)^{-1},
}
$$

$$
\boxed{
\Delta W_K
=
-r^\star\operatorname{rsgn}(G_K)^\top Z (Z^\top Z+\varepsilon I)^{-1},
}
$$

$$
\boxed{
\Delta W_V
=
-r^\star\operatorname{rsgn}(G_V)^\top Z (Z^\top Z+\varepsilon I)^{-1}.
}
$$

With damping $\eta$, replace $r^\star$ by $\eta r^\star$.

---

## 10. Complete proposed update

At iteration $t$, for one attention head:

1. **Forward pass.** Compute

   $$
   Q=ZW_Q^\top,\qquad K=ZW_K^\top,\qquad V=ZW_V^\top,
   $$

   $$
   S=QK^\top/\sqrt{d_h},\qquad P=\operatorname{softmax}_{\mathrm{row}}(S),\qquad Y=PV.
   $$

2. **Backward pass.** Obtain activation gradients

   $$
   G_Q=\nabla_Q\mathcal L,
   \qquad
   G_K=\nabla_K\mathcal L,
   \qquad
   G_V=\nabla_V\mathcal L,
   $$

   and the gradient at the head output $Y=PV$,

   $$
   G_Y=\nabla_Y\mathcal L.
   $$

3. **Compute row norms.**

   $$
   q=\|Q\|_{\infty,2},
   \qquad
   k=\|K\|_{\infty,2},
   \qquad
   v=\|V\|_{\infty,2}.
   $$

4. **Compute the dual gradient norms.**

   $$
   S_G=\|G_Q\|_{1,2}+\|G_K\|_{1,2}+\|G_V\|_{1,2},
   \qquad
   g_Y=\|G_Y\|_{1,2}.
   $$

5. **Define the radius-dependent curvature bound.**

   $$
   C_t(r)
   =
   h_\sigma (v+r)
   \frac{(q+k+2r)^2}{d_h}
   +
   \frac{2(v+r)}{\sqrt{d_h}}
   +
   \frac{2(q+k+2r)}{\sqrt{d_h}}.
   $$

6. **Find the scalar radius.** Solve

   $$
   -S_G+g_Y\!\left[rC_t(r)+\frac{1}{2}r^2C_t'(r)\right]=0
   $$

   for $r\ge 0$ by the bisection search above. Call the result $r^\star$.

7. **Construct activation-space targets.**

   $$
   T_Q=-\eta r^\star\operatorname{rsgn}(G_Q),
   \qquad
   T_K=-\eta r^\star\operatorname{rsgn}(G_K),
   \qquad
   T_V=-\eta r^\star\operatorname{rsgn}(G_V).
   $$

8. **Pull back to weights.**

   $$
   \Delta W_Q = T_Q^\top Z(Z^\top Z+\varepsilon I)^{-1},
   $$

   $$
   \Delta W_K = T_K^\top Z(Z^\top Z+\varepsilon I)^{-1},
   $$

   $$
   \Delta W_V = T_V^\top Z(Z^\top Z+\varepsilon I)^{-1}.
   $$

9. **Update.**

   $$
   W_Q \leftarrow W_Q+\Delta W_Q,
   \qquad
   W_K \leftarrow W_K+\Delta W_K,
   \qquad
   W_V \leftarrow W_V+\Delta W_V.
   $$

---

## 11. Why the pullback uses a Frobenius minimum

The row-norm derivation chooses the desired movement in **activation space**:

$$
Z\Delta W^\top = T.
$$

There are generally many weight-space perturbations that realize the same activation-space perturbation on the current batch. Once the activation-space target $T$ has been chosen, the remaining question is only how to pick a representative $\Delta W$ among all solutions.

The minimum-Frobenius solution

$$
\Delta W^\top = Z^+T
$$

is the least-energy representative in parameter space. This does **not** replace the row-norm geometry used to choose $T$. It only selects the smallest parameter update that realizes the chosen activation update on the current batch.

This is also the step that introduces the input Gram matrix

$$
Z^\top Z.
$$

With damping,

$$
\Delta W = T^\top Z(Z^\top Z+\varepsilon I)^{-1},
$$

which has the same input-whitening structure as Newton-style or Muon-style updates, but with a row-normalized activation-space target.

---

## 12. Relation to the static-$\lambda$ approximation

A simpler approximation would freeze the curvature at radius zero:

$$
\lambda_t := C_t(0)
=
h_\sigma v\frac{(q+k)^2}{d_h}
+
\frac{2v}{\sqrt{d_h}}
+
\frac{2(q+k)}{\sqrt{d_h}}.
$$

This gives

$$
r_{\mathrm{static}} = \frac{S_G}{g_Y\,\lambda_t}.
$$

But the correct model is

$$
\Phi_t(r)=-S_Gr+\frac{1}{2}\,g_Y\,C_t(r)\,r^2,
$$

with $C_t(r)$ increasing in $r$. Therefore the correct radius generally satisfies

$$
r^\star < \frac{S_G}{g_Y\,C_t(0)}
$$

unless the step is infinitesimal or curvature growth is negligible.

The static approximation is acceptable only as a cheap heuristic. The radius search is the curvature-consistent version.

---

## 13. Optional exact line search on the actual loss

The $r^\star$ above is exact for the certified scalar upper model. One may instead define the actual one-dimensional objective

$$
\ell_t(r)
:=
\mathcal L\left(
F\left(
Q-r\operatorname{rsgn}(G_Q),
K-r\operatorname{rsgn}(G_K),
V-r\operatorname{rsgn}(G_V)
\right)
\right).
$$

An exact line search on $\ell_t(r)$ generally has no closed form because of the softmax and the downstream loss. It can be approximated by backtracking, golden-section search over a bracket, or safeguarded interpolation. However, this searches the actual loss, not the smoothness-certified model.

A practical compromise is:

1. Compute the certified radius $r^\star$ from $\Phi_t'(r)=0$.
2. Try $r=\eta r^\star$.
3. If the actual minibatch loss or a standard Armijo condition fails, shrink $\eta$.

This preserves the curvature-aware scale while allowing the implementation to guard against looseness in the bound.

---

## 14. Final compact formula

For one head, the proposed update is

$$
\boxed{
\Delta W_B
=
-\eta r^\star
\operatorname{rsgn}(G_B)^\top
Z(Z^\top Z+\varepsilon I)^{-1},
\qquad
B\in\{Q,K,V\},
}
$$

where $r^\star$ is the unique nonnegative root of

$$
\boxed{
-S_G+g_Y\!\left[rC_t(r)+\frac{1}{2}r^2C_t'(r)\right]=0,
}
$$

with

$$
\boxed{
S_G=\|G_Q\|_{1,2}+\|G_K\|_{1,2}+\|G_V\|_{1,2},
\qquad
g_Y=\|\nabla_Y\mathcal L\|_{1,2},
}
$$

$$
\boxed{
C_t(r)
=
 h_\sigma (v+r)
\frac{(q+k+2r)^2}{d_h}
+
\frac{2(v+r)}{\sqrt{d_h}}
+
\frac{2(q+k+2r)}{\sqrt{d_h}},
}
$$

and

$$
\boxed{
C_t'(r)
=
\frac{h_\sigma}{d_h}
\left[(q+k+2r)^2+4(v+r)(q+k+2r)\right]
+
\frac{6}{\sqrt{d_h}}.
}
$$

This is the row-norm, radius-correct, activation-space update with a minimum-Frobenius pullback to parameter space.
