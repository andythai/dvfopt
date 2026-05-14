"""Rewrite the ``neighborhood-md`` cell with clean UTF-8 and extend it with
prose tying the continuous IFT radius formula to the discrete construction
used by ``dvfopt.jacobian.ift_radius_2d``.

Fixes prior mojibake (em-dash / section-sign / umlaut bytes stored as
latin-1 code points) by overwriting the cell source, and adds a new
subsection describing the per-pixel finite-difference stencil used at
runtime.
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "shoelace-artifact-example.ipynb"


NEIGHBORHOOD_MD = r"""<a id="sec-neighborhood"></a>
## Verification 2 — Neighborhood measure

* `cell_min_jdet_2d` — closed-form minimum of the bilinear-interpolant Jdet over each unit quad. Positivity ⇒ the continuous deformation is injective over the whole cell (sub-pixel certificate).
* `ift_radius_2d` — quantitative Inverse Function Theorem lower bound on the size of the neighbourhood where the deformation is provably injective. Small values mean the certified-invertible region shrinks toward a point.

<a id="sec-ift-formula"></a>
### The IFT radius formula

We want a **guaranteed-injective ball**: the largest $r$ such that the deformation $\varphi(x) = x + u(x)$ is one-to-one on $B(x_0, r)$. The quantitative inverse-function theorem gives

$$r(x_0) \;\lesssim\; \frac{\sigma_{\min}\bigl(I + \nabla u(x_0)\bigr)}{\lVert \nabla^2 u \rVert_{B(x_0,r)}}.$$

This follows from the fundamental theorem of calculus plus the Neumann / Banach invertibility lemma. Derivation below.

<a id="sec-ift-derivation"></a>
#### Step-by-step derivation

**Step 1 — chord form from the fundamental theorem of calculus.** For any $y, z$ in a convex set, $\varphi(z) - \varphi(y)$ is the integral of its derivative along the segment $\gamma(t) = y + t(z - y)$, $t \in [0, 1]$. Chain rule + FTC give

$$
\varphi(z) - \varphi(y) \;=\; \int_0^1 D\varphi\bigl(y + t(z-y)\bigr)(z - y)\,dt \;=\; \underbrace{\left[\int_0^1 D\varphi\bigl(y + t(z-y)\bigr)\,dt\right]}_{M(y, z)}\,(z - y).
$$

So if $\varphi(z) = \varphi(y)$ with $y \neq z$, then $M(y, z)$ must kill the nonzero vector $z - y$ — i.e. $M(y, z)$ is **singular**. Therefore **$\varphi$ is injective on $B(x_0, r)$ whenever $M(y, z)$ is invertible for every pair in that ball.**

**Step 2 — split $M$ into centre + perturbation.** Add and subtract $D\varphi(x_0)$:

$$
M(y, z) \;=\; D\varphi(x_0) \;+\; \underbrace{\int_0^1 \bigl[\,D\varphi\bigl(y + t(z-y)\bigr) - D\varphi(x_0)\,\bigr]\,dt}_{E(y, z)}.
$$

The "unperturbed" piece $D\varphi(x_0) = I + \nabla u(x_0)$ is exactly what appears in the numerator of the final formula. Everything else is dumped into $E$.

**Step 3 — bound $\lVert E \rVert$ using the Hessian.** Since $D^2 \varphi = D^2 u$, the mean-value theorem applied to $D\varphi$ gives

$$
\bigl\lVert D\varphi(w) - D\varphi(x_0) \bigr\rVert \;\le\; \lVert \nabla^2 u \rVert \cdot \lVert w - x_0 \rVert.
$$

If $y, z \in B(x_0, r)$ then by convexity $y + t(z - y) \in B(x_0, r)$ too, so $\lVert w - x_0 \rVert \le r$ on the whole segment. Plug in and take the sup over the ball:

$$
\lVert E(y, z) \rVert \;\le\; \int_0^1 \lVert \nabla^2 u \rVert \cdot r\, dt \;=\; r \cdot L, \qquad L \;:=\; \sup_{B(x_0, r)} \lVert \nabla^2 u \rVert.
$$

**Step 4 — Neumann / Banach invertibility lemma.** For any square matrix $A$ and perturbation $E$,

$$
\lVert E \rVert \;<\; \sigma_{\min}(A) \quad\Longrightarrow\quad A + E \text{ is invertible}.
$$

(Proof sketch: write $A + E = A(I + A^{-1}E)$. $\lVert A^{-1} E \rVert \le \lVert A^{-1} \rVert \cdot \lVert E \rVert = \lVert E \rVert / \sigma_{\min}(A) < 1$, so $I + A^{-1}E$ is invertible by its Neumann series $\sum_{k \ge 0} (-A^{-1}E)^k$.)

Apply with $A = D\varphi(x_0) = I + \nabla u(x_0)$ and $E = E(y, z)$:

$$
r \cdot L \;<\; \sigma_{\min}\bigl(I + \nabla u(x_0)\bigr) \quad\Longrightarrow\quad M(y, z) \text{ invertible for all } y, z \in B(x_0, r).
$$

**Step 5 — solve for $r$.** Rearranging gives the **sharp quantitative-IFT radius**:

$$
r \;<\; \frac{\sigma_{\min}\bigl(I + \nabla u(x_0)\bigr)}{\sup_{B(x_0, r)} \lVert \nabla^2 u \rVert}.
$$

Pointwise (take the sup shrinking to $\lVert \nabla^2 u(x_0) \rVert$ as $r \to 0$), this is the formula evaluated by `ift_radius_2d`. A conservative **factor of $\tfrac{1}{2}$** is often added — this is the contraction-mapping convention, which asks $\lVert E \rVert \le \tfrac{1}{2}\sigma_{\min}(A)$ so the contraction constant stays at most $1/2$, buying quantitative control on $\varphi^{-1}$ in addition to mere invertibility.

**What each term measures, in one line:**

* $\sigma_{\min}(I + \nabla u)$ — how far the linearised Jacobian is from being singular (large ⇒ strongly invertible at $x_0$).
* $\lVert \nabla^2 u \rVert$ — how fast the map deviates from that linearisation (the curvature of the displacement).
* $r$ — radius on which the linear part *still dominates* the curvature term, so no two distinct points collapse onto the same image.

**Why this complements pixel Jdet.** The pixel Jacobian determinant only certifies that the map is non-degenerate *at a point*. The IFT radius says *how far* that certificate extends before curvature could flip the sign. In our bowtie example the cell-min Jdet should go non-positive exactly on the folded cell, and the IFT radius should collapse there.

<a id="sec-ift-discrete"></a>
### How `ift_radius_2d` discretises the formula

The continuous bound above is evaluated pointwise on the grid by [`dvfopt.jacobian.injectivity_radius.ift_radius_2d`](../dvfopt/jacobian/injectivity_radius.py#L65). Given a displacement field with channels $(dy, dx)$ of shape $(H, W)$, it returns a per-pixel `(H, W)` radius map in three steps:

1. **Jacobian $I + \nabla u$ from central differences.** Four partials per pixel are computed with `np.gradient` (interior: second-order central; boundaries: first-order forward/backward):

$$
I + \nabla u \;=\; \begin{pmatrix} 1 + \partial_x dx & \partial_y dx \\[2pt] \partial_x dy & 1 + \partial_y dy \end{pmatrix}.
$$

   These are exactly the same central-difference stencils that produce the pixel Jdet critiqued earlier — so the IFT map inherits the same smoothing blind-spot on sharp bowties. That is by design: both maps are about the *linearised* differential, and the IFT map catches what the Jdet misses via its *denominator*, not its numerator.

2. **Smallest singular value in closed form.** The subscript $F$ is the **Frobenius norm** — square every entry, sum, square-root. For $J = \bigl(\begin{smallmatrix} a & b \\ c & d \end{smallmatrix}\bigr)$, $\lVert J \rVert_F^2 = a^2 + b^2 + c^2 + d^2$.

   Every $2 \times 2$ matrix has two singular values $\sigma_{\min} \le \sigma_{\max}$ satisfying two identities:

$$
\sigma_{\min}\,\sigma_{\max} \;=\; \lvert \det J \rvert, \qquad \sigma_{\min}^2 + \sigma_{\max}^2 \;=\; \lVert J \rVert_F^2.
$$

   So $\sigma_{\min}^2$ and $\sigma_{\max}^2$ are the two roots of $t^2 - \lVert J \rVert_F^2\,t + (\det J)^2 = 0$. The quadratic formula picks the smaller root:

$$
\sigma_{\min}^2 \;=\; \tfrac{1}{2}\Bigl(\lVert J \rVert_F^2 - \sqrt{\lVert J \rVert_F^4 - 4\,(\det J)^2}\Bigr).
$$

   In code this is just four arithmetic steps (see [`_sigma_min_2d`](../dvfopt/jacobian/injectivity_radius.py#L36)):

   ```python
   det       = a*d - b*c
   frob_sq   = a*a + b*b + c*c + d*d
   disc      = frob_sq**2 - 4 * det**2
   sigma_min = np.sqrt(0.5 * (frob_sq - np.sqrt(disc)))
   ```

   No SVD call — one vectorised expression over the whole $(H, W)$ grid. `np.clip(..., 0, None)` inside both `sqrt`s guards against tiny negative values from floating-point noise.

3. **Hessian norm from nested `np.gradient`.** The six independent second partials are formed by applying `np.gradient` twice (once along each axis of the first derivative):

$$
\lVert \nabla^2 u \rVert \;=\; \sqrt{(\partial_{xx} dx)^2 + (\partial_{xy} dx)^2 + (\partial_{yy} dx)^2 + (\partial_{xx} dy)^2 + (\partial_{xy} dy)^2 + (\partial_{yy} dy)^2}.
$$

   This is a 3-point stencil applied twice, so the effective support is a 5-point row/column window — large enough that the denominator *does* see the bowtie's sharp twist even when the central-diff Jacobian smooths it away. That asymmetry is what makes the IFT radius collapse at the fold while the pixel Jdet stays positive.

4. **Per-pixel radius.** The returned array is

$$
r[i, j] \;=\; \frac{\sigma_{\min}(I + \nabla u)[i, j]}{2\,\lVert \nabla^2 u \rVert[i, j] + \varepsilon},\qquad \varepsilon = 10^{-8},
$$

   with $\varepsilon$ keeping the map finite in constant-Jacobian regions where the true radius is $+\infty$ (there, $r$ saturates at $\sigma_{\min}/(2\varepsilon)$, which is effectively "huge — no nearby fold"). The heatmap plotted above is this array rendered with a log-scaled colourbar, so the bowtie cell appears as the dark spot where $r$ drops by several orders of magnitude.

**Practical reading.** Pixels with $r$ near machine-huge are surrounded by a guaranteed-injective ball of radius $\gtrsim r$; pixels where $r$ approaches zero are on (or next to) a fold, where no finite injective neighbourhood can be certified. In the bowtie example the ratio collapses on the self-intersecting quad, consistent with the fold diagnosis from `cell_min_jdet_2d`.

[Back to TOC](#sec-toc)"""


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "neighborhood-md":
            if c["cell_type"] != "markdown":
                raise SystemExit("neighborhood-md is not markdown")
            c["source"] = NEIGHBORHOOD_MD.splitlines(keepends=True)
            break
    else:
        raise SystemExit("neighborhood-md cell not found")
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NB}")


if __name__ == "__main__":
    main()
