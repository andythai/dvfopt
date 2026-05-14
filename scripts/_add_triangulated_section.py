"""One-shot helper: swap the triangulated before/after figure for a 3-row
version that also shows per-pixel central-diff Jdet at deformed vertices."""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "shoelace-artifact-example.ipynb"


TRI_BA_MD = r"""<a id="sec-tri-beforeafter"></a>
### Before / after — triangulated

Three columns: pre-correction bowtie, plain-shoelace correction, triangulated correction.
Top row: quads coloured by shoelace area. Middle row: per-cell `cell_min_jdet_2d` in the
first two columns, per-cell `min(T1, T2)` in the triangulated column (the metric the
constraint actually enforced). Bottom row: per-vertex central-difference pixel Jdet
scattered over the deformed quad wireframe — unchanged sign across all three variants,
reminding us that pixel Jdet alone can't distinguish the cases."""


TRI_BA = r"""fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)

cmap = plt.get_cmap("bwr")

# Shared norms so the three columns are directly comparable.
vmax_s3 = max(abs(shoe.min()), abs(shoe.max()),
              abs(shoe_corr.min()), abs(shoe_corr.max()),
              abs(shoe_tri.min()), abs(shoe_tri.max()), 1.0)
norm_s3 = mcolors.TwoSlopeNorm(vmin=-vmax_s3, vcenter=0, vmax=vmax_s3)

vmax_c3 = max(abs(cell_min.min()), abs(cell_min.max()),
              abs(cell_min_corr.min()), abs(cell_min_corr.max()),
              abs(cell_min_tri.min()), abs(cell_min_tri.max()), 1.0)
norm_c3 = mcolors.TwoSlopeNorm(vmin=-vmax_c3, vcenter=0, vmax=vmax_c3)

triA = np.minimum(T1_tri, T2_tri)
vmax_t = max(1.0, abs(triA.min()), abs(triA.max()))
norm_t = mcolors.TwoSlopeNorm(vmin=-vmax_t, vcenter=0, vmax=vmax_t)

vmax_j3 = max(abs(jdet.min()), abs(jdet.max()),
              abs(jdet_corr.min()), abs(jdet_corr.max()),
              abs(jdet_tri.min()), abs(jdet_tri.max()), 1.0)
norm_j3 = mcolors.TwoSlopeNorm(vmin=-vmax_j3, vcenter=0, vmax=vmax_j3)

# Shared view window across all panels.
_all_x = np.concatenate([def_x.ravel(), def_x_corr.ravel(), def_x_tri.ravel()])
_all_y = np.concatenate([def_y.ravel(), def_y_corr.ravel(), def_y_tri.ravel()])
_xlim = (_all_x.min() - 0.6, _all_x.max() + 0.6)
_ylim = (_all_y.max() + 0.6, _all_y.min() - 0.6)


def _draw_quads(ax, dxg, dyg, fill_vals, norm, title, bad_mask=None):
    xg = xx + dxg
    yg = yy + dyg
    for i in range(H - 1):
        for j in range(W - 1):
            corners = [
                (xg[i, j],         yg[i, j]),
                (xg[i, j + 1],     yg[i, j + 1]),
                (xg[i + 1, j + 1], yg[i + 1, j + 1]),
                (xg[i + 1, j],     yg[i + 1, j]),
            ]
            bad = bool(bad_mask[i, j]) if bad_mask is not None else fill_vals[i, j] <= 0
            ax.add_patch(Polygon(
                corners, closed=True,
                facecolor=cmap(norm(fill_vals[i, j])),
                edgecolor="yellow" if bad else "black",
                linewidth=2.5 if bad else 0.6,
                zorder=2 if bad else 1,
            ))
    _annotate_vertices(ax, xg, yg, fontsize=6)
    ax.set_xlim(*_xlim)
    ax.set_ylim(*_ylim)
    ax.set_aspect("equal")
    ax.set_title(title)


def _draw_vertex_jdet(ax, dxg, dyg, jdet_vals, title):
    xg = xx + dxg
    yg = yy + dyg
    for i in range(H - 1):
        for j in range(W - 1):
            corners = [
                (xg[i, j],         yg[i, j]),
                (xg[i, j + 1],     yg[i, j + 1]),
                (xg[i + 1, j + 1], yg[i + 1, j + 1]),
                (xg[i + 1, j],     yg[i + 1, j]),
            ]
            ax.add_patch(Polygon(corners, closed=True, facecolor="none",
                                 edgecolor="lightgray", linewidth=0.6, zorder=1))
    sc = ax.scatter(xg.ravel(), yg.ravel(),
                    c=jdet_vals.ravel(), cmap=cmap, norm=norm_j3,
                    s=260, edgecolor="k", linewidth=0.5, zorder=3)
    _annotate_vertices(ax, xg, yg, fontsize=6)
    ax.set_xlim(*_xlim)
    ax.set_ylim(*_ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    return sc


# Row 0: shoelace fill.
_draw_quads(axes[0, 0], dx,      dy,      shoe,      norm_s3,
            f"BEFORE (bowtie)  |  shoe min = {shoe.min():+.3f}")
_draw_quads(axes[0, 1], dx_corr, dy_corr, shoe_corr, norm_s3,
            f"SHOELACE  |  shoe min = {shoe_corr.min():+.3f}")
_draw_quads(axes[0, 2], dx_tri,  dy_tri,  shoe_tri,  norm_s3,
            f"TRIANGULATED  |  shoe min = {shoe_tri.min():+.3f}")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_s3),
             ax=axes[0, :], label="shoelace area", shrink=0.8)

# Row 1: cell_min_jdet on first two panels, min(T1,T2) on the third.
_draw_quads(axes[1, 0], dx,      dy,      cell_min,      norm_c3,
            f"BEFORE  |  cell_min = {cell_min.min():+.3f}")
_draw_quads(axes[1, 1], dx_corr, dy_corr, cell_min_corr, norm_c3,
            f"SHOELACE  |  cell_min = {cell_min_corr.min():+.3f}")
_draw_quads(axes[1, 2], dx_tri,  dy_tri,  triA, norm_t,
            f"TRIANGULATED  |  min(T1,T2) = {triA.min():+.3f}",
            bad_mask=(triA <= 0))
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_c3),
             ax=axes[1, 0:2], label="cell_min_jdet", shrink=0.8)
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_t),
             ax=axes[1, 2], label="min(T1, T2)", shrink=0.8)

# Row 2: per-vertex central-diff pixel Jdet.
sc_jdet = _draw_vertex_jdet(axes[2, 0], dx,      dy,      jdet,
                            f"BEFORE  |  pixel Jdet min = {jdet.min():+.3f}")
_draw_vertex_jdet(axes[2, 1], dx_corr, dy_corr, jdet_corr,
                  f"SHOELACE  |  pixel Jdet min = {jdet_corr.min():+.3f}")
_draw_vertex_jdet(axes[2, 2], dx_tri,  dy_tri,  jdet_tri,
                  f"TRIANGULATED  |  pixel Jdet min = {jdet_tri.min():+.3f}")
fig.colorbar(sc_jdet, ax=axes[2, :], label="central-diff pixel Jdet", shrink=0.8)

plt.show()"""


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    patched_md = patched_code = False
    for c in nb["cells"]:
        if c.get("id") == "tri-beforeafter-md":
            c["source"] = TRI_BA_MD.splitlines(keepends=True)
            patched_md = True
        elif c.get("id") == "tri-beforeafter-code":
            c["source"] = TRI_BA.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            patched_code = True
    if not (patched_md and patched_code):
        raise SystemExit("tri-beforeafter cells not found")
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Patched {NB}")


if __name__ == "__main__":
    main()
