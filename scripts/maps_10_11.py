"""Maps 10 and 11: polar sunlight and permanent shadow; crater density in shadowed versus lit cells.

Reads : data/processed/cells_illumination.csv, polar_cells_gistar.csv   (needs maps_6_7.py)
Writes: figures/map10_sunlight_shadow.png, map11_shadow_vs_lit_map.png, chart11_shadow_vs_lit_bars.png
Run   : python maps_10_11.py 2>&1 | tee maps_10_11.log

A cell is "shadowed" if more than 5% of its 240 m pixels are in permanent shadow.
The printed ratios (mean density in shadowed cells / mean density in lit cells, per ring, Robbins, 1 km
and above) should match the report: north 0.49 0.51 0.65 0.77 0.95, south 0.35 0.77 0.67 0.80 0.84.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from maps_6_7 import (load_cells, to_grid, polar_panel, pick_cat, is_north, CATS, BASE, FIG, INK, MUTED, CELL, HALF)

RING_EDGES = [65, 70, 75, 80, 85, 90.01]
RING_LAB = ["65–70°", "70–75°", "75–80°", "80–85°", "85–90°"]
LIT, SHD = "#7A8594", "#E08A1E"


def load_all():
    d = load_cells()
    il = pd.read_csv(BASE + "/cells_illumination.csv")
    print("illumination pole values:", sorted(il.pole.astype(str).unique()), "| rows", len(il), flush=True)
    il["north"] = il.pole.map(is_north)
    m = d.merge(il[["north", "id", "sun_mean", "psr_frac"]], on=["north", "id"], how="left")
    print("cells with illumination after the join: %d of %d" % (m.sun_mean.notna().sum(), len(m)), flush=True)
    m["ring"] = pd.cut(m.lat.abs(), RING_EDGES, right=False, labels=RING_LAB)
    m["shadowed"] = m.psr_frac > 0.05
    return m


def map10(m):
    cells = m[(m.thr_km == 1) & pick_cat(m.catalogue, "robbins")]
    smax = cells.sun_mean.quantile(0.99)
    pmax = cells.psr_frac.quantile(0.99)
    print("sunlight fraction: median %.3f, 99th pct %.3f | shadow fraction: median %.3f, 99th pct %.3f, cells above 5%%: N %d, S %d" %
          (cells.sun_mean.median(), smax, cells.psr_frac.median(), pmax,
           int(cells[cells.north].shadowed.sum()), int(cells[~cells.north].shadowed.sum())), flush=True)
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.4))
    for i, north in enumerate((True, False)):
        sub = cells[cells.north == north]
        im1 = polar_panel(axs[i, 0], to_grid(sub, "sun_mean"), north, "YlOrBr_r", 0, smax,
                          "Average sunlight fraction" if i == 0 else None)
        im2 = polar_panel(axs[i, 1], to_grid(sub, "psr_frac"), north, "Blues", 0, pmax,
                          "Permanent-shadow fraction" if i == 0 else None)
        axs[i, 0].set_ylabel("North cap" if north else "South cap", fontsize=10, color=INK)
    for im, ax, lab in ((im1, axs[:, 0], "share of time in sunlight (0 = never lit)"),
                        (im2, axs[:, 1], "share of the cell in permanent shadow")):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04, extend="max")
        cb.set_label(lab, fontsize=9, color=MUTED)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Sunlight and permanent shadow on the 20 km polar grid (65° to the pole)", fontsize=12, color=INK, y=0.95)
    fig.savefig(FIG + "/map10_sunlight_shadow.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved map10_sunlight_shadow.png", flush=True)


def map11(m):
    cells = m[(m.thr_km == 1) & pick_cat(m.catalogue, "robbins")]
    vmax = cells.density.quantile(0.99)
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.0))
    for i, north in enumerate((True, False)):
        sub = cells[cells.north == north].copy()
        cat = np.where(sub.shadowed, 2.0, 1.0)
        sub = sub.assign(cat=cat)
        from matplotlib.colors import ListedColormap
        cm = ListedColormap([LIT, SHD])
        polar_panel(axs[i, 0], to_grid(sub, "cat"), north, cm, 1, 2,
                    "Shadowed (> 5% permanent shadow) and lit cells" if i == 0 else None)
        im = polar_panel(axs[i, 1], to_grid(sub, "density"), north, "viridis", 0, vmax,
                         "Crater density, shadowed cells outlined" if i == 0 else None)
        sh = sub[sub.shadowed]
        axs[i, 1].scatter(sh.x / 1000, sh.y / 1000, s=14, marker="s", facecolors="none", edgecolors="#FF3B30", linewidths=0.5)
        axs[i, 0].set_ylabel("North cap" if north else "South cap", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=axs[:, 1], orientation="horizontal", fraction=0.04, pad=0.05, extend="max")
    cb.set_label("craters of 1 km and above per 1,000 km² (Robbins)", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    fig.legend(handles=[Patch(color=LIT, label="lit cell (5% shadow or less)"), Patch(color=SHD, label="shadowed cell (more than 5%)"),
                        Line2D([], [], marker="s", ls="none", mfc="none", mec="#FF3B30", label="shadowed cell on density map")],
               loc="lower left", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.08, 0.0))
    fig.suptitle("Where permanent shadow lies and how many craters those cells hold", fontsize=12, color=INK, y=0.95)
    fig.savefig(FIG + "/map11_shadow_vs_lit_map.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved map11_shadow_vs_lit_map.png", flush=True)


def bars(m):
    t = m[m.thr_km == 1]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
    for j, (key, lab) in enumerate(CATS):
        for i, north in enumerate((True, False)):
            sub = t[pick_cat(t.catalogue, key) & (t.north == north)]
            g = sub.groupby(["ring", "shadowed"], observed=True).density.mean().unstack()
            n = sub.groupby(["ring", "shadowed"], observed=True).size().unstack().fillna(0).astype(int)
            g = g.reindex(RING_LAB)
            n = n.reindex(RING_LAB)
            ratio = (g[True] / g[False]).round(2)
            print("  %-10s %s: shadowed/lit ratio per ring %s | shadowed cells per ring %s" %
                  (lab, "N" if north else "S", list(ratio.values), list(n[True].fillna(0).astype(int).values)), flush=True)
            ax = axs[i, j]
            x = np.arange(5)
            ax.bar(x - 0.2, g[False], 0.4, color=LIT, label="lit cells")
            ax.bar(x + 0.2, g[True], 0.4, color=SHD, label="shadowed cells (> 5%)")
            ax.set_xticks(x)
            ax.set_xticklabels(RING_LAB, fontsize=8)
            ax.tick_params(labelsize=8, colors=MUTED)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color="#E8E4DA", lw=0.8)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(lab, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(("North cap" if north else "South cap") + "\nmean craters per 1,000 km²", fontsize=9, color=MUTED)
    axs[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Crater density (1 km and above) in shadowed and lit cells, by latitude ring", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG + "/chart11_shadow_vs_lit_bars.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved chart11_shadow_vs_lit_bars.png", flush=True)


if __name__ == "__main__":
    m = load_all()
    print("===== RESULTS =====", flush=True)
    map10(m)
    map11(m)
    bars(m)
    print("all done", flush=True)