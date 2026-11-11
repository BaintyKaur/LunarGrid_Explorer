"""Maps 6 and 7: crater density on the 20 km equal-area polar grid (north and south, three catalogues).

Reads : data/processed/polar_cells_gistar.csv
Writes: figures/map06_07_polar_density_1km.png, map06_07_polar_density_2km.png
Run   : python maps_6_7.py 2>&1 | tee maps_6_7.log

This file also holds the helper functions that the later polar maps (8 to 12) re-use.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/lunargrid/data/processed")
FIG = os.path.expanduser("~/lunargrid/figures")
os.makedirs(FIG, exist_ok=True)
R = 1737400.0
CELL, HALF = 20000.0, 770000.0
INK, MUTED = "#14181F", "#4A5260"
plt.rcParams["font.family"] = "DejaVu Sans"
CATS = [("robbins", "Robbins"), ("lu5m", "LU5M812TGT"), ("luc", "LUC-GRAS200")]


def is_north(s):
    return str(s).strip().lower().startswith("n")


def pick_cat(series, key):
    return series.astype(str).str.lower().str.contains(key)


def load_cells():
    d = pd.read_csv(BASE + "/polar_cells_gistar.csv")
    print("pole values:", sorted(d.pole.astype(str).unique()), "| catalogue values:", sorted(d.catalogue.astype(str).unique()),
          "| thr_km values:", sorted(d.thr_km.unique()), flush=True)
    d["north"] = d.pole.map(is_north)
    return d


def to_grid(sub, col):
    """Cells -> 77 x 77 array (row 0 = most negative y), NaN outside the cap."""
    g = np.full((77, 77), np.nan)
    ix = np.floor((sub.x.to_numpy() + HALF) / CELL).astype(int)
    iy = np.floor((sub.y.to_numpy() + HALF) / CELL).astype(int)
    g[iy, ix] = sub[col].to_numpy()
    return g


def rho_km(lat_abs):
    return 2 * R * np.tan(np.radians(90 - lat_abs) / 2) / 1000.0


def polar_panel(ax, grid, north, cmap, vmin, vmax, title=None, bad="#E8E8E8"):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(bad)
    im = ax.imshow(grid, origin="lower", extent=[-770, 770, -770, 770], cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    t = np.linspace(0, 2 * np.pi, 200)
    for la in (70, 80):
        r = rho_km(la)
        ax.plot(r * np.cos(t), r * np.sin(t), color="white", lw=0.6, alpha=0.8)
        ax.text(r * 0.707 + 4, r * 0.707 + 4, "%d°" % la, fontsize=6.5, color="white" if la else MUTED, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # longitude 0 marker: south map has 0° at the top, north map has 0° at the bottom
    ax.text(0, 752 if not north else -752, "0°E", fontsize=7, color=MUTED, ha="center", va="top" if not north else "bottom")
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    return im


def main():
    d = load_cells()
    print("===== RESULTS =====", flush=True)
    for thr in sorted(d.thr_km.unique()):
        t = d[np.isclose(d.thr_km, thr)]
        vmax = np.percentile(t.density, 99)
        print("threshold %g km: cells %d, median density %.1f, 99th percentile %.1f (colour limit), max %.1f" %
              (thr, len(t), t.density.median(), vmax, t.density.max()), flush=True)
        fig, axs = plt.subplots(2, 3, figsize=(12.5, 8.6))
        for j, (key, lab) in enumerate(CATS):
            for i, north in enumerate((True, False)):
                sub = t[pick_cat(t.catalogue, key) & (t.north == north)]
                im = polar_panel(axs[i, j], to_grid(sub, "density"), north, "viridis", 0, vmax, lab if i == 0 else None)
                if j == 0:
                    axs[i, j].set_ylabel("North cap (65°N to pole)" if north else "South cap (65°S to pole)", fontsize=10, color=INK)
        cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.035, pad=0.03, extend="max")
        cb.set_label("craters of %g km and above per 1,000 km² (20 km equal-area cells; one scale for all six maps)" % thr, fontsize=9, color=MUTED)
        cb.ax.tick_params(labelsize=8)
        fig.suptitle("Polar crater density, craters of %g km and above" % thr, fontsize=12, color=INK, y=0.96)
        name = "map06_07_polar_density_%gkm.png" % thr
        fig.savefig(FIG + "/" + name, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("saved", name, flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()