import os, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

P = os.path.expanduser("~/lunargrid/data/processed")
OUT = os.path.expanduser("~/lunargrid/maps"); os.makedirs(OUT, exist_ok=True)
R, CELL, NC = 1737400.0, 20000.0, 77
HALF = NC * CELL / 2.0
THR = 61000.0
NAMES = {"robbins": "Robbins", "lu5m": "LU5M812TGT", "luc": "LUC-GRAS200"}
CATS = list(NAMES)
def log(*a): print(*a, flush=True)

def gistar(xy, val):
    n = len(val)
    nb = cKDTree(xy).query_ball_point(xy, r=THR)
    rows = np.repeat(np.arange(n), [len(a) for a in nb]); cols = np.concatenate([np.array(a) for a in nb])
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    sw = np.asarray(W.sum(axis=1)).ravel()
    xb, s = val.mean(), np.sqrt((val**2).mean() - val.mean()**2)
    return (W @ val - xb * sw) / (s * np.sqrt((n * sw - sw**2) / (n - 1)))

def to_grid(x, y, v):
    g = np.full((NC, NC), np.nan)
    ix = np.rint((x + HALF - CELL / 2) / CELL).astype(int)
    iy = np.rint((y + HALF - CELL / 2) / CELL).astype(int)
    g[iy, ix] = v
    return g

def decorate(ax, pole):
    t = np.linspace(0, 2 * np.pi, 240)
    for lat in (70, 75, 80, 85):
        rho = 2 * R * np.tan(np.pi / 4 - np.radians(lat) / 2) / 1000.0
        ax.plot(rho * np.cos(t), rho * np.sin(t), lw=0.5, color="k", alpha=0.35)
        ax.text(rho * np.cos(-0.785), rho * np.sin(-0.785), f"{lat}\u00b0{'S' if pole == 'south' else 'N'}", fontsize=6, alpha=0.8, ha="left", va="top")
    for lon, lab in ((0, "0\u00b0"), (90, "90\u00b0E"), (180, "180\u00b0"), (-90, "90\u00b0W")):
        lam = np.radians(lon); dx, dy = np.sin(lam), (np.cos(lam) if pole == "south" else -np.cos(lam))
        ax.plot([0, 770 * dx], [0, 770 * dy], lw=0.5, color="k", alpha=0.25)
        ax.text(815 * dx, 815 * dy, lab, ha="center", va="center", fontsize=8)
    ax.set_xlim(-870, 870); ax.set_ylim(-870, 870); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

d = pd.read_csv(P + "/polar_cells_gistar.csv")
ext = [-HALF / 1000, HALF / 1000, -HALF / 1000, HALF / 1000]
log("RESULTS")
for pole in ("south", "north"):
    for thr in (1, 2):
        sub = d[(d.pole == pole) & (d.thr_km == thr)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
        parts = {c: sub[sub.catalogue == c].sort_values("id") for c in CATS}
        vmax = float(np.nanpercentile(np.concatenate([parts[c].density.values for c in CATS]), 99))
        for j, c in enumerate(CATS):
            p = parts[c]
            im0 = axes[0, j].imshow(to_grid(p.x.values, p.y.values, p.density.values), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
            im1 = axes[1, j].imshow(to_grid(p.x.values, p.y.values, p.z_numpy.values), origin="lower", extent=ext, cmap="RdBu_r", vmin=-8, vmax=8)
            axes[0, j].set_title(NAMES[c]); decorate(axes[0, j], pole); decorate(axes[1, j], pole)
        fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.8, label="craters per 1000 km\u00b2 (20 km cells)")
        fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.8, label="Gi* Z-score (red = hot, blue = cold)")
        fig.suptitle(f"{pole.title()} polar cap (65\u00b0\u201390\u00b0), craters with D \u2265 {thr} km", fontsize=14)
        f = f"{OUT}/polar_maps_{pole}_D{thr}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight"); plt.close(fig)
        log("saved", f)

for pole in ("south", "north"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3))
    cmap = ListedColormap(["#eeeeee", "#fee08b", "#f46d43", "#a50026"]); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    for ax, thr in zip(axes, (1, 2)):
        sub = d[(d.pole == pole) & (d.thr_km == thr)]
        parts = {c: sub[sub.catalogue == c].sort_values("id").reset_index(drop=True) for c in CATS}
        base = parts["robbins"]
        ring = pd.cut(base.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False)
        k = int(np.ceil(0.05 * len(base)))
        count = np.zeros(len(base), dtype=int)
        for c in CATS:
            dens = parts[c].density.values
            rel = dens / pd.Series(dens).groupby(ring.values, observed=True).transform("mean").values
            z = gistar(base[["x", "y"]].values, rel)
            assert not np.isnan(z).any()
            count[np.argsort(-z)[:k]] += 1
        im = ax.imshow(to_grid(base.x.values, base.y.values, count.astype(float)), origin="lower", extent=ext, cmap=cmap, norm=norm)
        decorate(ax, pole); ax.set_title(f"D \u2265 {thr} km")
        r3 = pd.Series(count == 3).groupby(ring.values, observed=True).sum()
        log(f"{pole} D>={thr}: cells in the top 5% of all three catalogues, by ring: " + ", ".join(f"{str(i)}: {int(v)}" for i, v in r3.items()))
    cb = fig.colorbar(im, ax=axes.tolist(), ticks=[0, 1, 2, 3], shrink=0.8, label="catalogues (of 3) with the cell in the top 5% of ring-corrected Gi*")
    fig.suptitle(f"{pole.title()} polar cap: agreement between catalogues", fontsize=13)
    f = f"{OUT}/consensus_{pole}.png"
    fig.savefig(f, dpi=110, bbox_inches="tight"); plt.close(fig)
    log("saved", f)
log("DONE")
