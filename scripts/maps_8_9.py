"""Maps 8 and 9: Getis-Ord Gi* hotspot / coldspot maps and the hotspot agreement map.

Reads : data/processed/polar_cells_gistar.csv   (needs maps_6_7.py in the same folder)
Writes: figures/map08_gistar_1km.png, map08_gistar_2km.png, map09_hotspot_agreement.png
Run   : python maps_8_9.py 2>&1 | tee maps_8_9.log

Map 8 shows the Gi* Z-score of every cell (red = hotspot, blue = coldspot).
Map 9 uses exactly the method of hotspot_agreement.py: density is divided by the mean density of
its 5-degree latitude ring (removes the north-cap trend), Gi* is recomputed on this relative
density, and each catalogue's top 5% of cells (234 of 4,669) are its hotspots; the map shows in how many catalogues (0 to 3)
every cell is a hotspot. The printed consensus counts should be 172 / 205 (south 1 km, 2 km)
and 133 / 159 (north 1 km, 2 km), the numbers already in the report.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from maps_6_7 import (load_cells, to_grid, polar_panel, pick_cat, CATS, FIG, INK, MUTED)

THR = 61000.0


def gistar(xy, val):
    n = len(val)
    nb = cKDTree(xy).query_ball_point(xy, r=THR)
    rows = np.repeat(np.arange(n), [len(a) for a in nb])
    cols = np.concatenate([np.array(a) for a in nb])
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    sw = np.asarray(W.sum(axis=1)).ravel()
    xb, sd = val.mean(), np.sqrt((val ** 2).mean() - val.mean() ** 2)
    return (W @ val - xb * sw) / (sd * np.sqrt((n * sw - sw ** 2) / (n - 1)))


def map8(d):
    for thr in (1, 2):
        t = d[np.isclose(d.thr_km, thr)]
        fig, axs = plt.subplots(2, 3, figsize=(12.5, 8.6))
        for j, (key, lab) in enumerate(CATS):
            for i, north in enumerate((True, False)):
                sub = t[pick_cat(t.catalogue, key) & (t.north == north)]
                hot, cold = int((sub.z_sed > 2.58).sum()), int((sub.z_sed < -2.58).sum())
                print("  %g km %s %-8s: cells Z>2.58 (hot) %d | Z<-2.58 (cold) %d | Z range %.1f to %.1f" %
                      (thr, "N" if north else "S", key, hot, cold, sub.z_sed.min(), sub.z_sed.max()), flush=True)
                im = polar_panel(axs[i, j], to_grid(sub, "z_sed"), north, "RdBu_r", -10, 10, lab if i == 0 else None)
                if j == 0:
                    axs[i, j].set_ylabel("North cap" if north else "South cap", fontsize=10, color=INK)
        cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.035, pad=0.03, extend="both")
        cb.set_label("Getis-Ord Gi* Z-score (red: hotspot, Z > 2.58 is significant at 1%; blue: coldspot)", fontsize=9, color=MUTED)
        cb.ax.tick_params(labelsize=8)
        fig.suptitle("Crater-density hotspots and coldspots, craters of %g km and above" % thr, fontsize=12, color=INK, y=0.96)
        name = "map08_gistar_%gkm.png" % thr
        fig.savefig(FIG + "/" + name, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("saved", name, flush=True)


def map9(d):
    cm = ListedColormap(["#EDEDED", "#F9D48B", "#E4703A", "#9B1B1B"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cm.N)
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.0))
    for i, north in enumerate((True, False)):
        for j, thr in enumerate((1, 2)):
            t = d[np.isclose(d.thr_km, thr) & (d.north == north)]
            votes, geo = None, None
            for key, _ in CATS:
                sub = t[pick_cat(t.catalogue, key)].sort_values("id").reset_index(drop=True)
                ring = pd.cut(sub.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False)
                rel = sub.density.values / sub.density.groupby(ring.values, observed=True).transform("mean").values
                z = gistar(sub[["x", "y"]].values, rel)
                k = int(np.ceil(0.05 * len(sub)))
                hot = np.zeros(len(sub), int)
                hot[np.argsort(-z)[:k]] = 1
                votes = hot if votes is None else votes + hot
                geo = sub
            geo = geo.assign(votes=votes)
            counts = geo.votes.value_counts().reindex([0, 1, 2, 3]).fillna(0).astype(int).to_dict()
            print("  %s %g km: cells by number of catalogues %s | hot in all three: %d" % ("N" if north else "S", thr, counts, counts[3]), flush=True)
            polar_panel(axs[i, j], to_grid(geo, "votes"), north, cm, -0.5, 3.5, "%s cap, %g km and above" % ("North" if north else "South", thr))
            axs[i, j].images[0].set_norm(norm)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(["#EDEDED", "#F9D48B", "#E4703A", "#9B1B1B"],
               ["0 catalogues", "1 catalogue", "2 catalogues", "3 catalogues"])], loc="lower center", ncol=4, frameon=False, fontsize=9,
               title="number of catalogues that place the cell in their top 5% of ring-corrected Gi*", title_fontsize=9)
    fig.suptitle("Hotspot agreement across Robbins, LU5M812TGT and LUC-GRAS200", fontsize=12, color=INK, y=0.95)
    fig.savefig(FIG + "/map09_hotspot_agreement.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved map09_hotspot_agreement.png", flush=True)


if __name__ == "__main__":
    d = load_cells()
    print("===== RESULTS =====", flush=True)
    map8(d)
    map9(d)
    print("all done", flush=True)