"""Map 12: hotspots that survive the strict max-T permutation test.

Reads : data/processed/perm_hotspots.csv, polar_cells_gistar.csv   (needs maps_6_7.py)
Writes: figures/map12_permutation_1km.png, map12_permutation_2km.png
Run   : python map_12.py 2>&1 | tee map_12.log

Each panel has three kinds of cell:
  light grey   not a hotspot
  orange       hotspot by the basic Gi* threshold only (Z > 2.58)
  dark red     still significant after the max-T test (family-wise p < 0.05)
The last column counts in how many of the three catalogues a cell passes the max-T test.
The test shown is the first null model (shuffling inside latitude rings only); the second null
(rings and sunlight classes) is printed for comparison. Expected counts of cells significant in all
three catalogues (null A / null B): north 1 km 36/37, north 2 km 1/1, south 1 km 158/139, south 2 km 244/242.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from maps_6_7 import (load_cells, to_grid, polar_panel, pick_cat, is_north, CATS, BASE, FIG, INK, MUTED)

COL3 = ["#EDEDED", "#F4A259", "#B3202A"]
COL4 = ["#EDEDED", "#F9D48B", "#E4703A", "#9B1B1B"]


def main():
    d = load_cells()
    pm = pd.read_csv(BASE + "/perm_hotspots.csv")
    pm["north"] = pm.pole.map(is_north)
    nulls = sorted(pm["null"].astype(str).unique())
    print("null values:", nulls, "| using", nulls[0], "as null A (ring only) and", nulls[1], "as null B (ring x sunlight)", flush=True)
    A, B = nulls[0], nulls[1]
    print("===== RESULTS =====", flush=True)
    for thr in (1, 2):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8.4))
        for i, north in enumerate((True, False)):
            sig_votes = {A: 0, B: 0}
            for j, (key, lab) in enumerate(CATS):
                base = d[np.isclose(d.thr_km, thr) & pick_cat(d.catalogue, key) & (d.north == north)][["id", "x", "y", "z_sed"]]
                res = {}
                for nl in (A, B):
                    p = pm[np.isclose(pm.thr_km, thr) & pick_cat(pm.catalogue, key) & (pm.north == north) & (pm["null"].astype(str) == nl)]
                    res[nl] = base.merge(p[["id", "p_maxT"]], on="id", how="left")
                    sig_votes[nl] = sig_votes[nl] + (res[nl].p_maxT <= 0.0500001).astype(int).to_numpy()
                g = res[A].copy()
                g["cat"] = np.where(g.p_maxT <= 0.0500001, 2, np.where(g.z_sed > 2.58, 1, 0)).astype(float)
                print("  %g km %s %-8s: basic Gi* hot %d | max-T significant: null A %d, null B %d" %
                      (thr, "N" if north else "S", key, int((g.z_sed > 2.58).sum()),
                       int((res[A].p_maxT <= 0.0500001).sum()), int((res[B].p_maxT <= 0.0500001).sum())), flush=True)
                polar_panel(axs[i, j], to_grid(g, "cat"), north, ListedColormap(COL3), 0, 2, lab if i == 0 else None)
                if j == 0:
                    axs[i, j].set_ylabel("North cap" if north else "South cap", fontsize=10, color=INK)
                geo = g
            print("  %g km %s: significant in all three catalogues: null A %d | null B %d" %
                  (thr, "N" if north else "S", int((sig_votes[A] == 3).sum()), int((sig_votes[B] == 3).sum())), flush=True)
            geo = geo.assign(votes=sig_votes[A].astype(float))
            polar_panel(axs[i, 3], to_grid(geo, "votes"), north, ListedColormap(COL4), 0, 3,
                        "Passes max-T test in how many catalogues" if i == 0 else None)
        fig.legend(handles=[Patch(color=COL3[0], label="not a hotspot"), Patch(color=COL3[1], label="hotspot by basic Gi* only (Z > 2.58)"),
                            Patch(color=COL3[2], label="significant after max-T test (p ≤ 0.05)")],
                   loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.36, 0.02))
        fig.legend(handles=[Patch(color=c, label="%d" % n) for n, c in enumerate(COL4)], loc="lower center", ncol=4, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.86, 0.02), title="catalogues passing the test", title_fontsize=8.5)
        fig.suptitle("Statistically robust hotspots (strict permutation test), craters of %g km and above" % thr, fontsize=12, color=INK, y=0.95)
        name = "map12_permutation_%gkm.png" % thr
        fig.savefig(FIG + "/" + name, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("saved", name, flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()