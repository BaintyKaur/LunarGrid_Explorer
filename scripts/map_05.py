"""Map 5: crater density by latitude, north and south, three catalogues, beside a polar map.

Reads : data/processed/tiles_gistar.csv, polar_cells_gistar.csv   (needs maps_6_7.py)
Writes: figures/map05_latitude_profile.png
Run   : python map_05.py 2>&1 | tee map_05.log

Each row (north, south) has three panels:
  left   polar density map (Robbins, craters of 1 km and above, 20 km equal-area cells)
  middle median density of the 5-degree tiles in the five latitude bands (all three catalogues)
  right  mean density in 5-degree rings of the equal-area polar grid, relative to the outer ring (65-70 deg)
Expected ring ratios at 85-90 deg (Robbins / LU5M812TGT / LUC-GRAS200): north 2.08 / 1.96 / 1.67, south 1.15 / 1.04 / 0.92.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from maps_6_7 import (load_cells, to_grid, polar_panel, pick_cat, CATS, BASE, FIG, INK, MUTED)

C = {"robbins": "#2a78d6", "lu5m": "#eb6834", "luc": "#1baf7a"}
BANDS = [0, 30, 60, 70, 80, 90.01]
BLAB = ["0–30°", "30–60°", "60–70°", "70–80°", "80–90°"]
RINGS = [65, 70, 75, 80, 85, 90.01]
RLAB = ["65–70°", "70–75°", "75–80°", "80–85°", "85–90°"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9C3B4")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color="#E8E4DA", lw=0.8)
    ax.set_axisbelow(True)


def main():
    d = load_cells()
    t = pd.read_csv(BASE + "/tiles_gistar.csv")
    t["band"] = pd.cut(t.latc.abs(), BANDS, right=False, labels=BLAB)
    cells = d[np.isclose(d.thr_km, 1)].copy()
    cells["ring"] = pd.cut(cells.lat.abs(), RINGS, right=False, labels=RLAB)
    vmax = cells.density.quantile(0.99)
    print("===== RESULTS =====", flush=True)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    for i, north in enumerate((True, False)):
        hemi = "North" if north else "South"
        # left: polar map
        sub = cells[pick_cat(cells.catalogue, "robbins") & (cells.north == north)]
        im = polar_panel(axs[i, 0], to_grid(sub, "density"), north, "viridis", 0, vmax, "%s cap, Robbins (density per 1,000 km²)" % hemi)
        # middle: tile bands
        ax = axs[i, 1]
        tt = t[t.latc > 0] if north else t[t.latc < 0]
        for key, lab in CATS:
            med = tt.groupby("band", observed=True)["dens1_" + key].median().reindex(BLAB)
            print("  %s tile-band medians %-10s %s" % (hemi, lab, [round(v, 1) for v in med.values]), flush=True)
            ax.plot(range(5), med.values, marker="o", lw=2, ms=5.5, color=C[key], label=lab)
        ax.set_xticks(range(5))
        ax.set_xticklabels(BLAB)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("latitude band (towards the pole →)", fontsize=9, color=MUTED)
        ax.set_ylabel("median density, 5° tiles\n(craters ≥ 1 km per 1,000 km²)", fontsize=9, color=MUTED)
        ax.set_title("%s hemisphere: 5° tiles" % hemi, fontsize=10, color=INK)
        style(ax)
        if i == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        # right: rings relative to outer ring
        ax = axs[i, 2]
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        for key, lab in CATS:
            s = cells[pick_cat(cells.catalogue, key) & (cells.north == north)]
            mr = s.groupby("ring", observed=True).density.mean().reindex(RLAB)
            rel = mr / mr.iloc[0]
            print("  %s ring ratios %-10s %s" % (hemi, lab, [round(v, 2) for v in rel.values]), flush=True)
            ax.plot(range(5), rel.values, marker="o", lw=2, ms=5.5, color=C[key], label=lab)
        ax.set_xticks(range(5))
        ax.set_xticklabels(RLAB)
        ax.set_ylim(0.6, 2.4)
        ax.set_xlabel("latitude ring (towards the pole →)", fontsize=9, color=MUTED)
        ax.set_ylabel("density ÷ density of the 65–70° ring", fontsize=9, color=MUTED)
        ax.set_title("%s cap: equal-area 20 km cells" % hemi, fontsize=10, color=INK)
        style(ax)
    cb = fig.colorbar(im, ax=axs[:, 0], orientation="horizontal", fraction=0.04, pad=0.08, extend="max")
    cb.set_label("craters per 1,000 km²", fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("Crater density by latitude: a steady rise towards the north pole, no consistent rise towards the south pole", fontsize=12, color=INK, y=0.97)
    fig.savefig(FIG + "/map05_latitude_profile.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved map05_latitude_profile.png", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()