"""Step 1 of the improvements: attach sunlight and permanent shadow to every
20 km polar cell and relate them to crater density.

Reads : data/processed/polar_cells_gistar.csv   (the equal-area cells of Part 15)
        data/polar/AVGVISIB_65{N,S}_240M_201608.TIF  (sunlight fraction)
        data/polar/LPSR_65{N,S}_240M_201608.TIF      (permanent shadow, 1 = shadow)
Writes: data/processed/cells_illumination.csv
"""
import os
import numpy as np
import pandas as pd
import rasterio

BASE = os.path.expanduser("~/lunargrid/data")
CELL = 20000.0       # cell size in metres (same grid as polar_grid_gistar.py)
HALF_G = 770000.0    # cell grid half-width (77 x 77 cells)
HALF_M = 770400.0    # 240 m map half-width
PX = 240.0
N = 6420
RINGS = [65, 70, 75, 80, 85, 90.01]


def load_scaled(path):
    with rasterio.open(path) as src:
        raw = src.read(1)
        scale, offset, nodata = src.scales[0], src.offsets[0], src.nodata
    arr = raw.astype("float32") * np.float32(scale) + np.float32(offset)
    arr[raw == nodata] = np.nan
    return arr


def cell_key(x, y):
    ix = np.floor((x + HALF_G) / CELL).astype(np.int64)
    iy = np.floor((y + HALF_G) / CELL).astype(np.int64)
    return ix, iy


def main():
    cells_all = pd.read_csv(BASE + "/processed/polar_cells_gistar.csv")
    print("columns:", list(cells_all.columns), flush=True)
    print("rows:", len(cells_all), flush=True)

    # one row per cell and pole (geometry does not depend on catalogue)
    cells_all["north"] = cells_all["lat"] > 0
    geo = (cells_all.groupby(["north", "id"], as_index=False)
           .agg(x=("x", "first"), y=("y", "first"), lat=("lat", "first"),
                lon=("lon", "first"), area_km2=("area_km2", "first")))
    print("cells per pole:", geo.groupby("north").size().to_dict(), flush=True)

    # pixel centres of the 240 m maps
    cx = (np.arange(N) + 0.5) * PX - HALF_M
    cy = HALF_M - (np.arange(N) + 0.5) * PX
    X, Y = np.meshgrid(cx, cy)
    ix_p, iy_p = cell_key(X, Y)
    del X, Y

    out = []
    for north, hemi in ((True, "N"), (False, "S")):
        g = geo[geo.north == north].copy()
        ix_c, iy_c = cell_key(g.x.values, g.y.values)
        g["ix"], g["iy"] = ix_c, iy_c
        side = 77
        # map each pixel to a cell index (-1 if outside the cell list)
        lookup = -np.ones((side + 2, side + 2), dtype=np.int64)
        ok_c = (ix_c >= 0) & (ix_c < side) & (iy_c >= 0) & (iy_c < side)
        assert ok_c.all(), "cell centres outside 77 x 77 grid"
        lookup[iy_c, ix_c] = np.arange(len(g))
        inside = (ix_p >= 0) & (ix_p < side) & (iy_p >= 0) & (iy_p < side)
        pix_cell = np.where(inside, lookup[np.clip(iy_p, 0, side), np.clip(ix_p, 0, side)], -1)

        ill = load_scaled(BASE + "/polar/AVGVISIB_65%s_240M_201608.TIF" % hemi)
        psr = load_scaled(BASE + "/polar/LPSR_65%s_240M_201608.TIF" % hemi)
        m = pix_cell >= 0
        cid = pix_cell[m]
        ill_v, psr_v = ill[m], psr[m]
        ok_i, ok_p = ~np.isnan(ill_v), ~np.isnan(psr_v)
        k = len(g)
        n_pix = np.bincount(cid, minlength=k)
        n_i = np.bincount(cid[ok_i], minlength=k)
        n_p = np.bincount(cid[ok_p], minlength=k)
        sun = np.bincount(cid[ok_i], weights=ill_v[ok_i], minlength=k) / np.maximum(n_i, 1)
        shad = np.bincount(cid[ok_p], weights=(psr_v[ok_p] > 0.5).astype(float), minlength=k) / np.maximum(n_p, 1)
        sun_min = np.full(k, np.nan)
        g["sun_mean"] = np.where(n_i > 0, sun, np.nan)
        g["psr_frac"] = np.where(n_p > 0, shad, np.nan)
        g["n_pix"] = n_pix
        g["pole"] = hemi
        print("%s: cells %d, cells without pixels %d, pixels per cell median %d" %
              (hemi, k, int((n_pix == 0).sum()), int(np.median(n_pix))), flush=True)
        print("%s: sun_mean range %.3f..%.3f | cells with psr_frac>0: %d | >0.05: %d | >0.2: %d" %
              (hemi, np.nanmin(g.sun_mean), np.nanmax(g.sun_mean),
               int((g.psr_frac > 0).sum()), int((g.psr_frac > 0.05).sum()), int((g.psr_frac > 0.2).sum())), flush=True)
        out.append(g)

    il = pd.concat(out, ignore_index=True)
    il = il[["pole", "id", "ix", "iy", "sun_mean", "psr_frac", "n_pix"]]
    il.to_csv(BASE + "/processed/cells_illumination.csv", index=False)

    # join to densities
    cells_all["pole"] = np.where(cells_all.north, "N", "S")
    d = cells_all.merge(il, on=["pole", "id"], how="left", validate="many_to_one")
    assert d.sun_mean.notna().all(), "cells without illumination"
    ring = pd.cut(d.lat.abs(), RINGS, right=False, labels=False)
    assert ring.notna().all(), "cell outside ring bins"
    d["ring"] = ring

    print("\n===== RESULTS =====")
    print("Spearman correlation of crater density with sunlight and with permanent-shadow share.")
    print("'within rings' = average of the correlations inside each 5-degree ring (removes the latitude trend).\n")
    rows = []
    for (pole, cat, thr), s in d.groupby(["pole", "catalogue", "thr_km"]):
        r = {"pole": pole, "catalogue": cat, "D>=km": thr}
        r["rho_sun"] = s["density"].corr(s["sun_mean"], method="spearman")
        r["rho_psr"] = s["density"].corr(s["psr_frac"], method="spearman")
        rs, rp = [], []
        for _, q in s.groupby("ring"):
            if len(q) > 30:
                rs.append(q["density"].corr(q["sun_mean"], method="spearman"))
                if q.psr_frac.nunique() > 1:
                    rp.append(q["density"].corr(q["psr_frac"], method="spearman"))
        r["rho_sun_ring"] = np.nanmean(rs)
        r["rho_psr_ring"] = np.nanmean(rp) if rp else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nMean density (per 1000 km2) in cells with and without permanent shadow (psr_frac > 0.05):")
    rows = []
    for (pole, cat, thr), s in d.groupby(["pole", "catalogue", "thr_km"]):
        a, b = s[s.psr_frac > 0.05], s[s.psr_frac <= 0.05]
        rows.append({"pole": pole, "catalogue": cat, "D>=km": thr,
                     "cells_shadow": len(a), "dens_shadow": a.density.mean(),
                     "cells_lit": len(b), "dens_lit": b.density.mean(),
                     "ratio": a.density.mean() / b.density.mean() if len(a) else np.nan})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nSame comparison inside rings (ratio shadow/lit, ring by ring, Robbins D>=1 only):")
    s = d[(d.catalogue == d.catalogue.iloc[0]) & (d.thr_km == d.thr_km.min())]
    print("catalogue used:", s.catalogue.iloc[0], "threshold:", s.thr_km.iloc[0])
    for pole, sp in s.groupby("pole"):
        for rg, q in sp.groupby("ring"):
            a, b = q[q.psr_frac > 0.05], q[q.psr_frac <= 0.05]
            print("%s ring %d: shadow cells %4d lit cells %4d  density %6.1f vs %6.1f" %
                  (pole, rg, len(a), len(b), a.density.mean() if len(a) else np.nan, b.density.mean()))
    print("\nwrote", BASE + "/processed/cells_illumination.csv")


if __name__ == "__main__":
    main()