"""Dashboard data, step 1: gather everything the click-on-the-Moon dashboard needs into ONE small file.

Reads : data/processed/tiles_gistar.csv, tiles_terrain.csv, polar_cells_gistar.csv,
        cells_illumination.csv, cells_terrain.csv, perm_hotspots.csv
Writes: data/processed/dashboard_data.json   (about 1-3 MB)
Run   : python export_dashboard_data.py
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/lunargrid/data/processed")
CATS = ["robbins", "lu5m", "luc"]
NC = 77


def clean(v, nd=3):
    """Round, and turn NaN into None so the file is valid JSON."""
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), nd) for x in v]


def main():
    # ---------------- global 5-degree tiles ----------------
    g = pd.read_csv(BASE + "/tiles_gistar.csv")
    t = pd.read_csv(BASE + "/tiles_terrain.csv")[["tile_id", "lat_min", "lat_max", "lon_min", "lon_max",
                                                  "mean_elev_m", "mean_slope_deg"]]
    tiles = g.merge(t, on="tile_id", how="left").sort_values("tile_id")
    cols = ["tile_id", "lat_min", "lat_max", "lon_min", "lon_max", "area_km2", "mean_elev_m", "mean_slope_deg"]
    for c in CATS:
        cols += ["n_ge1_" + c, "n_ge2_" + c, "n_ge5_" + c, "n_ge20_" + c,
                 "dens1_" + c, "dens2_" + c, "z1_" + c, "z2_" + c]
    tiles_out = {c: clean(tiles[c].tolist(), 4) for c in cols}
    print("tiles:", len(tiles), flush=True)

    # ---------------- polar 20 km cells ----------------
    p = pd.read_csv(BASE + "/polar_cells_gistar.csv")
    p["pole"] = p["pole"].astype(str).str.lower().str[0].map({"n": "N", "s": "S"})
    base = p[["pole", "id", "x", "y", "lat", "lon", "area_km2"]].drop_duplicates(["pole", "id"]).copy()
    for c in CATS:
        for thr in (1, 2):
            q = p[(p.catalogue == c) & (p.thr_km == thr)][["pole", "id", "count", "density", "z_sed"]]
            q = q.rename(columns={"count": "n%d_%s" % (thr, c), "density": "d%d_%s" % (thr, c),
                                  "z_sed": "z%d_%s" % (thr, c)})
            base = base.merge(q, on=["pole", "id"], how="left")

    ill = pd.read_csv(BASE + "/cells_illumination.csv")[["pole", "id", "sun_mean", "psr_frac"]]
    ill["pole"] = ill["pole"].astype(str).str.upper().str[0]
    base = base.merge(ill, on=["pole", "id"], how="left")

    ter = pd.read_csv(BASE + "/cells_terrain.csv")
    ter["pole"] = ter["pole"].astype(str).str.upper().str[0]
    ter["id"] = ter["iy"] * NC + ter["ix"]
    base = base.merge(ter[["pole", "id", "elev_mean_m", "slope_mean_deg"]], on=["pole", "id"], how="left")

    pm = pd.read_csv(BASE + "/perm_hotspots.csv")
    pm["pole"] = pm["pole"].astype(str).str.upper().str[0]
    for c in CATS:
        for thr in (1, 2):
            for nul, short in (("A_ring", "pA"), ("B_ring_sun", "pB")):
                q = pm[(pm.catalogue == c) & (pm.thr_km == thr) & (pm["null"] == nul)][["pole", "id", "p_maxT"]]
                q = q.rename(columns={"p_maxT": "%s%d_%s" % (short, thr, c)})
                base = base.merge(q, on=["pole", "id"], how="left")

    base["ix"] = base["id"] % NC
    base["iy"] = base["id"] // NC
    base = base.sort_values(["pole", "id"])
    ccols = [c for c in base.columns if c != "pole"]
    cells_out = {"pole": base["pole"].tolist()}
    for c in ccols:
        cells_out[c] = clean(base[c].tolist(), 4)
    print("cells:", len(base), "| columns:", len(base.columns), flush=True)

    out = BASE + "/dashboard_data.json"
    with open(out, "w") as f:
        json.dump({"tiles": tiles_out, "cells": cells_out, "nc": NC, "cell_m": 20000, "half_m": 770000}, f,
                  separators=(",", ":"))
    print("saved:", out, "| %.2f MB" % (os.path.getsize(out) / 1e6), flush=True)


if __name__ == "__main__":
    main()