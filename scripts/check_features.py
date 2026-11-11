"""Checks on data/processed/craters_features.parquet (no recomputation)."""
import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/lunargrid/data/processed")
f = pd.read_parquet(BASE + "/craters_features.parquet")
print("===== RESULTS =====")
print("rows %d | columns %s" % (len(f), list(f.columns)))

# check 1: elevation against the earlier single-machine file
old = pd.read_parquet(BASE + "/craters_with_elev.parquet")[["crater_id", "elev_m"]].rename(columns={"elev_m": "elev_old"})
m = f[["crater_id", "elev_m"]].merge(old, on="crater_id")
d = (m.elev_m - m.elev_old).abs()
print("\ncheck 1: elevation against craters_with_elev.parquet")
print("  joined %d | max |difference| %.3f m | differing by > 0.01 m: %d | one-sided missing: %d" %
      (len(m), np.nanmax(d), int((d > 0.01).sum()), int((m.elev_m.isna() != m.elev_old.isna()).sum())))

# check 2: crater slope by size and latitude
g = f[f.slope_disc_deg.notna()].copy()
g["size"] = pd.cut(g.diam_km, [0, 2, 5, 10, 20, 50.1], labels=["1-2", "2-5", "5-10", "10-20", "20-50"], right=False)
print("\ncheck 2: slope over the crater disc (degrees) by diameter (km)")
print(g.groupby("size", observed=True).slope_disc_deg.agg(
    ["count", "median", lambda s: s.quantile(.1), lambda s: s.quantile(.9)]).round(2).rename(
    columns={"<lambda_0>": "p10", "<lambda_1>": "p90"}).to_string())
g["latband"] = pd.cut(g.lat.abs(), [0, 30, 60, 70, 80.01], labels=["0-30", "30-60", "60-70", "70-80"], right=False)
print("\nmedian slope by latitude band (disc and centre pixel):")
print(g.groupby("latband", observed=True).agg(disc=("slope_disc_deg", "median"),
                                              centre=("slope_centre_deg", "median"), n=("lat", "size")).round(2).to_string())
print("\nSpearman(slope_disc, slope_centre) = %.3f" % g.slope_disc_deg.corr(g.slope_centre_deg, method="spearman"))
print("stride used: %s" % g.stride.value_counts().sort_index().to_dict())

# check 3: tile-level consistency with the tile slope table
tl = pd.read_parquet(BASE + "/tiles_terrain.parquet")[["tile_id", "mean_slope_deg"]]
g["tile_id"] = np.floor((g.lat + 90) / 5).astype(int) * 72 + np.floor((g.lon + 180) / 5).astype(int)
tm = g.groupby("tile_id").slope_disc_deg.agg(["mean", "size"]).reset_index().merge(tl, on="tile_id").dropna()
tm = tm[tm["size"] >= 20]
print("\ncheck 3: tiles with >= 20 craters: %d" % len(tm))
print("  Spearman(tile mean of crater slope, tile mean slope) = %.3f" % tm["mean"].corr(tm.mean_slope_deg, method="spearman"))
print("  mean over tiles: crater disc slope %.2f deg | tile slope %.2f deg" % (tm["mean"].mean(), tm.mean_slope_deg.mean()))

# polar part
p = f[f.sunlight_240m.notna()]
print("\npolar craters with sunlight: %d | with slope: %d | in shadow (centre, >0.5): %.2f %%" %
      (len(p), p.slope_disc_deg.notna().sum(), 100 * (p.shadow_240m > 0.5).mean()))