import os
import numpy as np
import pandas as pd
P = os.path.expanduser("~/lunargrid/data/processed")
def log(*a): print(*a, flush=True)
log("RESULTS")
st = pd.read_parquet(P + "/tiles_stats_spark.parquet")
log("columns of tiles_stats_spark:", list(st.columns))
if "tile_id" not in st.columns:
    raise SystemExit("no tile_id column in tiles_stats_spark; tell me the column names above")
cand = [c for c in st.columns if "elev" in c.lower() and any(k in c.lower() for k in ("mean", "avg"))]
if not cand:
    cand = [c for c in st.columns if any(k in c.lower() for k in ("mean", "avg"))]
if not cand:
    raise SystemExit("could not find a mean-elevation column; tell me the column names above")
ecol = cand[0]
log("using elevation column:", ecol, "(candidates:", cand, ")")
t = pd.read_csv(P + "/tiles_gistar.csv").merge(st[["tile_id", ecol]], on="tile_id", how="left")
log("tiles without elevation:", int(t[ecol].isna().sum()))
q = t[ecol].quantile([1/3, 2/3]).values
log(f"elevation terciles of tile means: {q[0]:.0f} m and {q[1]:.0f} m")
t["elev_class"] = pd.cut(t[ecol], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
t["band"] = pd.cut(t.latc.abs(), [0, 30, 60, 70, 80, 90], right=False)
log("\ntiles per band and elevation class:")
log(pd.crosstab(t.band, t.elev_class).to_string())
for thr in (1, 2):
    for c in ["robbins", "lu5m", "luc"]:
        g = t.groupby(["band", "elev_class"], observed=True).agg(n=(f"n_ge{thr}_{c}", "sum"), a=("area_km2", "sum"))
        g["dens"] = g.n / g.a * 1000
        tab = g.dens.unstack("elev_class")
        log(f"\n--- {c}, D >= {thr} km: pooled density per 1000 km^2 by band and elevation class ---")
        log(tab.round(1).to_string())
        log("pole-cap / equatorial-band ratio: " + ", ".join(f"{k}: {tab.iloc[-1][k] / tab.iloc[0][k]:.2f}" for k in tab.columns))
log("\nDONE")
