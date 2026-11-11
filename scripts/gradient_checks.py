import os
import numpy as np
import pandas as pd
P = os.path.expanduser("~/lunargrid/data/processed")
def log(*a): print(*a, flush=True)
t = pd.read_csv(P + "/tiles_gistar.csv")
t["side"] = np.where(t.lonc.abs() < 90, "near", "far")
t["hemi"] = np.where(t.latc > 0, "N", "S")
t["band"] = pd.cut(t.latc.abs(), [0, 30, 60, 70, 80, 90], right=False)
log("RESULTS (pooled density = craters per 1000 km^2; columns: side, hemisphere)")
for thr in (1, 2):
    for c in ["robbins", "lu5m", "luc"]:
        g = t.groupby(["band", "side", "hemi"], observed=True).agg(n=(f"n_ge{thr}_{c}", "sum"), a=("area_km2", "sum"))
        g["dens"] = g.n / g.a * 1000
        tab = g.dens.unstack(["side", "hemi"])
        log(f"\n--- {c}, D >= {thr} km ---")
        log(tab.round(1).to_string())
        log("pole-cap / equatorial-band ratio: " + ", ".join(f"{a}-{b}: {tab.iloc[-1][(a, b)] / tab.iloc[0][(a, b)]:.2f}" for a, b in tab.columns))
log("\nDONE")
