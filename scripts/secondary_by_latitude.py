import os
import numpy as np
import pandas as pd
P = os.path.expanduser("~/lunargrid/data/processed")
d = pd.read_parquet(P + "/craters_luc.parquet", columns=["lat", "diam_km", "secondary_flag", "flag_resolution_boundary"])
d = d[(d.diam_km <= 2.0) & (d.secondary_flag >= 0)].copy()
d["size"] = pd.cut(d.diam_km, [0.19, 0.5, 1.0, 2.0001], labels=["0.2-0.5 km", "0.5-1 km", "1-2 km"])
d["band"] = pd.cut(d.lat.abs(), [0, 30, 60, 65, 70, 75, 80, 90.01], right=False)
d["possible"] = d.secondary_flag == 1
d["prob_cert"] = d.secondary_flag >= 2
print("RESULTS (percent of craters labelled secondary, among craters with D <= 2 km)")
for s in ["0.2-0.5 km", "0.5-1 km", "1-2 km"]:
    g = d[d["size"] == s].groupby("band", observed=True).agg(n=("lat", "size"), pct_possible=("possible", "mean"), pct_prob_or_certain=("prob_cert", "mean"))
    g[["pct_possible", "pct_prob_or_certain"]] *= 100
    print(f"\n--- size class {s}, by |latitude| ---")
    print(g.round(2).to_string())
g = d.groupby(["size", "flag_resolution_boundary"], observed=True).agg(n=("lat", "size"), pct_possible=("possible", "mean"), pct_prob_or_certain=("prob_cert", "mean"))
g[["pct_possible", "pct_prob_or_certain"]] *= 100
print("\n--- by LUC resolution flag (0 standard, 1 transition 65-75, 2 polar >70) ---")
print(g.round(2).to_string())
print("\nDONE")
