import os
import numpy as np
import pandas as pd
P = os.path.expanduser("~/lunargrid/data/processed")
d = pd.read_csv(P + "/polar_cells_gistar.csv")
d["lonbin"] = np.minimum(np.floor((d.lon + 180) / 30).astype(int), 11)
d["group"] = np.where(d.lat.abs() < 75, "outer 65-75", "inner 75-90")
labels = [f"{-180 + 30*i:>4}..{-150 + 30*i:>4}" for i in range(12)]
print("RESULTS (density in each 30-degree longitude bin / density of the whole group; 1.00 = average)")
for pole in ("south", "north"):
    for thr in (1, 2):
        sub = d[(d.pole == pole) & (d.thr_km == thr)]
        cols = {}
        for g in ("outer 65-75", "inner 75-90"):
            for c, short in (("robbins", "Rob"), ("lu5m", "LU5M"), ("luc", "LUC")):
                s = sub[(sub.group == g) & (sub.catalogue == c)]
                tot = s["count"].sum() / s.area_km2.sum()
                b = s.groupby("lonbin").agg(n=("count", "sum"), a=("area_km2", "sum"))
                cols[f"{g[:5]} {short}"] = (b.n / b.a / tot).reindex(range(12))
        t = pd.DataFrame(cols)
        t.index = labels
        print(f"\n--- {pole} cap, D >= {thr} km ---")
        print(t.round(2).to_string())
print("\nDONE")
