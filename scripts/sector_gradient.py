import os
import numpy as np
import pandas as pd
P = os.path.expanduser("~/lunargrid/data/processed")
d = pd.read_csv(P + "/polar_cells_gistar.csv")
edges = [(-180, -90), (-90, 0), (0, 90), (90, 180)]
d["sector"] = pd.cut(d.lon, [-180.001, -90, 0, 90, 180.001], labels=[f"{a}..{b}" for a, b in edges])
d["ring"] = pd.cut(d.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False)
d["grp"] = np.where(d.lat.abs() < 75, "outer", "inner")
print("RESULTS (pooled density per 1000 km^2; ratio = inner 75-90 / outer 65-75)")
for pole in ("south", "north"):
    for thr in (1, 2):
        sub = d[(d.pole == pole) & (d.thr_km == thr)]
        rows = []
        for sec in sub.sector.cat.categories:
            row = {"sector (lon)": sec}
            for c, short in (("robbins", "Rob"), ("lu5m", "LU5M"), ("luc", "LUC")):
                s = sub[(sub.catalogue == c) & (sub.sector == sec)]
                dens = {g: s[s.grp == g]["count"].sum() / s[s.grp == g].area_km2.sum() * 1000 for g in ("outer", "inner")}
                row[f"{short} outer"] = round(dens["outer"], 1); row[f"{short} inner"] = round(dens["inner"], 1)
                row[f"{short} ratio"] = round(dens["inner"] / dens["outer"], 2)
            rows.append(row)
        print(f"\n--- {pole} cap, D >= {thr} km ---")
        print(pd.DataFrame(rows).to_string(index=False))
print("\nDONE")
