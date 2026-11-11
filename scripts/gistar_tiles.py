import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

P = os.path.expanduser("~/lunargrid/data/processed")
R = 1737.4  # km
def log(*a): print(*a, flush=True)

t = pd.read_csv(P + "/tiles_catalogue_counts.csv")
row, col = t.tile_id // 72, t.tile_id % 72
lat0 = -90 + 5 * row
latc, lonc = lat0 + 2.5, -180 + 5 * col + 2.5
area = R**2 * np.radians(5) * (np.sin(np.radians(lat0 + 5)) - np.sin(np.radians(lat0)))   # km^2
t["latc"], t["lonc"], t["area_km2"] = latc, lonc, area

la, lo = np.radians(latc.values), np.radians(lonc.values)
xyz = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
tree = cKDTree(xyz)
chord = 2 * np.sin(np.radians(10) / 2)                  # neighbours within 10 deg of arc
nbr = tree.query_ball_point(xyz, r=chord)               # includes the tile itself
n = len(t)
rows = np.repeat(np.arange(n), [len(x) for x in nbr]); cols = np.concatenate([np.array(x) for x in nbr])
W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
sumw = np.asarray(W.sum(axis=1)).ravel()

def gistar(x):
    xbar, s = x.mean(), np.sqrt((x**2).mean() - x.mean()**2)
    wx = W @ x
    return (wx - xbar * sumw) / (s * np.sqrt((n * sumw - sumw**2) / (n - 1)))

log("RESULTS")
log(f"tiles {n}; neighbours per tile: min {int(sumw.min())}, median {int(np.median(sumw))}, max {int(sumw.max())}")
bands = pd.cut(t.latc.abs(), [0, 30, 60, 70, 80, 90], right=False)
cats = ["robbins", "lu5m", "luc"]
for thr in (1, 2):
    log(f"\n=============== craters with D >= {thr} km ===============")
    for c in cats:
        dens = t[f"n_ge{thr}_{c}"].values / area * 1000.0          # per 1000 km^2
        t[f"dens{thr}_{c}"] = dens
        t[f"z{thr}_{c}"] = gistar(dens)
        z = t[f"z{thr}_{c}"]
        log(f"\n  {c}: overall density {t[f'n_ge{thr}_{c}'].sum() / area.sum() * 1000:.2f} per 1000 km^2;  hot tiles (z>2.58): {(z>2.58).sum()},  cold tiles (z<-2.58): {(z<-2.58).sum()}")
        g = pd.DataFrame({"band": bands, "dens": dens, "z": z, "hot": z > 2.58, "cold": z < -2.58}).groupby("band", observed=True).agg(
            tiles=("z", "size"), median_density=("dens", "median"), mean_z=("z", "mean"), share_hot=("hot", "mean"), share_cold=("cold", "mean"))
        log("    " + g.round(3).to_string().replace("\n", "\n    "))
    zz = t[[f"z{thr}_{c}" for c in cats]]; zz.columns = cats
    log(f"\n  correlation of Gi* z-scores between catalogues (D >= {thr} km):")
    log("    " + zz.corr().round(3).to_string().replace("\n", "\n    "))
    if thr == 2:
        for c in cats:
            top = t.sort_values(f"z{thr}_{c}", ascending=False).head(5)[["tile_id", "latc", "lonc", f"n_ge{thr}_{c}", f"dens{thr}_{c}", f"z{thr}_{c}"]]
            log(f"\n  top 5 hot tiles, {c}, D >= 2 km:")
            log("    " + top.round(2).to_string(index=False).replace("\n", "\n    "))
t.to_csv(P + "/tiles_gistar.csv", index=False)
log("\nDONE (saved tiles_gistar.csv)")
