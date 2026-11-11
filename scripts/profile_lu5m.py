import os, time
import numpy as np
import pandas as pd

P = os.path.expanduser("~/lunargrid/data/lu5m812tgt/Moon_Diamond_catalog_4.csv")
def log(*a): print(*a, flush=True)
t0 = time.time()
a = pd.read_csv(P, usecols=["Longitude", "Latitude", "Diameter_h", "Diameter_w", "Confidence"], encoding="utf-8-sig")
log(f"rows {len(a):,}")

log("\n--- longitude outside -180..180 ---")
lo = a.Longitude
out = (lo < -180) | (lo > 180)
log(f"rows < -180: {(lo < -180).sum():,}   rows > 180: {(lo > 180).sum():,}   total outside: {out.sum():,} ({out.mean()*100:.4f}%)")
log("histogram of longitude (bin edges in degrees):")
edges = [-300, -180, -90, 0, 90, 180, 320]
h, _ = np.histogram(lo, bins=edges)
for i in range(len(h)):
    log(f"  [{edges[i]:>5}, {edges[i+1]:>4}): {h[i]:>10,}")
idx = np.flatnonzero(out.values)
if len(idx):
    log(f"first out-of-range row positions: {idx[:10].tolist()}   last: {idx[-5:].tolist()}")
    log("sample out-of-range rows:")
    log(a.iloc[idx[:5]].to_string())
    log(f"out-of-range rows with |lat|>=80: {(a.Latitude.values[idx].__abs__() >= 80).sum():,}")

log("\n--- by latitude band ---")
a["ratio"] = a.Diameter_h / a.Diameter_w
bands = pd.cut(a.Latitude.abs(), [0, 30, 60, 70, 80, 90], right=False)
g = a.groupby(bands, observed=True).agg(n=("ratio", "size"), med_ratio=("ratio", "median"),
                                         med_h=("Diameter_h", "median"), med_w=("Diameter_w", "median"),
                                         med_conf=("Confidence", "median"))
log(g.to_string())
a["ratio_lat"] = np.where(a.Latitude.abs() < 85, a.ratio * np.cos(np.radians(a.Latitude.abs())), np.nan)
log(f"median h/w times cos(lat), |lat|<85: {a.ratio_lat.median():.4f}  (h/w alone: {a.ratio.median():.4f})")

log("\n--- confidence by diameter (using sqrt(h*w)) ---")
d = np.sqrt(a.Diameter_h * a.Diameter_w)
db = pd.cut(d, [0, 0.5, 0.7, 1, 2, 5, 20, 1000])
g2 = a.groupby(db, observed=True).Confidence.agg(["size", "median", lambda s: (s >= 0.5).mean()])
g2.columns = ["n", "median_conf", "share_conf>=0.5"]
log(g2.to_string())

log("\n--- near-duplicate positions (lon, lat rounded to 0.001 deg) ---")
key = (np.round(a.Longitude.values, 3) * 1000).astype(np.int64) * 1_000_000 + (np.round(a.Latitude.values, 3) * 1000).astype(np.int64)
u = np.unique(key).size
log(f"unique positions {u:,} of {len(a):,}  -> duplicates {len(a)-u:,}")
log(f"DONE {time.time()-t0:.0f} s")
