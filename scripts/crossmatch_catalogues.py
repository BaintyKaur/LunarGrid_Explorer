import os, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

base = os.path.expanduser("~/lunargrid/data/processed")
R = 1737.4  # km
K = 16
def log(*a): print(*a, flush=True)
def wrap(lon): return ((lon + 180.0) % 360.0) - 180.0
def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
def pick(df, names, what):
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"cannot find {what} column; columns are: {list(df.columns)}")

t0 = time.time()
rob = pd.read_parquet(base + "/craters_clean.parquet")
latc = pick(rob, ["lat", "latitude", "LAT_CIRC_IMG", "lat_deg"], "latitude")
lonc = pick(rob, ["lon", "longitude", "LON_CIRC_IMG", "lon_deg", "lon_e"], "longitude")
dc = pick(rob, ["diam_km", "DIAM_CIRC_IMG", "diameter_km"], "diameter")
log(f"using lat={latc}, lon={lonc}, diameter={dc}")
rob = rob[rob[dc] >= 1.0].reset_index(drop=True)
rlon, rlat, rd = wrap(rob[lonc].values.astype(float)), rob[latc].values.astype(float), rob[dc].values.astype(float)
N = len(rob)
log(f"Robbins craters >= 1 km: {N:,}")
rx = xyz(rlon, rlat)
bins = [(1, 2), (2, 5), (5, 20), (20, 10000)]

def run(name, path, variants, size_col):
    df = pd.read_parquet(path)
    tree = cKDTree(xyz(df.lon.values, df.lat.values))
    chord, idx = tree.query(rx, k=K)
    dist_km = 2 * R * np.arcsin(np.clip(chord / 2, 0, 1))
    dsz = df[size_col].values[idx]
    cand = (dist_km <= 0.25 * rd[:, None]) & (dsz >= 0.5 * rd[:, None]) & (dsz <= 2.0 * rd[:, None])
    has = cand.any(axis=1)
    first = cand.argmax(axis=1)
    rows = np.arange(N)
    sel = idx[rows, first]
    dsel = dist_km[rows, first]
    log(f"\n=== {name} ({len(df):,} craters)   [{time.time()-t0:.0f} s] ===")
    for lo, hi in bins:
        b = (rd >= lo) & (rd < hi)
        m = b & has
        if b.sum() == 0:
            continue
        label = f"{lo}-{hi} km" if hi < 10000 else f">= {lo} km"
        off = np.median(dsel[m] / rd[m]) * 100 if m.sum() else float("nan")
        log(f"  Robbins {label:<9} n={b.sum():>8,}  matched {m.sum():>8,} ({m.sum()/b.sum()*100:5.1f}%)  median centre offset {off:4.1f}% of D")
        for vname, col in variants:
            ratio = df[col].values[sel[m]] / rd[m]
            if len(ratio):
                p25, p50, p75 = np.percentile(ratio, [25, 50, 75])
                log(f"       diameter {vname:<10} / Robbins: median {p50:.3f}   IQR {p25:.3f}-{p75:.3f}")

run("LU5M812TGT (window uses sqrt(h*w))", base + "/craters_lu5m.parquet",
    [("h", "diam_h_km"), ("w", "diam_w_km"), ("sqrt(h*w)", "diam_km")], "diam_km")
run("LUC-GRAS200", base + "/craters_luc.parquet", [("diam_km", "diam_km")], "diam_km")
log(f"\nDONE {time.time()-t0:.0f} s")
