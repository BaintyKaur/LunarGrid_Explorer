import os, time
import numpy as np
import pandas as pd

base = os.path.expanduser("~/lunargrid/data")
out = base + "/processed"
LU5M = base + "/lu5m812tgt/Moon_Diamond_catalog_4.csv"
FULL = base + "/LUC-GRAS200_crater_data/LUC-GRAS200_full/LUC-GRAS200_full.csv"
SEC  = base + "/LUC-GRAS200_crater_data/LUC-GRAS200_secondary_le2km/LUC-GRAS200_secondary_le2km.csv"
def log(*a): print(*a, flush=True)
def wrap(lon): return ((lon + 180.0) % 360.0) - 180.0
t0 = time.time()

# ---------- LU5M812TGT ----------
a = pd.read_csv(LU5M, usecols=["Longitude", "Latitude", "Diameter_h", "Diameter_w", "Confidence"], encoding="utf-8-sig")
n0 = len(a)
a = a.dropna(subset=["Longitude", "Latitude"]).reset_index(drop=True)
log(f"LU5M rows {n0:,}; after dropping missing coordinates {len(a):,}")
lon_raw = a.Longitude.values
w = wrap(lon_raw)
flag = (lon_raw < -180) | (lon_raw > 180)
r = pd.DataFrame({
    "cid": ("LU5M_" + pd.Series(np.arange(len(a))).astype(str)).values,
    "lon": w, "lat": a.Latitude.values,
    "diam_km": np.sqrt(a.Diameter_h.values * a.Diameter_w.values),   # provisional rule
    "diam_h_km": a.Diameter_h.values.astype(np.float32),
    "diam_w_km": a.Diameter_w.values.astype(np.float32),
    "hw_ratio": (a.Diameter_h.values / a.Diameter_w.values).astype(np.float32),
    "confidence": a.Confidence.values.astype(np.float32),
    "lon_wrapped": flag,
})
r["diam_km"] = r.diam_km.astype(np.float32)
r.to_parquet(out + "/craters_lu5m.parquet", index=False)
log(f"  wrapped longitudes: {flag.sum():,}; lon range now {r.lon.min():.3f} to {r.lon.max():.3f}; lat {r.lat.min():.3f} to {r.lat.max():.3f}")
log(f"  extreme boxes (hw_ratio <0.5 or >2): {((r.hw_ratio < 0.5) | (r.hw_ratio > 2)).sum():,}")
log(f"  diam_km median {r.diam_km.median():.3f}, saved craters_lu5m.parquet   [{time.time()-t0:.0f} s]")
del a, r

# ---------- LUC-GRAS200 ----------
cols = ["crater_id", "latitude", "longitude", "diameter_m", "depth_m", "rim_height_m", "slope_mean_deg",
        "slope_max_deg", "d_D_ratio", "confidence_level", "secondary_flag", "flag_resolution_boundary"]
f = pd.read_csv(FULL, usecols=cols, encoding="utf-8-sig")
log(f"LUC rows {len(f):,}")
f = f.dropna(subset=["latitude", "longitude", "diameter_m"]).reset_index(drop=True)
s = pd.read_csv(SEC, usecols=["crater_id", "secondary_confidence"], encoding="utf-8-sig")
f = f.merge(s, on="crater_id", how="left")
log(f"  after merge with secondary file: {len(f):,} rows; secondary_confidence present for {f.secondary_confidence.notna().sum():,}")
lon_raw = f.longitude.values
changed = int((np.abs(wrap(lon_raw) - lon_raw) > 1e-9).sum())
r = pd.DataFrame({
    "cid": f.crater_id.values,
    "lon": wrap(lon_raw), "lat": f.latitude.values,
    "diam_km": (f.diameter_m.values / 1000.0).astype(np.float32),
    "depth_m": f.depth_m.values.astype(np.float32),
    "rim_height_m": f.rim_height_m.values.astype(np.float32),
    "slope_mean_deg": f.slope_mean_deg.values.astype(np.float32),
    "slope_max_deg": f.slope_max_deg.values.astype(np.float32),
    "confidence_level": f.confidence_level.values,
    "secondary_flag": f.secondary_flag.values.astype(np.int8),
    "secondary_confidence": f.secondary_confidence.values.astype(np.float32),
    "flag_resolution_boundary": f.flag_resolution_boundary.values.astype(np.int8),
})
r.to_parquet(out + "/craters_luc.parquet", index=False)
log(f"  longitudes changed by wrapping (180 -> -180): {changed:,}; lon range {r.lon.min():.3f} to {r.lon.max():.3f}")
log(f"  diam_km median {r.diam_km.median():.3f}, saved craters_luc.parquet   [{time.time()-t0:.0f} s]")
log("DONE")
