import os, time
import numpy as np
import pandas as pd

base = os.path.expanduser("~/lunargrid/data")
LU5M = base + "/lu5m812tgt/Moon_Diamond_catalog_4.csv"
FULL = base + "/LUC-GRAS200_crater_data/LUC-GRAS200_full/LUC-GRAS200_full.csv"
SEC  = base + "/LUC-GRAS200_crater_data/LUC-GRAS200_secondary_le2km/LUC-GRAS200_secondary_le2km.csv"

def log(*a):
    print(*a, flush=True)

def quant(name, s):
    q = s.quantile([0, .01, .25, .5, .75, .99, 1]).values
    log(f"  {name:<12} min={q[0]:.4g} p1={q[1]:.4g} p25={q[2]:.4g} median={q[3]:.4g} p75={q[4]:.4g} p99={q[5]:.4g} max={q[6]:.4g}")

def thresholds(label, d_km, lat):
    log(f"  counts using {label}:")
    alat = np.abs(lat)
    for t in [0.2, 0.4, 1, 2, 5, 20]:
        m = d_km >= t
        log(f"    D>={t:>4} km: {m.sum():>10,}   |lat|>=65: {(m & (alat >= 65)).sum():>9,}   |lat|>=80: {(m & (alat >= 80)).sum():>8,}")

t0 = time.time()

# ---------- LU5M812TGT ----------
log("=== LU5M812TGT ===")
a = pd.read_csv(LU5M, usecols=["Longitude", "Latitude", "Diameter_h", "Diameter_w", "Confidence"], encoding="utf-8-sig")
log(f"  rows: {len(a):,}   read in {time.time()-t0:.0f} s")
log(f"  longitude range: {a.Longitude.min():.3f} to {a.Longitude.max():.3f}   share > 180: {(a.Longitude > 180).mean():.3f}")
log(f"  latitude range:  {a.Latitude.min():.3f} to {a.Latitude.max():.3f}")
quant("Diameter_h", a.Diameter_h)
quant("Diameter_w", a.Diameter_w)
quant("h / w", a.Diameter_h / a.Diameter_w)
quant("Confidence", a.Confidence)
log(f"  Diameter_h > Diameter_w for {(a.Diameter_h > a.Diameter_w).mean()*100:.1f}% of rows")
thresholds("Diameter_h", a.Diameter_h, a.Latitude)
thresholds("Diameter_w", a.Diameter_w, a.Latitude)
for c in [0.3, 0.5, 0.7, 0.9]:
    m = a.Confidence >= c
    log(f"  Confidence>={c}: {m.sum():>10,} rows; of those Diameter_h>=0.4: {(m & (a.Diameter_h >= 0.4)).sum():,}")
del a

# ---------- LUC-GRAS200 full ----------
log("")
log("=== LUC-GRAS200 full ===")
cols = ["crater_id", "latitude", "longitude", "diameter_m", "confidence_level", "secondary_flag", "flag_resolution_boundary"]
f = pd.read_csv(FULL, usecols=cols, encoding="utf-8-sig")
log(f"  rows: {len(f):,}   elapsed {time.time()-t0:.0f} s")
log(f"  longitude range: {f.longitude.min():.3f} to {f.longitude.max():.3f}")
log(f"  latitude range:  {f.latitude.min():.3f} to {f.latitude.max():.3f}")
f["d_km"] = f.diameter_m / 1000.0
quant("d_km", f.d_km)
thresholds("diameter_m/1000", f.d_km, f.latitude)
log("  confidence_level:\n" + f.confidence_level.value_counts().to_string())
log("  flag_resolution_boundary:\n" + f.flag_resolution_boundary.value_counts().to_string())
log("  secondary_flag (top values):\n" + f.secondary_flag.value_counts().head(8).to_string())
f["idn"] = f.crater_id.str.slice(12).astype(np.int64)
log(f"  id range: {f.idn.min():,} to {f.idn.max():,}   unique ids: {f.idn.nunique():,}")
n_le2 = int((f.diameter_m <= 2000).sum())
log(f"  rows with diameter <= 2000 m: {n_le2:,}   rows above: {len(f)-n_le2:,}")
full_ids = set(f.idn.values)
full_small_ids = f.idn[f.diameter_m <= 2000].values
del f

# ---------- LUC-GRAS200 secondary file ----------
log("")
log("=== LUC-GRAS200 secondary_le2km ===")
cols = ["crater_id", "diameter_m", "confidence_level", "secondary_flag", "secondary_label", "secondary_confidence", "flag_resolution_boundary"]
s = pd.read_csv(SEC, usecols=cols, encoding="utf-8-sig")
log(f"  rows: {len(s):,}   elapsed {time.time()-t0:.0f} s")
quant("d_km", s.diameter_m / 1000.0)
log("  secondary_label:\n" + s.secondary_label.value_counts().to_string())
log("  confidence_level:\n" + s.confidence_level.value_counts().to_string())
quant("sec_conf", s.secondary_confidence)
s["idn"] = s.crater_id.str.slice(12).astype(np.int64)
sec_ids = set(s.idn.values)
log("")
log("=== relationship between the two LUC files ===")
log(f"  secondary ids found in full file: {len(sec_ids & full_ids):,} of {len(sec_ids):,}")
log(f"  full-file craters <=2000 m: {len(full_small_ids):,}; of these present in secondary file: {len(sec_ids & set(full_small_ids)):,}")
log(f"  secondary id range: {s.idn.min():,} to {s.idn.max():,}")
log(f"DONE in {time.time()-t0:.0f} s")
