import os
import numpy as np
import pandas as pd
import rasterio

base = os.path.expanduser("~/lunargrid/data")
R = 1737400.0

df = pd.read_parquet(base + "/processed/craters_clean.parquet")
print("craters loaded:", len(df), flush=True)

with rasterio.open(base + "/dem/ldem_118m.tif") as src:
    t = src.transform
    nodata = src.nodata
    print("reading the DEM into memory ...", flush=True)
    dem = src.read(1)
print("DEM shape:", dem.shape, flush=True)

x = R * np.radians(df["lon"].values)
y = R * np.radians(df["lat"].values)
col = np.clip(np.floor((x - t.c) / t.a).astype(int), 0, dem.shape[1] - 1)
row = np.clip(np.floor((y - t.f) / t.e).astype(int), 0, dem.shape[0] - 1)

raw = dem[row, col].astype("float32")
raw[raw == nodata] = np.nan
df["elev_m"] = raw * 0.5

out = base + "/processed/craters_with_elev.parquet"
df.to_parquet(out, index=False)
print("saved:", out, flush=True)
print("missing elevations:", int(df["elev_m"].isna().sum()))
print(df["elev_m"].describe())
big = df[df["diam_km"] >= 300]
print("craters >= 300 km:", len(big),
      "| median elevation at their centres (m):", float(np.nanmedian(big["elev_m"])))
print("median elevation of all craters (m):", float(np.nanmedian(df["elev_m"])))
