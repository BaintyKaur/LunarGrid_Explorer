import os
import numpy as np
import pandas as pd

base = os.path.expanduser("~/lunargrid/data")
src = (base + "/craters/lunar_crater_database_robbins_2018_bundle/data/"
       "lunar_crater_database_robbins_2018.csv")
df = pd.read_csv(src, usecols=["CRATER_ID", "LAT_CIRC_IMG",
                               "LON_CIRC_IMG", "DIAM_CIRC_IMG"])
df = df.rename(columns={"CRATER_ID": "crater_id", "LAT_CIRC_IMG": "lat",
                        "LON_CIRC_IMG": "lon360", "DIAM_CIRC_IMG": "diam_km"})

df["lon"] = ((df["lon360"] + 180.0) % 360.0) - 180.0
df["tile_id"] = (np.floor((df["lat"] + 90) / 5).astype(int) * 72
                 + np.floor((df["lon"] + 180) / 5).astype(int))

os.makedirs(base + "/processed", exist_ok=True)
out = base + "/processed/craters_clean.parquet"
df[["crater_id", "lat", "lon", "diam_km", "tile_id"]].to_parquet(out, index=False)

chk = pd.read_parquet(out)
print("rows written:", len(chk))
print("lon range:", chk.lon.min(), "to", chk.lon.max())
print("tile_id range:", chk.tile_id.min(), "to", chk.tile_id.max(),
      "| distinct tiles:", chk.tile_id.nunique())
print(chk.head(3))
