import os
import pandas as pd

f = os.path.expanduser(
    "~/lunargrid/data/craters/lunar_crater_database_robbins_2018_bundle/data/"
    "lunar_crater_database_robbins_2018.csv")
use = ["CRATER_ID", "LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"]
df = pd.read_csv(f, usecols=use)

print("rows:", len(df))
print(df[use[1:]].describe().T[["min", "max", "mean"]])
print("missing values per column:")
print(df.isna().sum())
print("duplicate crater IDs:", df.CRATER_ID.duplicated().sum())
print("diameter percentiles (1%, 50%, 99%):")
print(df.DIAM_CIRC_IMG.quantile([0.01, 0.5, 0.99]))
print("longitudes below 0:", (df.LON_CIRC_IMG < 0).sum(),
      "| above 180:", (df.LON_CIRC_IMG > 180).sum())
