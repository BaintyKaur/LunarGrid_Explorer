import os, shutil
import numpy as np
import pandas as pd
base = os.path.expanduser("~/lunargrid/data")
p = base + "/processed/craters_lu5m.parquet"
shutil.copy(p, base + "/processed_backup/craters_lu5m_sqrt_version.parquet")
d = pd.read_parquet(p)
d["diam_sqrt_km"] = d["diam_km"].astype(np.float32)      # old provisional rule, kept
d["diam_km"] = d["diam_h_km"].astype(np.float32)          # new rule: Diameter_h
d.to_parquet(p, index=False)
print("rows", len(d), "columns", list(d.columns))
print("diam_km median", round(float(d.diam_km.median()), 4))
print("diam_km >= 1 km:", int((d.diam_km >= 1).sum()), "(expect 1,998,112)")
print("diam_km >= 2 km:", int((d.diam_km >= 2).sum()), "(expect 696,037)")
print("DONE")
