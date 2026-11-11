"""Map plan, step 0: list the tables and maps we can draw from (no computation, a few seconds)."""
import os, glob
import pandas as pd
import rasterio

B = os.path.expanduser("~/lunargrid/data")
print("===== RESULTS =====")
for p in sorted(glob.glob(B + "/processed/*")):
    name = os.path.basename(p)
    size = os.path.getsize(p) / 1e6 if os.path.isfile(p) else 0
    line = "%-40s %8.1f MB" % (name, size)
    try:
        if name.endswith(".parquet") and os.path.isfile(p):
            d = pd.read_parquet(p)
            line += "  rows %d  cols %s" % (len(d), list(d.columns))
        elif name.endswith(".csv"):
            d = pd.read_csv(p, nrows=3)
            line += "  cols %s" % list(d.columns)
    except Exception as e:
        line += "  (could not open: %s)" % e
    print(line)
print("\nrasters:")
for p in sorted(glob.glob(B + "/dem/*.tif") + glob.glob(B + "/polar/*.TIF") + glob.glob(B + "/processed/*.tif")):
    with rasterio.open(p) as s:
        print("%-40s %d x %d  dtype %s  scale %s  nodata %s" % (os.path.basename(p), s.width, s.height, s.dtypes[0], s.scales[0], s.nodata))