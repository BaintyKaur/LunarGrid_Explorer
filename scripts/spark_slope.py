import os
import sys
import time

import numpy as np

TILE_DEG = 5
SLOPE_MAX_LAT = 80.0


def band_slope(p, lat_deg, px):
    """Horn slope (degrees) for the middle rows of p, with the latitude correction.
    p: float32 (n+2, ncols) in metres; lat_deg: latitude of each middle row."""
    a, b, c = p[:-2], p[1:-1], p[2:]
    dx = (px * np.cos(np.radians(lat_deg))).astype("float32")[:, None]
    t = a + 2 * b + c
    dzdx = (np.roll(t, -1, axis=1) - np.roll(t, 1, axis=1)) / (8 * dx)

    def rowfilter(v):
        return np.roll(v, 1, axis=1) + 2 * v + np.roll(v, -1, axis=1)

    dzdy = (rowfilter(c) - rowfilter(a)) / (8 * px)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def band_sums(read_rows, bi, nrows, ncols, px, nodata, step, tcols, chunk):
    """Sum and count of slope per tile for latitude band bi (0 = northernmost).
    read_rows(a, b) returns the raw stored values of DEM rows a..b-1.
    Works on `chunk` rows at a time so memory stays small.
    Returns (sums, counts) with one entry per tile column, or None if the band is beyond 80 degrees."""
    nbands = nrows // step
    ntc = ncols // tcols
    lat_lo = -90 + (nbands - 1 - bi) * TILE_DEG
    lat_hi = lat_lo + TILE_DEG
    if max(abs(lat_lo), abs(lat_hi)) > SLOPE_MAX_LAT:
        return None
    deg_per_row = 180.0 / nrows
    r0, r1 = bi * step, (bi + 1) * step
    sums = np.zeros(ntc)
    counts = np.zeros(ntc)
    for s0 in range(r0, r1, chunk):
        s1 = min(s0 + chunk, r1)
        a, b = max(s0 - 1, 0), min(s1 + 1, nrows)
        raw = read_rows(a, b)
        if s0 == 0:                      # top of the Moon: repeat the first row
            raw = np.vstack([raw[:1], raw])
        if s1 == nrows:                  # bottom of the Moon: repeat the last row
            raw = np.vstack([raw, raw[-1:]])
        p = raw.astype("float32")
        p[raw == nodata] = np.nan
        p *= 0.5
        lat_rows = 90.0 - (np.arange(s0, s1) + 0.5) * deg_per_row
        sl = band_slope(p, lat_rows, px).reshape(s1 - s0, ntc, tcols)
        sums += np.nansum(sl, axis=(0, 2), dtype=np.float64)
        counts += np.sum(~np.isnan(sl), axis=(0, 2))
    return sums, counts


def process_band(bi):
    """Runs inside a Spark worker: open the DEM, compute one band."""
    import rasterio
    path = os.path.expanduser("~/lunargrid/data/dem/ldem_118m.tif")
    with rasterio.open(path) as src:
        px, nodata = src.transform.a, src.nodata
        nrows, ncols = src.height, src.width

        def read_rows(a, b):
            return src.read(1, window=((a, b), (0, ncols)))

        res = band_sums(read_rows, bi, nrows, ncols, px, nodata, 1280, 1280, 320)
    nbands = nrows // 1280
    lat_lo_idx = nbands - 1 - bi
    out = []
    for cj in range(ncols // 1280):
        if res is None:
            out.append((lat_lo_idx * 72 + cj, None))
        else:
            out.append((lat_lo_idx * 72 + cj, float(res[0][cj] / res[1][cj])))
    return out


def main():
    t_start = time.time()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession
    base = os.path.expanduser("~/lunargrid")
    spark = (SparkSession.builder.appName("lunargrid-slope").master("local[16]")
             .config("spark.driver.memory", "16g")
             .config("spark.local.dir", base + "/spark-tmp").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark ready in %.1f s" % (time.time() - t_start), flush=True)

    t0 = time.time()
    rows = spark.sparkContext.parallelize(range(36), 36).flatMap(process_band).collect()
    t_slope = time.time() - t0
    print("slope for all bands took %.1f s" % t_slope, flush=True)

    import pandas as pd
    sp = pd.DataFrame(rows, columns=["tile_id", "mean_slope_spark"]).sort_values("tile_id").reset_index(drop=True)
    sp.to_parquet(base + "/data/processed/tiles_slope_spark.parquet", index=False)
    bt = pd.read_parquet(base + "/data/processed/tiles_terrain.parquet")[["tile_id", "mean_slope_deg"]]
    m = bt.merge(sp, on="tile_id")
    both = m.dropna()
    d = (both.mean_slope_deg - both.mean_slope_spark).abs()
    print("tiles compared:", len(m))
    print("tiles with slope: baseline %d | spark %d" % (m.mean_slope_deg.notna().sum(), m.mean_slope_spark.notna().sum()))
    print("tiles where only one has slope:", int((m.mean_slope_deg.notna() != m.mean_slope_spark.notna()).sum()))
    print("max |spark - baseline| mean slope (degrees): %.6f" % d.max())
    print("tiles differing by more than 0.001 degrees:", int((d > 0.001).sum()))
    print("TOTAL script time %.1f s" % (time.time() - t_start))
    spark.stop()


if __name__ == "__main__":
    main()
