"""Step 4: one table with one row per crater: size, elevation, slope, sunlight, shadow.

New per-crater slope: the mean Horn slope (degrees, 118 m elevation map, with the
latitude correction used for the tile table) over the crater's own disc (radius
= diameter / 2). For craters wider than about 10 km the disc would be too many
pixels, so we sample every k-th pixel inside it (k = 1 for small craters).
Craters wider than 50 km are skipped (slope left empty), as are craters poleward
of 80 degrees (same limit as the tile slope table).

Reads : data/dem/ldem_118m.tif, data/processed/craters_clean.parquet,
        data/processed/craters_polar.parquet, data/processed/craters_with_elev.parquet,
        data/processed/tiles_terrain.parquet
Writes: data/processed/craters_features.parquet
Spark : one task per block of 128 DEM rows (about 360 tasks) on local[16].
"""
import os
import sys
import time
import threading

import numpy as np

BASE = os.path.expanduser("~/lunargrid")
DEM = BASE + "/data/dem/ldem_118m.tif"
PROGRESS = BASE + "/logs/features_progress.txt"
CH = 128              # DEM rows per task
MAX_D_KM = 50.0       # craters wider than this are skipped
TARGET_R = 40         # sampled radius in pixels (at most)
SLOPE_MAX_LAT = 80.0


def band_slope(p, lat_deg, px):
    """Horn slope (degrees) for the middle rows of p, latitude-corrected (same as spark_slope.py)."""
    a, b, c = p[:-2], p[1:-1], p[2:]
    dx = (px * np.cos(np.radians(lat_deg))).astype("float32")[:, None]
    t = a + 2 * b + c
    dzdx = (np.roll(t, -1, axis=1) - np.roll(t, 1, axis=1)) / (8 * dx)

    def rowfilter(v):
        return np.roll(v, 1, axis=1) + 2 * v + np.roll(v, -1, axis=1)

    dzdy = (rowfilter(c) - rowfilter(a)) / (8 * px)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def disc_offsets(r_full, stride):
    """Row and column offsets of the sampled pixels inside a disc of radius r_full pixels."""
    k = np.arange(-(r_full // stride), r_full // stride + 1) * stride
    dy, dx = np.meshgrid(k, k, indexing="ij")
    m = dy ** 2 + dx ** 2 <= r_full ** 2
    return dy[m], dx[m]


def compute_chunk(raw, a, nrows, ncols, nodata, px, rows, cols, diam_km):
    """raw: stored DEM values for rows a .. a+len(raw)-1 (already padded at the Moon's top/bottom).
    rows, cols: DEM pixel row/column of each crater in this chunk.
    Returns elevation (m), slope over the disc, slope at the centre, r_full (px), stride, all arrays."""
    n = len(rows)
    elev = np.full(n, np.nan)
    s_disc = np.full(n, np.nan)
    s_cen = np.full(n, np.nan)
    r_out = np.zeros(n, dtype=np.int32)
    st_out = np.zeros(n, dtype=np.int32)
    p = raw.astype("float32")
    p[raw == nodata] = np.nan
    p *= 0.5
    deg_per_row = 180.0 / nrows
    srows = np.arange(a + 1, a + len(raw) - 1)          # DEM rows that have a slope
    lat_rows = 90.0 - (srows + 0.5) * deg_per_row
    sl = band_slope(p, lat_rows, px)                     # shape (len(raw)-2, ncols)
    cache = {}
    for i in range(n):
        r0, c0 = int(rows[i]), int(cols[i])
        elev[i] = p[r0 - a, c0]
        lat = 90.0 - (r0 + 0.5) * deg_per_row
        if abs(lat) > SLOPE_MAX_LAT or diam_km[i] > MAX_D_KM:
            continue
        r_full = int(np.clip(round(diam_km[i] * 500.0 / px), 2, 10 ** 6))
        stride = max(1, int(np.ceil(r_full / TARGET_R)))
        r_out[i], st_out[i] = r_full, stride
        li = r0 - (a + 1)                                # row index inside sl
        s_cen[i] = sl[li, c0] if 0 <= li < sl.shape[0] else np.nan
        key = (r_full, stride)
        if key not in cache:
            cache[key] = disc_offsets(r_full, stride)
        dy, dx = cache[key]
        ri = li + dy
        if ri.min() < 0 or ri.max() >= sl.shape[0]:
            continue
        v = sl[ri, (c0 + dx) % ncols]
        if np.isfinite(v).any():
            s_disc[i] = np.nanmean(v)
    return elev, s_disc, s_cen, r_out, st_out


def worker(task):
    import rasterio
    k, rows, cols, diam, ids = task
    with rasterio.open(DEM) as src:
        nrows, ncols, nodata, px = src.height, src.width, src.nodata, src.transform.a
        margin = int(np.ceil(MAX_D_KM * 500.0 / px)) + 2
        r_a, r_b = k * CH, min((k + 1) * CH, nrows)
        a, b = r_a - margin - 1, r_b + margin + 1
        raw = src.read(1, window=((max(a, 0), min(b, nrows)), (0, ncols)))
    if a < 0:
        raw = np.vstack([np.repeat(raw[:1], -a, axis=0), raw])
    if b > nrows:
        raw = np.vstack([raw, np.repeat(raw[-1:], b - nrows, axis=0)])
    out = compute_chunk(raw, a, nrows, ncols, nodata, px, rows, cols, diam)
    fd = os.open(PROGRESS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, ("%d\n" % k).encode())
    os.close(fd)
    return (ids,) + out


def main():
    import pandas as pd
    import rasterio
    t0 = time.time()
    os.makedirs(BASE + "/logs", exist_ok=True)
    open(PROGRESS, "w").close()
    cr = pd.read_parquet(BASE + "/data/processed/craters_clean.parquet", columns=["crater_id", "lat", "lon", "diam_km"])
    with rasterio.open(DEM) as src:
        t = src.transform
        ncols, nrows = src.width, src.height
        A, C, E, F = t.a, t.c, t.e, t.f
    R = 1737400.0
    col = np.floor((R * np.radians(cr.lon.values) - C) / A).astype(np.int64)
    row = np.floor((R * np.radians(cr.lat.values) - F) / E).astype(np.int64)
    ok = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    print("craters %d | inside DEM %d | outside %d" % (len(cr), ok.sum(), (~ok).sum()), flush=True)
    cr = cr[ok].reset_index(drop=True)
    col, row = col[ok], row[ok]
    chunk = row // CH
    nchunks = (nrows + CH - 1) // CH
    tasks = []
    order = np.argsort(chunk, kind="stable")
    bounds = np.searchsorted(chunk[order], np.arange(nchunks + 1))
    for k in range(nchunks):
        idx = order[bounds[k]:bounds[k + 1]]
        if len(idx):
            tasks.append((k, row[idx], col[idx], cr.diam_km.values[idx], cr.crater_id.values[idx]))
    print("tasks: %d (each = %d DEM rows plus margin)" % (len(tasks), CH), flush=True)

    os.environ["PYSPARK_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("lunargrid-features").master("local[16]")
             .config("spark.driver.memory", "16g").config("spark.local.dir", BASE + "/spark-tmp")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark ready after %.1f s" % (time.time() - t0), flush=True)

    stop = threading.Event()

    def watch():
        while not stop.wait(15):
            with open(PROGRESS) as f:
                done = sum(1 for _ in f)
            print("  progress: %d / %d tasks done, %.0f s" % (done, len(tasks), time.time() - t0), flush=True)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t1 = time.time()
    res = spark.sparkContext.parallelize(tasks, len(tasks)).map(worker).collect()
    stop.set()
    print("Spark part took %.1f s" % (time.time() - t1), flush=True)
    spark.stop()

    ids = np.concatenate([r[0] for r in res])
    f = pd.DataFrame({"crater_id": ids,
                      "elev_m": np.concatenate([r[1] for r in res]),
                      "slope_disc_deg": np.concatenate([r[2] for r in res]),
                      "slope_centre_deg": np.concatenate([r[3] for r in res]),
                      "r_px": np.concatenate([r[4] for r in res]),
                      "stride": np.concatenate([r[5] for r in res])})
    base_cols = pd.read_parquet(BASE + "/data/processed/craters_clean.parquet", columns=["crater_id", "lat", "lon", "diam_km"])
    f = base_cols.merge(f, on="crater_id", how="left", validate="one_to_one")
    pol = pd.read_parquet(BASE + "/data/processed/craters_polar.parquet")
    pol = pol[["crater_id", "illum", "psr240"]].rename(columns={"illum": "sunlight_240m", "psr240": "shadow_240m"})
    f = f.merge(pol, on="crater_id", how="left", validate="one_to_one")
    out = BASE + "/data/processed/craters_features.parquet"
    f.to_parquet(out, index=False)

    print("\n===== RESULTS =====")
    print("rows: %d | saved %s" % (len(f), out))
    print("elevation missing: %d | slope_disc missing: %d | slope_centre missing: %d" %
          (f.elev_m.isna().sum(), f.slope_disc_deg.isna().sum(), f.slope_centre_deg.isna().sum()))
    print("  slope missing because poleward of 80 deg: %d | wider than %.0f km: %d" %
          ((f.lat.abs() > SLOPE_MAX_LAT).sum(), MAX_D_KM, (f.diam_km > MAX_D_KM).sum()))
    print("polar craters with sunlight value: %d" % f.sunlight_240m.notna().sum())

    # check 1: elevation against the earlier single-machine file
    old = pd.read_parquet(BASE + "/data/processed/craters_with_elev.parquet")
    ecol = [c for c in old.columns if "elev" in c.lower()]
    print("\ncheck 1: elevation against craters_with_elev.parquet, column(s):", ecol)
    if ecol:
        m = f[["crater_id", "elev_m"]].merge(old[["crater_id", ecol[0]]], on="crater_id")
        d = (m.elev_m - m[ecol[0]]).abs()
        print("  joined %d | max |difference| %.3f m | differing by > 0.01 m: %d | one-sided missing: %d" %
              (len(m), np.nanmax(d), int((d > 0.01).sum()), int((m.elev_m.isna() != m[ecol[0]].isna()).sum())))

    # check 2: crater slope summary by size and latitude
    g = f[f.slope_disc_deg.notna()].copy()
    g["size"] = pd.cut(g.diam_km, [0, 2, 5, 10, 20, 50.1], labels=["1-2", "2-5", "5-10", "10-20", "20-50"], right=False)
    print("\ncheck 2: slope over the crater disc (degrees) by diameter (km): count, median, 10th and 90th percentile")
    print(g.groupby("size", observed=True).slope_disc_deg.agg(["count", "median", lambda s: s.quantile(.1), lambda s: s.quantile(.9)]).round(2).to_string())
    g["latband"] = pd.cut(g.lat.abs(), [0, 30, 60, 70, 80.01], labels=["0-30", "30-60", "60-70", "70-80"], right=False)
    print("\nmedian slope over the disc by latitude band, and centre-pixel slope for comparison:")
    print(g.groupby("latband", observed=True).agg(disc=("slope_disc_deg", "median"), centre=("slope_centre_deg", "median"), n=("lat", "size")).round(2).to_string())
    print("\nSpearman correlation of slope_disc with slope_centre: %.3f" % g.slope_disc_deg.corr(g.slope_centre_deg, method="spearman"))

    # check 3: tile-level consistency with the tile slope table
    tl = pd.read_parquet(BASE + "/data/processed/tiles_terrain.parquet")[["tile_id", "mean_slope_deg"]]
    g["tile_id"] = (np.floor((g.lat + 90) / 5).astype(int) * 72 + np.floor((g.lon + 180) / 5).astype(int))
    tm = g.groupby("tile_id").slope_disc_deg.agg(["mean", "size"]).reset_index().merge(tl, on="tile_id").dropna()
    tm = tm[tm["size"] >= 20]
    print("\ncheck 3: tiles with >= 20 craters: %d | Spearman(tile mean of crater slope, tile mean slope) = %.3f" %
          (len(tm), tm["mean"].corr(tm.mean_slope_deg, method="spearman")))
    print("  mean over tiles: crater disc slope %.2f deg | tile slope %.2f deg" % (tm["mean"].mean(), tm.mean_slope_deg.mean()))
    print("\nTOTAL %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()