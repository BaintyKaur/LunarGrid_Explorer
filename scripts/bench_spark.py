"""Step 3: benchmark single machine (pandas) against Spark SQL with 1-16 cores.

Usage (one configuration per run, because Spark's core count is fixed at start):
    python bench_spark.py pandas
    python bench_spark.py 1        # Spark local[1]
    python bench_spark.py 16r      # Spark local[16] with repartition(64) first
    python bench_spark.py summary  # print the final tables from the results file

Two jobs, identical in every configuration (each starts by reading the file):
    J1  tile count: number of craters, and craters >= 2 km, per 5-degree tile
    J2  polar cell count: craters poleward of 65 degrees, polar stereographic
        x/y, 20 km cell id, count and mean diameter per cell
Four inputs: Robbins 1.3 M, LU5M812TGT 5.7 M, LUC-GRAS200 10.7 M, and the
LUC-GRAS200 table repeated 4 times (42.8 M rows) to test a larger input.
Each job is run 4 times; run 0 is a warm-up and is reported separately.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/lunargrid")
OUT = BASE + "/data/processed/bench_results.csv"
R = 1737400.0
FILES = {
    "robbins": BASE + "/data/processed/craters_clean.parquet",
    "lu5m": BASE + "/data/processed/craters_lu5m.parquet",
    "luc": BASE + "/data/processed/craters_luc.parquet",
}
DATASETS = ["robbins", "lu5m", "luc", "luc_x4"]
RUNS = 4  # run 0 = warm-up
COLS = ["lon", "lat", "diam_km"]


# ---------------------------------------------------------------- pandas
def pandas_load(name):
    if name == "luc_x4":
        d = pd.read_parquet(FILES["luc"], columns=COLS)
        return pd.concat([d] * 4, ignore_index=True)
    return pd.read_parquet(FILES[name], columns=COLS)


def pandas_j1(name):
    d = pandas_load(name)
    tile = (np.floor((d.lat.values + 90) / 5).astype(np.int64) * 72 +
            np.floor((d.lon.values + 180) / 5).astype(np.int64))
    g = pd.DataFrame({"t": tile, "ge2": (d.diam_km.values >= 2).astype(np.int64)}).groupby("t").agg(
        n=("ge2", "size"), n2=("ge2", "sum"))
    return len(g), int(g.n.sum()), int(g.n2.sum())


def pandas_j2(name):
    d = pandas_load(name)
    d = d[np.abs(d.lat.values) >= 65]
    phi, lam = np.radians(d.lat.values), np.radians(d.lon.values)
    s = np.where(d.lat.values > 0, -1.0, 1.0)
    rho = 2 * R * np.tan(np.pi / 4 + s * phi / 2)
    x, y = rho * np.sin(lam), s * rho * np.cos(lam)
    ix = np.floor((x + 770000.0) / 20000.0).astype(np.int64)
    iy = np.floor((y + 770000.0) / 20000.0).astype(np.int64)
    key = np.where(d.lat.values > 0, 1, 0) * 100000 + (iy + 1) * 100 + (ix + 1)
    g = pd.DataFrame({"k": key, "ge2": (d.diam_km.values >= 2).astype(np.int64), "dk": d.diam_km.values}).groupby("k").agg(
        n=("ge2", "size"), n2=("ge2", "sum"), m=("dk", "mean"))
    return len(g), int(g.n.sum()), int(g.n2.sum())


# ----------------------------------------------------------------- Spark
def spark_df(spark, name):
    def rd(p):
        return spark.read.parquet("file://" + p).select(*COLS)
    if name == "luc_x4":
        d = rd(FILES["luc"])
        return d.union(d).union(d).union(d)
    return rd(FILES[name])


def spark_j1(spark, name, repart):
    d = spark_df(spark, name)
    if repart:
        d = d.repartition(repart)
    d.createOrReplaceTempView("c")
    rows = spark.sql("""
        SELECT CAST(floor((lat + 90) / 5) AS BIGINT) * 72 + CAST(floor((lon + 180) / 5) AS BIGINT) AS t,
               COUNT(*) AS n, SUM(CASE WHEN diam_km >= 2 THEN 1 ELSE 0 END) AS n2
        FROM c GROUP BY 1""").collect()
    return len(rows), int(sum(r.n for r in rows)), int(sum(r.n2 for r in rows))


def spark_j2(spark, name, repart):
    d = spark_df(spark, name)
    if repart:
        d = d.repartition(repart)
    d.createOrReplaceTempView("c")
    rows = spark.sql("""
        WITH p AS (
          SELECT lat, lon, diam_km,
                 CASE WHEN lat > 0 THEN -1.0 ELSE 1.0 END AS s,
                 radians(lat) AS phi, radians(lon) AS lam
          FROM c WHERE abs(lat) >= 65),
        q AS (
          SELECT diam_km, lat, s, lam, 2 * 1737400.0 * tan(pi() / 4 + s * phi / 2) AS rho FROM p),
        r AS (
          SELECT diam_km, lat,
                 CAST(floor((rho * sin(lam) + 770000.0) / 20000.0) AS BIGINT) AS ix,
                 CAST(floor((s * rho * cos(lam) + 770000.0) / 20000.0) AS BIGINT) AS iy
          FROM q)
        SELECT CASE WHEN lat > 0 THEN 1 ELSE 0 END * 100000 + (iy + 1) * 100 + (ix + 1) AS k,
               COUNT(*) AS n, SUM(CASE WHEN diam_km >= 2 THEN 1 ELSE 0 END) AS n2, AVG(diam_km) AS m
        FROM r GROUP BY 1""").collect()
    return len(rows), int(sum(r.n for r in rows)), int(sum(r.n2 for r in rows))


def make_spark(cores):
    from pyspark.sql import SparkSession
    tmp = BASE + "/spark-tmp"
    t0 = time.time()
    spark = (SparkSession.builder.appName("lunargrid-bench").master("local[%d]" % cores)
             .config("spark.driver.memory", "32g")
             .config("spark.local.dir", tmp)
             .config("spark.sql.shuffle.partitions", "32")
             .config("spark.sql.files.maxPartitionBytes", str(16 * 1024 * 1024))
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark, time.time() - t0


# ------------------------------------------------------------------ main
def summary():
    r = pd.read_csv(OUT)
    t = r[r.run > 0]
    med = t.groupby(["dataset", "job", "config"]).seconds.median().unstack("config")
    cfg_order = [c for c in ["pandas", "1", "2", "4", "8", "16", "16r"] if c in med.columns]
    print("Median seconds over runs 1-3 (read + compute + collect):\n")
    print(med[cfg_order].round(2).to_string())
    print("\nSpeed-up against pandas (pandas seconds / config seconds; above 1 = faster than pandas):\n")
    print((med["pandas"].values[:, None] / med[cfg_order]).round(2).to_string())
    sp = [c for c in cfg_order if c.isdigit()]
    if "1" in sp:
        print("\nScaling against Spark with 1 core (1-core seconds / config seconds):\n")
        print((med["1"].values[:, None] / med[sp]).round(2).to_string())
    print("\nWarm-up run (run 0) seconds:\n")
    print(r[r.run == 0].groupby(["dataset", "job", "config"]).seconds.median().unstack("config")[cfg_order].round(2).to_string())
    print("\nSpread (max - min of runs 1-3, seconds), largest 8 cases:")
    sp_ = t.groupby(["dataset", "job", "config"]).seconds.agg(lambda s: s.max() - s.min()).sort_values(ascending=False).head(8)
    print(sp_.round(2).to_string())
    print("\nStartup seconds of Spark by configuration:")
    print(r.groupby("config").startup_s.first().round(1).to_string())
    print("\nCorrectness: results (groups, total, n>=2km) must be identical across configs:")
    chk = r.groupby(["dataset", "job"]).apply(lambda g: g[["groups", "total", "n2"]].drop_duplicates().shape[0])
    print(chk.to_string(), "\n(1 = identical in every config and run)")
    print("\nInput partitions Spark used:")
    print(r[r.config != "pandas"].groupby(["config", "dataset"]).partitions.first().unstack("dataset").to_string())


def main():
    mode = sys.argv[1]
    if mode == "summary":
        summary()
        return
    repart = 64 if mode.endswith("r") else 0
    if mode == "pandas":
        spark, startup = None, 0.0
    else:
        cores = int(mode.rstrip("r"))
        spark, startup = make_spark(cores)
        print("Spark local[%d] ready in %.1f s" % (cores, startup), flush=True)
    rows = []
    for name in DATASETS:
        for run in range(RUNS):
            for job in ("J1", "J2"):
                t0 = time.time()
                if spark is None:
                    res = pandas_j1(name) if job == "J1" else pandas_j2(name)
                    parts = 0
                else:
                    res = (spark_j1 if job == "J1" else spark_j2)(spark, name, repart)
                sec = time.time() - t0
                if spark is not None:
                    parts = repart or spark_df(spark, name).rdd.getNumPartitions()
                rows.append({"config": mode, "dataset": name, "job": job, "run": run, "seconds": sec,
                             "groups": res[0], "total": res[1], "n2": res[2], "partitions": parts, "startup_s": startup})
                print("%-7s %-7s %s run %d: %7.2f s  groups %d total %d >=2km %d partitions %s" %
                      (mode, name, job, run, sec, res[0], res[1], res[2], parts), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print("appended %d rows to %s" % (len(df), OUT), flush=True)


if __name__ == "__main__":
    main()