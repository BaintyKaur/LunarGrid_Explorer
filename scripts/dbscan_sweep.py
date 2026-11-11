import os, time
import numpy as np
import pandas as pd
from sedona.spark import SedonaContext

base = os.path.expanduser("~/lunargrid/data")
P = base + "/processed"
R = 1737400.0  # m
def log(*a): print(*a, flush=True)
t0 = time.time()

config = (SedonaContext.builder().appName("dbscan_sweep").master("local[16]")
          .config("spark.driver.memory", "32g").config("spark.local.dir", base + "/spark-tmp")
          .config("spark.sql.shuffle.partitions", "32")
          .config("spark.jars.packages", "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.9.1,org.datasyslab:geotools-wrapper:1.9.1-33.5")
          .config("spark.jars.repositories", "https://artifacts.unidata.ucar.edu/repository/unidata-all").getOrCreate())
spark = SedonaContext.create(config)
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir(base + "/spark-tmp/ckpt")

cats = {"robbins": P + "/craters_clean.parquet", "lu5m": P + "/craters_lu5m.parquet", "luc": P + "/craters_luc.parquet"}
eps_list_km = [2, 5, 10, 20]
minpts_list = [5, 10]
rows = []
log("RESULTS")
for pole in ("south", "north"):
    for name, path in cats.items():
        d = pd.read_parquet(path, columns=["lon", "lat", "diam_km"])
        sel = (d.diam_km >= 1.0) & ((d.lat <= -80) if pole == "south" else (d.lat >= 80))
        d = d[sel].reset_index(drop=True)
        lam, phi = np.radians(d.lon.values), np.radians(d.lat.values)
        if pole == "south":
            rho = 2 * R * np.tan(np.pi / 4 + phi / 2); x = rho * np.sin(lam); y = rho * np.cos(lam)
        else:
            rho = 2 * R * np.tan(np.pi / 4 - phi / 2); x = rho * np.sin(lam); y = -rho * np.cos(lam)
        n = len(d)
        pts = spark.createDataFrame(pd.DataFrame({"id": np.arange(n), "x": x, "y": y}))
        pts.createOrReplaceTempView("p0")
        spark.sql("SELECT id, ST_Point(x, y) AS geom FROM p0").cache().createOrReplaceTempView("pts")
        log(f"\n{pole} / {name}: {n:,} craters >= 1 km  [{time.time()-t0:.0f} s]")
        for eps in eps_list_km:
            for mp in minpts_list:
                q = f"SELECT id, ST_DBSCAN(geom, {eps*1000.0}, {mp}, false) AS r FROM pts"
                res = spark.sql(q).selectExpr("id", "r.isCore AS core", "r.cluster AS cl").toPandas()
                cl = res.cl.values
                real = cl[cl != -1]
                ncl = int(np.unique(real).size)
                noise = float((cl == -1).mean())
                big = float(pd.Series(real).value_counts().iloc[0] / n) if ncl else 0.0
                rows.append(dict(pole=pole, catalogue=name, n=n, eps_km=eps, minpts=mp, n_clusters=ncl, noise_share=round(noise, 4), largest_cluster_share=round(big, 4)))
                log(f"  eps={eps:>2} km minPts={mp:>2}: clusters {ncl:>5}   noise {noise*100:5.1f}%   biggest cluster {big*100:5.1f}% of craters   [{time.time()-t0:.0f} s]")
        spark.catalog.clearCache()
pd.DataFrame(rows).to_csv(P + "/dbscan_sweep.csv", index=False)
log(f"\nDONE {time.time()-t0:.0f} s")
spark.stop()
