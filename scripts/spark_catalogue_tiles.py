import os, time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

base = os.path.expanduser("~/lunargrid/data")
P = base + "/processed"
def log(*a): print(*a, flush=True)
t0 = time.time()

spark = (SparkSession.builder.appName("catalogue_tiles").master("local[16]")
         .config("spark.driver.memory", "32g").config("spark.local.dir", base + "/spark-tmp")
         .config("spark.sql.shuffle.partitions", "32").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

cats = {"robbins": P + "/craters_clean.parquet", "lu5m": P + "/craters_lu5m.parquet", "luc": P + "/craters_luc.parquet"}
thr = [1, 2, 5, 20]
parts = []
for name, path in cats.items():
    spark.read.parquet(path).select("lon", "lat", "diam_km").createOrReplaceTempView("c")
    sums = ", ".join([f"SUM(CASE WHEN diam_km>={t} THEN 1 ELSE 0 END) AS n_ge{t}" for t in thr])
    q = f"""SELECT tile_id, COUNT(*) AS n_all, {sums} FROM (
              SELECT CAST(diam_km AS DOUBLE) AS diam_km,
                     CAST(least(35, greatest(0, floor((lat+90)/5))) AS INT)*72
                     + CAST(least(71, greatest(0, floor(pmod(lon+180, 360)/5))) AS INT) AS tile_id
              FROM c) GROUP BY tile_id"""
    pdf = spark.sql(q).toPandas().set_index("tile_id")
    pdf.columns = [f"{c}_{name}" for c in pdf.columns]
    parts.append(pdf)
    log(f"counted {name} [{time.time()-t0:.0f} s]")

t = pd.concat(parts, axis=1).reindex(range(2592)).fillna(0).astype(np.int64)
t.index.name = "tile_id"
t["lat_row"] = t.index // 72
t.reset_index().to_csv(P + "/tiles_catalogue_counts.csv", index=False)

log("\nRESULTS")
polar = t.lat_row.isin([0, 1, 34, 35])
for tt in thr:
    log(f"\n--- craters with D >= {tt} km ---")
    for name in cats:
        tot = t[f"n_ge{tt}_{name}"].sum()
        pol = t.loc[polar, f"n_ge{tt}_{name}"].sum()
        log(f"  {name:<8} total {tot:>10,}   poleward of 80 deg: {pol:>8,}  ({pol/tot*100:4.2f}% of craters; caps are 1.52% of area)")
    cols = [f"n_ge{tt}_{n}" for n in cats]
    sub = t[cols]; sub.columns = list(cats)
    log("  Pearson correlation of per-tile counts:")
    log("    " + sub.corr().round(3).to_string().replace("\n", "\n    "))
    log("  Spearman correlation of per-tile counts:")
    log("    " + sub.corr(method="spearman").round(3).to_string().replace("\n", "\n    "))
    if tt in (1, 2):
        ok = t[f"n_ge{tt}_robbins"] >= 50
        for n in ["lu5m", "luc"]:
            r = (t.loc[ok, f"n_ge{tt}_{n}"] / t.loc[ok, f"n_ge{tt}_robbins"]).replace([np.inf, -np.inf], np.nan).dropna()
            log(f"  tiles with >=50 Robbins craters: {ok.sum()}; ratio {n}/robbins per tile: median {r.median():.2f}  p10 {r.quantile(.1):.2f}  p90 {r.quantile(.9):.2f}")
log(f"\nDONE {time.time()-t0:.0f} s")
spark.stop()
