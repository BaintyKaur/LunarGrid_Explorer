"""Step 5b: K-means on crater properties (Spark MLlib).

Usage:  python kmeans_craters.py 16          # Spark local[16], prints everything
        python kmeans_craters.py 1 quiet     # Spark local[1], prints only the timing

Features (each rescaled to mean 0 and standard deviation 1):
  log10 diameter, slope over the crater disc, elevation, absolute latitude
Craters used: the 1.26 million with a slope (not poleward of 80 deg, not wider than 50 km).
Tries k = 2..8, scores each with the silhouette (-1..1; above 0.5 = clear groups,
0.25-0.5 = weak structure, below 0.25 = no real groups), then profiles the groups.

Reads : data/processed/craters_features.parquet
Writes: data/processed/kmeans_sweep.csv, kmeans_profile_k{k}.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/lunargrid/data/processed")
FEATS = ["log_d", "slope", "elev", "abs_lat"]


def main():
    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    quiet = len(sys.argv) > 2 and sys.argv[2] == "quiet"
    t_start = time.time()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession, functions as F
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator

    spark = (SparkSession.builder.appName("lunargrid-kmeans").master("local[%d]" % cores)
             .config("spark.driver.memory", "16g").config("spark.local.dir", os.path.expanduser("~/lunargrid/spark-tmp"))
             .config("spark.sql.shuffle.partitions", "32").config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = (spark.read.parquet("file://" + BASE + "/craters_features.parquet")
          .where(F.col("slope_disc_deg").isNotNull())
          .select("crater_id", "lat", "lon", "diam_km", "elev_m", "slope_disc_deg",
                  F.log10("diam_km").alias("log_d"), F.col("slope_disc_deg").alias("slope"),
                  F.col("elev_m").alias("elev"), F.abs(F.col("lat")).alias("abs_lat")))
    vec = VectorAssembler(inputCols=FEATS, outputCol="raw").transform(df)
    sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(vec)
    data = sc.transform(vec).repartition(max(cores * 2, 8)).cache()
    n = data.count()
    print("===== RESULTS (local[%d]) =====" % cores)
    print("craters used: %d | prepared in %.1f s" % (n, time.time() - t_start), flush=True)
    ev = ClusteringEvaluator(metricName="silhouette", distanceMeasure="squaredEuclidean", featuresCol="features")

    rows, models = [], {}
    t_sweep = time.time()
    for k in range(2, 9):
        t0 = time.time()
        m = KMeans(k=k, seed=1, maxIter=50, featuresCol="features").fit(data)
        sil = ev.evaluate(m.transform(data))
        rows.append({"k": k, "silhouette": sil, "cost": m.summary.trainingCost, "iterations": m.summary.numIter, "seconds": time.time() - t0})
        models[k] = m
        print("  k=%d  silhouette %.3f  cost %.0f  iterations %d  (%.1f s)" % (k, sil, m.summary.trainingCost, m.summary.numIter, time.time() - t0), flush=True)
    t_sweep = time.time() - t_sweep
    sw = pd.DataFrame(rows)
    sw.to_csv(BASE + "/kmeans_sweep.csv", index=False)
    best = int(sw.loc[sw.silhouette.idxmax(), "k"])
    print("\nbest silhouette at k=%d (%.3f) | sweep time %.1f s" % (best, sw.silhouette.max(), t_sweep), flush=True)

    if not quiet:
        # stability of the best k over three seeds (silhouette only)
        sils = []
        for seed in (2, 3):
            m = KMeans(k=best, seed=seed, maxIter=50, featuresCol="features").fit(data)
            sils.append(ev.evaluate(m.transform(data)))
        print("same k with two more seeds: silhouette %.3f, %.3f" % tuple(sils))

        pd.set_option("display.width", 250)
        for k in sorted({best, 5}):
            pred = models[k].transform(data)
            prof = (pred.groupBy("prediction").agg(
                F.count("*").alias("n"),
                F.round(F.expr("percentile_approx(diam_km, 0.5)"), 2).alias("diam_med_km"),
                F.round(F.avg("slope"), 2).alias("slope_deg"),
                F.round(F.avg("elev"), 0).alias("elev_m"),
                F.round(F.avg("abs_lat"), 1).alias("abs_lat"),
                F.round(F.avg(F.when(F.abs("lon") >= 90, 1.0).otherwise(0.0)) * 100, 1).alias("far_side_%"),
                F.round(F.avg(F.when(F.col("abs_lat") >= 65, 1.0).otherwise(0.0)) * 100, 1).alias("poleward_65_%"),
            ).orderBy("prediction").toPandas())
            prof["share_%"] = (100 * prof.n / n).round(1)
            prof.to_csv(BASE + "/kmeans_profile_k%d.csv" % k, index=False)
            print("\nprofile of the %d groups (k=%d):" % (k, k))
            print(prof[["prediction", "n", "share_%", "diam_med_km", "slope_deg", "elev_m", "abs_lat", "far_side_%", "poleward_65_%"]].to_string(index=False))
    print("\nTIMING local[%d]: sweep k=2..8 took %.1f s | whole script %.1f s" % (cores, t_sweep, time.time() - t_start))
    spark.stop()


if __name__ == "__main__":
    main()