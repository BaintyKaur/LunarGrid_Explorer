import os
import time
import traceback

t_start = time.time()
base = os.path.expanduser("~/lunargrid")

from sedona.spark import SedonaContext

config = (
    SedonaContext.builder()
    .appName("lunargrid-tilestats")
    .master("local[16]")
    .config("spark.driver.memory", "32g")
    .config("spark.local.dir", base + "/spark-tmp")
    .config("spark.sql.shuffle.partitions", "32")
    .config("spark.jars.packages",
            "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.9.1,"
            "org.datasyslab:geotools-wrapper:1.9.1-33.5")
    .config("spark.jars.repositories",
            "https://artifacts.unidata.ucar.edu/repository/unidata-all")
    .getOrCreate()
)
spark = SedonaContext.create(config)
spark.sparkContext.setLogLevel("ERROR")
print("Spark + Sedona ready in %.1f s" % (time.time() - t_start), flush=True)

try:
    dem = (spark.read.format("raster")
           .option("retile", "true")
           .option("tileWidth", "1280")
           .option("tileHeight", "1280")
           .option("pathGlobFilter", "dem_band_*.tif")
           .load("file://" + base + "/data/dem_bands"))
    dem.createOrReplaceTempView("dem_raw")

    stats = spark.sql("""
        SELECT (35 - CAST(regexp_extract(name, 'dem_band_([0-9]+)', 1) AS INT)) * 72 + x AS tile_id,
               st.count AS n_pixels, st.mean AS mean_elev_spark, st.stddev AS std_elev_spark,
               st.min AS min_elev_spark, st.max AS max_elev_spark
        FROM (SELECT name, x, RS_SummaryStatsAll(rast, 1, true) AS st FROM dem_raw)
    """)

    out = base + "/data/processed/tiles_stats_spark.parquet"
    t0 = time.time()
    stats.write.mode("overwrite").parquet("file://" + out)
    t_stats = time.time() - t0
    print("tile statistics for all tiles + write took %.1f s" % t_stats, flush=True)

    spark.read.parquet("file://" + out).createOrReplaceTempView("ts")
    spark.read.parquet("file://" + base + "/data/processed/tiles_terrain.parquet") \
        .createOrReplaceTempView("tb")
    r = spark.sql("""
        SELECT COUNT(*) AS n,
               MIN(ts.n_pixels) AS min_px, MAX(ts.n_pixels) AS max_px,
               MAX(ABS(ts.mean_elev_spark - tb.mean_elev_m)) AS max_d_mean,
               MAX(ABS(ts.std_elev_spark - tb.std_elev_m)) AS max_d_std,
               SUM(CASE WHEN ABS(ts.mean_elev_spark - tb.mean_elev_m) > 0.01 THEN 1 ELSE 0 END) AS n_mean_differ,
               SUM(CASE WHEN ABS(ts.std_elev_spark - tb.std_elev_m) > 0.01 THEN 1 ELSE 0 END) AS n_std_differ,
               percentile_approx(CASE WHEN ABS(ts.mean_elev_spark) > 5 THEN tb.mean_elev_m / ts.mean_elev_spark END, 0.5) AS median_ratio_mean
        FROM ts JOIN tb ON ts.tile_id = tb.tile_id
    """).collect()[0]
    print("tiles compared:", r["n"])
    print("pixels per tile: min", r["min_px"], "| max", r["max_px"], "(full tile = 1638400)")
    print("max |spark - baseline| mean elevation (m):", r["max_d_mean"])
    print("max |spark - baseline| std of elevation (m):", r["max_d_std"])
    print("tiles whose mean differs by > 0.01 m:", r["n_mean_differ"], "| whose std differs:", r["n_std_differ"])
    print("median ratio baseline/spark for the mean:", r["median_ratio_mean"])
except Exception:
    print("FAILED:")
    print(traceback.format_exc()[-2200:])

print("TOTAL script time %.1f s" % (time.time() - t_start))
spark.stop()
