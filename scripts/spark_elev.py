import os
import time
import traceback

t_start = time.time()

import rasterio
base = os.path.expanduser("~/lunargrid")
with rasterio.open(base + "/data/dem/ldem_118m.tif") as src:
    t = src.transform
    NCOLS, NROWS = src.width, src.height
A, C, E, F = t.a, t.c, t.e, t.f          # same transform the single-machine script used
R = 1737400.0
print("pixel width %.6f m | origin x %.3f y %.3f | %d x %d" % (A, C, F, NCOLS, NROWS), flush=True)

from sedona.spark import SedonaContext

config = (
    SedonaContext.builder()
    .appName("lunargrid-elev")
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
    # 1. DEM tiles (36 band files -> 2,592 tiles of 1,280 x 1,280 pixels)
    dem = (spark.read.format("raster")
           .option("retile", "true")
           .option("tileWidth", "1280")
           .option("tileHeight", "1280")
           .option("pathGlobFilter", "dem_band_*.tif")
           .load("file://" + base + "/data/dem_bands"))
    dem.createOrReplaceTempView("dem_raw")
    spark.sql("""
        SELECT CAST(regexp_extract(name, 'dem_band_([0-9]+)', 1) AS INT) AS b, x AS tx, rast
        FROM dem_raw
    """).createOrReplaceTempView("dem_t")

    # 2. each crater -> pixel column/row in the full DEM -> (band, tile column, position inside the tile)
    spark.read.parquet("file://" + base + "/data/processed/craters_clean.parquet") \
        .createOrReplaceTempView("craters_raw")
    spark.sql("""
        SELECT crater_id,
               CAST(floor(({R} * radians(lon) - ({C})) / {A}) AS INT) AS col,
               CAST(floor(({R} * radians(lat) - ({F})) / ({E})) AS INT) AS row
        FROM craters_raw
    """.format(R=repr(R), C=repr(C), A=repr(A), F=repr(F), E=repr(E))).createOrReplaceTempView("cr_px")
    spark.sql("""
        SELECT crater_id,
               CAST(floor(row / 1280) AS INT) AS b,
               CAST(floor(col / 1280) AS INT) AS tx,
               col - CAST(floor(col / 1280) AS INT) * 1280 AS lc,
               row - CAST(floor(row / 1280) AS INT) * 1280 AS lr
        FROM cr_px
        WHERE col >= 0 AND col < {NC} AND row >= 0 AND row < {NR}
    """.format(NC=NCOLS, NR=NROWS)).createOrReplaceTempView("cr_loc")
    n_all = spark.sql("SELECT COUNT(*) AS n FROM craters_raw").collect()[0]["n"]
    n_ok = spark.sql("SELECT COUNT(*) AS n FROM cr_loc").collect()[0]["n"]
    print("craters: %d | inside the DEM grid: %d | outside: %d" % (n_all, n_ok, n_all - n_ok), flush=True)

    # 3. group craters per DEM tile, then read all their pixels from that tile in one call
    spark.sql("""
        SELECT b, tx, collect_list(struct(crater_id AS id, lc AS c, lr AS r)) AS items
        FROM cr_loc GROUP BY b, tx
    """).createOrReplaceTempView("per_tile")
    sampled = spark.sql("""
        SELECT /*+ BROADCAST(g) */
               transform(g.items, s -> s.id) AS ids,
               RS_Values(d.rast, transform(g.items, s -> s.c), transform(g.items, s -> s.r), 1) AS vals
        FROM per_tile g JOIN dem_t d ON g.b = d.b AND g.tx = d.tx
    """)
    sampled.createOrReplaceTempView("sampled")
    result = spark.sql("""
        SELECT ids[pos] AS crater_id,
               CASE WHEN val IS NULL OR val = -32768 THEN CAST(NULL AS DOUBLE) ELSE val END AS elev_spark
        FROM (SELECT ids, posexplode(vals) AS (pos, val) FROM sampled)
    """)

    # 4. run it and save (this is the timed step)
    out = base + "/data/processed/craters_elev_spark.parquet"
    t0 = time.time()
    result.write.mode("overwrite").parquet("file://" + out)
    t_elev = time.time() - t0
    print("elevation sampling for all craters + write took %.1f s" % t_elev, flush=True)

    # 5. compare with the single-machine result
    spark.read.parquet("file://" + out).createOrReplaceTempView("elev_spark")
    base_df = spark.read.parquet("file://" + base + "/data/processed/craters_with_elev.parquet")
    cands = [c for c in base_df.columns if "elev" in c.lower()]
    print("baseline columns:", base_df.columns, "| using:", cands[:1], flush=True)
    ecol = cands[0]
    base_df.createOrReplaceTempView("elev_base")
    cmp = spark.sql("""
        SELECT COUNT(*) AS n_joined,
               SUM(CASE WHEN s.elev_spark IS NULL AND (b.{e} IS NULL OR isnan(b.{e})) THEN 1 ELSE 0 END) AS both_empty,
               SUM(CASE WHEN (s.elev_spark IS NULL) <> (b.{e} IS NULL OR isnan(b.{e})) THEN 1 ELSE 0 END) AS empty_mismatch,
               MAX(ABS(s.elev_spark - b.{e})) AS max_abs_diff,
               SUM(CASE WHEN ABS(s.elev_spark - b.{e}) > 0.01 THEN 1 ELSE 0 END) AS differ_over_0_01
        FROM elev_spark s JOIN elev_base b ON s.crater_id = b.crater_id
    """.format(e=ecol)).collect()[0]
    print("joined on crater_id:", cmp["n_joined"])
    print("both empty (no data):", cmp["both_empty"], "| empty in one only:", cmp["empty_mismatch"])
    print("max |spark - baseline| in metres:", cmp["max_abs_diff"])
    print("craters differing by more than 0.01 m:", cmp["differ_over_0_01"])
except Exception:
    print("FAILED:")
    print(traceback.format_exc()[-2200:])

print("TOTAL script time %.1f s" % (time.time() - t_start))
spark.stop()
