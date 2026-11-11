import os
import time
import traceback

t_start = time.time()

from sedona.spark import SedonaContext

base = os.path.expanduser("~/lunargrid")
config = (
    SedonaContext.builder()
    .appName("lunargrid-dem-bands")
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
    t0 = time.time()
    dem = (spark.read.format("raster")
           .option("retile", "true")
           .option("tileWidth", "1280")
           .option("tileHeight", "1280")
           .option("pathGlobFilter", "dem_band_*.tif")
           .load("file://" + base + "/data/dem_bands"))
    print("partitions:", dem.rdd.getNumPartitions(), flush=True)
    dem.createOrReplaceTempView("dem")

    spark.sql("""
        SELECT (35 - CAST(regexp_extract(name, 'dem_band_([0-9]+)', 1) AS INT)) * 72 + x AS tile_id,
               x, y, name, rast
        FROM dem
    """).createOrReplaceTempView("dem_tiles")

    r = spark.sql("""
        SELECT COUNT(*) AS n, COUNT(DISTINCT tile_id) AS n_ids,
               MIN(tile_id) AS id0, MAX(tile_id) AS id1,
               MIN(x) AS x0, MAX(x) AS x1, MIN(y) AS y0, MAX(y) AS y1
        FROM dem_tiles
    """).collect()[0]
    print("tiles: %d | distinct tile_id: %d | tile_id %d..%d | x %d..%d | y %d..%d | took %.1f s"
          % (r["n"], r["n_ids"], r["id0"], r["id1"], r["x0"], r["x1"], r["y0"], r["y1"], time.time() - t0), flush=True)

    spark.sql("""
        SELECT tile_id, x, y, name, RS_Width(rast) AS w, RS_Height(rast) AS h,
               RS_NumBands(rast) AS bands, RS_SRID(rast) AS srid
        FROM dem_tiles ORDER BY tile_id LIMIT 5
    """).show(truncate=False)
except Exception:
    print("RASTER READ FAILED:")
    print(traceback.format_exc()[-1800:])

print("TOTAL script time %.1f s" % (time.time() - t_start))
spark.stop()
