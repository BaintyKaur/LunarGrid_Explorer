import os
import time
import traceback

t_start = time.time()

import rasterio
dem_path = os.path.expanduser("~/lunargrid/data/dem/ldem_118m.tif")
with rasterio.open(dem_path) as src:
    print("DEM internal block shape:", src.block_shapes[0], "| compression:", src.compression,
          "| size:", src.width, "x", src.height, flush=True)

from sedona.spark import SedonaContext

base = os.path.expanduser("~/lunargrid")
config = (
    SedonaContext.builder()
    .appName("lunargrid-dem-open")
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
           .load("file://" + dem_path))
    print("columns:", dem.columns, flush=True)
    dem.createOrReplaceTempView("dem")

    r = spark.sql("SELECT COUNT(*) AS n, MIN(x) AS x0, MAX(x) AS x1, MIN(y) AS y0, MAX(y) AS y1 FROM dem").collect()[0]
    print("tiles: %d | x %d..%d | y %d..%d | took %.1f s" % (r["n"], r["x0"], r["x1"], r["y0"], r["y1"], time.time() - t0), flush=True)

    spark.sql("""
        SELECT x, y, RS_Width(rast) AS w, RS_Height(rast) AS h,
               RS_NumBands(rast) AS bands, RS_SRID(rast) AS srid
        FROM dem ORDER BY y, x LIMIT 5
    """).show(truncate=False)
except Exception:
    print("RASTER READ FAILED:")
    print(traceback.format_exc()[-1800:])

print("TOTAL script time %.1f s" % (time.time() - t_start))
spark.stop()
