import os
import time

t_start = time.time()

from sedona.spark import SedonaContext

base = os.path.expanduser("~/lunargrid")
tmp = base + "/spark-tmp"

config = (
    SedonaContext.builder()
    .appName("lunargrid-tiles")
    .master("local[16]")
    .config("spark.driver.memory", "32g")
    .config("spark.local.dir", tmp)
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
t_ready = time.time()
print("Spark + Sedona ready in %.1f s" % (t_ready - t_start), flush=True)

# 1. read the crater table (Parquet) in parallel partitions
df = spark.read.parquet("file://" + base + "/data/processed/craters_clean.parquet")
print("partitions:", df.rdd.getNumPartitions(), "| columns:", df.columns, flush=True)
df.createOrReplaceTempView("craters")

# 2. make a point geometry for every crater (lon = x, lat = y) and a tile id
spark.sql("""
    SELECT crater_id, lat, lon, diam_km,
           ST_Point(lon, lat) AS geom,
           CAST(floor((lat + 90) / 5) AS INT) * 72 + CAST(floor((lon + 180) / 5) AS INT) AS tile_id_spark
    FROM craters
""").createOrReplaceTempView("craters_geo")

# 3. check: the point geometry must hold the same numbers we put in
chk = spark.sql("""
    SELECT COUNT(*) AS n,
           MAX(ABS(ST_X(geom) - lon)) AS max_dx,
           MAX(ABS(ST_Y(geom) - lat)) AS max_dy
    FROM craters_geo
""").collect()[0]
print("points built:", chk["n"], "| max |x-lon|:", chk["max_dx"], "| max |y-lat|:", chk["max_dy"], flush=True)

# 4. count craters per tile (all, and >= 2 km)
t0 = time.time()
rows = spark.sql("""
    SELECT tile_id_spark AS tile_id,
           COUNT(*) AS n_craters,
           SUM(CASE WHEN diam_km >= 2 THEN 1 ELSE 0 END) AS n_craters_ge2km
    FROM craters_geo
    GROUP BY tile_id_spark
""").collect()
t_count = time.time() - t0
print("per-tile counting took %.2f s" % t_count, flush=True)

import pandas as pd
sp = pd.DataFrame([r.asDict() for r in rows])

# 5. compare with the single-machine tile table
base_tiles = pd.read_parquet(base + "/data/processed/tiles_terrain.parquet")[
    ["tile_id", "n_craters", "n_craters_ge2km"]]
m = base_tiles.merge(sp, on="tile_id", how="outer", suffixes=("_pandas", "_spark")).fillna(0)
bad = m[(m.n_craters_pandas != m.n_craters_spark) | (m.n_craters_ge2km_pandas != m.n_craters_ge2km_spark)]
print("tiles compared:", len(m))
print("craters  pandas: %d | spark: %d" % (m.n_craters_pandas.sum(), m.n_craters_spark.sum()))
print("tiles that differ:", len(bad))
if len(bad):
    print(bad.head(10).to_string())
print("TOTAL script time %.1f s" % (time.time() - t_start))
spark.stop()
