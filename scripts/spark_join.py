import os
import time

t_start = time.time()

from sedona.spark import SedonaContext

base = os.path.expanduser("~/lunargrid")
tmp = base + "/spark-tmp"

config = (
    SedonaContext.builder()
    .appName("lunargrid-join")
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
print("Spark + Sedona ready in %.1f s" % (time.time() - t_start), flush=True)

# craters as points
spark.read.parquet("file://" + base + "/data/processed/craters_clean.parquet") \
    .createOrReplaceTempView("craters_raw")
spark.sql("""
    SELECT monotonically_increasing_id() AS rid, crater_id, lat, lon, diam_km,
           ST_Point(lon, lat) AS geom
    FROM craters_raw
""").createOrReplaceTempView("craters_geo")

# tiles as polygons: one rectangle per 5 x 5 degree tile
spark.read.parquet("file://" + base + "/data/processed/tiles_terrain.parquet") \
    .createOrReplaceTempView("tiles_raw")
spark.sql("""
    SELECT tile_id,
           ST_PolygonFromEnvelope(lon_min, lat_min, lon_max, lat_max) AS geom
    FROM tiles_raw
""").createOrReplaceTempView("tiles_geo")
print("tile polygons:", spark.sql("SELECT COUNT(*) AS n FROM tiles_geo").collect()[0]["n"], flush=True)

# spatial join: which tile polygon contains each crater point?
t0 = time.time()
rows = spark.sql("""
    SELECT tile_id,
           COUNT(*) AS n_craters,
           SUM(CASE WHEN diam_km >= 2 THEN 1 ELSE 0 END) AS n_craters_ge2km
    FROM (
        SELECT c.rid, MAX(t.tile_id) AS tile_id, MAX(c.diam_km) AS diam_km
        FROM craters_geo c
        JOIN tiles_geo t ON ST_Intersects(t.geom, c.geom)
        GROUP BY c.rid
    )
    GROUP BY tile_id
""").collect()
t_join = time.time() - t0
print("spatial join + count took %.2f s" % t_join, flush=True)

# same count with the arithmetic formula, in the same session, for a rough side-by-side
t0 = time.time()
spark.sql("""
    SELECT CAST(floor((lat + 90) / 5) AS INT) * 72 + CAST(floor((lon + 180) / 5) AS INT) AS tile_id,
           COUNT(*) AS n
    FROM craters_raw GROUP BY 1
""").collect()
t_formula = time.time() - t0
print("formula count took %.2f s (same session, rough comparison only)" % t_formula, flush=True)

import pandas as pd
sp = pd.DataFrame([r.asDict() for r in rows])
base_tiles = pd.read_parquet(base + "/data/processed/tiles_terrain.parquet")[
    ["tile_id", "n_craters", "n_craters_ge2km"]]
m = base_tiles.merge(sp, on="tile_id", how="outer", suffixes=("_pandas", "_spark")).fillna(0)
bad = m[(m.n_craters_pandas != m.n_craters_spark) | (m.n_craters_ge2km_pandas != m.n_craters_ge2km_spark)]
print("tiles compared:", len(m))
print("craters  pandas: %d | spark join: %d | not matched by the join: %d"
      % (m.n_craters_pandas.sum(), m.n_craters_spark.sum(), m.n_craters_pandas.sum() - m.n_craters_spark.sum()))
print("tiles that differ:", len(bad))
if len(bad):
    print(bad.head(10).to_string())

# diagnostic: craters sitting exactly on a tile edge (multiple of 5 degrees)
edge = spark.sql("""
    SELECT COUNT(*) AS n FROM craters_raw
    WHERE (lat % 5 = 0) OR (lon % 5 = 0)
""").collect()[0]["n"]
print("craters exactly on a 5-degree edge:", edge)
print("TOTAL script time %.1f s" % (time.time() - t_start))
spark.stop()
