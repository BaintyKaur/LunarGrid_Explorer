import os
from sedona.spark import SedonaContext
base = os.path.expanduser("~/lunargrid")
config = (SedonaContext.builder().appName("test_dbscan").master("local[4]")
          .config("spark.driver.memory", "8g").config("spark.local.dir", base + "/spark-tmp")
          .config("spark.sql.shuffle.partitions", "4")
          .config("spark.jars.packages", "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.9.1,org.datasyslab:geotools-wrapper:1.9.1-33.5")
          .config("spark.jars.repositories", "https://artifacts.unidata.ucar.edu/repository/unidata-all").getOrCreate())
spark = SedonaContext.create(config)
spark.sparkContext.setLogLevel("ERROR")
print("RESULTS", flush=True)

try:
    spark._jvm.java.lang.Class.forName("org.graphframes.GraphFrame")
    print("GraphFrames on classpath: YES", flush=True)
except Exception:
    print("GraphFrames on classpath: NO", flush=True)

def run(title, q):
    print("\n---", title, flush=True)
    try:
        spark.sql(q).show(truncate=False)
    except Exception as e:
        lines = str(e).splitlines()
        keep = [l.strip()[:300] for l in lines if any(k in l for k in ("Exception", "Caused by", "Error", "graphframes", "checkpoint", "ClassNotFound"))]
        print("  FAILED. Key lines:", flush=True)
        for l in keep[:10]:
            print("   ", l, flush=True)

spark.sql("""SELECT id, ST_Point(CAST(x AS DOUBLE), CAST(0.0 AS DOUBLE)) AS geom
             FROM VALUES (1,0.0),(2,100.0),(3,200.0),(4,10000.0),(5,10100.0),(6,50000.0) AS t(id, x)""").createOrReplaceTempView("a")
spark.sql("""SELECT id, ST_Point(CAST(lon AS DOUBLE), CAST(0.0 AS DOUBLE)) AS geom
             FROM VALUES (1,0.00),(2,0.01),(3,0.02),(4,0.03) AS t(id, lon)""").createOrReplaceTempView("b")
qa = "SELECT id, ST_X(geom) AS x, ST_DBSCAN(geom, 150.0, 2, false) AS res FROM a ORDER BY id"

run("Test A (no checkpoint dir): planar, metres, eps=150, minPoints=2 (expect ids 1-3 one cluster, 4-5 another, 6 noise)", qa)

spark.sparkContext.setCheckpointDir(base + "/spark-tmp/ckpt")
run("Test A2 (checkpoint dir set): same query", qa)

for eps in (400.0, 1200.0):
    run(f"Test B (checkpoint dir set): useSpheroid=true, eps={eps} m (spacing ~1113 m on Earth, ~303 m on the Moon)",
        f"SELECT id, ST_X(geom) AS lon, ST_DBSCAN(geom, {eps}, 2, true) AS res FROM b ORDER BY id")
print("\nDONE", flush=True)
spark.stop()
