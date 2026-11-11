import os
from sedona.spark import SedonaContext

tmp = os.path.expanduser("~/lunargrid/spark-tmp")
os.makedirs(tmp, exist_ok=True)

config = (
    SedonaContext.builder()
    .appName("lunargrid-test")
    .master("local[16]")
    .config("spark.driver.memory", "32g")
    .config("spark.local.dir", tmp)
    .config("spark.jars.packages",
            "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.9.1,"
            "org.datasyslab:geotools-wrapper:1.9.1-33.5")
    .config("spark.jars.repositories",
            "https://artifacts.unidata.ucar.edu/repository/unidata-all")
    .getOrCreate()
)
sedona = SedonaContext.create(config)
sedona.sql("SELECT ST_Point(1.0, 2.0) AS pt").show()
print("SEDONA OK")
