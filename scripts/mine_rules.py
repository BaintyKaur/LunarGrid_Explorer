"""Step 5a: association rules (FP-Growth, Spark MLlib) on crater features.

Usage:  python mine_rules.py 16          # Spark local[16], prints the rules
        python mine_rules.py 1 quiet     # Spark local[1], prints only counts and timing

Every crater becomes a small set of labels ("transaction"). FP-Growth finds label
combinations that occur together often (support) and rules A => B with
  confidence = share of A-craters that also have B
  lift       = confidence / share of all craters with B   (1 = no association)

Two passes:
  ALL    all craters with the labels size, slope, elevation, latitude, side, hemisphere
  POLAR  craters poleward of 65 degrees, adding sunlight and shadow labels

Slope classes are terciles INSIDE each size class (slope depends on size by construction).
Reads : data/processed/craters_features.parquet
Writes: data/processed/rules_all.csv, rules_polar.csv, transactions_*.parquet
"""
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/lunargrid/data/processed")
SIZE_BINS = [0, 2, 5, 20, 1e9]
SIZE_LAB = ["size 1-2km", "size 2-5km", "size 5-20km", "size 20km+"]


def tercile_labels(x, name):
    q = pd.qcut(x, 3, labels=False, duplicates="drop")
    return q.map({0: name + " low", 1: name + " mid", 2: name + " high"})


def within_group_terciles(x, g, name):
    pct = x.groupby(g, observed=True).rank(pct=True)
    out = pd.Series(np.where(pct <= 1 / 3, name + " low", np.where(pct <= 2 / 3, name + " mid", name + " high")), index=x.index)
    out[x.isna()] = None
    return out


def to_items(cols):
    lists = zip(*[c.tolist() for c in cols])
    return [[v for v in row if isinstance(v, str)] for row in lists]


def build_all(f):
    size = pd.cut(f.diam_km, SIZE_BINS, labels=SIZE_LAB, right=False).astype(object)
    sizec = pd.Series(size, index=f.index)
    slope = within_group_terciles(f.slope_disc_deg, sizec, "slope")
    elev = tercile_labels(f.elev_m, "elev")
    lat = pd.cut(f.lat.abs(), [0, 30, 60, 70, 80, 90.01], labels=["lat 0-30", "lat 30-60", "lat 60-70", "lat 70-80", "lat 80-90"], right=False).astype(object)
    side = pd.Series(np.where(f.lon.abs() < 90, "side near", "side far"), index=f.index)
    hemi = pd.Series(np.where(f.lat > 0, "north", "south"), index=f.index)
    return to_items([sizec, slope, elev, pd.Series(lat, index=f.index), side, hemi])


def build_polar(f):
    p = f[f.sunlight_240m.notna()].copy()
    sizec = pd.Series(pd.cut(p.diam_km, SIZE_BINS, labels=SIZE_LAB, right=False).astype(object), index=p.index)
    slope = within_group_terciles(p.slope_disc_deg, sizec, "slope")
    elev = tercile_labels(p.elev_m, "elev")
    sun = tercile_labels(p.sunlight_240m, "sunlight")
    shadow = pd.Series(np.where(p.shadow_240m > 0.5, "in shadow", None), index=p.index)
    lat = pd.cut(p.lat.abs(), [65, 70, 75, 80, 90.01], labels=["lat 65-70", "lat 70-75", "lat 75-80", "lat 80-90"], right=False).astype(object)
    side = pd.Series(np.where(p.lon.abs() < 90, "side near", "side far"), index=p.index)
    hemi = pd.Series(np.where(p.lat > 0, "north", "south"), index=p.index)
    return to_items([sizec, slope, elev, sun, shadow, pd.Series(lat, index=p.index), side, hemi])


def run_pass(spark, name, items, min_support, min_conf, quiet):
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql import functions as F
    path = BASE + "/transactions_%s.parquet" % name
    pd.DataFrame({"items": items}).to_parquet(path, index=False)
    df = spark.read.parquet("file://" + path)
    n = df.count()
    print("\n--- pass %s: %d transactions, %.1f labels per transaction on average" % (name.upper(), n, np.mean([len(i) for i in items[:200000]])), flush=True)
    t0 = time.time()
    model = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_conf).fit(df)
    n_sets = model.freqItemsets.count()
    t_fit = time.time() - t0
    t1 = time.time()
    rules = model.associationRules
    n_rules = rules.count()
    t_rules = time.time() - t1
    print("frequent itemsets: %d (%.1f s) | rules: %d (%.1f s)" % (n_sets, t_fit, n_rules, t_rules), flush=True)
    r = rules.select(F.concat_ws(" & ", "antecedent").alias("if"), F.concat_ws(" & ", "consequent").alias("then"),
                     F.size("antecedent").alias("n_if"), "support", "confidence", "lift").toPandas()
    r.sort_values("lift", ascending=False).to_csv(BASE + "/rules_%s.csv" % name, index=False)
    if not quiet:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 70)
        fmt = lambda d: d[["if", "then", "support", "confidence", "lift"]].round(3).to_string(index=False)
        print("\nstrongest positive associations, one or two labels in the IF part (top 15 by lift):")
        print(fmt(r[r.n_if <= 2].sort_values("lift", ascending=False).head(15)))
        print("\nstrongest positive associations, any length (top 12 by lift):")
        print(fmt(r.sort_values("lift", ascending=False).head(12)))
        print("\nlabels that avoid each other (lowest lift, one or two labels in the IF part, 10 rows):")
        print(fmt(r[r.n_if <= 2].sort_values("lift").head(10)))
    return n_sets, n_rules, t_fit, t_rules


def main():
    cores = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    quiet = len(sys.argv) > 2 and sys.argv[2] == "quiet"
    t0 = time.time()
    f = pd.read_parquet(BASE + "/craters_features.parquet")
    items_all = build_all(f)
    items_pol = build_polar(f)
    print("labels built in %.1f s | craters %d | polar craters %d" % (time.time() - t0, len(items_all), len(items_pol)), flush=True)

    os.environ["PYSPARK_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("lunargrid-rules").master("local[%d]" % cores)
             .config("spark.driver.memory", "16g").config("spark.local.dir", os.path.expanduser("~/lunargrid/spark-tmp"))
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    print("===== RESULTS (local[%d]) =====" % cores, flush=True)
    a = run_pass(spark, "all", items_all, 0.02, 0.1, quiet)
    b = run_pass(spark, "polar", items_pol, 0.03, 0.1, quiet)
    print("\nTIMING local[%d]: ALL fit %.1f s + rules %.1f s | POLAR fit %.1f s + rules %.1f s | total %.0f s" %
          (cores, a[2], a[3], b[2], b[3], time.time() - t0))
    spark.stop()


if __name__ == "__main__":
    main()