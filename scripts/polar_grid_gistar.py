import os, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sedona.spark import SedonaContext
from sedona.spark.stats.weighting import add_binary_distance_band_column
from sedona.spark.stats.hotspot_detection.getis_ord import g_local

base = os.path.expanduser("~/lunargrid/data")
P = base + "/processed"
R = 1737400.0
CELL, NC = 20000.0, 77
HALF = NC * CELL / 2.0            # 770,000 m  (cap down to about 65 deg)
THR = 61000.0
def log(*a): print(*a, flush=True)
t0 = time.time()

config = (SedonaContext.builder().appName("polar_grid_gistar").master("local[16]")
          .config("spark.driver.memory", "32g").config("spark.local.dir", base + "/spark-tmp")
          .config("spark.sql.shuffle.partitions", "32")
          .config("spark.jars.packages", "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.9.1,org.datasyslab:geotools-wrapper:1.9.1-33.5")
          .config("spark.jars.repositories", "https://artifacts.unidata.ucar.edu/repository/unidata-all").getOrCreate())
spark = SedonaContext.create(config)
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir(base + "/spark-tmp/ckpt")

cats = {"robbins": P + "/craters_clean.parquet", "lu5m": P + "/craters_lu5m.parquet", "luc": P + "/craters_luc.parquet"}
data = {k: pd.read_parquet(v, columns=["lon", "lat", "diam_km"]) for k, v in cats.items()}

def gistar_numpy(xy, val):
    n = len(val)
    nb = cKDTree(xy).query_ball_point(xy, r=THR)
    rows = np.repeat(np.arange(n), [len(a) for a in nb]); cols = np.concatenate([np.array(a) for a in nb])
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    sw = np.asarray(W.sum(axis=1)).ravel()
    xb, s = val.mean(), np.sqrt((val**2).mean() - val.mean()**2)
    return (W @ val - xb * sw) / (s * np.sqrt((n * sw - sw**2) / (n - 1)))

def pick(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise SystemExit(f"column {name} not in {list(df.columns)}")

xs = -HALF + CELL / 2 + CELL * np.arange(NC)
GX, GY = np.meshgrid(xs, xs)                       # GY[row=iy, col=ix]
rho_all = np.hypot(GX, GY).ravel()
mask = rho_all <= HALF
out_rows, summaries = [], []
log("RESULTS")
for pole in ("south", "north"):
    sign = -1 if pole == "south" else 1
    colat = 2 * np.arctan(rho_all[mask] / (2 * R))
    latc = sign * (90 - np.degrees(colat))
    k = 2 / (1 + np.sin(np.radians(np.abs(latc))))
    area_km2 = (CELL / 1000.0) ** 2 / k**2
    cx, cy = GX.ravel()[mask], GY.ravel()[mask]
    lonc = np.degrees(np.arctan2(cx, cy if pole == "south" else -cy))
    cell_id = np.flatnonzero(mask)
    for name, d in data.items():
        for thr in (1, 2):
            sel = (d.diam_km >= thr) & ((d.lat <= -65) if pole == "south" else (d.lat >= 65))
            lam, phi = np.radians(d.lon[sel].values), np.radians(d.lat[sel].values)
            if pole == "south":
                rho = 2 * R * np.tan(np.pi / 4 + phi / 2); x = rho * np.sin(lam); y = rho * np.cos(lam)
            else:
                rho = 2 * R * np.tan(np.pi / 4 - phi / 2); x = rho * np.sin(lam); y = -rho * np.cos(lam)
            ix, iy = np.floor((x + HALF) / CELL).astype(int), np.floor((y + HALF) / CELL).astype(int)
            ok = (ix >= 0) & (ix < NC) & (iy >= 0) & (iy < NC)
            cnt = np.bincount((iy[ok] * NC + ix[ok]), minlength=NC * NC)[cell_id]
            dens = cnt / area_km2 * 1000.0
            z_np = gistar_numpy(np.column_stack([cx, cy]), dens)
            sdf = spark.createDataFrame(pd.DataFrame({"id": cell_id.astype(int), "x": cx, "y": cy, "density": dens}))
            sdf.createOrReplaceTempView("c0")
            cells = spark.sql("SELECT id, CAST(density AS DOUBLE) AS density, ST_Point(x, y) AS geometry FROM c0")
            w = add_binary_distance_band_column(cells, THR, include_self=True, saved_attributes=["id", "density"])
            h = g_local(w, "density", star=True)
            hp = h.toPandas()
            idc, zc, pc = pick(hp, "id"), pick(hp, "Z"), pick(hp, "P")
            m = pd.DataFrame({"id": cell_id.astype(int), "z_np": z_np}).merge(hp[[idc, zc, pc]].rename(columns={idc: "id", zc: "z_sed", pc: "p_sed"}), on="id", how="left")
            diff = float(np.nanmax(np.abs(m.z_np - m.z_sed))) if m.z_sed.notna().any() else float("nan")
            log(f"{pole}/{name}/D>={thr}km: cells {len(m):,}, craters in grid {int(cnt.sum()):,}; Sedona cells returned {int(m.z_sed.notna().sum()):,}; max |Z numpy - Z Sedona| = {diff:.2e}   [{time.time()-t0:.0f} s]")
            df = pd.DataFrame({"pole": pole, "catalogue": name, "thr_km": thr, "id": cell_id, "x": cx, "y": cy, "lat": latc, "lon": lonc,
                               "count": cnt, "area_km2": area_km2, "density": dens, "z_numpy": z_np})
            df = df.merge(m[["id", "z_sed", "p_sed"]], on="id", how="left")
            out_rows.append(df)
res = pd.concat(out_rows, ignore_index=True)
res.to_csv(P + "/polar_cells_gistar.csv", index=False)
res["ring"] = pd.cut(res.lat.abs(), [65, 70, 75, 80, 85, 90], right=False)
for pole in ("south", "north"):
    for thr in (1, 2):
        log(f"\n===== {pole} cap, D >= {thr} km: by latitude ring (density per 1000 km^2; share of cells with Z>2.58 using numpy Z) =====")
        for name in cats:
            r = res[(res.pole == pole) & (res.catalogue == name) & (res.thr_km == thr)]
            g = r.groupby("ring", observed=True).agg(cells=("z_numpy", "size"), density=("density", "mean"), mean_z=("z_numpy", "mean"),
                                                     hot=("z_numpy", lambda s: (s > 2.58).mean()), cold=("z_numpy", lambda s: (s < -2.58).mean()))
            log(f"  {name}\n    " + g.round(3).to_string().replace("\n", "\n    "))
            top = r.sort_values("z_numpy", ascending=False).head(3)
            log("    top 3 hot cells (lat, lon, density, Z): " + "; ".join(f"({a:.1f}, {b:.0f}, {c:.0f}, {e:.1f})" for a, b, c, e in zip(top.lat, top.lon, top.density, top.z_numpy)))
log(f"\nDONE {time.time()-t0:.0f} s (saved polar_cells_gistar.csv)")
spark.stop()
