"""Research questions, step A: elevation and slope for every 20 km equal-area polar cell.

The 118 m DEM (equirectangular) is projected pixel by pixel onto the polar stereographic grid
of the polar analysis and averaged per cell. Pixels are weighted by cos(latitude), because
a DEM pixel covers less ground near the pole. Slope is the Horn slope with latitude correction,
only used up to 80 degrees (poleward of that it is left empty, as everywhere in the project).

Reads : data/dem/ldem_118m.tif   (needs tile_terrain.py in the same folder)
Writes: data/processed/cells_terrain.csv   (pole, ix, iy, elev_mean_m, elev_std_m, slope_mean_deg, n_pix)
Run   : nohup python cell_terrain.py > cell_terrain.log 2>&1 &   then   tail -f cell_terrain.log
"""
import os
import time
import numpy as np
import pandas as pd
import rasterio
from tile_terrain import band_slope

R = 1737400.0
CELL, HALF, NC = 20000.0, 770000.0, 77
BASE = os.path.expanduser("~/lunargrid/data")
RAD = np.pi / 180.0


def project(lat, lon, north):
    rho = 2 * R * np.tan((90.0 - np.abs(lat)) * RAD / 2)
    x = rho * np.sin(lon * RAD)
    y = rho * np.cos(lon * RAD) * (-1.0 if north else 1.0)
    return x, y


def accumulate(acc, lat_rows, lon, z, s, north):
    """Add one band of pixels to the per-cell sums. lat_rows: (n,), z and s: (n, ncols)."""
    n, nc = z.shape
    lat = np.repeat(lat_rows[:, None], nc, axis=1)
    lo = np.broadcast_to(lon[None, :], (n, nc))
    x, y = project(lat, lo, north)
    ix = np.floor((x + HALF) / CELL).astype(np.int64)
    iy = np.floor((y + HALF) / CELL).astype(np.int64)
    ok = (ix >= 0) & (ix < NC) & (iy >= 0) & (iy < NC)
    cell = (iy * NC + ix)[ok]
    w = np.cos(lat * RAD)[ok]
    zz, ss = z[ok], s[ok]
    zok, sok = ~np.isnan(zz), ~np.isnan(ss)
    m = NC * NC
    acc["w_z"] += np.bincount(cell[zok], weights=w[zok], minlength=m)
    acc["z"] += np.bincount(cell[zok], weights=(w * zz)[zok], minlength=m)
    acc["z2"] += np.bincount(cell[zok], weights=(w * zz * zz)[zok], minlength=m)
    acc["w_s"] += np.bincount(cell[sok], weights=w[sok], minlength=m)
    acc["s"] += np.bincount(cell[sok], weights=(w * ss)[sok], minlength=m)
    acc["n"] += np.bincount(cell, minlength=m)


def main():
    t0 = time.time()
    with rasterio.open(BASE + "/dem/ldem_118m.tif") as src:
        px, nodata, nrows, ncols = src.transform.a, src.nodata, src.height, src.width
        dpr = 180.0 / nrows
        lon = -180.0 + (np.arange(ncols) + 0.5) * 360.0 / ncols
        rmax = int(25.0 / dpr)                       # rows from the pole down to 65 degrees
        print("===== RESULTS =====\nDEM %d x %d | rows per pole cap: %d" % (ncols, nrows, rmax), flush=True)
        out = []
        for north in (True, False):
            acc = {k: np.zeros(NC * NC) for k in ("w_z", "z", "z2", "w_s", "s", "n")}
            rows = range(0, rmax, 640) if north else range(nrows - rmax, nrows, 640)
            for r0 in rows:
                r1 = min(r0 + 640, rmax if north else nrows)
                w0, w1 = max(r0 - 1, 0), min(r1 + 1, nrows)
                p = src.read(1, window=((w0, w1), (0, ncols))).astype("float32")
                p[p == nodata] = np.nan
                p *= 0.5
                if r0 == 0:
                    p = np.vstack([p[:1], p])
                if r1 == nrows:
                    p = np.vstack([p, p[-1:]])
                lat_rows = 90.0 - (np.arange(r0, r1) + 0.5) * dpr
                s = band_slope(p, lat_rows, px)
                s[np.abs(lat_rows) > 80.0] = np.nan
                accumulate(acc, lat_rows, lon, p[1:-1], s, north)
                print("  %s rows %d-%d done (%.0f s)" % ("north" if north else "south", r0, r1, time.time() - t0), flush=True)
            idx = np.arange(NC * NC)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = acc["z"] / acc["w_z"]
                std = np.sqrt(np.maximum(acc["z2"] / acc["w_z"] - mean ** 2, 0))
                sl = acc["s"] / acc["w_s"]
            df = pd.DataFrame({"pole": "north" if north else "south", "ix": idx % NC, "iy": idx // NC,
                               "elev_mean_m": mean, "elev_std_m": std, "slope_mean_deg": sl, "n_pix": acc["n"].astype(int)})
            df = df[df.n_pix > 0]
            print("%s: %d cells | elevation %.0f to %.0f m | slope cells %d, mean %.1f deg" %
                  ("north" if north else "south", len(df), df.elev_mean_m.min(), df.elev_mean_m.max(),
                   df.slope_mean_deg.notna().sum(), df.slope_mean_deg.mean()), flush=True)
            out.append(df)
    pd.concat(out).to_csv(BASE + "/processed/cells_terrain.csv", index=False)
    print("saved cells_terrain.csv | total %.0f s\nall done" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()