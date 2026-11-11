import os
import numpy as np

R = 1737400.0          # Moon radius in metres
TILE_DEG = 5           # tile size in degrees
SLOPE_MAX_LAT = 80.0   # slope is only trusted up to this latitude


def band_slope(p, lat_deg, px):
    """Horn slope (degrees) for the middle rows of p, corrected for latitude."""
    a, b, c = p[:-2], p[1:-1], p[2:]
    dx = (px * np.cos(np.radians(lat_deg))).astype("float32")[:, None]
    t = a + 2 * b + c
    dzdx = (np.roll(t, -1, axis=1) - np.roll(t, 1, axis=1)) / (8 * dx)

    def rowfilter(v):
        return np.roll(v, 1, axis=1) + 2 * v + np.roll(v, -1, axis=1)

    dzdy = (rowfilter(c) - rowfilter(a)) / (8 * px)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def main():
    import pandas as pd
    import rasterio

    base = os.path.expanduser("~/lunargrid/data")
    with rasterio.open(base + "/dem/ldem_118m.tif") as src:
        px = src.transform.a
        nodata = src.nodata
        print("reading the DEM into memory ...", flush=True)
        dem = src.read(1)
    nrows, ncols = dem.shape
    step = nrows // (180 // TILE_DEG)
    tcols = ncols // (360 // TILE_DEG)
    nbands = nrows // step
    ntilecols = ncols // tcols
    deg_per_row = 180.0 / nrows
    print("DEM", dem.shape, "| tile", step, "x", tcols, "| bands", nbands, flush=True)

    recs = []
    for bi in range(nbands):
        r0, r1 = bi * step, (bi + 1) * step
        top = dem[max(r0 - 1, 0):r0]
        bot = dem[r1:min(r1 + 1, nrows)]
        mid = dem[r0:r1]
        if r0 == 0:
            top = mid[:1]
        if r1 == nrows:
            bot = mid[-1:]
        p = np.vstack([top, mid, bot]).astype("float32")
        p[p == nodata] = np.nan
        p *= 0.5

        lat_rows = 90.0 - (np.arange(r0, r1) + 0.5) * deg_per_row
        lat_lo_idx = nbands - 1 - bi
        lat_lo = -90 + lat_lo_idx * TILE_DEG
        lat_hi = lat_lo + TILE_DEG
        slope_ok = max(abs(lat_lo), abs(lat_hi)) <= SLOPE_MAX_LAT

        z = p[1:-1].reshape(step, ntilecols, tcols)
        elev_mean = np.nanmean(z, axis=(0, 2))
        elev_std = np.nanstd(z, axis=(0, 2))
        if slope_ok:
            s = band_slope(p, lat_rows, px).reshape(step, ntilecols, tcols)
            slope_mean = np.nanmean(s, axis=(0, 2))
        else:
            slope_mean = np.full(ntilecols, np.nan)

        for cj in range(ntilecols):
            recs.append((lat_lo_idx * ntilecols + cj, lat_lo, lat_hi,
                         -180 + cj * TILE_DEG, -180 + (cj + 1) * TILE_DEG,
                         float(elev_mean[cj]), float(elev_std[cj]), float(slope_mean[cj])))
        print("band", bi + 1, "of", nbands, "| lat", lat_lo, "to", lat_hi,
              "| slope", "ok" if slope_ok else "skipped", flush=True)

    tiles = pd.DataFrame(recs, columns=["tile_id", "lat_min", "lat_max", "lon_min", "lon_max",
                                        "mean_elev_m", "std_elev_m", "mean_slope_deg"])
    tiles = tiles.sort_values("tile_id").reset_index(drop=True)

    dlon = np.radians(TILE_DEG)
    tiles["area_km2"] = ((R / 1000.0) ** 2 * dlon *
                         (np.sin(np.radians(tiles.lat_max)) - np.sin(np.radians(tiles.lat_min))))

    cr = pd.read_parquet(base + "/processed/craters_with_elev.parquet")
    n_all = cr.groupby("tile_id").size().rename("n_craters")
    n_ge2 = cr[cr.diam_km >= 2].groupby("tile_id").size().rename("n_craters_ge2km")
    tiles = tiles.join(n_all, on="tile_id").join(n_ge2, on="tile_id")
    tiles[["n_craters", "n_craters_ge2km"]] = tiles[["n_craters", "n_craters_ge2km"]].fillna(0).astype(int)
    tiles["density_per_1000km2"] = tiles.n_craters / tiles.area_km2 * 1000
    tiles["density_ge2km_per_1000km2"] = tiles.n_craters_ge2km / tiles.area_km2 * 1000

    out = base + "/processed/tiles_terrain.parquet"
    tiles.to_parquet(out, index=False)
    tiles.to_csv(base + "/processed/tiles_terrain.csv", index=False)
    print("saved:", out)
    print("tiles:", len(tiles), "| craters counted:", int(tiles.n_craters.sum()))
    print(tiles[["mean_elev_m", "std_elev_m", "mean_slope_deg", "area_km2",
                 "density_per_1000km2"]].describe().T[["count", "min", "50%", "max"]])
    print("slope available for", int(tiles.mean_slope_deg.notna().sum()), "tiles")


if __name__ == "__main__":
    main()
