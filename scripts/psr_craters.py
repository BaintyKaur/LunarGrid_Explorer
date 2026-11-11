import os
import numpy as np

R = 1737400.0
HALF = 304000.0      # map half-width in metres (corner coordinate from gdalinfo)
PX = 20.0            # pixel size in metres
LAT_MAX = -80.0      # only craters south of this latitude are on the map


def south_polar_xy(lat_deg, lon_deg):
    """Polar stereographic (south, sphere R, scale 1) -> x, y in metres."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    rho = 2 * R * np.tan(np.pi / 4 + phi / 2)
    return rho * np.sin(lam), rho * np.cos(lam)


def main():
    import pandas as pd
    import rasterio

    base = os.path.expanduser("~/lunargrid/data")
    cr = pd.read_parquet(base + "/processed/craters_clean.parquet")
    cr = cr[cr.lat <= LAT_MAX].copy()
    print("craters south of 80S:", len(cr), flush=True)

    x, y = south_polar_xy(cr.lat.values, cr.lon.values)
    col = np.floor((x + HALF) / PX).astype(int)
    row = np.floor((HALF - y) / PX).astype(int)

    print("reading the shadow map ...", flush=True)
    with rasterio.open(base + "/polar/LPSR_80S_20MPP_ADJ.TIF") as src:
        z = src.read(1)
    n = z.shape[0]
    ok = (col >= 0) & (col < n) & (row >= 0) & (row < n)
    val = np.full(len(cr), np.nan, dtype="float32")
    val[ok] = z[row[ok], col[ok]]
    cr["psr_value"] = val
    cr["in_psr"] = cr.psr_value > 0.5

    # share of map area in shadow, inside the 80S circle only
    inside = 0
    shadow = 0
    cx = (np.arange(n) + 0.5) * PX - HALF
    r80 = 2 * R * np.tan(np.radians(5.0))
    for r0 in range(0, n, 1000):
        zz = z[r0:r0 + 1000]
        yy = HALF - (np.arange(r0, r0 + zz.shape[0]) + 0.5) * PX
        m = np.hypot(cx[None, :], yy[:, None]) <= r80
        inside += int(m.sum())
        shadow += int(((zz > 0.5) & m).sum())
    print("area in shadow, inside 80S circle: %.2f %%" % (100.0 * shadow / inside))

    print("craters with a valid pixel:", int(ok.sum()), "of", len(cr))
    print("craters whose centre is in shadow: %d (%.2f %%)" % (cr.in_psr.sum(), 100 * cr.in_psr.mean()))
    for lo in (0, 2, 5, 10, 20):
        s = cr[cr.diam_km >= lo]
        print("diam >= %2d km: %6d craters, %5.2f %% centre in shadow" % (lo, len(s), 100 * s.in_psr.mean()))

    out = base + "/processed/craters_psr_south.parquet"
    cr.to_parquet(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
