import os
import numpy as np

R = 1737400.0
HALF = 770400.0      # map half-width in metres (from gdalinfo)
PX = 240.0           # pixel size in metres
LAT_EDGE = 65.0      # maps cover this latitude to the pole


def polar_xy(lat_deg, lon_deg, north):
    """Polar stereographic on a sphere (scale 1 at the pole) -> x, y in metres."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    if north:
        rho = 2 * R * np.tan(np.pi / 4 - phi / 2)
        return rho * np.sin(lam), -rho * np.cos(lam)
    rho = 2 * R * np.tan(np.pi / 4 + phi / 2)
    return rho * np.sin(lam), rho * np.cos(lam)


def load_scaled(path):
    import rasterio
    with rasterio.open(path) as src:
        raw = src.read(1)
        scale, offset, nodata = src.scales[0], src.offsets[0], src.nodata
    arr = raw.astype("float32") * np.float32(scale) + np.float32(offset)
    arr[raw == nodata] = np.nan
    return arr


def main():
    import pandas as pd

    base = os.path.expanduser("~/lunargrid/data")
    cr_all = pd.read_parquet(base + "/processed/craters_clean.parquet")
    n = 6420
    r_edge = 2 * R * np.tan(np.radians((90 - LAT_EDGE) / 2))
    cx = (np.arange(n) + 0.5) * PX - HALF
    yy = HALF - (np.arange(n) + 0.5) * PX
    circle = np.hypot(cx[None, :], yy[:, None]) <= r_edge

    parts = []
    for hemi, north in (("N", True), ("S", False)):
        cr = cr_all[cr_all.lat >= LAT_EDGE] if north else cr_all[cr_all.lat <= -LAT_EDGE]
        cr = cr.copy()
        x, y = polar_xy(cr.lat.values, cr.lon.values, north)
        col = np.floor((x + HALF) / PX).astype(int)
        row = np.floor((HALF - y) / PX).astype(int)
        ok = (col >= 0) & (col < n) & (row >= 0) & (row < n)
        print("=== %s pole | craters poleward of 65: %d (on map: %d)" % (hemi, len(cr), ok.sum()), flush=True)

        ill = load_scaled(base + "/polar/AVGVISIB_65%s_240M_201608.TIF" % hemi)
        psr = load_scaled(base + "/polar/LPSR_65%s_240M_201608.TIF" % hemi)
        cr["illum"] = np.where(ok, ill[np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)], np.nan)
        cr["psr240"] = np.where(ok, psr[np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)], np.nan)
        cr["in_psr240"] = cr.psr240 > 0.5
        cr["hemi"] = hemi

        m_ill = np.nanmean(ill[circle])
        m_psr = np.nanmean(psr[circle] > 0.5)
        print("map inside 65 deg circle: mean sunlight fraction %.3f | area in shadow %.2f %%" % (m_ill, 100 * m_psr))
        print("craters: mean sunlight fraction %.3f | centre in shadow %.2f %% | missing values %d"
              % (np.nanmean(cr.illum), 100 * cr.in_psr240.mean(), cr.illum.isna().sum()))
        for lo in (0, 2, 5, 10, 20):
            s = cr[cr.diam_km >= lo]
            print("  diam >= %2d km: %6d craters | mean sunlight %.3f | %5.2f %% in shadow"
                  % (lo, len(s), s.illum.mean(), 100 * s.in_psr240.mean()))
        parts.append(cr)
        del ill, psr

    out = base + "/processed/craters_polar.parquet"
    pd.concat(parts).to_parquet(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
