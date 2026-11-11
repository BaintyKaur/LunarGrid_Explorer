"""Maps 1-4: global crater distribution, density per catalogue, elevation, slope.

Reads : data/processed/craters_clean.parquet, tiles_gistar.csv, data/dem/ldem_118m.tif
Writes: figures/map01_crater_distribution.png, map02_density_global.png,
        map03_elevation.png, map04_slope.png  (+ small arrays dem_small_*.npy for re-use)
Run   : python maps_1_4.py 2>&1 | tee maps_1_4.log
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tile_terrain import band_slope

BASE = os.path.expanduser("~/lunargrid/data")
FIG = os.path.expanduser("~/lunargrid/figures")
os.makedirs(FIG, exist_ok=True)
INK, MUTED = "#14181F", "#4A5260"
plt.rcParams["font.family"] = "DejaVu Sans"
EXT = [-180, 180, -90, 90]


def base_axes(ax, title):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(color="#D5D5D5", lw=0.5)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_xlabel("longitude (°E)", fontsize=9, color=MUTED)
    ax.set_ylabel("latitude (°N)", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")


def save(fig, name):
    fig.savefig(FIG + "/" + name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


# ---------------- map 1: crater distribution (Robbins) ----------------
def map1():
    t = time.time()
    c = pd.read_parquet(BASE + "/processed/craters_clean.parquet", columns=["lat", "lon", "diam_km"])
    classes = [(1, 2, 0.15, "#A9BBD0", "1 to 2 km"), (2, 5, 0.9, "#3F73B0", "2 to 5 km"),
               (5, 20, 4.0, "#E08A1E", "5 to 20 km"), (20, 1e9, 14.0, "#C0392B", "20 km and above")]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    for lo, hi, s, col, lab in classes:       # small first, large on top
        m = (c.diam_km >= lo) & (c.diam_km < hi)
        ax.scatter(c.lon[m], c.lat[m], s=s, c=col, linewidths=0, rasterized=True)
        print("  %-16s %8d craters" % (lab, m.sum()), flush=True)
    base_axes(ax, "Global distribution of %s lunar craters of 1 km and above (Robbins catalogue)" % format(len(c), ","))
    handles = [Line2D([], [], marker="o", ls="none", ms=max(2, s ** 0.5 * 1.6), color=col, label=lab)
               for lo, hi, s, col, lab in classes]
    ax.legend(handles=handles, title="crater diameter", loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=4, frameon=False, fontsize=9, title_fontsize=9)
    save(fig, "map01_crater_distribution.png")
    print("map 1 done in %.0f s" % (time.time() - t), flush=True)


# ---------------- map 2: tile density, three catalogues ----------------
def map2():
    t = time.time()
    d = pd.read_csv(BASE + "/processed/tiles_gistar.csv").sort_values("tile_id")
    names = [("robbins", "Robbins"), ("lu5m", "LU5M812TGT"), ("luc", "LUC-GRAS200")]
    grids = {}
    for k, _ in names:
        g = d["dens1_" + k].to_numpy().reshape(36, 72)       # row 0 = south, col 0 = lon -180
        grids[k] = g
        print("  %s: median %.1f, 99th pct %.1f, max %.1f per 1,000 km2" % (k, np.nanmedian(g), np.nanpercentile(g, 99), np.nanmax(g)), flush=True)
    vmax = max(np.nanpercentile(g, 99) for g in grids.values())
    fig, axs = plt.subplots(3, 1, figsize=(10, 14))
    for ax, (k, lab) in zip(axs, names):
        im = ax.imshow(grids[k], origin="lower", extent=EXT, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        base_axes(ax, "%s: craters of 1 km and above per 1,000 km² (5° × 5° tiles)" % lab)
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.025, pad=0.05, extend="max")
    cb.set_label("craters per 1,000 km² (same scale in all three maps; top 1% of tiles saturate)", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    save(fig, "map02_density_global.png")
    print("map 2 done in %.0f s" % (time.time() - t), flush=True)


# ---------------- maps 3 and 4: elevation and slope ----------------
def dem_small(f=32):
    """Block-average the DEM and the Horn slope (blocks of f x f pixels, about 3.8 km)."""
    import rasterio
    ep, sp = BASE + "/processed/dem_small_elev.npy", BASE + "/processed/dem_small_slope.npy"
    if os.path.exists(ep) and os.path.exists(sp):
        return np.load(ep), np.load(sp)
    with rasterio.open(BASE + "/dem/ldem_118m.tif") as src:
        px, nodata = src.transform.a, src.nodata
        nrows, ncols = src.height, src.width
        step = 1280
        nb = nrows // step
        elev = np.full((nrows // f, ncols // f), np.nan, "float32")
        slope = np.full_like(elev, np.nan)
        dpr = 180.0 / nrows
        for bi in range(nb):                                   # bi = 0 is the northernmost band
            t = time.time()
            r0, r1 = bi * step, (bi + 1) * step
            w0, w1 = max(r0 - 1, 0), min(r1 + 1, nrows)
            p = src.read(1, window=((w0, w1), (0, ncols))).astype("float32")
            p[p == nodata] = np.nan
            p *= 0.5
            if r0 == 0:
                p = np.vstack([p[:1], p])
            if r1 == nrows:
                p = np.vstack([p, p[-1:]])
            lat_rows = 90.0 - (np.arange(r0, r1) + 0.5) * dpr
            ob = slice(bi * step // f, (bi + 1) * step // f)
            z = p[1:-1].reshape(step // f, f, ncols // f, f)
            elev[ob] = np.nanmean(z, axis=(1, 3))
            lat_lo = -90 + (nb - 1 - bi) * 5
            if max(abs(lat_lo), abs(lat_lo + 5)) <= 80:
                s = band_slope(p, lat_rows, px).reshape(step // f, f, ncols // f, f)
                slope[ob] = np.nanmean(s, axis=(1, 3))
            print("  DEM band %d of %d done (%.0f s)" % (bi + 1, nb, time.time() - t), flush=True)
    np.save(ep, elev)
    np.save(sp, slope)
    return elev, slope


def map3_4():
    t = time.time()
    elev, slope = dem_small()
    print("  elevation km: min %.2f max %.2f | slope deg: median %.1f, 99th pct %.1f" %
          (np.nanmin(elev) / 1000, np.nanmax(elev) / 1000, np.nanmedian(slope), np.nanpercentile(slope, 99)), flush=True)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    im = ax.imshow(elev / 1000.0, extent=EXT, cmap="cividis", interpolation="nearest")
    base_axes(ax, "Lunar surface elevation (LOLA 118 m model, averaged to about 3.8 km)")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.14)
    cb.set_label("elevation above the reference sphere (km)", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    save(fig, "map03_elevation.png")

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_facecolor("#E3E3E3")
    im = ax.imshow(slope, extent=EXT, cmap="magma_r", vmin=0, vmax=20, interpolation="nearest")
    base_axes(ax, "Lunar terrain steepness: Horn slope with latitude correction (mean per 3.8 km block)")
    ax.axhline(80, color=MUTED, lw=0.6, ls="--")
    ax.axhline(-80, color=MUTED, lw=0.6, ls="--")
    ax.text(178, 83, "slope not computed poleward of 80°", fontsize=7.5, color=MUTED, ha="right")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.14, extend="max")
    cb.set_label("slope (degrees); low = gentle, high = steep", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8)
    save(fig, "map04_slope.png")
    print("maps 3 and 4 done in %.0f s" % (time.time() - t), flush=True)


if __name__ == "__main__":
    print("===== RESULTS =====", flush=True)
    map1()
    map2()
    map3_4()
    print("all done", flush=True)