import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

P = os.path.expanduser("~/lunargrid/data/processed")
R, CELL, NC = 1737400.0, 20000.0, 77
HALF = NC * CELL / 2.0
THR = 61000.0
def log(*a): print(*a, flush=True)

def gistar(xy, val):
    n = len(val)
    nb = cKDTree(xy).query_ball_point(xy, r=THR)
    rows = np.repeat(np.arange(n), [len(a) for a in nb]); cols = np.concatenate([np.array(a) for a in nb])
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    sw = np.asarray(W.sum(axis=1)).ravel()
    xb, s = val.mean(), np.sqrt((val**2).mean() - val.mean()**2)
    return (W @ val - xb * sw) / (s * np.sqrt((n * sw - sw**2) / (n - 1)))

cells = pd.read_csv(P + "/polar_cells_gistar.csv")
luc = pd.read_parquet(P + "/craters_luc.parquet", columns=["lon", "lat", "diam_km", "secondary_flag"])
band = luc[(luc.diam_km >= 1.0) & (luc.diam_km <= 2.0)]
log("RESULTS")
log("secondary_flag counts for LUC craters of 1-2 km (-1 = not assessed, 0 = non-secondary, 1 = possible, 2 = probable, 3 = certain):")
log("  " + band.secondary_flag.value_counts().sort_index().to_string().replace("\n", "\n  "))

for pole in ("south", "north"):
    c = cells[(cells.pole == pole) & (cells.catalogue == "robbins") & (cells.thr_km == 1)].sort_values("id").reset_index(drop=True)
    ids, xy, area = c.id.values.astype(int), c[["x", "y"]].values, c.area_km2.values
    ring = pd.cut(c.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False)
    sel = band[(band.lat <= -65) if pole == "south" else (band.lat >= 65)]
    lam, phi = np.radians(sel.lon.values), np.radians(sel.lat.values)
    if pole == "south":
        rho = 2 * R * np.tan(np.pi / 4 + phi / 2); x = rho * np.sin(lam); y = rho * np.cos(lam)
    else:
        rho = 2 * R * np.tan(np.pi / 4 - phi / 2); x = rho * np.sin(lam); y = -rho * np.cos(lam)
    ix, iy = np.floor((x + HALF) / CELL).astype(int), np.floor((y + HALF) / CELL).astype(int)
    ok = (ix >= 0) & (ix < NC) & (iy >= 0) & (iy < NC)
    flat = iy[ok] * NC + ix[ok]
    flags = sel.secondary_flag.values[ok]
    def counts(mask): return np.bincount(flat[mask], minlength=NC * NC)[ids]
    variants = {"all craters": counts(np.ones(len(flat), bool)),
                "without probable+certain": counts((flags <= 1)),
                "non-secondary only": counts(flags == 0)}
    zs = {}
    for name, cnt in variants.items():
        dens = cnt / area * 1000.0
        rel = dens / pd.Series(dens).groupby(ring.values, observed=True).transform("mean").values
        zs[name] = gistar(xy, rel)
        log(f"\n[{pole}] {name}: {int(cnt.sum()):,} craters (1-2 km) in the cap")
    k = int(np.ceil(0.05 * len(ids)))
    top_all = set(np.argsort(-zs["all craters"])[:k])
    log(f"\n===== {pole} cap, craters of 1-2 km, top 5% = {k} cells =====")
    for name in ("without probable+certain", "non-secondary only"):
        tv = set(np.argsort(-zs[name])[:k])
        log(f"  {name}: Z correlation with 'all craters' = {np.corrcoef(zs['all craters'], zs[name])[0, 1]:.3f}; top-5% cells kept = {len(top_all & tv)} of {k} ({len(top_all & tv) / k * 100:.0f}%)")
    hot_flat = set(ids[list(top_all)])
    in_hot = np.isin(flat, list(hot_flat))
    sec = flags >= 2
    log(f"  probable+certain secondaries as share of all 1-2 km craters: inside hot cells {sec[in_hot].mean() * 100:.1f}% ({int(in_hot.sum()):,} craters), elsewhere {sec[~in_hot].mean() * 100:.1f}% ({int((~in_hot).sum()):,} craters)")
log("\nDONE")
