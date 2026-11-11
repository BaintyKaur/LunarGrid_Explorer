import os, itertools
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

P = os.path.expanduser("~/lunargrid/data/processed")
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

d = pd.read_csv(P + "/polar_cells_gistar.csv")
cats = ["robbins", "lu5m", "luc"]
log("RESULTS")
for pole in ("south", "north"):
    for thr in (1, 2):
        sub = d[(d.pole == pole) & (d.thr_km == thr)]
        base = sub[sub.catalogue == "robbins"].sort_values("id").reset_index(drop=True)
        n = len(base)
        xy = base[["x", "y"]].values
        ring = pd.cut(base.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False)
        k = int(np.ceil(0.05 * n))
        e_i = k * k / n
        j_rand = e_i / (2 * k - e_i)
        dens, dens_rel, z_raw, z_rel = {}, {}, {}, {}
        for c in cats:
            s = sub[sub.catalogue == c].sort_values("id").reset_index(drop=True)
            assert (s.id.values == base.id.values).all()
            dens[c] = s.density.values
            ringmean = pd.Series(dens[c]).groupby(ring.values, observed=True).transform("mean").values
            dens_rel[c] = dens[c] / ringmean
            z_raw[c] = s.z_numpy.values
            z_rel[c] = gistar(xy, dens_rel[c])
            log(f'    check {pole} D>={thr} {c}: NaN ring={int(ring.isna().sum())}, NaN ringmean={int(np.isnan(ringmean).sum())}, '
                f'NaN dens_rel={int(np.isnan(dens_rel[c]).sum())}, NaN z_rel={int(np.isnan(z_rel[c]).sum())}, '
                f'z_rel min/max={np.nanmin(z_rel[c]):.2f}/{np.nanmax(z_rel[c]):.2f}, dens_rel min/max={np.nanmin(dens_rel[c]):.2f}/{np.nanmax(dens_rel[c]):.2f}')
            if np.isnan(z_rel[c]).any():
                raise SystemExit('NaN in z_rel: stopping so we do not rank NaNs')
        log("    z_rel correlation between catalogues (Pearson): " + ", ".join(f"{a}-{b}: {np.corrcoef(z_rel[a], z_rel[b])[0,1]:.3f}" for a, b in itertools.combinations(cats, 2)))
        log(f"\n===== {pole} cap, D >= {thr} km: {n} cells, top 5% = {k} cells (random Jaccard about {j_rand:.3f}) =====")
        for label, zz, dd in (("raw density", z_raw, dens), ("density relative to its ring mean", z_rel, dens_rel)):
            tops = {c: set(np.argsort(-zz[c])[:k]) for c in cats}
            parts = []
            for a, b in itertools.combinations(cats, 2):
                i = len(tops[a] & tops[b]); parts.append(f"{a}-{b}: {i / (2 * k - i):.3f}")
            all3 = len(tops["robbins"] & tops["lu5m"] & tops["luc"])
            rho = [f"{a}-{b}: {pd.Series(dd[a]).corr(pd.Series(dd[b]), method='spearman'):.3f}" for a, b in itertools.combinations(cats, 2)]
            log(f"  [{label}]")
            log("    Jaccard of top-5% hot cells : " + ", ".join(parts))
            log(f"    cells hot in all three      : {all3}   (expected by chance: {k**3 / n**2:.1f})")
            log("    Spearman of cell densities  : " + ", ".join(rho))
log("\nDONE")
