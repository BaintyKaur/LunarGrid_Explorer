"""Step 2: permutation test for the polar hotspots.

Null hypothesis: inside groups of comparable cells, crater density has no
spatial arrangement (cells are interchangeable). We shuffle densities inside
the groups, recompute Gi*, and see how extreme the real values are.

Two kinds of groups:
  A  ring only                 (keeps the north-cap trend with latitude)
  B  ring x sunlight tercile   (also keeps the sunlight effect found in step 1)

Two kinds of answer for every cell:
  p_cell  : one-cell p-value (not corrected for looking at 4,669 cells)
  p_maxT  : corrected p-value: how often does ANY cell in a shuffled map
            reach a Z as high as this cell (family-wise error control)
Also the Benjamini-Hochberg count (false-discovery rate 5%).

Reads : data/processed/polar_cells_gistar.csv, data/processed/cells_illumination.csv
Writes: data/processed/perm_hotspots.csv
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse

BASE = os.path.expanduser("~/lunargrid/data/processed")
THR = 61000.0
P = 999
RINGS = [65, 70, 75, 80, 85, 90.01]
SEED = 20261109


def build_w(x, y):
    tree = cKDTree(np.c_[x, y])
    nb = tree.query_ball_point(np.c_[x, y], r=THR)
    rows = np.repeat(np.arange(len(x)), [len(a) for a in nb])
    cols = np.concatenate([np.asarray(a) for a in nb])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(x), len(x)))


def gi_denominator(W, v):
    n = len(v)
    xbar = v.mean()
    s = np.sqrt((v ** 2).mean() - xbar ** 2)
    m = np.asarray(W.sum(axis=1)).ravel()
    den = s * np.sqrt((n * m - m ** 2) / (n - 1))
    return xbar * m, den


def shuffle_within(v, groups, rng):
    X = np.empty((len(v), P))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        order = np.argsort(rng.random((len(idx), P)), axis=0)
        X[idx, :] = v[idx][order]
    return X


def bh_count(p, q=0.05):
    ps = np.sort(p)
    m = len(p)
    ok = np.where(ps <= q * (np.arange(1, m + 1) / m))[0]
    return 0 if len(ok) == 0 else int(ok.max() + 1)


def main():
    t0 = time.time()
    cells = pd.read_csv(BASE + "/cells_illumination.csv")
    allc = pd.read_csv(BASE + "/polar_cells_gistar.csv")
    allc["pole"] = np.where(allc.lat > 0, "N", "S")
    d = allc.merge(cells[["pole", "id", "sun_mean", "psr_frac"]], on=["pole", "id"], how="left", validate="many_to_one")
    assert d.sun_mean.notna().all()
    rng = np.random.default_rng(SEED)

    out, summ = [], []
    for pole in ("N", "S"):
        dp = d[d.pole == pole]
        geo = dp.drop_duplicates("id").sort_values("id").reset_index(drop=True)
        W = build_w(geo.x.values, geo.y.values)
        ring = pd.cut(geo.lat.abs(), RINGS, right=False, labels=False).values
        assert not pd.isna(ring).any()
        ring = ring.astype(int)
        terc = pd.qcut(geo.sun_mean, 3, labels=False, duplicates="drop").values.astype(int)
        gA = ring
        gB = ring * 10 + terc
        print("%s: cells %d, mean neighbours %.1f, groups A %d, groups B %d" %
              (pole, len(geo), W.sum() / len(geo), len(np.unique(gA)), len(np.unique(gB))), flush=True)

        for (cat, thr), s in dp.groupby(["catalogue", "thr_km"]):
            s = s.sort_values("id").reset_index(drop=True)
            assert (s.id.values == geo.id.values).all()
            v = s.density.values.astype(float)
            assert np.isfinite(v).all()
            shift, den = gi_denominator(W, v)
            z_obs = (W @ v - shift) / den
            for name, grp in (("A_ring", gA), ("B_ring_sun", gB)):
                X = shuffle_within(v, grp, rng)
                Zp = (W @ X - shift[:, None]) / den[:, None]
                p_cell = (1 + (Zp >= z_obs[:, None]).sum(axis=1)) / (P + 1)
                maxz = Zp.max(axis=0)
                p_max = (1 + (maxz[None, :] >= z_obs[:, None]).sum(axis=1)) / (P + 1)
                thr_max = np.quantile(maxz, 0.95)
                n_cell = int((p_cell <= 0.05).sum())
                n_bh = bh_count(p_cell)
                n_max = int((p_max <= 0.05).sum())
                summ.append({"pole": pole, "catalogue": cat, "D>=km": thr, "null": name,
                             "Z>2.58": int((z_obs > 2.58).sum()), "p_cell<=.05": n_cell,
                             "BH_q.05": n_bh, "maxT_p<=.05": n_max, "Z95_of_max": round(thr_max, 2)})
                out.append(pd.DataFrame({"pole": pole, "catalogue": cat, "thr_km": thr, "null": name,
                                         "id": s.id.values, "lat": s.lat.values, "lon": s.lon.values,
                                         "z": z_obs, "p_cell": p_cell, "p_maxT": p_max}))
            print("  done %s %s D>=%s  (%.0f s)" % (pole, cat, thr, time.time() - t0), flush=True)

    res = pd.concat(out, ignore_index=True)
    res.to_csv(BASE + "/perm_hotspots.csv", index=False)
    sm = pd.DataFrame(summ)

    print("\n===== RESULTS =====")
    print("Cells (of 4,669 per pole) called hot, by method. Null A = shuffle inside rings;")
    print("null B = shuffle inside ring x sunlight groups. maxT is the strict test.\n")
    print(sm.to_string(index=False))

    print("\nCells significant by maxT (p<=0.05) in ALL THREE catalogues:")
    for (pole, thr, null), q in res.groupby(["pole", "thr_km", "null"]):
        sig = q[q.p_maxT <= 0.05].groupby("id").catalogue.nunique()
        print("  %s D>=%s %-10s all three: %4d" % (pole, thr, null, int((sig == 3).sum())))

    print("\nStrongest all-three consensus cells (null B, maxT; lat, lon, mean Z):")
    for (pole, thr), q in res[res.null == "B_ring_sun"].groupby(["pole", "thr_km"]):
        sig = q[q.p_maxT <= 0.05].groupby("id").catalogue.nunique()
        keep = sig[sig == 3].index
        if len(keep) == 0:
            print("  %s D>=%s: none" % (pole, thr))
            continue
        top = q[q.id.isin(keep)].groupby("id").agg(lat=("lat", "first"), lon=("lon", "first"), z=("z", "mean")).sort_values("z", ascending=False).head(5)
        for _, r in top.iterrows():
            print("  %s D>=%s: lat %6.1f lon %7.1f  meanZ %.1f" % (pole, thr, r.lat, r.lon, r.z))
    print("\nwrote", BASE + "/perm_hotspots.csv", " total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()