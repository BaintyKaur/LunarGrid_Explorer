"""Research questions 1-4 on the 20 km equal-area polar cells.

Q1  Do permanently shadowed cells have fewer craters than lit cells, and does this depend on crater size?
Q2  Does terrain slope (65 to 80 degrees) change crater density and the mix of crater sizes?
Q3  Do the strict hotspots share elevation, slope or illumination conditions?
Q4  Where do the three catalogues disagree most, and is this linked to shadow, slope, elevation or latitude?

Reads : craters_clean / craters_lu5m / craters_luc .parquet, polar_cells_gistar.csv, cells_illumination.csv,
        cells_terrain.csv, perm_hotspots.csv   (needs maps_6_7.py)
Writes: figures/rq1_size_shadow.png, rq2_slope.png, rq3_hotspot_terrain.png, rq4_disagreement_map.png, rq4_disagreement_groups.png
        data/processed/rq1_size_shadow.csv, rq2_slope.csv, rq3_hotspots.csv, rq4_disagreement.csv
Run   : python research_questions.py 2>&1 | tee research_questions.log

Method notes
 - "Shadowed" cell: more than 5% permanent shadow. Ring = 5-degree latitude ring (65-70 ... 85-90).
 - Ring-adjusted density: a cell's density divided by the mean density of all cells in its ring, so that the
   trend with latitude does not decide the comparison. A ratio of 1 means no difference.
 - Intervals are bootstrap intervals over cells (500 resamples). Neighbouring cells are related, so the real
   uncertainty is larger than the intervals suggest: read them as indicative.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import maps_6_7 as M
from maps_6_7 import pick_cat, is_north, CATS, FIG, INK, MUTED, CELL, HALF, R

RL = ["65–70°", "70–75°", "75–80°", "80–85°", "85–90°"]
SIZES = [(1, 2, "1–2 km"), (2, 5, "2–5 km"), (5, 10, "5–10 km"), (10, 1e9, "10 km and above")]
COL = {"robbins": "#2a78d6", "lu5m": "#eb6834", "luc": "#1baf7a"}
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- data
def cell_index(lat, lon, north):
    rho = 2 * R * np.tan(np.radians(90 - np.abs(lat)) / 2)
    x = rho * np.sin(np.radians(lon))
    y = rho * np.cos(np.radians(lon)) * (-1.0 if north else 1.0)
    ix = np.floor((x + HALF) / CELL).astype(np.int64)
    iy = np.floor((y + HALF) / CELL).astype(np.int64)
    ok = (ix >= 0) & (ix < 77) & (iy >= 0) & (iy < 77)
    return np.where(ok, iy * 77 + ix, -1)


def read_craters(name):
    cols = {"robbins": ("craters_clean", "diam_km"), "lu5m": ("craters_lu5m", "diam_h_km"), "luc": ("craters_luc", "diam_km")}[name]
    df = pd.read_parquet(M.BASE + "/" + cols[0] + ".parquet", columns=["lat", "lon", cols[1]]).rename(columns={cols[1]: "diam"})
    return df[(df.diam >= 1) & (df.lat.abs() >= 65)]


def class_counts(df, north):
    d = df[df.lat > 0] if north else df[df.lat < 0]
    idx = cell_index(d.lat.to_numpy(), d.lon.to_numpy(), north)
    out = np.zeros((4, 77 * 77))
    for k, (lo, hi, _) in enumerate(SIZES):
        m = (d.diam.to_numpy() >= lo) & (d.diam.to_numpy() < hi) & (idx >= 0)
        out[k] = np.bincount(idx[m], minlength=77 * 77)
    return out


def build():
    d = M.load_cells()
    g = d[np.isclose(d.thr_km, 1) & pick_cat(d.catalogue, "robbins")][["north", "id", "x", "y", "lat", "lon", "area_km2"]].copy()
    g["ix"] = np.floor((g.x + HALF) / CELL).astype(int)
    g["iy"] = np.floor((g.y + HALF) / CELL).astype(int)
    g["cellno"] = g.iy * 77 + g.ix
    g["ring"] = pd.cut(g.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False, labels=RL)
    il = pd.read_csv(M.BASE + "/cells_illumination.csv")
    il["north"] = il.pole.map(is_north)
    g = g.merge(il[["north", "id", "sun_mean", "psr_frac"]], on=["north", "id"], how="left")
    te = pd.read_csv(M.BASE + "/cells_terrain.csv")
    te["north"] = te.pole.map(is_north)
    g = g.merge(te[["north", "ix", "iy", "elev_mean_m", "slope_mean_deg"]], on=["north", "ix", "iy"], how="left")
    g["shadowed"] = g.psr_frac > 0.05
    # slope and elevation classes: terciles inside each ring
    for col, name in (("slope_mean_deg", "slope_cls"), ("elev_mean_m", "elev_cls")):
        pct = g.groupby(["north", "ring"], observed=True)[col].rank(pct=True)
        g[name] = np.where(pct.isna(), None, np.where(pct <= 1 / 3, "low", np.where(pct <= 2 / 3, "mid", "high")))
    # counts per size class and catalogue
    for key, _ in CATS:
        cr = read_craters(key)
        cn = {True: class_counts(cr, True), False: class_counts(cr, False)}
        print("  %-8s craters poleward of 65: %d" % (key, len(cr)), flush=True)
        for k, (_, _, lab) in enumerate(SIZES):
            g["n_%s_%d" % (key, k)] = [cn[nh][k][c] for nh, c in zip(g.north, g.cellno)]
        g["n_%s_all" % key] = sum(g["n_%s_%d" % (key, k)] for k in range(4))
        g["dens_%s" % key] = g["n_%s_all" % key] / g.area_km2 * 1000
        for k in range(4):
            g["d_%s_%d" % (key, k)] = g["n_%s_%d" % (key, k)] / g.area_km2 * 1000
        chk = d[np.isclose(d.thr_km, 1) & pick_cat(d.catalogue, key)][["north", "id", "count"]].merge(g[["north", "id", "n_%s_all" % key]], on=["north", "id"])
        print("  check %s: cell counts equal the earlier table in %d of %d cells (sum difference %d)" %
              (key, int((chk["count"] == chk["n_%s_all" % key]).sum()), len(chk), int(chk["count"].sum() - chk["n_%s_all" % key].sum())), flush=True)
    return g


# ---------------------------------------------------------------- helpers
def rel(g, col):
    return g[col] / g.groupby(["north", "ring"], observed=True)[col].transform("mean")


def ratio_ci(a, b, n=500):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5 or b.mean() == 0:
        return np.nan, np.nan, np.nan
    r = a.mean() / b.mean()
    bs = [RNG.choice(a, len(a)).mean() / max(RNG.choice(b, len(b)).mean(), 1e-12) for _ in range(n)]
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9C3B4")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis=grid, color="#E8E4DA", lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG + "/" + name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


# ---------------------------------------------------------------- Q1
def q1(g):
    print("\n--- Q1: shadowed versus lit cells, by crater size (ring-adjusted ratio, 1 = no difference) ---", flush=True)
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, north in zip(axs, (True, False)):
        s = g[g.north == north]
        for ci, (key, lab) in enumerate(CATS):
            for k, (_, _, sl) in enumerate(SIZES):
                col = "d_%s_%d" % (key, k)
                r = rel(s, col)
                a, b = r[s.shadowed], r[~s.shadowed]
                est, lo, hi = ratio_ci(a, b)
                rows.append(dict(pole="north" if north else "south", catalogue=key, size=sl, shadowed_cells=int(s.shadowed.sum()), ratio=est, lo=lo, hi=hi))
                ax.bar(k + (ci - 1) * 0.26, est, 0.24, color=COL[key], label=lab if k == 0 else None)
                ax.plot([k + (ci - 1) * 0.26] * 2, [lo, hi], color=INK, lw=1)
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(range(4))
        ax.set_xticklabels([x[2] for x in SIZES], fontsize=9)
        ax.set_title("%s cap (%d shadowed cells)" % ("North" if north else "South", int(s.shadowed.sum())), fontsize=10, color=INK)
        style(ax)
    axs[0].set_ylabel("ring-adjusted density\nshadowed ÷ lit cells", fontsize=9, color=MUTED)
    axs[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Q1: crater density in shadowed versus lit cells, by crater size (bars: ratio; lines: bootstrap 95% interval)", fontsize=11, color=INK)
    save(fig, "rq1_size_shadow.png")
    out = pd.DataFrame(rows)
    out.to_csv(M.BASE + "/rq1_size_shadow.csv", index=False)
    print(out.round(2).to_string(index=False), flush=True)


# ---------------------------------------------------------------- Q2
def q2(g):
    print("\n--- Q2: slope classes (terciles inside each ring, 65-80 degrees only) ---", flush=True)
    s0 = g[g.slope_cls.notna()]
    print("  cells with slope: %d (north %d, south %d)" % (len(s0), int(s0.north.sum()), int((~s0.north).sum())), flush=True)
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
    classes = ["low", "mid", "high"]
    for i, north in enumerate((True, False)):
        s = s0[s0.north == north]
        ax = axs[i, 0]
        for ci, (key, lab) in enumerate(CATS):
            r = rel(s, "dens_%s" % key)
            for k, c in enumerate(classes):
                v = r[s.slope_cls == c]
                m = v.mean()
                se = v.std() / np.sqrt(len(v))
                rows.append(dict(pole="north" if north else "south", catalogue=key, slope_class=c, cells=len(v), rel_density=m, se=se,
                                 mean_slope=s[s.slope_cls == c].slope_mean_deg.mean()))
                ax.bar(k + (ci - 1) * 0.26, m, 0.24, color=COL[key], label=lab if k == 0 else None)
                ax.plot([k + (ci - 1) * 0.26] * 2, [m - 1.96 * se, m + 1.96 * se], color=INK, lw=1)
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["low slope", "medium slope", "high slope"], fontsize=9)
        ax.set_ylabel(("North" if north else "South") + " cap\nring-adjusted density", fontsize=9, color=MUTED)
        ax.set_ylim(0.6, 1.4)
        style(ax)
        if i == 0:
            ax.set_title("Crater density (1 km and above) by slope class", fontsize=10, color=INK)
            ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
        # size mix: share of craters of 2 km and above
        ax = axs[i, 1]
        for ci, (key, lab) in enumerate(CATS):
            for k, c in enumerate(classes):
                m = s[s.slope_cls == c]
                tot = m["n_%s_all" % key].sum()
                big = sum(m["n_%s_%d" % (key, j)].sum() for j in (1, 2, 3))
                ax.bar(k + (ci - 1) * 0.26, 100 * big / tot, 0.24, color=COL[key])
                rows.append(dict(pole="north" if north else "south", catalogue=key, slope_class=c, share_ge2km_pct=100 * big / tot))
        ax.set_xticks(range(3))
        ax.set_xticklabels(["low slope", "medium slope", "high slope"], fontsize=9)
        ax.set_ylabel("% of craters that are 2 km or larger", fontsize=9, color=MUTED)
        if i == 0:
            ax.set_title("Crater size mix by slope class", fontsize=10, color=INK)
        style(ax)
    fig.suptitle("Q2: slope and crater population near the poles (65° to 80°; bars with approximate 95% intervals)", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "rq2_slope.png")
    out = pd.DataFrame(rows)
    out.to_csv(M.BASE + "/rq2_slope.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)
    for north in (True, False):
        s = s0[s0.north == north]
        rho = [stats.spearmanr(x.slope_mean_deg, x.dens_robbins)[0] for _, x in s.groupby("ring", observed=True) if len(x) > 30]
        print("  %s: Spearman(slope, Robbins density) inside rings: %s" % ("north" if north else "south", [round(v, 2) for v in rho]), flush=True)


# ---------------------------------------------------------------- Q3
def q3(g):
    print("\n--- Q3: hotspot cells (significant in all three catalogues, max-T, ring-only null) versus other cells ---", flush=True)
    pm = pd.read_csv(M.BASE + "/perm_hotspots.csv")
    pm["north"] = pm.pole.map(is_north)
    nulls = sorted(pm["null"].astype(str).unique())
    pm = pm[pm["null"].astype(str) == nulls[0]]
    vars_ = [("elev_mean_m", "elevation"), ("slope_mean_deg", "slope"), ("sun_mean", "sunlight"), ("psr_frac", "shadow fraction")]
    zc = {}
    for v, _ in vars_:
        grp = g.groupby(["north", "ring"], observed=True)[v]
        zc[v] = (g[v] - grp.transform("mean")) / grp.transform("std")
    rows = []
    cases = [(True, 1), (False, 1), (False, 2), (True, 2)]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    xs, cl = [], []
    for ci, (north, thr) in enumerate(cases):
        sig = None
        for key, _ in CATS:
            p = pm[np.isclose(pm.thr_km, thr) & pick_cat(pm.catalogue, key) & (pm.north == north)][["id", "p_maxT"]]
            ok = set(p.id[p.p_maxT <= 0.0500001])
            sig = ok if sig is None else sig & ok
        mask = g.north.eq(north) & g.id.isin(sig)
        n_h = int(mask.sum())
        label = "%s, %d km+ (%d cells)" % ("north" if north else "south", thr, n_h)
        print("  %s" % label, flush=True)
        if n_h < 10:
            print("    too few cells for a comparison", flush=True)
            continue
        for vi, (v, vl) in enumerate(vars_):
            hz = zc[v][mask].dropna()
            oth = g[v][g.north.eq(north) & ~mask].dropna()
            hv = g[v][mask].dropna()
            lo, hi = (np.percentile([RNG.choice(hz, len(hz)).mean() for _ in range(500)], [2.5, 97.5]) if len(hz) > 5 else (np.nan, np.nan))
            u = stats.mannwhitneyu(hz, zc[v][g.north.eq(north) & ~mask].dropna())[1] if len(hz) > 5 else np.nan
            rows.append(dict(case=label, variable=vl, hotspot_median=hv.median(), other_median=oth.median(), ring_z_mean=hz.mean(), lo=lo, hi=hi, mannwhitney_p=u, n_hot=len(hz)))
            xpos = vi + (ci - 1.5) * 0.2
            ax.errorbar(xpos, hz.mean(), yerr=[[hz.mean() - lo], [hi - hz.mean()]], fmt="o", color=["#2a78d6", "#eb6834", "#c0392b", "#1baf7a"][ci], ms=6, lw=1.4,
                        label=label if vi == 0 else None)
    ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(vars_)))
    ax.set_xticklabels([v[1] for v in vars_], fontsize=10)
    ax.set_ylabel("hotspot cells minus ring average\n(in standard deviations of the ring)", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Q3: terrain and illumination of hotspot cells compared with other cells in the same latitude ring", fontsize=11, color=INK)
    style(ax)
    save(fig, "rq3_hotspot_terrain.png")
    out = pd.DataFrame(rows)
    out.to_csv(M.BASE + "/rq3_hotspots.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)


# ---------------------------------------------------------------- Q4
def q4(g):
    print("\n--- Q4: disagreement between the three catalogues (coefficient of variation of the three densities, 1 km and above) ---", flush=True)
    D = g[["dens_robbins", "dens_lu5m", "dens_luc"]].to_numpy()
    g["cv"] = D.std(axis=1) / D.mean(axis=1)
    g["lr_lu5m"] = np.log2((g.dens_lu5m + 0.5) / (g.dens_robbins + 0.5))
    g["lr_luc"] = np.log2((g.dens_luc + 0.5) / (g.dens_robbins + 0.5))
    print("  mean CV north %.3f, south %.3f | mean log2(LU5M/Robbins) %.2f | mean log2(LUC/Robbins) %.2f" %
          (g.cv[g.north].mean(), g.cv[~g.north].mean(), g.lr_lu5m.mean(), g.lr_luc.mean()), flush=True)
    # map
    vmax = g.cv.quantile(0.98)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, north in zip(axs, (True, False)):
        im = M.polar_panel(ax, M.to_grid(g[g.north == north], "cv"), north, "magma_r", 0, vmax, "North cap" if north else "South cap")
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.045, pad=0.04, extend="max")
    cb.set_label("disagreement between catalogues (spread of the three densities ÷ their mean)", fontsize=9, color=MUTED)
    fig.suptitle("Q4: where the three crater catalogues disagree most (craters of 1 km and above)", fontsize=11, color=INK, y=0.97)
    save(fig, "rq4_disagreement_map.png")
    # groups
    groups = [("ring", "latitude ring", RL), ("shadowed", "shadow", [False, True]), ("slope_cls", "slope class", ["low", "mid", "high"]),
              ("elev_cls", "elevation class", ["low", "mid", "high"])]
    labs = {"shadowed": ["lit", "shadowed"]}
    rows = []
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.4), sharey="row")
    for i, north in enumerate((True, False)):
        s = g[g.north == north]
        for j, (col, title, order) in enumerate(groups):
            ax = axs[i, j]
            vals = []
            for o in order:
                m = s[s[col] == o]
                vals.append(m.cv.mean())
                rows.append(dict(pole="north" if north else "south", group=title, level=str(o), cells=len(m), mean_cv=m.cv.mean(),
                                 log2_lu5m_over_robbins=m.lr_lu5m.mean(), log2_luc_over_robbins=m.lr_luc.mean()))
            ax.bar(range(len(order)), vals, color="#6b5b95", width=0.62)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(labs.get(col, [str(o) for o in order]), fontsize=8)
            if i == 0:
                ax.set_title(title, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(("North" if north else "South") + " cap\nmean disagreement", fontsize=9, color=MUTED)
            style(ax)
    fig.suptitle("Q4: catalogue disagreement by latitude, shadow, slope and elevation (slope and elevation classes are terciles inside each ring)", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "rq4_disagreement_groups.png")
    out = pd.DataFrame(rows)
    out.to_csv(M.BASE + "/rq4_disagreement.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)
    # which crater sizes disagree most
    print("  disagreement by crater size (CV of the three catalogue totals poleward of 65):", flush=True)
    for k, (_, _, lab) in enumerate(SIZES):
        tot = [g["n_%s_%d" % (key, k)].sum() for key, _ in CATS]
        print("    %-16s Robbins %d | LU5M812TGT %d | LUC-GRAS200 %d | CV %.2f" % (lab, tot[0], tot[1], tot[2], np.std(tot) / np.mean(tot)), flush=True)


if __name__ == "__main__":
    print("===== RESULTS =====", flush=True)
    g = build()
    print("cells: %d | shadowed %d | with slope %d | with elevation %d" % (len(g), int(g.shadowed.sum()), int(g.slope_mean_deg.notna().sum()), int(g.elev_mean_m.notna().sum())), flush=True)
    q1(g)
    q2(g)
    q3(g)
    q4(g)
    print("all done", flush=True)