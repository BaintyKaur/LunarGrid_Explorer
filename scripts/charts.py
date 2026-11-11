"""Summary charts: catalogue counts, diameter distribution, north vs south density, K-means groups, association rules.

Reads : tiles_catalogue_counts.csv, craters_clean/lu5m/luc.parquet (diameter column only),
        polar_cells_gistar.csv, kmeans_profile_k3.csv, rules_all.csv, rules_polar.csv   (needs maps_6_7.py)
Writes: figures/chart01_catalogue_counts.png, chart02_diameter_distribution.png, chart03_north_vs_south.png,
        chart04_kmeans_groups.png, chart05_association_rules.png
Run   : python charts.py 2>&1 | tee charts.log
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from maps_6_7 import load_cells, pick_cat, CATS, BASE, FIG, INK, MUTED

C = {"robbins": "#2a78d6", "lu5m": "#eb6834", "luc": "#1baf7a"}
NAME = dict(CATS)


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9C3B4")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color="#E8E4DA", lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG + "/" + name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def chart1():
    t = pd.read_csv(BASE + "/tiles_catalogue_counts.csv")
    sizes = [("ge1", "1 km and above"), ("ge2", "2 km and above"), ("ge5", "5 km and above"), ("ge20", "20 km and above")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (s, lab) in zip(axs, sizes):
        v = [int(t["n_%s_%s" % (s, k)].sum()) for k, _ in CATS]
        print("  %-16s %s" % (lab, dict(zip([n for _, n in CATS], v))), flush=True)
        bars = ax.bar(range(3), v, color=[C[k] for k, _ in CATS], width=0.65)
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, format(x, ","), ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for _, n in CATS], fontsize=8, rotation=15)
        ax.set_ylim(0, max(v) * 1.15)
        ax.set_title(lab, fontsize=10, color=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
        style(ax)
    fig.suptitle("Number of craters in each catalogue, by size limit", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "chart01_catalogue_counts.png")


def chart2():
    rob = pd.read_parquet(BASE + "/craters_clean.parquet", columns=["diam_km"]).diam_km.to_numpy()
    lu = pd.read_parquet(BASE + "/craters_lu5m.parquet", columns=["diam_h_km"]).diam_h_km.to_numpy()
    luc = pd.read_parquet(BASE + "/craters_luc.parquet", columns=["diam_km"]).diam_km.to_numpy()
    edges = np.logspace(0, np.log10(300), 40)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for (k, lab), arr in zip(CATS, (rob, lu, luc)):
        arr = arr[arr >= 1]
        print("  %-12s craters >= 1 km: %d | median %.2f km | largest %.0f km" % (lab, len(arr), np.median(arr), arr.max()), flush=True)
        h, _ = np.histogram(arr, bins=edges)
        mid = np.sqrt(edges[:-1] * edges[1:])
        axs[0].plot(mid[h > 0], h[h > 0] / np.diff(edges)[h > 0], marker="o", ms=3.5, lw=1.6, color=C[k], label=lab)
        s = np.sort(arr)[::-1]
        step = max(1, len(s) // 2000)
        axs[1].plot(s[::step], np.arange(1, len(s) + 1)[::step], lw=1.8, color=C[k], label=lab)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("crater diameter (km)", fontsize=9, color=MUTED)
        style(ax, grid="both")
    axs[0].set_ylabel("craters per km of diameter", fontsize=9, color=MUTED)
    axs[0].set_title("Size histogram (logarithmic bins)", fontsize=10, color=INK)
    axs[0].legend(frameon=False, fontsize=8.5)
    axs[1].set_ylabel("number of craters at least this large", fontsize=9, color=MUTED)
    axs[1].set_title("Cumulative size distribution", fontsize=10, color=INK)
    save(fig, "chart02_diameter_distribution.png")


def chart3():
    d = load_cells()
    d = d[np.isclose(d.thr_km, 1)].copy()
    rl = ["65–70°", "70–75°", "75–80°", "80–85°", "85–90°"]
    d["ring"] = pd.cut(d.lat.abs(), [65, 70, 75, 80, 85, 90.01], right=False, labels=rl)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, (k, lab) in zip(axs, CATS):
        s = d[pick_cat(d.catalogue, k)]
        g = s.groupby(["ring", "north"], observed=True).density.mean().unstack()
        print("  %-10s north %s | south %s" % (lab, [round(v, 1) for v in g[True]], [round(v, 1) for v in g[False]]), flush=True)
        x = np.arange(5)
        ax.bar(x - 0.2, g[True], 0.4, color="#2a78d6", label="north cap")
        ax.bar(x + 0.2, g[False], 0.4, color="#eb6834", label="south cap")
        ax.set_xticks(x)
        ax.set_xticklabels(rl, fontsize=8)
        ax.set_title(lab, fontsize=10, color=INK)
        style(ax)
    axs[0].set_ylabel("mean craters per 1,000 km²\n(1 km and above)", fontsize=9, color=MUTED)
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("North versus south: crater density in 5° rings of the equal-area polar grid", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "chart03_north_vs_south.png")


def chart4():
    p = pd.read_csv(BASE + "/kmeans_profile_k3.csv").sort_values("prediction")
    print(p.round(2).to_string(index=False), flush=True)
    cols = [("share_%", "share of craters (%)"), ("diam_med_km", "median diameter (km)"), ("slope_deg", "mean slope (°)"),
            ("elev_m", "mean elevation (m)"), ("abs_lat", "mean |latitude| (°)"), ("far_side_%", "on the far side (%)")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for ax, (c, lab) in zip(axs.ravel(), cols):
        b = ax.bar(range(len(p)), p[c], color=colors[:len(p)], width=0.6)
        for bb, v in zip(b, p[c]):
            ax.text(bb.get_x() + bb.get_width() / 2, v, ("%.1f" % v) if abs(v) < 1000 else "%.0f" % v, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels(["Group %d" % (i + 1) for i in range(len(p))], fontsize=8.5)
        ax.set_title(lab, fontsize=9.5, color=INK)
        style(ax)
    fig.suptitle("K-means with three groups (silhouette 0.38: weak structure)", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "chart04_kmeans_groups.png")


def chart5():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, (name, title, minsup) in zip(axs, (("rules_all", "All craters (1.26 million with slope)", 0.02), ("rules_polar", "Polar craters (poleward of 65°)", 0.03))):
        r = pd.read_csv(BASE + "/" + name + ".csv")
        r = r[(r.n_if <= 2) & (r.lift > 1)].sort_values("lift", ascending=False).head(10).iloc[::-1]
        print("  %s top rule: %s => %s (support %.3f, confidence %.2f, lift %.2f)" % (name, r.iloc[-1]["if"], r.iloc[-1]["then"], r.iloc[-1].support, r.iloc[-1].confidence, r.iloc[-1].lift), flush=True)
        labs = ["%s  →  %s" % (a, b) for a, b in zip(r["if"], r["then"])]
        labs = [l if len(l) <= 62 else l[:60] + "…" for l in labs]
        ax.barh(range(len(r)), r.lift, color="#2a78d6", height=0.65)
        for i, (lf, cf) in enumerate(zip(r.lift, r.confidence)):
            ax.text(lf, i, "  lift %.2f, conf. %.0f%%" % (lf, cf * 100), va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(r)))
        ax.set_yticklabels(labs, fontsize=7.5)
        ax.axvline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xlim(0, r.lift.max() * 1.45)
        ax.set_xlabel("lift (1 = no association)", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK)
        style(ax, grid="x")
    fig.suptitle("Strongest association rules (FP-Growth, one or two labels in the IF part)", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "chart05_association_rules.png")


if __name__ == "__main__":
    print("===== RESULTS =====", flush=True)
    chart1()
    chart2()
    chart3()
    chart4()
    chart5()
    print("all done", flush=True)