"""Repo step 2: build a clean, small folder (~/lunargrid_repo) that is ready to push to GitHub.

Copies : all scripts (*.py, *.sh) from ~/lunargrid  -> scripts/
         all figures (*.png)                         -> figures/
         the small result tables                     -> results/
Writes : README.md, .gitignore, requirements.txt, docs/ (empty, the dashboard goes here)
Skips  : raw data, the elevation model, parquet files, logs.
Run    : python prepare_repo.py
"""
import os
import re
import shutil
import glob

HOME = os.path.expanduser("~/lunargrid")
OUT = os.path.expanduser("~/lunargrid_repo")
RESULTS = [
    "tiles_catalogue_counts.csv", "tiles_gistar.csv", "tiles_terrain.csv", "polar_cells_gistar.csv",
    "cells_illumination.csv", "cells_terrain.csv", "perm_hotspots.csv", "dbscan_sweep.csv",
    "kmeans_sweep.csv", "kmeans_profile_k3.csv", "kmeans_profile_k5.csv", "rules_all.csv", "rules_polar.csv",
    "bench_results.csv", "rq1_size_shadow.csv", "rq2_slope.csv", "rq3_hotspots.csv", "rq4_disagreement.csv",
]

README = """# LunarGrid: Distributed Analytics of the Lunar Surface

Where are lunar craters crowded, where are they sparse, and do independent crater catalogues agree?
This project answers those questions for more than 10 million craters, using Apache Spark and Apache Sedona
for the spatial work, and a small interactive dashboard to explore the results.

**Live dashboard:** https://YOUR-USERNAME.github.io/lunargrid-explorer/  (replace YOUR-USERNAME after publishing)

## What the project does

- Combines three global crater catalogues (Robbins, LU5M812TGT and LUC-GRAS200) with a 118 m lunar elevation model.
- Places every crater on equal-area grids (5 degree tiles for the whole Moon, 20 km cells poleward of 65 degrees),
  so that densities are fair between the equator and the poles.
- Calculates slope with a latitude correction, and joins elevation, slope, sunlight and permanent shadow to each place.
- Finds hotspots and coldspots with Getis-Ord Gi*, and tests them with a permutation test that shuffles densities
  inside latitude rings.
- Compares the catalogues with each other, runs DBSCAN, FP-Growth association rules and K-means, and benchmarks
  Spark and Sedona against a single machine.

## Main findings

- The catalogues agree strongly on where craters are crowded (tile-by-tile correlation 0.94 to 0.97) but differ on how
  many small craters exist.
- Crater density rises steadily towards the north pole but not towards the south pole.
- After the latitude trend is removed, the clearest local hotspots are in the south polar region.
- Permanently shadowed ground holds fewer small craters than lit ground in the same latitude ring. This may be real
  or may reflect how hard craters are to detect in the dark.

See the report in `report/` for the full method, results and limitations.

## Repository layout

| Folder | Contents |
| --- | --- |
| `scripts/` | All analysis scripts, from data cleaning to maps, tests and the benchmark |
| `results/` | The small result tables the figures and the dashboard are built from |
| `figures/` | Maps and charts used in the report |
| `docs/` | The interactive dashboard (`index.html`), published with GitHub Pages |
| `report/` | The written report |

## Data sources (not included, too large for GitHub)

- Robbins lunar crater database (NASA Planetary Data System).
- LU5M812TGT crater catalogue (Zenodo record 13990480).
- LUC-GRAS200 crater catalogue (Chinese Lunar and Planetary Data System).
- 118 m lunar elevation model from the Lunar Orbiter Laser Altimeter (NASA Planetary Geodesy Data Archive).

## Running the scripts

Each script starts with a short description of what it reads and writes. Place the downloaded data in a `data/`
folder as described in those headers, install the packages in `requirements.txt`, and run the scripts in the
order of the analysis: data cleaning, grids and Gi*, terrain and illumination, permutation test, maps and charts.

## Using the dashboard

Open `docs/index.html` in any modern browser. It is a single file with all the data inside, so it needs no server.
Click anywhere on the map to see crater densities, hotspot tests, terrain and sunlight for that place.
"""

GITIGNORE = """# data and large files
data/
*.tif
*.parquet
*.zip
*.log
nohup.out
spark-tmp/
__pycache__/
.ipynb_checkpoints/
.env
*.pem
"""

REQ = """numpy
pandas
scipy
scikit-learn
matplotlib
rasterio
pyarrow
pyspark
apache-sedona
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for d in ("scripts", "results", "figures", "docs", "report"):
        os.makedirs(os.path.join(OUT, d))

    flagged = []
    n = 0
    for pat in ("*.py", "*.sh"):
        for p in sorted(glob.glob(os.path.join(HOME, pat))):
            txt = open(p, errors="ignore").read()
            if re.search(r"(password|passwd|token|secret)\s*[=:]", txt, re.I):
                flagged.append(os.path.basename(p))
            shutil.copy(p, os.path.join(OUT, "scripts"))
            n += 1
    print("scripts copied:", n, flush=True)
    if flagged:
        print("CHECK THESE SCRIPTS for a password or token before pushing:", flagged, flush=True)

    figs = sorted(glob.glob(os.path.join(HOME, "figures", "*.png")))
    for p in figs:
        shutil.copy(p, os.path.join(OUT, "figures"))
    print("figures copied:", len(figs), flush=True)

    k = 0
    for name in RESULTS:
        p = os.path.join(HOME, "data", "processed", name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(OUT, "results"))
            k += 1
        else:
            print("  missing:", name, flush=True)
    print("result tables copied:", k, flush=True)

    open(os.path.join(OUT, "README.md"), "w").write(README)
    open(os.path.join(OUT, ".gitignore"), "w").write(GITIGNORE)
    open(os.path.join(OUT, "requirements.txt"), "w").write(REQ)

    total = 0
    biggest = ("", 0)
    for r, _, fs in os.walk(OUT):
        for f in fs:
            s = os.path.getsize(os.path.join(r, f))
            total += s
            if s > biggest[1]:
                biggest = (os.path.join(r, f), s)
    print("repo folder:", OUT, "| total %.1f MB | largest file: %s (%.1f MB)" % (total / 1e6, os.path.basename(biggest[0]), biggest[1] / 1e6), flush=True)


if __name__ == "__main__":
    main()