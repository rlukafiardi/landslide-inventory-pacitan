"""Detection test with site information controlled: do Satellite Embedding CHANGE features detect Cempaka
landslides better than Sentinel-2 dNDVI when both get the same 11 site factors (Samodra et al. release)?

Feature sets (all spatial block CV, 3 km blocks, pooled out-of-fold AUC, logreg and rf):
  F   site factors only (the floor: how far the site alone gets)
  B+  dNDVI set (pre, postA, postB, dA, dB) + factors
  D+  embedding change (2018 - 2017, 64) + factors
  E   embedding change + dNDVI set + factors
  A+  2018 embedding + factors (reference, not in the criteria)
Criteria (fixed before the first run), each required for BOTH models:
  D0  sanity: label-permutation AUC of D+ (logreg) within 0.5 +/- 0.05
  D1  AUC(D+) - AUC(B+): block-bootstrap CI lower bound > 0   (embedding change detects better than dNDVI)
  D2  AUC(E) - max(AUC(D+), AUC(B+)): CI lower bound > 0       (embeddings add to dNDVI)
  Keep embeddings for detection: D0 and (D1 or D2). Otherwise use the dNDVI / S1 / terrain route.
Known open risk: the 2017 layer is dataset v1.1 and 2018 is v1.0, so part of the change may be a version artifact.
Factors: aspect (sin, cos), distriv, distroad, elev, planc, profc, slope, spi, twi, plus one-hot geologi and landuse.
Usage: python -m src.site_controlled_check <features_csv> <factor_dir>
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.cv_utils import MODELS, SEED, block_boot, blocks, oof, pooled_auc
from src.embedding_utils import BANDS, auc

feat_csv, fdir = sys.argv[1:3]
d = pd.read_csv(feat_csv, index_col="pid")
xy = gpd.GeoSeries(gpd.points_from_xy(d.lon, d.lat), index=d.index, crs=4326).to_crs(32749)
coords = list(zip(xy.x, xy.y, strict=True))

NUM = ["distriv", "distroad", "elev", "planc", "profc", "slope", "spi", "twi"]
CAT = ["geologi", "landuse"]
cols = {}
for name in NUM + CAT + ["aspect"]:
    with rasterio.open(f"{fdir}/map_{name}.tif") as r:
        v = np.array([x[0] for x in r.sample(coords)], dtype=float)
        v[(v == r.nodata) | ~np.isfinite(v)] = np.nan
        cols[name] = v
site = pd.DataFrame(cols, index=d.index)
print("missing per factor:", site.isna().sum()[lambda s: s > 0].to_dict() or "none")
ok = site.dropna(subset=NUM + CAT).index
asp = site.aspect.loc[ok]
siteX = pd.concat([site.loc[ok, NUM], pd.get_dummies(site.loc[ok, CAT].astype(int).astype(str), dtype=float)], axis=1)
siteX["asp_sin"], siteX["asp_cos"] = np.sin(np.deg2rad(asp)).fillna(0), np.cos(np.deg2rad(asp)).fillna(0)
d = d.loc[ok].join(siteX)
d = d.assign(dA=d.postA - d.pre, dB=d.postB - d.pre, **{f"x_{b}": d[f"e18_{b}"] - d[f"e17_{b}"] for b in BANDS})
y = (d.grp == "landslide").to_numpy()
area = d.area.to_numpy()
print(f"points {len(d)} (landslide {y.sum()}), dropped for missing factors: {len(site) - len(ok)}")

SITE = list(siteX.columns)
NDVI = ["pre", "postA", "postB", "dA", "dB"]
DIFF = [f"x_{b}" for b in BANDS]
SETS = {
    "F": SITE,
    "B+": NDVI + SITE,
    "D+": DIFF + SITE,
    "E": DIFF + NDVI + SITE,
    "A+": [f"e18_{b}" for b in BANDS] + SITE,
}
g3 = blocks(xy.loc[d.index].x, xy.loc[d.index].y, 3000)
print(f"3 km blocks: {len(set(g3))}")

verdict = {}
for model in MODELS:
    P = {k: oof(d, c, model, g3, y) for k, c in SETS.items()}
    print(f"\n=== {model}: out-of-fold AUC (increment over F) ===")
    f = pooled_auc(P["F"], y)
    for k, v in P.items():
        a = pooled_auc(v, y)
        print(f"  {k:<3} {a:.3f}" + ("" if k == "F" else f"  ({a - f:+.3f})"))
    print("  by polygon size class        " + "   ".join(f"{k:>4}" for k in ("F", "B+", "D+", "E")))
    for lab, lo, hi in [("<5 px (<500 m2)", 0, 500), ("5-20 px", 500, 2000), (">20 px (>2000 m2)", 2000, 1e12)]:
        m = ~y | ((area >= lo) & (area < hi))
        print(
            f"    {lab:<22} n={int((m & y).sum()):<4} "
            + "   ".join(f"{pooled_auc(P[k][m], y[m]):.3f}" for k in ("F", "B+", "D+", "E"))
        )
    ci1 = block_boot(P, g3, y, lambda p, yy: auc(p["D+"][yy], p["D+"][~yy]) - auc(p["B+"][yy], p["B+"][~yy]))
    ci2 = block_boot(
        P,
        g3,
        y,
        lambda p, yy: (
            auc(p["E"][yy], p["E"][~yy]) - max(auc(p["D+"][yy], p["D+"][~yy]), auc(p["B+"][yy], p["B+"][~yy]))
        ),
    )
    d1 = pooled_auc(P["D+"], y) - pooled_auc(P["B+"], y)
    d2 = pooled_auc(P["E"], y) - max(pooled_auc(P["D+"], y), pooled_auc(P["B+"], y))
    print(f"  D1 D+ - B+: {d1:+.3f}, CI [{ci1[0]:+.3f}, {ci1[1]:+.3f}]")
    print(f"  D2 E - max(D+, B+): {d2:+.3f}, CI [{ci2[0]:+.3f}, {ci2[1]:+.3f}]")
    perm = np.random.default_rng(SEED).permutation(y)
    perm_auc = pooled_auc(oof(d, SETS["D+"], model, g3, perm), perm)
    print(f"  D0 label-permutation AUC of D+: {perm_auc:.3f}")
    verdict[model] = (abs(perm_auc - 0.5) <= 0.05, ci1[0] > 0, ci2[0] > 0)

d0 = all(v[0] for v in verdict.values())
d1 = all(v[1] for v in verdict.values())
d2 = all(v[2] for v in verdict.values())
print(
    f"\nD0 {'PASS' if d0 else 'FAIL'} | D1 {'PASS' if d1 else 'FAIL'} | D2 {'PASS' if d2 else 'FAIL'} | "
    f"keep embeddings for detection: {'YES' if d0 and (d1 or d2) else 'NO'}"
)
