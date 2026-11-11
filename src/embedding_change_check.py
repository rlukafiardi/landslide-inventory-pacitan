"""POC risk #2: do annual Satellite Embedding changes separate Cempaka landslide points from background?

Pass criteria (fixed before the first run):
  P1  AUC of cosine distance 2017->2018, landslide vs background points  >= 0.70
  P2  version artifact check: background median distance 2017->2018 <= 1.5 x background median 2018->2019
      (2017 is dataset v1.1, 2018 is v1.0; a large ratio means the version change, not the world, moves the vectors)
Usage: python -I embedding_change_check.py <landslide_gpkg> <out_csv>
"""
import os
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SEED, N_MIN_DIST_M = 42, 100
BANDS = [f"A{i:02d}" for i in range(64)]
YEARS = (2017, 2018, 2019)


def auc(pos, neg):
    """Mann-Whitney AUC: P(score_pos > score_neg), ties count half."""
    s = pd.concat([pd.Series(pos), pd.Series(neg)], ignore_index=True)
    r = s.rank()
    n_pos, n_neg = len(pos), len(neg)
    return (r.iloc[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# known edge cases before touching data
assert auc([3, 4, 5], [0, 1, 2]) == 1.0
assert auc([0, 1, 2], [3, 4, 5]) == 0.0
assert auc([1, 1, 1], [1, 1, 1]) == 0.5

gpkg, out_csv = sys.argv[1:3]
ls = gpd.read_file(gpkg).to_crs(32749)
rng = np.random.default_rng(SEED)

pos = ls.assign(pid=[f"ls{i}" for i in range(len(ls))], grp="landslide", area=ls.area)
pos["geometry"] = pos.geometry.representative_point()

keep_out = ls.geometry.buffer(N_MIN_DIST_M).union_all()
minx, miny, maxx, maxy = ls.total_bounds
bg = []
while len(bg) < len(ls):
    p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
    if not keep_out.contains(p):
        bg.append(p)
neg = gpd.GeoDataFrame({"pid": [f"bg{i}" for i in range(len(bg))], "grp": "background", "area": np.nan},
                       geometry=bg, crs=32749)

pts = pd.concat([pos[["pid", "grp", "area", "geometry"]], neg]).pipe(gpd.GeoDataFrame, crs=32749).to_crs(4326)
ee.Initialize(project=os.environ["EARTHENGINE_PROJECT"])
fc = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([g.x, g.y]), {"pid": r.pid}) for r, g in zip(pts.itertuples(), pts.geometry)
])

coll = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
emb = {}
for y in YEARS:
    img = coll.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic()
    rows = img.sampleRegions(collection=fc, scale=10, geometries=False).getInfo()["features"]
    emb[y] = pd.DataFrame([{"pid": f["properties"]["pid"], **{b: f["properties"][b] for b in BANDS}} for f in rows]).set_index("pid")
    print(y, "sampled", len(emb[y]), "of", len(pts))

common = emb[2017].index.intersection(emb[2018].index).intersection(emb[2019].index)
meta = pts.set_index("pid").loc[common, ["grp", "area"]]
for a, b in [(2017, 2018), (2018, 2019)]:
    va, vb = emb[a].loc[common, BANDS].to_numpy(), emb[b].loc[common, BANDS].to_numpy()
    va, vb = va / np.linalg.norm(va, axis=1, keepdims=True), vb / np.linalg.norm(vb, axis=1, keepdims=True)
    meta[f"d{a % 100}{b % 100}"] = 1 - (va * vb).sum(axis=1)
meta.to_csv(out_csv)

print(f"\npoints valid in all years: {len(common)} (landslide {(meta.grp == 'landslide').sum()}, background {(meta.grp == 'background').sum()})")
L, B = meta[meta.grp == "landslide"], meta[meta.grp == "background"]
print("\nmedian cosine distance            landslide  background")
for c in ("d1718", "d1819"):
    print(f"  {c}                            {L[c].median():.4f}     {B[c].median():.4f}")
print("\nAUC landslide vs background")
for c in ("d1718", "d1819"):
    print(f"  {c}: {auc(L[c], B[c]):.3f}")
print("\nAUC d1718 by polygon size class")
for name, lo, hi in [("<5 px (<500 m2)", 0, 500), ("5-20 px", 500, 2000), (">20 px (>2000 m2)", 2000, 1e12)]:
    sub = L[(L.area >= lo) & (L.area < hi)]
    print(f"  {name}: n={len(sub)}, AUC={auc(sub['d1718'], B['d1718']):.3f}")

p1 = auc(L["d1718"], B["d1718"]) >= 0.70
ratio = B["d1718"].median() / B["d1819"].median()
p2 = ratio <= 1.5
print(f"\nP1 AUC(2017->2018) >= 0.70: {'PASS' if p1 else 'FAIL'}")
print(f"P2 background median ratio 17->18 / 18->19 = {ratio:.2f} <= 1.5: {'PASS' if p2 else 'FAIL'}")
