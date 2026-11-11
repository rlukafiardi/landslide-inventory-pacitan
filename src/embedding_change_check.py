"""POC risk #2: do annual Satellite Embedding changes separate Cempaka landslide points from background?

Pass criteria (fixed before the first run):
  P1  AUC of cosine distance 2017->2018, landslide vs background points  >= 0.70
  P2  version artifact check: background median distance 2017->2018 <= 1.5 x background median 2018->2019
      (2017 is dataset v1.1, 2018 is v1.0; a large ratio means the version change, not the world, moves the vectors)
Usage: python -m src.embedding_change_check <landslide_gpkg> <out_csv>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.embedding_utils import auc, embedding_distances, init_ee

SEED, N_MIN_DIST_M = 42, 100


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
init_ee()
dist = embedding_distances(pts)
meta = pts.set_index("pid").loc[dist.index, ["grp", "area"]].join(dist)
meta.to_csv(out_csv)
common = meta.index

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
