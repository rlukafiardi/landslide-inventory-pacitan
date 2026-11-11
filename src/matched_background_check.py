"""POC risk #2, confounder test: is the 2017->2018 embedding signal still there when the background is
matched to the landslide points on slope and elevation (SRTM 30 m)?

Pass criteria (fixed before the first run):
  M1  balance: |median slope diff| <= 1 deg and |median elevation diff| <= 25 m, landslide vs matched background
  M2  AUC of cosine distance 2017->2018, landslide vs matched background  >= 0.70
  M3  matched background median distance 2017->2018 <= 1.5 x its 2018->2019 median (version artifact check, as P2)
Usage: python -m src.matched_background_check <landslide_gpkg> <out_csv>
"""
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.embedding_utils import auc, embedding_distances, init_ee, sample_image

SEED, MIN_DIST_M, POOL = 42, 100, 20000

gpkg, out_csv = sys.argv[1:3]
ls = gpd.read_file(gpkg).to_crs(32749)
rng = np.random.default_rng(SEED)

pos = gpd.GeoDataFrame({"pid": [f"ls{i}" for i in range(len(ls))], "grp": "landslide", "area": ls.area.to_numpy()},
                       geometry=ls.geometry.representative_point(), crs=32749)

keep_out = ls.geometry.buffer(MIN_DIST_M).union_all()
minx, miny, maxx, maxy = ls.total_bounds
cand = []
while len(cand) < POOL:
    p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
    if not keep_out.contains(p):
        cand.append(p)
pool = gpd.GeoDataFrame({"pid": [f"bg{i}" for i in range(POOL)]}, geometry=cand, crs=32749)

init_ee()
dem = ee.Image("USGS/SRTMGL1_003")
terrain = dem.select("elevation").addBands(ee.Terrain.slope(dem))
pos4326, pool4326 = pos.to_crs(4326), pool.to_crs(4326)
t_pos = sample_image(terrain, pos4326, 30, ["slope", "elevation"])
t_pool = sample_image(terrain, pool4326, 30, ["slope", "elevation"])
t_pool = t_pool[t_pool.elevation > 0]
print(f"terrain: landslide {len(t_pos)}/{len(pos)}, pool {len(t_pool)}/{POOL} (ocean / nodata dropped)")

# greedy nearest neighbour without replacement in standardized (slope, elevation) space
sd = t_pos[["slope", "elevation"]].std()
P = (t_pos[["slope", "elevation"]] / sd).to_numpy()
C = (t_pool[["slope", "elevation"]] / sd).to_numpy()
used = np.zeros(len(C), bool)
picked = []
for i in rng.permutation(len(P)):
    d = ((C - P[i]) ** 2).sum(axis=1)
    d[used] = np.inf
    j = int(d.argmin())
    used[j] = True
    picked.append(t_pool.index[j])
matched = t_pool.loc[picked]

print("\nterrain balance            landslide  matched bg  random pool")
for c in ("slope", "elevation"):
    print(f"  median {c:<10}       {t_pos[c].median():8.1f}  {matched[c].median():10.1f}  {t_pool[c].median():10.1f}")
print(f"  slope-only AUC landslide vs random pool:   {auc(t_pos.slope, t_pool.slope):.3f}")
print(f"  slope-only AUC landslide vs matched bg:    {auc(t_pos.slope, matched.slope):.3f}")

neg = pool4326.set_index("pid").loc[picked].reset_index().assign(grp="background", area=np.nan)
pts = pd.concat([pos4326, neg[["pid", "grp", "area", "geometry"]]]).pipe(gpd.GeoDataFrame, crs=4326)
dist = embedding_distances(pts)
meta = pts.set_index("pid").loc[dist.index, ["grp", "area"]].join(dist).join(
    pd.concat([t_pos, matched])[["slope", "elevation"]])
meta.to_csv(out_csv)

L, B = meta[meta.grp == "landslide"], meta[meta.grp == "background"]
print(f"\npoints valid in all years: landslide {len(L)}, background {len(B)}")
print("\nmedian cosine distance            landslide  matched bg")
for c in ("d1718", "d1819"):
    print(f"  {c}                            {L[c].median():.4f}     {B[c].median():.4f}")
print("\nAUC landslide vs matched background")
for c in ("d1718", "d1819"):
    print(f"  {c}: {auc(L[c], B[c]):.3f}")
print("\nAUC d1718 by polygon size class")
for name, lo, hi in [("<5 px (<500 m2)", 0, 500), ("5-20 px", 500, 2000), (">20 px (>2000 m2)", 2000, 1e12)]:
    sub = L[(L.area >= lo) & (L.area < hi)]
    print(f"  {name}: n={len(sub)}, AUC={auc(sub['d1718'], B['d1718']):.3f}")

m1 = abs(t_pos.slope.median() - matched.slope.median()) <= 1 and abs(t_pos.elevation.median() - matched.elevation.median()) <= 25
m2 = auc(L["d1718"], B["d1718"]) >= 0.70
ratio = B["d1718"].median() / B["d1819"].median()
print(f"\nM1 terrain balance within 1 deg / 25 m: {'PASS' if m1 else 'FAIL'}")
print(f"M2 AUC(2017->2018) >= 0.70: {'PASS' if m2 else 'FAIL'}")
print(f"M3 matched bg median ratio 17->18 / 18->19 = {ratio:.2f} <= 1.5: {'PASS' if ratio <= 1.5 else 'FAIL'}")
