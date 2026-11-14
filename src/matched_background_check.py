"""POC risk #2, confounder test: is the 2017->2018 embedding signal still there when the background is
matched to the landslide points on slope and elevation (SRTM 30 m)?

Pass criteria (fixed before the first run):
  M1  balance: |median slope diff| <= 1 deg and |median elevation diff| <= 25 m, landslide vs matched background
  M2  AUC of cosine distance 2017->2018, landslide vs matched background  >= 0.70
  M3  matched background median distance 2017->2018 <= 1.5 x its 2018->2019 median (version artifact check, as P2)
With a landuse raster as third argument the match is exact on land-use class first, then nearest on
(slope, elevation); that run's M1 also requires every landslide to find a same-class background point.
Usage: python -m src.matched_background_check <landslide_gpkg> <out_csv> [landuse_tif]
"""

import os
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

from src.embedding_utils import auc, embedding_distances, init_ee, sample_image

SEED, MIN_DIST_M, POOL = 42, 100, 20000

gpkg, out_csv = sys.argv[1:3]
landuse_tif = sys.argv[3] if len(sys.argv) > 3 else None
POOL_CACHE = "data/processed/pool_terrain.csv"
ls = gpd.read_file(gpkg).to_crs(32749)
rng = np.random.default_rng(SEED)

pos = gpd.GeoDataFrame(
    {"pid": [f"ls{i}" for i in range(len(ls))], "grp": "landslide", "area": ls.area.to_numpy()},
    geometry=ls.geometry.representative_point(),
    crs=32749,
)

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
if os.path.exists(POOL_CACHE):
    t_pool = pd.read_csv(POOL_CACHE, index_col="pid")
else:
    t_pool = sample_image(terrain, pool4326, 30, ["slope", "elevation"])
    t_pool.to_csv(POOL_CACHE)
t_pool = t_pool[t_pool.elevation > 0]
print(f"terrain: landslide {len(t_pos)}/{len(pos)}, pool {len(t_pool)}/{POOL} (ocean / nodata dropped)")

if landuse_tif:
    with rasterio.open(landuse_tif) as r:

        def lu(gdf):
            return pd.Series([v[0] for v in r.sample([(g.x, g.y) for g in gdf.geometry])], index=gdf.pid)

        t_pos["lu"] = lu(pos).loc[t_pos.index]
        t_pool["lu"] = lu(pool).loc[t_pool.index]
        nodata = r.nodata
    t_pool = t_pool[t_pool.lu != nodata]
    t_pos = t_pos[t_pos.lu != nodata]
    tab = pd.DataFrame({"landslide": t_pos.lu.value_counts(), "pool": t_pool.lu.value_counts()}).fillna(0).astype(int)
    print("\nland-use class counts (code: landslide / pool)\n", tab.to_string())

# greedy nearest neighbour without replacement in standardized (slope, elevation) space,
# restricted to the same land-use class when a landuse raster is given
sd = t_pos[["slope", "elevation"]].std()
P = (t_pos[["slope", "elevation"]] / sd).to_numpy()
C = (t_pool[["slope", "elevation"]] / sd).to_numpy()
used = np.zeros(len(C), bool)
picked, matched_pos = [], []
for i in rng.permutation(len(P)):
    d = ((C - P[i]) ** 2).sum(axis=1)
    d[used] = np.inf
    if landuse_tif:
        d[t_pool.lu.to_numpy() != t_pos.lu.iloc[i]] = np.inf
    j = int(d.argmin())
    if not np.isfinite(d[j]):
        continue
    used[j] = True
    picked.append(t_pool.index[j])
    matched_pos.append(t_pos.index[i])
matched = t_pool.loc[picked]
t_pos = t_pos.loc[matched_pos]
print(f"\nmatched {len(picked)} of {len(P)} landslide points")

print("\nterrain balance            landslide  matched bg  random pool")
for c in ("slope", "elevation"):
    print(f"  median {c:<10}       {t_pos[c].median():8.1f}  {matched[c].median():10.1f}  {t_pool[c].median():10.1f}")
print(f"  slope-only AUC landslide vs random pool:   {auc(t_pos.slope, t_pool.slope):.3f}")
print(f"  slope-only AUC landslide vs matched bg:    {auc(t_pos.slope, matched.slope):.3f}")

pos4326 = pos4326.set_index("pid").loc[matched_pos].reset_index()
neg = pool4326.set_index("pid").loc[picked].reset_index().assign(grp="background", area=np.nan)
pts = pd.concat([pos4326, neg[["pid", "grp", "area", "geometry"]]]).pipe(gpd.GeoDataFrame, crs=4326)
dist = embedding_distances(pts)
pi = pts.set_index("pid").loc[dist.index]
meta = (
    pi[["grp", "area"]]
    .assign(lon=pi.geometry.x, lat=pi.geometry.y)
    .join(dist)
    .join(pd.concat([t_pos, matched])[["slope", "elevation"]])
)
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

m1 = (
    abs(t_pos.slope.median() - matched.slope.median()) <= 1
    and abs(t_pos.elevation.median() - matched.elevation.median()) <= 25
)
m2 = auc(L["d1718"], B["d1718"]) >= 0.70
ratio = B["d1718"].median() / B["d1819"].median()
m1 = m1 and len(picked) == len(pos)
print(
    f"\nM1 terrain balance within 1 deg / 25 m{' and all landslides matched' if landuse_tif else ''}: {'PASS' if m1 else 'FAIL'}"
)
print(f"M2 AUC(2017->2018) >= 0.70: {'PASS' if m2 else 'FAIL'}")
print(f"M3 matched bg median ratio 17->18 / 18->19 = {ratio:.2f} <= 1.5: {'PASS' if ratio <= 1.5 else 'FAIL'}")
