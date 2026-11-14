"""Phase 1e: pixel labels on the factor grid, the evaluation domain and the 3 km block ids.

Outputs (data/interim/):
  label.tif   uint8  1 = pixel centre inside a landslide polygon, 0 = background, 255 = outside the domain or in the
              20 m exclusion ring around a polygon
  block.tif   int32  3 km block id (row * 100 + col) on the factor grid
Polygons are first shifted onto the Sentinel-2 grid (SHIFT_E_M, SHIFT_N_M).
Domain = valid land-use pixels with a geology value, minus the 3 km blocks flagged as likely unmapped
(src/data/coverage_map.py rule, listed in UNMAPPED_BLOCKS). Usage: python -m src.data.labels <factor_dir>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features

from src.data.constants import RING_M, SHIFT_E_M, SHIFT_N_M, UNMAPPED_BLOCKS, B

FDIR = sys.argv[1]
OUT = "data/interim"

with rasterio.open(f"{FDIR}/map_landuse.tif") as r:
    valid, tr, crs, H, W = r.read(1) != r.nodata, r.transform, r.crs, r.height, r.width
with rasterio.open(f"{FDIR}/map_geologi.tif") as g:
    has_geo = g.read(1) != g.nodata
ls = gpd.read_file(f"{FDIR}/landslide_pacitan.gpkg").to_crs(crs)
ls["geometry"] = ls.geometry.translate(xoff=SHIFT_E_M, yoff=SHIFT_N_M)

rows, cols = np.indices((H, W))
block = (rows // B) * 100 + (cols // B)
unmapped = np.isin(block, [i * 100 + j for i, j in UNMAPPED_BLOCKS])
domain = valid & has_geo & ~unmapped

pos = features.rasterize(((g, 1) for g in ls.geometry), out_shape=(H, W), transform=tr, fill=0, dtype="uint8")  # centre rule
ring = features.rasterize(((g, 1) for g in ls.geometry.buffer(RING_M)), out_shape=(H, W), transform=tr, fill=0, dtype="uint8") == 1
label = np.where(~domain, 255, np.where(pos == 1, 1, np.where(ring, 255, 0))).astype("uint8")

prof = dict(driver="GTiff", height=H, width=W, count=1, crs=crs, transform=tr, compress="deflate", tiled=True,
            blockxsize=256, blockysize=256)
with rasterio.open(f"{OUT}/label.tif", "w", dtype="uint8", nodata=255, **prof) as d:
    d.write(label, 1)
with rasterio.open(f"{OUT}/block.tif", "w", dtype="int32", nodata=-1, **prof) as d:
    d.write(np.where(domain, block, -1).astype("int32"), 1)

n_dom = int(domain.sum())
print(f"valid land-use pixels      {int(valid.sum()):>12,}")
print(f"  minus no geology         {int((valid & ~has_geo).sum()):>12,}")
print(f"  minus 5 unmapped blocks  {int((valid & has_geo & unmapped).sum()):>12,}")
print(f"domain pixels              {n_dom:>12,} ({n_dom * 100 / 1e6:.0f} km2)")
print(f"positive pixels (centre rule, in domain) {int((label == 1).sum()):,}  = {(label == 1).sum() / n_dom:.3%} of the domain")
print(f"ring pixels excluded (in domain, not positive) {int((domain & ring & (pos == 0)).sum()):,}")
lab = features.rasterize(((g, i + 1) for i, g in enumerate(ls.geometry)), out_shape=(H, W), transform=tr, fill=0, dtype="int32")
sizes = pd.Series(lab[lab > 0]).value_counts()
cls = pd.cut(sizes, [0, 4, 20, 10**9], labels=["<5 px", "5-20 px", ">20 px"])
print(f"polygons with >= 1 centre pixel {len(sizes)} of {len(ls)}; no centre pixel: {len(ls) - len(sizes)}")
print("polygons by size class (pixels under the centre rule):", cls.value_counts().sort_index().to_dict())
inside = np.array([domain[min(H - 1, max(0, int((tr.f - p.y) / 10))), min(W - 1, max(0, int((p.x - tr.c) / 10)))] for p in ls.geometry.representative_point()])
print(f"polygons whose representative point is outside the domain (dropped with the 5 blocks / no geology): {int((~inside).sum())}")
