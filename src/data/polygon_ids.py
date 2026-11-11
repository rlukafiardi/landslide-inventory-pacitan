"""Phase 4: polygon id raster and polygon table for the object-level metrics.

data/interim/poly_id.tif       int32, 0 = no polygon, else polygon id (1-based); rasterized with all_touched so that polygons
                               without a pixel centre are still detectable (a later polygon overwrites the one overlapping pair)
data/interim/polygons.parquet  id, area_m2, size_class (<5 px, 5-20 px, >20 px by area), block (3 km block of the representative
                               point, -1 outside the domain), in_train
Polygons are shifted onto the Sentinel-2 grid like the labels. Usage: python -m src.data.polygon_ids <factor_dir>
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features

from src.data.constants import SHIFT_E_M, SHIFT_N_M
from src.data.load import block_mask

FDIR = sys.argv[1]
OUT = "data/interim"
CLASS_EDGES, CLASS_NAMES = [0, 500, 2000, np.inf], ["<5 px", "5-20 px", ">20 px"]

with rasterio.open(f"{OUT}/label.tif") as r:
    tr, crs, H, W = r.transform, r.crs, r.height, r.width
with rasterio.open(f"{OUT}/block.tif") as r:
    block = r.read(1)
ls = gpd.read_file(f"{FDIR}/landslide_pacitan.gpkg").to_crs(crs)
ls["geometry"] = ls.geometry.translate(SHIFT_E_M, SHIFT_N_M)
ids = np.arange(1, len(ls) + 1)
poly = features.rasterize(zip(ls.geometry, ids), out_shape=(H, W), transform=tr, fill=0, dtype="int32", all_touched=True)
rp = ls.geometry.representative_point()
pr = np.clip(((tr.f - rp.y) / 10).astype(int), 0, H - 1)
pc = np.clip(((rp.x - tr.c) / 10).astype(int), 0, W - 1)
train = block_mask("train")
tab = pd.DataFrame({"id": ids, "area_m2": ls.area.to_numpy(), "block": block[pr, pc], "in_train": train[pr, pc]})
tab["size_class"] = pd.cut(tab.area_m2, CLASS_EDGES, labels=CLASS_NAMES, right=False).astype(str)
tab["touched_px"] = np.bincount(poly.ravel(), minlength=len(ls) + 1)[1:]
prof = dict(driver="GTiff", height=H, width=W, count=1, crs=crs, transform=tr, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
with rasterio.open(f"{OUT}/poly_id.tif", "w", dtype="int32", nodata=0, **prof) as d:
    d.write(poly, 1)
tab.to_parquet(f"{OUT}/polygons.parquet", index=False)
print(f"{len(tab)} polygons; in training blocks {int(tab.in_train.sum())}; outside the domain {int((tab.block < 0).sum())}")
print(tab.groupby("size_class").agg(polygons=("id", "size"), in_train=("in_train", "sum"), median_touched_px=("touched_px", "median")).to_string())
print("polygons hidden by an overlap (touched_px == 0):", int((tab.touched_px == 0).sum()))
