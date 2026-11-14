"""Are the five 3 km blocks flagged as likely unmapped really different? Run the final model inside them (it never saw them) and compare the
density of candidate objects with the mapped blocks.

Rule used to exclude them (src/data/coverage_map.py): an empty block whose expected count, from the landslide density of its modal lithology
class, is >= 4.6 slides. Here: candidates per km2 (frozen threshold and minimum size, large = 20 px or more) in each of the five blocks, against the
distribution over the mapped blocks (honest scores there: out-of-fold on training blocks, final model on test blocks). Candidates in mapped blocks
split into those overlapping the inventory (true) and those that do not (false alarms or unmapped scars); the fair baseline for an unmapped block
is the second group. Also reports the authors' non-landslide points per block (their sampling frame, a hint of the study footprint).
Writes reports/tables/unmapped_blocks_check.csv and data/processed/unmapped_blocks_candidates.gpkg. Usage: python -m src.analysis.unmapped_check
"""
import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rfeat
from scipy import ndimage as ndi
from shapely.geometry import shape

from src.data.constants import UNMAPPED_BLOCKS, B
from src.features.build import FDIR, INTERIM, TILE, feature_window
from src.features.registry import NAMES
from src.modeling.metrics import objects
from src.modeling.prep import CONFIGS, SETS, design

op = json.load(open("models/operating_points.json"))["c_hgb"]
thr, a_min = np.float64(op["threshold"]), int(op["a_min"])
with rasterio.open(f"{FDIR}/map_landuse.tif") as r:
    valid, tr, H, W = r.read(1) != r.nodata, r.transform, r.height, r.width
with rasterio.open(f"{FDIR}/map_geologi.tif") as g:
    has_geo = g.read(1) != g.nodata
rows, cols = np.indices((H, W))
blk = (rows // B) * 100 + (cols // B)
ids = [i * 100 + j for i, j in UNMAPPED_BLOCKS]
um = np.isin(blk, ids) & valid & has_geo
del rows, cols

# 1. score the five blocks with the final model
model = joblib.load("models/c_hgb.joblib")
kind, fset = CONFIGS["c_hgb"]
score = np.full((H, W), np.nan, dtype="float32")
for r0 in range(0, H, TILE):
    for c0 in range(0, W, TILE):
        h, w = min(TILE, H - r0), min(TILE, W - c0)
        m = um[r0:r0 + h, c0:c0 + w]
        if not m.any():
            continue
        feats = feature_window(r0, c0, h, w)
        ii, jj = np.nonzero(m)
        df = pd.DataFrame({n: feats[n][ii, jj] for n in NAMES})
        score[r0 + ii, c0 + jj] = model.predict_proba(design(df, SETS[fset], kind))[:, 1]
lab, n = objects(score, thr, a_min, um)
sizes = np.bincount(lab.ravel(), minlength=n + 1)
geoms = {int(v): shape(g) for g, v in rfeat.shapes(lab.astype("int32"), mask=lab > 0, transform=tr, connectivity=8)}
cand = gpd.GeoDataFrame({"size_px": sizes[1:], "pmax": ndi.maximum(score, lab, index=np.arange(1, n + 1)), "geometry": [geoms[i] for i in range(1, n + 1)]}, crs="EPSG:32749")
c = cand.geometry.centroid
cand["block"] = (((tr.f - c.y) / 10).astype(int) // B) * 100 + (((c.x - tr.c) / 10).astype(int) // B)
cand["large"] = cand.size_px >= 20
cand.sort_values("pmax", ascending=False).to_file("data/processed/unmapped_blocks_candidates.gpkg", layer="candidates_s2_grid", driver="GPKG")

# 2. mapped blocks: the delivered candidates, per block
mp = gpd.read_file("data/processed/candidates_cempaka_pacitan.gpkg", layer="candidates_s2_grid")
c = mp.geometry.centroid
mp["block"] = (((tr.f - c.y) / 10).astype(int) // B) * 100 + (((c.x - tr.c) / 10).astype(int) // B)
mp["large"] = mp.size_px >= 20
with rasterio.open(f"{INTERIM}/block.tif") as r:
    bk = r.read(1)
area = pd.Series(np.bincount(bk[bk >= 0], minlength=bk.max() + 1)[np.unique(bk[bk >= 0])] * 100 / 1e6, index=np.unique(bk[bk >= 0]))
area = area[area >= 3.0]  # blocks of at least 3 km2 (drops coastal slivers, whose density is noisy)
tab = pd.DataFrame({"area_km2": area,
                    "large_all": mp[mp.large].groupby("block").size().reindex(area.index, fill_value=0),
                    "large_false": mp[mp.large & ~mp.overlaps_inventory].groupby("block").size().reindex(area.index, fill_value=0),
                    "all": mp.groupby("block").size().reindex(area.index, fill_value=0)})
for k in ("large_all", "large_false", "all"):
    tab[k + "_per_km2"] = tab[k] / tab.area_km2

# 3. the authors' non-landslide points per block
nl = gpd.read_file(f"{FDIR}/non_ls.gpkg")
nlb = (((tr.f - nl.geometry.y) / 10).astype(int) // B) * 100 + (((nl.geometry.x - tr.c) / 10).astype(int) // B)
nl_per = nlb.value_counts()
ls = gpd.read_file(f"{FDIR}/landslide_pacitan.gpkg").to_crs("EPSG:32749")
rp = ls.geometry.representative_point()
lsb = (((tr.f - rp.y) / 10).astype(int) // B) * 100 + (((rp.x - tr.c) / 10).astype(int) // B)
mapped_nl = float(nl_per.reindex(tab.index, fill_value=0).mean())

rows_out = []
for b in ids:
    a = float(um[blk == b].sum() * 100 / 1e6)
    sub = cand[cand.block == b]
    row = {"block_row_col": f"({b // 100}, {b % 100})", "area_km2": round(a, 2), "candidates": len(sub), "large": int(sub.large.sum()),
           "large_per_km2": round(sub.large.sum() / a, 2), "all_per_km2": round(len(sub) / a, 2),
           "pct_rank_vs_mapped_large_false": round(100 * float((tab.large_false_per_km2 < sub.large.sum() / a).mean()), 0),
           "pct_rank_vs_mapped_large_all": round(100 * float((tab.large_all_per_km2 < sub.large.sum() / a).mean()), 0),
           "non_landslide_points": int(nl_per.get(b, 0)), "inventory_polygons": int((lsb == b).sum())}
    rows_out.append(row)
res = pd.DataFrame(rows_out)
res.to_csv("reports/tables/unmapped_blocks_check.csv", index=False)
def q(s):
    return f"median {s.median():.2f}, IQR {s.quantile(.25):.2f} to {s.quantile(.75):.2f}"
print(res.to_string(index=False))
print(f"\nmapped blocks (n = {len(tab)}, area >= 3 km2), large candidates per km2: all {q(tab.large_all_per_km2)}; false alarms only {q(tab.large_false_per_km2)}")
print(f"mapped blocks: non-landslide points per block, mean {mapped_nl:.1f}; inventory polygons per block, mean {float(pd.Series(lsb).value_counts().reindex(tab.index, fill_value=0).mean()):.1f}")
print(f"five blocks together: {int(cand.large.sum())} large candidates in {res.area_km2.sum():.1f} km2 = {cand.large.sum() / res.area_km2.sum():.2f} per km2; all candidates {len(cand)}")
