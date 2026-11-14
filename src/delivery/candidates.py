"""Phase 7: candidate landslide polygons for expert review, as a GeoPackage (data/processed/candidates_cempaka_pacitan.gpkg).

Score raster: the out-of-fold c_hgb probabilities (repeat 0) on the training blocks and the final model on the test blocks, so every candidate
comes from a model that did not see its own block's labels (the final model's scores on training blocks are in-sample and are not used).
Objects: 8-connected pixels with probability >= the frozen threshold (models/operating_points.json), at least the frozen minimum size.
Layers (same attributes, two frames):
  candidates_s2_grid          geometry on the Sentinel-2 / factor grid (where the model works)
  candidates_inventory_frame  shifted back by (-8.4 m, -9.4 m) so they overlay the manual inventory as published (the shift was estimated from
                              the data, cause unverified, see docs/decisions-log.md)
Attributes: id, rank (1 = highest probability), size_px, area_m2, size_class, pmax, slope_deg, landuse, dist_river_m, overlaps_inventory (touches a
mapped polygon or its 20 m ring), dist_mapped_m, source (oof_train | final_test), verdict and reviewer_note (empty, for the expert).
Review the large objects first: the test blocks show objects of 20 px or more are right about one time in five, smaller ones 1% to 4%.
Usage: python -m src.delivery.candidates
"""

import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import shape

from src.data.constants import SHIFT_E_M, SHIFT_N_M
from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.objects import Scene, score_objects

OUT = "data/processed/candidates_cempaka_pacitan.gpkg"
op = json.load(open("models/operating_points.json"))["c_hgb"]
thr, a_min = np.float64(op["threshold"]), int(op["a_min"])

scene = Scene()
domain = scene.block >= 0
train = block_mask("train")
oof = np.asarray(np.load(f"{INTERIM}/oof/c_hgb_r0.npy", mmap_mode="r"))
final = np.asarray(np.load(f"{INTERIM}/final_c_hgb.npy", mmap_mode="r"))
score = np.where(train, oof, final)
score[~domain] = np.nan
obj, _, lab = score_objects(scene, score, thr, a_min, domain, scene.polygons.iloc[0:0])
print(
    f"threshold {thr:.4f}, minimum {a_min} px: {len(obj):,} candidate objects ({int((lab > 0).sum()):,} pixels, {100 * (lab > 0).sum() / domain.sum():.2f}% of the domain)"
)

with rasterio.open(f"{INTERIM}/label.tif") as r:
    tr, crs = r.transform, r.crs
geoms = {}
for geom, v in features.shapes(lab.astype("int32"), mask=lab > 0, transform=tr, connectivity=8):
    geoms[int(v)] = make_valid(
        shape(geom)
    )  # 8-connected objects pinched at a pixel corner trace self-touching rings; this splits them into parts, area unchanged
obj["label"] = np.arange(1, len(obj) + 1)
obj["geometry"] = obj.label.map(geoms)
obj["source"] = np.where(np.isin(obj["block"], np.unique(scene.block[train])), "oof_train", "final_test")
obj["rank"] = obj.pmax.rank(ascending=False, method="first").astype(int)
obj["area_m2"] = obj["size"] * 100
obj["size_class"] = np.where(obj["size"] >= 20, "large (20 px or more)", "small (3 to 19 px)")
gdf = gpd.GeoDataFrame(
    obj.rename(
        columns={
            "size": "size_px",
            "slope": "slope_deg",
            "distriv": "dist_river_m",
            "tp": "overlaps_inventory",
            "dist_poly_m": "dist_mapped_m",
        }
    ),
    crs=crs,
)
gdf["id"] = gdf["rank"]
gdf["verdict"], gdf["reviewer_note"] = "", ""
gdf = gdf.sort_values("rank")[
    [
        "id",
        "rank",
        "size_px",
        "area_m2",
        "size_class",
        "pmax",
        "slope_deg",
        "landuse",
        "dist_river_m",
        "overlaps_inventory",
        "dist_mapped_m",
        "source",
        "verdict",
        "reviewer_note",
        "geometry",
    ]
].round({"pmax": 4, "slope_deg": 1, "dist_river_m": 0, "dist_mapped_m": 0})
gdf.to_file(OUT, layer="candidates_s2_grid", driver="GPKG")
shifted = gdf.copy()
shifted["geometry"] = [translate(g, -SHIFT_E_M, -SHIFT_N_M) for g in gdf.geometry]
shifted.to_file(OUT, layer="candidates_inventory_frame", driver="GPKG")
summ = gdf.groupby(["source", "size_class", "overlaps_inventory"]).size().rename("candidates").reset_index()
summ.to_csv("reports/tables/candidates_summary.csv", index=False)
print(summ.to_string(index=False))
print(f"written {OUT}: {len(gdf):,} candidates, layers candidates_s2_grid and candidates_inventory_frame")
