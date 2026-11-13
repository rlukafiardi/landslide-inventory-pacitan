"""Blinded visual review of the large candidates in block (8, 10), the one likely-unmapped block with an excess of candidates
(src/analysis/unmapped_check.py). Page format and chips: review_page.py.

The candidates are read from data/processed/unmapped_blocks_candidates.gpkg (the final raster excludes these blocks) and rasterized back onto the
Sentinel-2 grid. Mixes three kinds of chips in random order so the reviewer cannot tell which is which: the 33 large (>= 20 px) objects of the final c_hgb model in
block (8, 10) at the frozen operating point (frozen threshold and minimum size), N_CTRL large true objects and N_FA large false alarms from the mapped training blocks (out-of-fold
c_hgb repeat 0 at its 80%-recall point). The reference false alarms show what a typical "not a scar" looks like; if the block-(8, 10) verdicts
are no more often "scar" than those, the block is not under-mapped. Key (not in the page): data/interim/review_unmapped_key.csv.
Needs the Earth Engine login (.env). Usage: python -m src.modeling.review_unmapped
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.metrics import EIGHT
from src.modeling.review_page import Chips, page, read

BLOCK, MIN_PX, N_CTRL, N_FA, SEED = 810, 20, 10, 10, 13
raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
oof = raw[(raw.config == "c_hgb") & (raw.repeat == 0) & (raw.recall_target == 0.8)].iloc[0]


def label(prob, thr, a_min, mask):
    lab, n = ndi.label((prob >= thr) & mask, structure=EIGHT)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= a_min
    keep[0] = False
    return lab, sizes, np.flatnonzero(keep & (sizes >= MIN_PX))


blockr = read(f"{INTERIM}/block.tif")
poly, lab_tif = read(f"{INTERIM}/poly_id.tif"), read(f"{INTERIM}/label.tif")
ring = (lab_tif == 255) & (blockr >= 0)  # label.tif: 255 is the ring, read as stored

cand = gpd.read_file("data/processed/unmapped_blocks_candidates.gpkg")
cand = cand[(cand.block == BLOCK) & (cand.size_px >= MIN_PX)].reset_index(drop=True)
with rasterio.open(f"{INTERIM}/block.tif") as r:
    tr, shape = r.transform, r.shape
lab_u = rasterize(((g, i + 1) for i, g in enumerate(cand.geometry)), out_shape=shape, transform=tr, dtype="int32")
sz_u = np.bincount(lab_u.ravel(), minlength=len(cand) + 1)
assert (sz_u[1:] == cand.size_px.to_numpy()).all(), "rasterized object sizes differ from the candidate table"
p_fin = np.zeros(shape, "float32")
p_fin[lab_u > 0] = cand.pmax.to_numpy()[lab_u[lab_u > 0] - 1]
big_u = np.arange(1, len(cand) + 1)
p_oof = np.asarray(np.load(f"{INTERIM}/oof/c_hgb_r0.npy", mmap_mode="r"))
lab_m, sz_m, big_m = label(p_oof, np.float64(oof.thr), int(oof.a_min), block_mask("train"))
true_m = np.zeros(len(sz_m), bool)
true_m[np.unique(lab_m[((poly > 0) | ring) & (lab_m > 0)])] = True
rng = np.random.default_rng(SEED)
ctrl = rng.choice(big_m[true_m[big_m]], N_CTRL, replace=False)
fa = rng.choice(big_m[~true_m[big_m]], N_FA, replace=False)
items = [("candidate_810", lab_u, p_fin, sz_u, o) for o in big_u] + [("control_true", lab_m, p_oof, sz_m, o) for o in ctrl] + \
        [("false_alarm_mapped", lab_m, p_oof, sz_m, o) for o in fa]
print(f"block {BLOCK}: {len(big_u)} large candidates (expected 33); sampling {N_CTRL} true + {N_FA} false alarms from mapped blocks")

chips = Chips()
rows, cards = [], []
for k, i in enumerate(np.random.default_rng(SEED + 1).permutation(len(items)), 1):
    kind, lab, prob, sizes, o = items[i]
    html, row = chips.card(lab, o, prob, k, int(sizes[o]))
    cards.append(html)
    rows.append({**row, "kind": kind, "object": int(o)})
key = pd.DataFrame(rows)[["id", "kind", "object", "size_px", "lat", "lon", "slope_deg", "pmax"]]
key.to_csv(f"{INTERIM}/review_unmapped_key.csv", index=False)
open("reports/review_unmapped.html", "w").write(page(cards, "landslide_review_unmapped_v1", "landslide_review_unmapped_answers.csv"))
print(f"written reports/review_unmapped.html ({len(rows)} chips); key in {INTERIM}/review_unmapped_key.csv")
