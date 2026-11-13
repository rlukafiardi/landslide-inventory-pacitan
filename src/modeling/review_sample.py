"""Phase 5: blinded visual review of large false alarms, as a self-contained HTML page (no GIS software needed).

Samples N_FA large false alarms (>= MIN_PX pixels, no overlap with a mapped polygon or its ring) and N_CTRL large true objects from the
c_hgb out-of-fold predictions at the 80%-recall operating point, shuffles them, and renders for each a 64 x 64 px chip (640 m) of
Sentinel-2 true colour before (Aug-Oct 2017) and after (Apr-May 2018), the same with the flagged object outlined, and the NDVI change.
The page asks for a verdict per chip (landslide scar / not a scar / unsure), keeps answers in the browser and downloads them as CSV.
The key (which chips are false alarms or controls) is written to data/interim/review_key.csv and not shown in the page.
Imagery is the same Sentinel-2 composite as the model input, not the 50 cm imagery behind the inventory, so only scar-sized objects
are judgeable; the page links each chip to a satellite view in Google Maps (grid coordinates are Sentinel-2-true, no shift needed).
Chip and page code: review_page.py. Needs the Earth Engine login (.env). Usage: python -m src.modeling.review_sample [config_r<seed>]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.metrics import EIGHT
from src.modeling.review_page import Chips, page, read

NAME = sys.argv[1] if len(sys.argv) > 1 else "c_hgb_r0"
MIN_PX, N_FA, N_CTRL, CHIP, SEED = 20, 30, 10, 64, 11
model, rep = NAME.rsplit("_r", 1)
raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
op = raw[(raw.config == model) & (raw.repeat == int(rep)) & (raw.recall_target == 0.8)].iloc[0]
THR, AMIN = np.float64(op.thr), int(op.a_min)  # np.float64, not float: see the NumPy 2 note in stage2.py


prob = np.asarray(np.load(f"{INTERIM}/oof/{NAME}.npy", mmap_mode="r"))
poly, block, label = (read(f"{INTERIM}/{n}.tif") for n in ("poly_id", "block", "label"))  # label: 255 is the ring, read as stored
ring = (label == 255) & (block >= 0)
train = block_mask("train")
lab, n = ndi.label((prob >= THR) & train, structure=EIGHT)
sizes = np.bincount(lab.ravel(), minlength=n + 1)
keep = sizes >= AMIN
keep[0] = False
is_true = np.zeros(n + 1, bool)
is_true[np.unique(lab[((poly > 0) | ring) & (lab > 0)])] = True
big = np.flatnonzero(keep & (sizes >= MIN_PX))
rng = np.random.default_rng(SEED)
fa = rng.choice(big[~is_true[big]], N_FA, replace=False)
ctrl = rng.choice(big[is_true[big]], N_CTRL, replace=False)
sample = rng.permutation(np.r_[fa, ctrl])
print(f"{NAME}: threshold {THR:.3f}; large objects {len(big)} ({int((~is_true[big]).sum())} false alarms, {int(is_true[big].sum())} true); sampling {N_FA} + {N_CTRL}")

chips = Chips()
rows, cards = [], []
for k, o in enumerate(sample, 1):
    html, row = chips.card(lab, o, prob, k, int(sizes[o]))
    cards.append(html)
    rows.append({**row, "object": int(o), "kind": "control_true" if is_true[o] else "false_alarm"})
key = pd.DataFrame(rows)[["id", "object", "kind", "size_px", "lat", "lon", "slope_deg", "pmax"]]
key.to_csv(f"{INTERIM}/review_key.csv", index=False)
json.dump(key[["id", "lat", "lon", "size_px"]].to_dict("records"), open(f"{INTERIM}/review_points.json", "w"))
open("reports/review_sample.html", "w").write(page(cards, "landslide_review_v1", "landslide_review_answers.csv"))
print(f"written reports/review_sample.html ({len(rows)} chips); key in {INTERIM}/review_key.csv")
