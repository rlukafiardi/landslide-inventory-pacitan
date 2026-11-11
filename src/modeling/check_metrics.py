"""Phase 4 checks: the object metrics on synthetic scenes with known answers, and the block folds on the real split.
Usage: python -m src.modeling.check_metrics
"""
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.modeling.folds import block_folds
from src.modeling.metrics import choose_operating_point, object_metrics

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}: {name} {detail}")
    if not cond:
        failures.append(name)


# synthetic scene: A 6x6 (>20 px class), B 2x2 (<5 px), C 3x3 (5-20 px); ring = 1-pixel border around polygons
H = W = 80
poly = np.zeros((H, W), int)
poly[10:16, 10:16], poly[40:42, 40:42], poly[60:63, 60:63] = 1, 2, 3
cls = {1: ">20 px", 2: "<5 px", 3: "5-20 px"}
ring = ndi.binary_dilation(poly > 0, structure=np.ones((3, 3))) & (poly == 0)
dom = np.ones((H, W), bool)
ids = [1, 2, 3]


def run(prob, a_min=1, eval_mask=dom, eval_ids=ids):
    return object_metrics(prob, 0.5, a_min, poly, cls, ring, eval_mask, eval_ids)


perfect = (poly > 0).astype(float)
m = run(perfect)
check("perfect prediction: recall, strict recall and precision all 1",
      m["recall_ge5"] == 1 and all(m[f"recall_{c}"] == 1 and m[f"strict_{c}"] == 1 for c in ("<5 px", "5-20 px", ">20 px")) and m["precision"] == 1)
m = run(np.zeros((H, W)))
check("empty prediction: recall 0, no objects, nothing flagged, precision undefined", m["recall_ge5"] == 0 and m["pred_objects"] == 0 and m["flagged_area_share"] == 0 and np.isnan(m["precision"]))
one = np.zeros((H, W))
one[12, 12] = 1
m = run(one)
check("one pixel inside the big polygon: detected leniently, not strictly (IoU 1/36)", m["recall_>20 px"] == 1 and m["strict_>20 px"] == 0 and m["recall_ge5"] == 0.5)
shifted = np.roll(perfect, 1, axis=1)
m = run(shifted)
check("prediction displaced by one pixel: every polygon still detected, all objects overlap polygon or ring", m["recall_<5 px"] == 1 and m["recall_ge5"] == 1 and m["precision"] == 1)
fa = perfect.copy()
fa[70:73, 5:8] = 1
m = run(fa)
check("one false alarm far from any polygon: precision 3/4, recall unchanged", m["pred_objects"] == 4 and m["tp_objects"] == 3 and m["precision"] == 0.75 and m["recall_ge5"] == 1)
speck = perfect.copy()
speck[5, 70] = 1
check("minimum object area removes a one-pixel speck", run(speck, a_min=1)["pred_objects"] == 4 and run(speck, a_min=2)["pred_objects"] == 3)
ring_only = np.zeros((H, W))
ring_only[9, 12] = 1
m = run(ring_only)
check("a hit only in the exclusion ring is a true object but does not detect the polygon", m["tp_objects"] == 1 and m["recall_>20 px"] == 0)
left = dom.copy()
left[:, 40:] = False
m = run(perfect, eval_mask=left, eval_ids=[1])
check("scoring only the left blocks: B and C outside, one polygon scored", m["polygons"] == 1 and m["pred_objects"] == 1 and m["recall_>20 px"] == 1)
only_c = np.zeros((H, W))
only_c[poly == 3] = 1
m = run(only_c)
check("only the 5-20 px polygon found: class recalls 0, 1, 0 and recall_ge5 0.5", (m["recall_<5 px"], m["recall_5-20 px"], m["recall_>20 px"], m["recall_ge5"]) == (0, 1, 0, 0.5))
sw = pd.DataFrame({"recall_ge5": [0.9, 0.85, 0.7], "precision": [0.2, 0.3, 0.6], "flagged_area_share": [0.03, 0.02, 0.01]})
op = choose_operating_point(sw)
check("operating point: highest precision among rows reaching 80% recall", op["precision"] == 0.3 and op["reached"])
check("operating point: reports reached=False when no row reaches the target", not choose_operating_point(sw.assign(recall_ge5=[0.5, 0.4, 0.3]))["reached"])

# folds on the real training split
test_ids = set(json.load(open("data/interim/test_blocks.json"))["test_blocks"])
test_groups = {(t // 100) * 1000 + t % 100 for t in test_ids}
prev = None
for seed in (0, 1, 2):
    f = block_folds(seed)
    pos = f.groupby("fold").pos_px.sum()
    check(f"folds seed {seed}: no test block, 5 folds, positives balanced",
          not (set(f.group) & test_groups) and f.fold.nunique() == 5 and pos.max() / pos.mean() < 1.1,
          f"(positives per fold {pos.min()} to {pos.max()}, groups {len(f)})")
    sig = tuple(f.sort_values("group").fold)
    check(f"folds seed {seed}: differs from the previous seed", prev is None or sig != prev)
    prev = sig
poly_tab = pd.read_parquet("data/interim/polygons.parquet")
f = block_folds(0)
tr_poly = poly_tab[poly_tab.in_train]
g = (tr_poly.block // 100) * 1000 + tr_poly.block % 100
per = tr_poly.assign(fold=g.map(dict(zip(f.group, f.fold))))
check("every training polygon lands in a fold, each fold holds all three size classes",
      per.fold.notna().all() and (per.groupby("fold").size_class.nunique() == 3).all(),
      "(polygons per fold " + str(per.groupby("fold").size().tolist()) + ")")
print("ALL CHECKS PASS" if not failures else f"FAILED: {failures}")
