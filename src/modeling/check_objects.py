"""Parity check on real training data: the object tables of objects.py reproduce metrics.object_metrics and the evaluate_oof results.

Uses the out-of-fold raster c_hgb_r0 at its own operating point and the training blocks only (never the test blocks).
Usage: python -m src.modeling.check_objects
"""

import numpy as np
import pandas as pd

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.metrics import CLASSES, object_metrics
from src.modeling.objects import Scene, score_objects

raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
op = raw[(raw.config == "c_hgb") & (raw.repeat == 0) & (raw.recall_target == 0.8)].iloc[0]
thr, a_min = np.float64(op.thr), int(op.a_min)  # np.float64, not float: see the NumPy 2 note in stage2.py
scene = Scene()
prob = np.asarray(np.load(f"{INTERIM}/oof/c_hgb_r0.npy", mmap_mode="r"))
train = block_mask("train")
pol = scene.polygons[scene.polygons.in_train]
obj, poly, _ = score_objects(scene, prob, thr, a_min, train, pol)
ref = object_metrics(
    prob, thr, a_min, scene.poly, dict(zip(pol.id, pol.size_class, strict=True)), scene.ring, train, pol.id.to_numpy()
)

big = poly.size_class != "<5 px"
mine = {
    "pred_objects": len(obj),
    "tp_objects": int(obj.tp.sum()),
    "precision": obj.tp.mean(),
    "recall_ge5": poly.detected[big].mean(),
    "strict_ge5": poly.strict[big].mean(),
}
for c in CLASSES:
    m = poly.size_class == c
    mine[f"recall_{c}"], mine[f"strict_{c}"] = poly.detected[m].mean(), poly.strict[m].mean()
bad = {k: (v, ref[k]) for k, v in mine.items() if not np.isclose(v, ref[k], rtol=0, atol=1e-12)}
print("objects.py vs metrics.object_metrics:", "all equal" if not bad else f"DIFFER {bad}")
vs_oof = {"precision": (mine["precision"], op.precision_at_target), "recall_ge5": (mine["recall_ge5"], op.recall_ge5)}
bad2 = {k: v for k, v in vs_oof.items() if not np.isclose(v[0], v[1], rtol=0, atol=1e-9)}
print(
    "objects.py vs evaluate_oof (c_hgb_r0):",
    "all equal" if not bad2 else f"DIFFER {bad2}",
    {k: round(float(v[0]), 4) for k, v in vs_oof.items()},
)
assert not bad and not bad2
print("CHECK PASS")
