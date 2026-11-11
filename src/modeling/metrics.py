"""Object-level metrics for the screening detector (definitions fixed in docs/validation-design.md before any model).

All functions work on plain arrays so they can be tested on synthetic scenes (src/modeling/check_metrics.py).
  prob       (H, W) float probability map
  poly_id    (H, W) int, 0 = no polygon, else polygon id (all_touched rasterization)
  ring       (H, W) bool, the exclusion ring around polygons inside the domain
  eval_mask  (H, W) bool, pixels of the blocks being scored
  eval_ids   polygon ids scored (representative point in the scored blocks); classes: "<5 px", "5-20 px", ">20 px" by area

Predicted objects: pixels with prob >= thr inside eval_mask, 8-connected, objects smaller than a_min pixels removed.
Lenient recall: a polygon counts as detected when at least one predicted pixel touches it.
Strict recall: IoU between the polygon and the union of the predicted objects overlapping it >= iou_min (0.3).
Object precision: the share of predicted objects that overlap a polygon or its exclusion ring (the ring is boundary tolerance).
Unmapped real slides count as false alarms, so precision is a floor.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score

CLASSES = ["<5 px", "5-20 px", ">20 px"]
EIGHT = np.ones((3, 3), dtype=int)


def objects(prob, thr, a_min, eval_mask):
    """Label 8-connected objects of (prob >= thr) inside eval_mask; objects under a_min pixels are dropped. Returns (labels, n)."""
    lab, n = ndi.label((prob >= thr) & eval_mask, structure=EIGHT)
    if n == 0:
        return lab, 0
    keep = np.bincount(lab.ravel()) >= a_min
    keep[0] = False
    return (np.cumsum(keep) * keep)[lab], int(keep.sum())


def object_metrics(prob, thr, a_min, poly_id, poly_class, ring, eval_mask, eval_ids, iou_min=0.3):
    """poly_class: Series or dict mapping polygon id -> size class. Returns a flat dict of counts and rates."""
    lab, n_pred = objects(prob, thr, a_min, eval_mask)
    pred = lab > 0
    sizes = np.bincount(lab.ravel(), minlength=n_pred + 1)
    eval_ids = np.asarray(eval_ids)
    hit = np.bincount(poly_id[pred], minlength=int(poly_id.max()) + 1)
    classes = np.array([poly_class[i] for i in eval_ids])
    detected = hit[eval_ids] > 0
    strict = np.zeros(len(eval_ids), bool)
    boxes = ndi.find_objects(poly_id, max_label=int(poly_id.max()))
    for k, pid in enumerate(eval_ids):
        if not detected[k] or boxes[pid - 1] is None:
            continue
        sl = boxes[pid - 1]
        inside = poly_id[sl] == pid
        objs = np.unique(lab[sl][inside])
        objs = objs[objs > 0]
        inter = int(np.isin(lab[sl][inside], objs).sum())
        union = int(inside.sum()) + int(sizes[objs].sum()) - inter
        strict[k] = inter / union >= iou_min
    tp_objects = len(np.unique(lab[((poly_id > 0) | ring) & pred])) if n_pred else 0
    out = {"thr": thr, "a_min": a_min, "pred_objects": n_pred, "tp_objects": tp_objects,
           "precision": tp_objects / n_pred if n_pred else np.nan,
           "flagged_area_share": float(pred.sum() / max(eval_mask.sum(), 1)), "polygons": len(eval_ids)}
    for c in CLASSES:
        m = classes == c
        out[f"n_{c}"] = int(m.sum())
        out[f"recall_{c}"] = float(detected[m].mean()) if m.any() else np.nan
        out[f"strict_{c}"] = float(strict[m].mean()) if m.any() else np.nan
    big = classes != "<5 px"
    out["recall_ge5"] = float(detected[big].mean()) if big.any() else np.nan
    out["strict_ge5"] = float(strict[big].mean()) if big.any() else np.nan
    return out


def sweep(prob, thresholds, a_mins, poly_id, poly_class, ring, eval_mask, eval_ids):
    return pd.DataFrame([object_metrics(prob, t, a, poly_id, poly_class, ring, eval_mask, eval_ids) for t in thresholds for a in a_mins])


def choose_operating_point(df, recall_target=0.8):
    """Highest precision among the rows that reach recall_ge5 >= recall_target (ties: smaller flagged area).
    If none reaches it, the row with the highest recall_ge5 is returned with reached=False."""
    ok = df[df.recall_ge5 >= recall_target]
    if ok.empty:
        return df.sort_values(["recall_ge5", "precision"], ascending=False).iloc[0].to_dict() | {"reached": False}
    return ok.sort_values(["precision", "flagged_area_share"], ascending=[False, True]).iloc[0].to_dict() | {"reached": True}


def pixel_ap(y, p, w=None):
    """Average precision at true prevalence: negatives carry their sampling weight."""
    return float(average_precision_score(y, p, sample_weight=w))
