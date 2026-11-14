"""Phase 6: score the frozen models on the held-out test blocks, once (protocol: docs/decisions-log.md, "MODEL CHOICE AND FROZEN TEST PROTOCOL").

Per model (c_hgb, and for context simple_logreg and naive -dB_min5): objects at the operating point frozen from the out-of-fold repeats
(models/operating_points.json), scored on the test blocks with the object definitions of docs/validation-design.md:
  recall (lenient and strict IoU >= 0.3) for polygons >= 5 px, recall by size class, object precision, flagged share of the area, false objects
  per true object; each with a 95% interval from resampling whole test blocks (2,000 resamples). For c_hgb also an object precision-recall
  curve over thresholds (minimum object size as frozen) and the error-analysis tables (error_analysis.tables).
Nothing is selected here. The test blocks are read only through data.load.block_mask("test"), which needs LANDSLIDE_FINAL_EVAL=1.
--rehearsal runs the same code on the TRAINING blocks with the out-of-fold repeat-0 rasters and each repeat-0 operating point (no test data,
no guard needed): it checks the pipeline and gives the out-of-fold curve that the report puts next to the test curve.
Writes reports/tables/{test|rehearsal}_results.csv, _curve.csv, _error_analysis_*.csv.
Usage: python -m src.modeling.evaluate_test --rehearsal   |   LANDSLIDE_FINAL_EVAL=1 python -m src.modeling.evaluate_test
"""

import json
import sys

import numpy as np
import pandas as pd

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.error_analysis import tables
from src.modeling.objects import Scene, score_objects

REHEARSAL = "--rehearsal" in sys.argv
PREFIX = "rehearsal" if REHEARSAL else "test"
N_BOOT, SEED = 2000, 7
CLASSES = ["<5 px", "5-20 px", ">20 px"]
MODELS = ["c_hgb", "simple_logreg", "naive_dBmin5"]
SHARES = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08]


def operating_point(cfg):
    if REHEARSAL:  # each model's own repeat-0 point, so the numbers can be compared with evaluate_oof
        raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
        row = raw[(raw.config == cfg) & (raw.repeat.isin([0, -1])) & (raw.recall_target == 0.8)].iloc[0]
        return np.float64(row.thr), int(row.a_min)
    op = json.load(open("models/operating_points.json"))[cfg]
    return np.float64(op["threshold"]), int(op["a_min"])


def raster(cfg):
    if REHEARSAL:
        return np.asarray(np.load(f"{INTERIM}/oof/{cfg if cfg == 'naive_dBmin5' else cfg + '_r0'}.npy", mmap_mode="r"))
    return np.asarray(np.load(f"{INTERIM}/final_{cfg}.npy", mmap_mode="r"))


def per_block(obj, poly, blocks, domain_px):
    """Counts per block (aligned to `blocks`) that the point estimate and the bootstrap are both built from."""

    def agg(s):
        return s.reindex(blocks, fill_value=0).to_numpy(dtype=float)

    big = poly[poly.size_class != "<5 px"]
    cols = {
        "objects": agg(obj.groupby("block").size()),
        "true_objects": agg(obj[obj.tp].groupby("block").size()),
        "flagged_px": agg(obj.groupby("block")["size"].sum()),
        "domain_px": domain_px,
        "big": agg(big.groupby("block").size()),
        "big_det": agg(big[big.detected].groupby("block").size()),
        "big_strict": agg(big[big.strict].groupby("block").size()),
    }
    for c in CLASSES:
        p = poly[poly.size_class == c]
        cols[f"n_{c}"], cols[f"det_{c}"] = agg(p.groupby("block").size()), agg(p[p.detected].groupby("block").size())
    return pd.DataFrame(cols, index=blocks)


def stats(t):
    """Ratios from summed per-block counts (works on one set of weights or a (B, blocks) weighted matrix via the caller)."""
    return {
        "object_recall_ge5": t.big_det / t.big,
        "strict_recall_ge5": t.big_strict / t.big,
        "object_precision": t.true_objects / t.objects,
        "flagged_share": t.flagged_px / t.domain_px,
        "false_per_true": (t.objects - t.true_objects) / t.true_objects,
        **{f"recall_{c}": t[f"det_{c}"] / t[f"n_{c}"] for c in CLASSES},
    }


def evaluate(scene, prob, thr, a_min, mask, polygons, blocks, domain_px, W):
    obj, poly, _ = score_objects(scene, prob, thr, a_min, mask, polygons)
    pb = per_block(obj, poly, blocks, domain_px)
    point = stats(pb.sum())
    boot = pd.DataFrame(stats(pd.DataFrame(W @ pb.to_numpy(), columns=pb.columns)))
    row = {
        "threshold": float(thr),
        "a_min": a_min,
        "objects": int(pb.objects.sum()),
        "true_objects": int(pb.true_objects.sum()),
        "polygons_ge5": int(pb.big.sum()),
        "polygons_all": len(poly),
    }
    for k, v in point.items():
        row[k] = float(v)
        row[f"{k}_ci_low"], row[f"{k}_ci_high"] = (float(x) for x in np.nanpercentile(boot[k], [2.5, 97.5]))
    return row, obj, poly


if __name__ == "__main__":
    mask = block_mask(
        "train" if REHEARSAL else "test"
    )  # the test side raises PermissionError without LANDSLIDE_FINAL_EVAL=1
    scene = Scene()
    blocks = np.unique(scene.block[mask])
    domain_px = np.bincount(np.searchsorted(blocks, scene.block[mask]), minlength=len(blocks)).astype(float)
    pols = scene.polygons[np.isin(scene.polygons.block, blocks)]
    print(
        f"{PREFIX}: {len(blocks)} blocks, {int(mask.sum()):,} pixels, {len(pols)} polygons ({int((pols.size_class != '<5 px').sum())} >= 5 px)",
        flush=True,
    )
    W = (
        np.random.default_rng(SEED)
        .multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=N_BOOT)
        .astype(float)
    )
    rows, keep = [], {}
    for cfg in MODELS:
        thr, a_min = operating_point(cfg)
        prob = raster(cfg)
        row, obj, poly = evaluate(scene, prob, thr, a_min, mask, pols, blocks, domain_px, W)
        rows.append({"model": cfg, **row})
        keep[cfg] = (obj, poly)
        print(
            f"{cfg:14s} thr {thr:.4f} a_min {a_min}: recall(>=5 px) {row['object_recall_ge5']:.3f} ({row['object_recall_ge5_ci_low']:.3f}-{row['object_recall_ge5_ci_high']:.3f}), "
            f"strict {row['strict_recall_ge5']:.3f}, precision {row['object_precision']:.3f} ({row['object_precision_ci_low']:.3f}-{row['object_precision_ci_high']:.3f}), "
            f"flagged {row['flagged_share']:.4f}, false per true {row['false_per_true']:.1f}",
            flush=True,
        )
    res = pd.DataFrame(rows)
    res.to_csv(f"reports/tables/{PREFIX}_results.csv", index=False)

    prob = raster("c_hgb")
    _, a_min = operating_point("c_hgb")
    curve = []
    for share in SHARES:
        thr = np.float64(np.quantile(prob[mask & ~np.isnan(prob)], 1 - share))
        row, _, _ = evaluate(scene, prob, thr, a_min, mask, pols, blocks, domain_px, W[:1])
        curve.append(
            {
                "flagged_share_target": share,
                **{
                    k: row[k]
                    for k in (
                        "threshold",
                        "objects",
                        "true_objects",
                        "object_recall_ge5",
                        "strict_recall_ge5",
                        "object_precision",
                        "flagged_share",
                    )
                },
            }
        )
    pd.DataFrame(curve).to_csv(f"reports/tables/{PREFIX}_curve.csv", index=False)
    obj, poly = keep["c_hgb"]
    for k, df in tables(scene, obj, poly, mask).items():
        df.to_csv(f"reports/tables/{PREFIX}_error_analysis_{k}.csv")
    c = res[res.model == "c_hgb"].iloc[0]
    print(
        f"\ncriterion (object recall >= 80% at object precision >= 30%), c_hgb on {PREFIX}: recall {c.object_recall_ge5:.3f} -> {'met' if c.object_recall_ge5 >= 0.8 else 'NOT met'}; "
        f"precision {c.object_precision:.3f} -> {'met' if c.object_precision >= 0.3 else 'NOT met'}"
    )
