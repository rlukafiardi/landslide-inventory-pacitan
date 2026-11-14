"""Phase 5: error analysis of predicted objects at an operating point: where the false alarms and the misses sit.

tables(scene, obj, poly, mask) takes the object and polygon tables of objects.score_objects and returns DataFrames:
  size_px, slope_deg, landuse, distriv_m   predicted objects by size, mean slope, modal land use, minimum distance to a river:
                                           count, true objects, false alarms, precision, share of the false alarms
  false_alarm_distance_to_polygon          false alarms by distance to the nearest mapped polygon, against the share of the domain area
  miss_by_area_m2, miss_by_slope, miss_by_landuse, miss_by_distriv_m   polygons >= 5 px by the same factors: detected, missed, recall
CLI: the out-of-fold raster of one repeat on the training blocks -> reports/tables/error_analysis_*.csv. evaluate_test.py calls
tables() on the test blocks. Usage: python -m src.modeling.error_analysis [config_r<seed>]
"""

import sys

import numpy as np
import pandas as pd

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.objects import Scene, score_objects

SL = [0, 8, 15, 25, 40, 90]
SZ = [0, 2, 3, 5, 10, 20, 10_000]
RIV = [0, 25, 50, 100, 250, 500, 100_000]
NEAR = [0, 1, 50, 100, 300, 1000, 100_000]


def _by(df, col, bins=None):
    k = pd.cut(df[col], bins, right=False) if bins else df[col]
    g = df.groupby(k, observed=True).agg(objects=("tp", "size"), true_objects=("tp", "sum"))
    g["false_alarms"] = g.objects - g.true_objects
    g["precision"] = g.true_objects / g.objects
    g["share_of_false_alarms"] = g.false_alarms / g.false_alarms.sum()
    return g.round(3)


def _det_by(big, col, bins=None):
    k = pd.cut(big[col], bins, right=False) if bins else big[col]
    g = big.groupby(k, observed=True).agg(polygons=("detected", "size"), detected=("detected", "sum"))
    g["recall"] = (g.detected / g.polygons).round(3)
    g["missed"] = g.polygons - g.detected
    return g


def tables(scene, obj, poly, mask):
    obj = obj.rename(columns={"size": "size_px", "distriv": "distriv_m"})
    out = {
        "size_px": _by(obj, "size_px", SZ),
        "slope_deg": _by(obj, "slope", SL).rename_axis("slope"),
        "landuse": _by(obj, "landuse"),
        "distriv_m": _by(obj, "distriv_m", RIV),
    }
    dom = (
        pd.Series(scene.landuse[mask & ~np.isnan(scene.landuse)].astype(int))
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )
    out["landuse"]["share_of_domain_area"] = dom.reindex(out["landuse"].index).to_numpy()
    fa = obj[~obj.tp]
    base = pd.cut(
        pd.Series(scene.dist_poly_m[mask].ravel()).sample(2_000_000, random_state=0, replace=mask.sum() < 2_000_000),
        NEAR,
        right=False,
    )
    out["false_alarm_distance_to_polygon"] = pd.DataFrame(
        {
            "false_alarm_share": pd.cut(fa.dist_poly_m, NEAR, right=False)
            .value_counts(normalize=True)
            .sort_index()
            .round(3),
            "domain_area_share": base.value_counts(normalize=True).sort_index().round(3),
        }
    )
    big = poly[poly.size_class != "<5 px"].rename(columns={"distriv": "distriv_m"})
    out["miss_by_area_m2"] = _det_by(big, "area_m2", [0, 500, 1000, 2000, 5000, 1e9])
    out["miss_by_slope"] = _det_by(big, "slope", SL)
    out["miss_by_landuse"] = _det_by(big, "landuse")
    out["miss_by_distriv_m"] = _det_by(big, "distriv_m", RIV)
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "c_hgb_r0"
    model, rep = name.rsplit("_r", 1)
    raw = pd.read_csv("reports/tables/modeling_results_raw.csv")
    op = raw[(raw.config == model) & (raw.repeat == int(rep)) & (raw.recall_target == 0.8)].iloc[0]
    thr, a_min = np.float64(op.thr), int(op.a_min)  # np.float64, not float: see the NumPy 2 note in stage2.py
    scene = Scene()
    prob = np.asarray(np.load(f"{INTERIM}/oof/{name}.npy", mmap_mode="r"))
    train = block_mask("train")
    obj, poly, _ = score_objects(scene, prob, thr, a_min, train, scene.polygons[scene.polygons.in_train])
    big = poly[poly.size_class != "<5 px"]
    print(
        f"{name}: threshold {thr:.3f}, a_min {a_min}; objects {len(obj):,}: true {int(obj.tp.sum()):,}, false alarms {int((~obj.tp).sum()):,} "
        f"(precision {obj.tp.mean():.3f}); polygons >= 5 px {len(big)}, detected {int(big.detected.sum())} (recall {big.detected.mean():.3f})"
    )
    for k, df in tables(scene, obj, poly, train).items():
        df.to_csv(f"reports/tables/error_analysis_{k}.csv")
        print(f"\n== {k}\n{df.to_string()}")
