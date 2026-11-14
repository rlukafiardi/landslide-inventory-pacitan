"""Object tables for scoring and error analysis, shared by error_analysis.py and evaluate_test.py.

score_objects(scene, prob, thr, a_min, eval_mask, polygons) returns
  obj   one row per predicted object (8-connected pixels with prob >= thr inside eval_mask, objects under a_min pixels dropped):
        size, block (3 km block of its first row-major pixel), tp (overlaps a polygon or its 20 m ring), slope, landuse, distriv, pmax,
        dist_poly_m (metres to the nearest mapped polygon pixel)
  poly  one row per scored polygon: id, block, area_m2, size_class, detected (>= 1 predicted pixel touches it), strict (IoU >= 0.3 between
        the polygon and the union of the objects overlapping it), slope, landuse, distriv, pmax
Definitions follow docs/validation-design.md and src/modeling/metrics.py; check_objects.py proves the two agree on real data.
"""

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

from src.features.build import FDIR, INTERIM
from src.modeling.metrics import objects

IOU_MIN = 0.3


def read(path, mask_nodata=True):
    with rasterio.open(path) as r:
        a = r.read(1).astype("float32")
        nd = r.nodata
    if mask_nodata and nd is not None:
        a[a == nd] = np.nan
    return a


class Scene:
    """The rasters and the polygon table, read once. label.tif uses 255 both as nodata and as the exclusion-ring code, so label, polygon
    and block rasters are read as stored."""

    def __init__(self):
        self.poly = read(f"{INTERIM}/poly_id.tif", mask_nodata=False).astype(int)
        self.block = read(f"{INTERIM}/block.tif", mask_nodata=False).astype(int)
        label = read(f"{INTERIM}/label.tif", mask_nodata=False)
        self.ring = (label == 255) & (self.block >= 0)
        self.slope, self.landuse, self.distriv = (read(f"{FDIR}/map_{n}.tif") for n in ("slope", "landuse", "distriv"))
        self.polygons = pd.read_parquet(f"{INTERIM}/polygons.parquet")
        self._dist = None

    @property
    def dist_poly_m(self):
        if self._dist is None:
            self._dist = ndi.distance_transform_edt(self.poly == 0).astype("float32") * 10
        return self._dist


def _modal(values):
    v = values[~np.isnan(values)]
    return int(np.bincount(v.astype(int)).argmax()) if len(v) else -1


def score_objects(scene, prob, thr, a_min, eval_mask, polygons):
    poly, block = scene.poly, scene.block
    lab, n = objects(prob, thr, a_min, eval_mask)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    tp_lab = np.zeros(n + 1, bool)
    tp_lab[np.unique(lab[((poly > 0) | scene.ring) & (lab > 0)])] = True
    boxes = ndi.find_objects(lab)
    rows = []
    for o in range(1, n + 1):
        sl = boxes[o - 1]
        m = lab[sl] == o
        first = np.argwhere(m)[0]
        rows.append(
            {
                "size": int(sizes[o]),
                "block": int(block[sl][tuple(first)]),
                "tp": bool(tp_lab[o]),
                "slope": float(np.nanmean(scene.slope[sl][m])) if not np.isnan(scene.slope[sl][m]).all() else np.nan,
                "landuse": _modal(scene.landuse[sl][m]),
                "distriv": float(np.nanmin(scene.distriv[sl][m]))
                if not np.isnan(scene.distriv[sl][m]).all()
                else np.nan,
                "pmax": float(prob[sl][m].max()),
                "dist_poly_m": float(scene.dist_poly_m[sl][m].min()),
            }
        )
    obj = pd.DataFrame(rows)

    pred = lab > 0
    hit = np.bincount(poly[pred], minlength=int(poly.max()) + 1)
    pboxes = ndi.find_objects(poly, max_label=int(poly.max()))
    prow = []
    for p in polygons.itertuples():
        sl = pboxes[p.id - 1]
        inside = poly[sl] == p.id
        detected = bool(hit[p.id] > 0)
        strict = False
        if detected:
            objs = np.unique(lab[sl][inside])
            objs = objs[objs > 0]
            inter = int(np.isin(lab[sl][inside], objs).sum())
            union = int(inside.sum()) + int(sizes[objs].sum()) - inter
            strict = inter / union >= IOU_MIN
        sv = scene.slope[sl][inside]
        prow.append(
            {
                "id": p.id,
                "block": int(p.block),
                "area_m2": float(p.area_m2),
                "size_class": p.size_class,
                "detected": detected,
                "strict": bool(strict),
                "slope": float(np.nanmean(sv)) if not np.isnan(sv).all() else np.nan,
                "landuse": _modal(scene.landuse[sl][inside]),
                "distriv": float(np.nanmin(scene.distriv[sl][inside]))
                if not np.isnan(scene.distriv[sl][inside]).all()
                else np.nan,
                "pmax": float(np.nanmax(prob[sl][inside])) if not np.isnan(prob[sl][inside]).all() else np.nan,
            }
        )
    return obj, pd.DataFrame(prow), lab
