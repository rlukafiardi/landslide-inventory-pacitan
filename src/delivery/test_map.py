"""Phase 7: what the frozen model did on the held-out test blocks (reports/figures/test_blocks_map.png).

Six panels, one per test block, chosen by one rule before looking at outcomes: the six test blocks with the most reference polygons of
500 m2 or more. Each panel: vegetation loss (NDVI pre minus post, darker = more lost) in grey as background, the reference polygons (black outline), the
flagged objects at the frozen point split into true objects (blue outline: overlap a polygon or its ring) and false alarms (orange outline).
The test blocks are read through data.load.block_mask("test"), so this needs LANDSLIDE_FINAL_EVAL=1 (the test score is already spent).
Usage: LANDSLIDE_FINAL_EVAL=1 python -m src.delivery.test_map
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D

from src.data.load import block_mask
from src.features.build import INTERIM
from src.modeling.objects import Scene, score_objects

BLUE, ORANGE, INK = "#2a78d6", "#e8871e", "#0b0b0b"
op = json.load(open("models/operating_points.json"))["c_hgb"]
thr, a_min = np.float64(op["threshold"]), int(op["a_min"])
scene = Scene()
test = block_mask("test")
blocks = np.unique(scene.block[test])
pols = scene.polygons[np.isin(scene.polygons.block, blocks)]
prob = np.asarray(np.load(f"{INTERIM}/final_c_hgb.npy", mmap_mode="r"))
obj, poly, lab = score_objects(scene, prob, thr, a_min, test, pols)
tp_ids = obj.index[obj.tp].to_numpy() + 1  # objects are numbered 1..n in row order
true_mask, fa_mask = np.isin(lab, tp_ids), (lab > 0) & ~np.isin(lab, tp_ids)
with rasterio.open(f"{INTERIM}/ndvi.tif") as r:
    nd = r.read().astype("float32")
nd[nd == -9999.0] = np.nan
dB = nd[2] - nd[0]

big = poly[poly.size_class != "<5 px"]
top = big.groupby("block").size().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(11, 7.6), dpi=110)
for ax, b in zip(axes.ravel(), top, strict=False):
    rr, cc = np.nonzero(scene.block == b)
    sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    ax.imshow(-dB[sl], cmap="Greys", vmin=-0.1, vmax=0.45, interpolation="nearest")
    for mask, col, lw in ((scene.poly[sl] > 0, INK, 0.9), (true_mask[sl], BLUE, 1.3), (fa_mask[sl], ORANGE, 1.1)):
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=col, linewidths=lw)
    pb, ob = big[big.block == b], obj[obj.block == b]
    ax.set_title(f"block {b}\n{int(pb.detected.sum())} of {len(pb)} polygons found · {int(ob.tp.sum())} true · {int((~ob.tp).sum())} false objects", fontsize=8.5, loc="left", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([8, 18], [sl[0].stop - sl[0].start - 10] * 2, color=INK, lw=2.5)  # 100 m scale bar (10 px)
    ax.text(8, sl[0].stop - sl[0].start - 16, "100 m", fontsize=7, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.8))
fig.legend(handles=[Line2D([], [], color=INK, lw=1.2, label="reference polygon (inventory)"), Line2D([], [], color=BLUE, lw=1.6, label="flagged object, true (overlaps a polygon or its ring)"),
                    Line2D([], [], color=ORANGE, lw=1.6, label="flagged object, false alarm")], loc="lower center", ncol=3, frameon=False, fontsize=8.5)
fig.suptitle("Held-out test blocks with the most reference polygons: most scars are found, and most flagged objects around them are false alarms", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.text(0.01, 0.045, "Background: Sentinel-2 vegetation loss (NDVI Aug-Oct 2017 minus Apr-May 2018), darker = more lost. 3 km blocks, north up, Sentinel-2 grid. Frozen threshold 0.1956, minimum 3 px.", fontsize=7, color="#52514e")
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig("reports/figures/test_blocks_map.png")
print("reports/figures/test_blocks_map.png; blocks", top)
