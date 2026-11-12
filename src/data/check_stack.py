"""Phase 1d: range and coverage checks on the downloaded stack, inside the evaluation domain.

Checks (each printed with a PASS/FAIL):
  embeddings  every domain pixel has data; dequantized L2 norm within 1 +/- 0.05 (uint8 rounding error)
  ndvi        values in [-1, 1]; share of domain pixels without a cloud-free observation, per window (counted, not dropped)
Usage: python -m src.data.check_stack
"""
import numpy as np
import rasterio
from rasterio.windows import Window

OUT = "data/interim"
STEP = 1000
with rasterio.open(f"{OUT}/block.tif") as b:
    H, W = b.height, b.width
    dom = b.read(1) >= 0
n_dom = int(dom.sum())
ok = True

for layer in ("emb2018",):
    missing, norms = 0, []
    with rasterio.open(f"{OUT}/{layer}.tif") as r:
        for i in range(0, H, STEP):
            w = Window(0, i, W, min(STEP, H - i))
            a = r.read(window=w).astype("float32")
            d = dom[i:i + STEP]
            nodata = (a == 0).all(axis=0)
            missing += int((nodata & d).sum())
            valid = d & ~nodata
            if valid.any():
                norms.append(np.linalg.norm((a[:, valid] - 128) / 127, axis=0))
    n = np.concatenate(norms)
    good = missing == 0 and abs(n.mean() - 1) < 0.05 and n.min() > 0.8 and n.max() < 1.2
    ok &= good
    print(f"{layer}: domain pixels without data {missing:,} of {n_dom:,}; norm mean {n.mean():.3f} min {n.min():.3f} max {n.max():.3f} "
          f"-> {'PASS' if good else 'FAIL'}")

with rasterio.open(f"{OUT}/ndvi.tif") as r:
    names = r.descriptions
    gap = np.zeros(len(names), int)
    out_of_range = np.zeros(len(names), int)
    allgap = 0
    for i in range(0, H, STEP):
        a = r.read(window=Window(0, i, W, min(STEP, H - i)))
        d = dom[i:i + STEP]
        nod = a == -9999
        gap += (nod & d).sum(axis=(1, 2))
        out_of_range += (((a < -1) | (a > 1)) & ~nod & d).sum(axis=(1, 2))
        allgap += int((nod.all(axis=0) & d).sum())
for k, nme in enumerate(names):
    good = out_of_range[k] == 0
    ok &= good
    print(f"ndvi {nme}: no cloud-free observation {gap[k]:,} px ({gap[k] / n_dom:.2%} of the domain), outside [-1, 1]: {out_of_range[k]} "
          f"-> {'PASS' if good else 'FAIL'}")
print(f"ndvi pixels with no observation in any window: {allgap:,} ({allgap / n_dom:.2%})")
print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
