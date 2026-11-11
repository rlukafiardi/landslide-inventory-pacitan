"""Phase 1a: where are the 3 km blocks without a landslide polygon? Map for judging inventory completeness.

Rule for "likely unmapped": an empty block whose expected landslide count is >= 4.6 (Poisson P(0) < 1%), with the
expected count = landslide density of the block's modal lithology class (polygons per km2 over the whole domain) x block area.
Usage: python -m src.data.coverage_map <factor_dir> <out_png>
"""
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fdir, out = sys.argv[1:3]
B = 300
with rasterio.open(f"{fdir}/map_landuse.tif") as r:
    valid = r.read(1) != 15
    tr = r.transform
with rasterio.open(f"{fdir}/map_slope.tif") as s:
    sl = s.read(1).astype("float32")
    sl[(sl == s.nodata) | ~np.isfinite(sl)] = np.nan
with rasterio.open(f"{fdir}/map_geologi.tif") as g:
    geo = g.read(1).astype(int)
    gnd = int(g.nodata)
H, W = valid.shape
ls = gpd.read_file(f"{fdir}/landslide_pacitan.gpkg")
p = ls.geometry.representative_point()
rows, cols = ((tr.f - p.y.to_numpy()) / 10).astype(int), ((p.x.to_numpy() - tr.c) / 10).astype(int)
hit = {(a // B, b // B) for a, b in zip(rows, cols)}
inreg = valid & (geo != gnd)
cls_km2 = {c: (inreg & (geo == c)).sum() * 100 / 1e6 for c in np.unique(geo[inreg])}
cls_n = {c: int((geo[rows, cols] == c).sum()) for c in cls_km2}
density = {c: cls_n[c] / cls_km2[c] for c in cls_km2 if cls_km2[c] > 1}
suspects = []

fig, ax = plt.subplots(figsize=(9, 6.6))
ax.imshow(np.where(valid, 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.2, alpha=0.5, extent=(0, W, H, 0))
for i in range(0, H, B):
    for j in range(0, W, B):
        v = valid[i:i + B, j:j + B]
        if v.sum() > 0.5 * B * B and (i // B, j // B) not in hit:
            steep = float(np.nanmean(sl[i:i + B, j:j + B][v] > 25))
            m = geo[i:i + B, j:j + B][v]
            m = m[m != gnd]
            code = int(np.bincount(m).argmax()) if len(m) else None
            expected = density.get(code, 0) * v.sum() * 100 / 1e6
            sus = expected >= 4.6
            if sus:
                suspects.append((i // B, j // B, round(expected, 1), f"{steep:.0%}", code))
            ax.add_patch(Rectangle((j, i), B, B, fill=sus, fc="#d95f02", alpha=0.45 if sus else 0, ec="#d95f02" if sus else "#999", lw=1.6 if sus else 0.6))
            ax.text(j + B / 2, i + B / 2, f"{steep:.0%}", ha="center", va="center", fontsize=6.5, color="#7f2704" if sus else "#666")
ax.scatter(cols, rows, s=2.5, c="#1b1b1b", lw=0)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Empty blocks: a low-density south-west cluster, and 5 inland blocks that look unmapped", loc="left", fontsize=11)
ax.text(0, H + 150, "3 km blocks (>50% inside the land-use mask) with no polygon centroid; number = share of pixels steeper than 25°; "
        "orange = likely unmapped (expected count >= 4.6 from lithology class density). Dots = landslide centroids (743).", fontsize=7.5, color="#444")
fig.savefig(out, dpi=140, bbox_inches="tight")

print("likely unmapped blocks (row, col, expected count, steep share, lithology code):", suspects)
