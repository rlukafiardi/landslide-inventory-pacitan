"""Phase 7: overview map of the candidates (reports/figures/candidates_map.png). Context in grey, the large candidates (the ones worth
reviewing first) in the focus colour, test blocks outlined. Usage: python -m src.delivery.overview_map"""
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D

from src.data.load import block_mask

BLUE, GREY, INK = "#2a78d6", "#b9b8b2", "#0b0b0b"
g = gpd.read_file("data/processed/candidates_cempaka_pacitan.gpkg", layer="candidates_s2_grid")
g["pt"] = g.geometry.centroid
with rasterio.open("data/interim/block.tif") as r:
    block, tr = r.read(1), r.transform
dom = block >= 0
test = dom & ~block_mask("train")
ext = [tr.c, tr.c + tr.a * block.shape[1], tr.f + tr.e * block.shape[0], tr.f]
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=110)
ax.imshow(np.where(dom, 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=4, interpolation="nearest")
ax.contour(np.linspace(ext[0], ext[1], block.shape[1]), np.linspace(ext[3], ext[2], block.shape[0]), test.astype(float), levels=[0.5], colors=INK, linewidths=0.7)
small, large = g[g.size_class.str.startswith("small")], g[g.size_class.str.startswith("large")]
ax.scatter(small.pt.x, small.pt.y, s=1.5, color=GREY, linewidths=0, zorder=3)
ax.scatter(large.pt.x, large.pt.y, s=7, color=BLUE, linewidths=0, zorder=4)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"{len(g):,} candidates for review in 1,370 km2 of Pacitan; the {len(large):,} large ones (20 pixels or more) come first", loc="left", fontsize=11, color=INK)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, markersize=6, label=f"large candidate ({len(large):,})"),
                   Line2D([], [], marker="o", ls="", color=GREY, markersize=4, label=f"small candidate, 3 to 19 px ({len(small):,})"),
                   Line2D([], [], color=INK, lw=1, label="held-out test blocks")], loc="upper left", frameon=False, fontsize=9)
credit = ("Candidates: out-of-fold scores on training blocks, final model on test blocks. Blank squares: five blocks excluded as likely unmapped. UTM 49S, north up.\n"
          "Labels: Samodra et al. 2024 (CC BY 4.0). Sentinel-2: Copernicus. AlphaEarth Foundations Satellite Embedding: Google and Google DeepMind (CC BY 4.0).")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.text(0.01, 0.012, credit, ha="left", va="bottom", fontsize=7, color="#52514e")
fig.savefig("reports/figures/candidates_map.png")
print("reports/figures/candidates_map.png")
