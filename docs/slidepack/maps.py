"""Map images for the deck: hillshade base, inventory, held-out blocks, one zoomed test block, the candidate file.

Everything is EPSG:32749 (UTM 49S, metres). Inventory polygons and the candidates_inventory_frame layer are in the inventory's own
frame; the block grid, polygons.parquet and the model use the Sentinel-2 frame (inventory shifted by SHIFT_E_M, SHIFT_N_M).
Join counts are printed so a silent loss shows up in the build log.
"""

import json
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import box, shape

warnings.filterwarnings("ignore")

B, RES = 300, 10  # block size in pixels (src/data/constants.py), pixel size in metres
UNMAPPED_BLOCKS = [(8, 10), (9, 15), (9, 16), (9, 18), (10, 14)]
NAVY, SUN, BLUE, GREY = "#1D3557", "#E63946", "#2171B5", "#8A8A84"
CRS = "EPSG:32749"


def _read_masked(path, step=1):
    with rasterio.open(path) as r:
        assert str(r.crs) == CRS, (path, r.crs)
        a = r.read(1).astype("float64")
        left, top = r.transform.c, r.transform.f
    a[a < -1e30] = np.nan
    if step > 1:
        h, w = a.shape[0] // step * step, a.shape[1] // step * step
        a = np.nanmean(a[:h, :w].reshape(h // step, step, w // step, step), axis=(1, 3))
    return a, left, top


def hillshade_rgba(z, step):
    from matplotlib.colors import LightSource

    land = np.isfinite(z)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        np.where(land, z, np.nanmin(z)), vert_exag=2, dx=RES * step, dy=RES * step
    )
    g = 0.66 + 0.30 * shade
    return np.dstack([g, g, g, land.astype(float)])


def block_box(i, j, left, top):
    x0, y1 = left + j * B * RES, top - i * B * RES
    return box(x0, y1 - B * RES, x0 + B * RES, y1)


def new_ax(extent, max_w, max_h):
    """Axes filling a figure whose aspect equals the data aspect, as large as fits in max_w x max_h points."""
    x0, x1, y0, y1 = extent
    asp = (x1 - x0) / (y1 - y0)
    w, h = (max_w, max_w / asp) if max_w / asp <= max_h else (max_h * asp, max_h)
    fig = plt.figure(figsize=(w / 72, h / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def scale_bar(ax, extent, length_m, label):
    x0, x1, y0, y1 = extent
    x, y = x0 + (x1 - x0) * 0.03, y0 + (y1 - y0) * 0.04
    ax.plot([x, x + length_m], [y, y], color=NAVY, lw=2.5, solid_capstyle="butt")
    ax.text(x + length_m / 2, y + (y1 - y0) * 0.015, label, ha="center", va="bottom", color=NAVY, fontsize=9)


def build_maps(root, img, slot_w=540, slot_h=345):
    S = root / "data/spatial/mendeley_vrtx3w6mjd"
    out, st = {}, {}
    cfg = json.load(open(root / "data/interim/test_blocks.json"))
    test_ids = set(cfg["test_blocks"])
    sliver_ids = set(cfg["sliver_blocks_train_only"])

    z, left, top = _read_masked(S / "map_elev.tif", step=4)
    extent = (left, left + z.shape[1] * RES * 4, top - z.shape[0] * RES * 4, top)
    base = hillshade_rgba(z, 4)

    # blocks: ids of the domain (the five likely-unmapped blocks are -1 in the raster)
    with rasterio.open(root / "data/interim/block.tif") as r:
        blk = r.read(1)
        assert str(r.crs) == CRS
    ids = np.unique(blk[blk >= 0])
    st["n_blocks_total"] = len(ids)
    st["n_blocks_eligible"] = len(set(ids) - sliver_ids)
    st["n_blocks_test"] = len(test_ids)
    print(
        f"blocks: {len(ids)} in domain, {st['n_blocks_eligible']} without the train-only slivers, {len(test_ids)} test"
    )
    step = 5
    blk_s = blk[: blk.shape[0] // step * step : step, : blk.shape[1] // step * step : step]
    tr = Affine(RES * step, 0, left, 0, -RES * step, top)
    shp = [(shape(g), int(v)) for g, v in features.shapes(blk_s.astype("int32"), mask=blk_s >= 0, transform=tr)]
    blocks = (
        gpd.GeoDataFrame({"id": [v for _, v in shp]}, geometry=[g for g, _ in shp], crs=CRS)
        .dissolve(by="id")
        .reset_index()
    )
    blocks["test"] = blocks.id.isin(test_ids)
    unm = gpd.GeoDataFrame(geometry=[block_box(i, j, left, top) for i, j in UNMAPPED_BLOCKS], crs=CRS)

    # inventory (inventory frame) joined to polygons.parquet (S2 frame) for block and size class
    inv = gpd.read_file(S / "landslide_pacitan.gpkg")
    pq = pd.read_parquet(root / "data/interim/polygons.parquet")
    print("inventory", inv.crs, len(inv), "parquet", len(pq))
    inv = inv.merge(pq[["id", "block", "in_train", "size_class", "area_m2"]], left_on="kode", right_on="id", how="left")
    assert len(inv) == len(pq) == inv.id.notna().sum(), "inventory / parquet join lost rows"
    inv["px"], inv["py"] = inv.geometry.centroid.x, inv.geometry.centroid.y
    st["n_poly"], st["n_poly_test"] = len(inv), int((~inv.in_train).sum())
    st["n_poly_test_ge5"] = int(((~inv.in_train) & (inv.size_class != "<5 px")).sum())

    # slope at the polygon centroids against the study area
    sl, _, _ = _read_masked(S / "map_slope.tif", step=1)
    inside = (blk >= 0) & np.isfinite(sl)
    cols = ((inv.px - left) / RES).astype(int)
    rows = ((top - inv.py) / RES).astype(int)
    s_poly = sl[rows, cols]
    st["slope_med_poly"], st["slope_med_area"] = float(np.nanmedian(s_poly)), float(np.median(sl[inside]))
    st["share_poly_ge25"], st["share_area_ge25"] = float(np.nanmean(s_poly >= 25)), float((sl[inside] >= 25).mean())
    print(
        "slope median at polygons / area:",
        st["slope_med_poly"],
        st["slope_med_area"],
        "share >=25 deg:",
        st["share_poly_ge25"],
        st["share_area_ge25"],
    )

    # 1. where the scars are
    fig, ax = new_ax(extent, slot_w, slot_h)
    ax.imshow(base, extent=extent, origin="upper", interpolation="bilinear")
    ax.scatter(inv.px, inv.py, s=9, color=SUN, alpha=0.85, linewidths=0)
    scale_bar(ax, extent, 10_000, "10 km")
    out["inventory"] = _save(fig, img / "map_inventory.png")

    # 2. blocks
    fig, ax = new_ax(extent, slot_w, slot_h)
    ax.imshow(base, extent=extent, origin="upper", interpolation="bilinear")
    blocks[~blocks.test].boundary.plot(ax=ax, color="white", linewidth=0.5)
    blocks[blocks.test].plot(ax=ax, color=SUN, alpha=0.35, linewidth=0)
    blocks[blocks.test].boundary.plot(ax=ax, color=SUN, linewidth=1.2)
    unm.plot(ax=ax, facecolor="none", edgecolor=NAVY, hatch="////", linewidth=0.6)
    ax.scatter(inv.px, inv.py, s=2.5, color=NAVY, linewidths=0)
    scale_bar(ax, extent, 10_000, "10 km")
    out["blocks"] = _save(fig, img / "map_blocks.png")

    # 3. zoom on the test block with the most polygons
    per = inv[~inv.in_train].groupby("block").size().sort_values(ascending=False)
    print("test blocks with most polygons:\n", per.head(6).to_string())
    bid = int(per.index[0])
    bi, bj = bid // 100, bid % 100
    bb = block_box(bi, bj, left, top)
    pad = 120
    zext = (bb.bounds[0] - pad, bb.bounds[2] + pad, bb.bounds[1] - pad, bb.bounds[3] + pad)
    zz, zl, zt = _read_masked(S / "map_elev.tif", step=1)
    c0, c1 = int((zext[0] - zl) / RES), int((zext[1] - zl) / RES)
    r0, r1 = int((zt - zext[3]) / RES), int((zt - zext[2]) / RES)
    zbase = hillshade_rgba(zz[r0:r1, c0:c1], 1)
    zext = (zl + c0 * RES, zl + c1 * RES, zt - r1 * RES, zt - r0 * RES)
    cand = gpd.read_file(root / "data/processed/candidates_cempaka_pacitan.gpkg", layer="candidates_inventory_frame")
    print("candidates", cand.crs, len(cand))
    cand = cand[cand.source == "final_test"]
    in_blk = cand[cand.geometry.representative_point().within(bb)]
    true_o, false_o = in_blk[in_blk.overlaps_inventory], in_blk[~in_blk.overlaps_inventory]
    iz = inv[inv.block == bid]
    hit = gpd.sjoin(iz[["kode", "geometry"]], cand[["geometry"]], how="left", predicate="intersects")
    detected_ids = set(hit[hit.index_right.notna()].kode)
    iz = iz.assign(detected=iz.kode.isin(detected_ids))
    ge5 = iz[iz.size_class != "<5 px"]
    st.update(
        zoom_block=(bi, bj),
        zoom_poly=len(iz),
        zoom_det_all=int(iz.detected.sum()),
        zoom_poly_ge5=len(ge5),
        zoom_det_ge5=int(ge5.detected.sum()),
        zoom_objects=len(in_blk),
        zoom_true=len(true_o),
        zoom_false=len(false_o),
    )
    print("zoom block", (bi, bj), {k: v for k, v in st.items() if k.startswith("zoom")})
    fig, ax = new_ax(zext, slot_h, slot_h)
    ax.imshow(zbase, extent=zext, origin="upper", interpolation="nearest")
    false_o.plot(ax=ax, facecolor=SUN, edgecolor=SUN, alpha=0.45, linewidth=0.8)
    true_o.plot(ax=ax, facecolor=BLUE, edgecolor=BLUE, alpha=0.45, linewidth=0.8)
    iz.plot(ax=ax, facecolor="none", edgecolor=NAVY, linewidth=1.1)
    iz[~iz.detected].plot(ax=ax, facecolor=NAVY, edgecolor=NAVY, alpha=0.9, linewidth=0.8)
    gpd.GeoSeries([bb], crs=CRS).boundary.plot(ax=ax, color=NAVY, linewidth=0.8, linestyle="--")
    scale_bar(ax, zext, 500, "500 m")
    out["zoom"] = _save(fig, img / "map_zoom.png")

    # 4. candidates, whole area
    cand_all = gpd.read_file(
        root / "data/processed/candidates_cempaka_pacitan.gpkg", layer="candidates_inventory_frame"
    )
    cp = cand_all.geometry.representative_point()
    large = cand_all.size_class.str.startswith("large").to_numpy()
    st["n_cand"], st["n_large"] = len(cand_all), int(large.sum())
    st["n_cand_test"] = int((cand_all.source == "final_test").sum())
    fig, ax = new_ax(extent, slot_w, slot_h)
    ax.imshow(base, extent=extent, origin="upper", interpolation="bilinear")
    ax.scatter(cp.x[~large], cp.y[~large], s=1.2, color=GREY, linewidths=0, alpha=0.7)
    ax.scatter(cp.x[large], cp.y[large], s=7, color=SUN, linewidths=0, alpha=0.9)
    blocks[blocks.test].boundary.plot(ax=ax, color=NAVY, linewidth=0.7)
    unm.plot(ax=ax, facecolor="none", edgecolor=NAVY, hatch="////", linewidth=0.5)
    scale_bar(ax, extent, 10_000, "10 km")
    out["candidates"] = _save(fig, img / "map_candidates.png")
    return out, st


def _save(fig, path):
    fig.savefig(path, dpi=250, transparent=True)
    plt.close(fig)
    return path
