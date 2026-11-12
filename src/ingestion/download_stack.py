"""Phase 1c: download the feature stack from Earth Engine onto the Mendeley factor grid (EPSG:32749, 10 m).

Layers (one GeoTIFF each in data/interim/):
  emb2017, emb2018  64 bands, uint8 = round(value * 127) + 128, nodata 0   (Satellite Embedding annual; value = (v - 128) / 127)
  ndvi              3 bands (pre, postA, postB), float32, nodata -9999 (Sentinel-2 L1C + Cloud Score+ median NDVI)
  ndvi_prev         2 bands (pre16, postB17): the same seasons one year earlier, same recipe
  ndvi_post         2 bands (postC, postD): Apr-May 2019 and 2020, same recipe (needs imagery 1-2 years after the event)
Tiles (560 px for embeddings, 800 for NDVI) stay under the 48 MiB computePixels limit; only tiles that touch the valid land-use
mask are requested; finished tiles are listed in <layer>.done.json, so a rerun resumes. Sequential and polite; stops
on the first quota or permission error. Usage: python -m src.ingestion.download_stack <layer> [--pilot N]
"""
import json
import os
import sys
import time

import ee
import numpy as np
import rasterio
from rasterio.windows import Window

from src.embedding_utils import BANDS, COLLECTION, S2_WINDOWS, S2_WINDOWS_POST, S2_WINDOWS_PREV, init_ee, ndvi_composite

FDIR = "data/spatial/mendeley_vrtx3w6mjd"
OUT = "data/interim"
PAUSE_S = 0.5

layer = sys.argv[1]
pilot = int(sys.argv[sys.argv.index("--pilot") + 1]) if "--pilot" in sys.argv else None

with rasterio.open(f"{FDIR}/map_landuse.tif") as r:
    valid, tr, H, W, crs = r.read(1) != r.nodata, r.transform, r.height, r.width, r.crs

if layer in ("emb2017", "emb2018"):
    year = int(layer[3:])
    names, dtype, nodata, bytes_px, TILE = BANDS, "uint8", 0, 2, 560  # the server counted 2 bytes per uint8 value
else:
    WINDOWS = {"ndvi": S2_WINDOWS, "ndvi_prev": S2_WINDOWS_PREV, "ndvi_post": S2_WINDOWS_POST}[layer]
    names, dtype, nodata, bytes_px, TILE = list(WINDOWS), "float32", -9999.0, 4, 800
assert TILE * TILE * len(names) * bytes_px <= 48 * 2**20, "tile exceeds the computePixels request limit"


def image():
    if layer in ("ndvi", "ndvi_prev", "ndvi_post"):
        return ee.Image.cat([ndvi_composite(*WINDOWS[k]).rename(k) for k in names]).unmask(nodata).toFloat()
    img = ee.ImageCollection(COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic().select(names)
    return img.multiply(127).round().clamp(-127, 127).add(128).unmask(nodata).toUint8()


path, done_path = f"{OUT}/{layer}.tif", f"{OUT}/{layer}.done.json"
if not os.path.exists(path):
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=len(names), dtype=dtype, crs=crs, transform=tr,
                       nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="YES") as dst:
        dst.descriptions = tuple(names)
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

tiles = [(i, j) for i in range(0, H, TILE) for j in range(0, W, TILE) if valid[i:i + TILE, j:j + TILE].any()]
todo = [t for t in tiles if f"{t[0]}_{t[1]}" not in done]
print(f"{layer}: {len(tiles)} tiles touch the domain, {len(done)} done, {len(todo)} to fetch")
if pilot:
    todo = todo[len(todo) // 2:][:pilot]

init_ee()
img = image()
t0 = time.time()
with rasterio.open(path, "r+") as dst:
    for n, (i, j) in enumerate(todo, 1):
        h, w = min(TILE, H - i), min(TILE, W - j)
        grid = {"dimensions": {"width": w, "height": h}, "crsCode": "EPSG:32749",
                "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": tr.c + j * 10,
                                    "shearY": 0, "scaleY": -10, "translateY": tr.f - i * 10}}
        try:
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        except ee.EEException as e:
            print(f"stopping at tile {i}_{j}: {e}")
            break
        dst.write(np.stack([arr[b] for b in names]).astype(dtype), window=Window(j, i, w, h))
        done.add(f"{i}_{j}")
        json.dump(sorted(done), open(done_path, "w"))
        print(f"  tile {i}_{j} ok ({n}/{len(todo)}, {time.time() - t0:.0f}s elapsed)", flush=True)
        time.sleep(PAUSE_S)
