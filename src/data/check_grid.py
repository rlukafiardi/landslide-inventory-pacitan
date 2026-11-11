"""Phase 1b: every factor raster must share CRS, transform, shape and nodata with the land-use grid.
Usage: python -m src.data.check_grid <factor_dir>
"""
import sys

import rasterio

fdir = sys.argv[1]
NAMES = ["landuse", "aspect", "distriv", "distroad", "elev", "geologi", "planc", "profc", "slope", "spi", "twi"]
ref = None
bad = 0
for n in NAMES:
    with rasterio.open(f"{fdir}/map_{n}.tif") as r:
        a = r.read(1)
        invalid = int((a == r.nodata).sum()) if r.nodata is not None else 0
        key = (str(r.crs), tuple(round(v, 6) for v in r.transform[:6]), r.shape)
        ref = ref or key
        same = key == ref
        bad += not same
        print(f"{n:<9} {'OK ' if same else 'DIFF'} dtype={r.dtypes[0]:<8} nodata={r.nodata} invalid_px={invalid:>9,} shape={r.shape}")
print("grid identical across all layers:", bad == 0)
