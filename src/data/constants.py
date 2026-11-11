"""Constants shared by the data steps (one place, so the shift and the block size cannot drift apart)."""
B = 300  # 3 km block in pixels (10 m grid)
RING_M = 20  # exclusion ring around polygons
# Inventory polygons sit about 12.6 m south-west of the Sentinel-2 / embedding grid (src/data/estimate_shift.py: the vegetation-loss
# AUC peaks 9.4 m north and 8.4 m east of the polygons; 7.5 to 10.5 m across four regional halves). Applied to the polygons.
SHIFT_E_M, SHIFT_N_M = 8.4, 9.4
# (block row, block col) with no polygon although the lithology-class density predicts >= 4.6 slides (src/data/coverage_map.py)
UNMAPPED_BLOCKS = [(8, 10), (9, 15), (9, 16), (9, 18), (10, 14)]
