"""Phase 1f: carve the final test set once, as whole 3 km blocks (about 20%), stratified by landslide-pixel count.

Strata: blocks with no positive pixel, then terciles of the blocks that have some. Seed 42. Refuses to overwrite
data/interim/test_blocks.json (gitignored). Run only on the frozen data. Usage: python -m src.data.split
"""
import json
import os

import numpy as np
import pandas as pd
import rasterio

OUT = "data/interim/test_blocks.json"
SEED, TEST_FRAC = 42, 0.20

if os.path.exists(OUT):
    raise SystemExit(f"{OUT} exists: the test set is carved once; refusing to overwrite")

with rasterio.open("data/interim/block.tif") as b, rasterio.open("data/interim/label.tif") as lab:
    block, label = b.read(1), lab.read(1)
m = block >= 0
tab = pd.DataFrame({"block": block[m], "pos": (label[m] == 1).astype(int)}).groupby("block").agg(px=("block", "size"), pos=("pos", "sum"))
tab = tab[tab.px >= 0.2 * 300 * 300]  # skip slivers of coast blocks: they join the training side below
slivers = sorted(set(np.unique(block[m]).tolist()) - set(tab.index.tolist()))
tab["stratum"] = np.where(tab.pos == 0, "none", pd.qcut(tab.pos.where(tab.pos > 0), 3, labels=["low", "mid", "high"]).astype(str))
rng = np.random.default_rng(SEED)
test = []
for _, grp in tab.groupby("stratum"):
    k = max(1, round(TEST_FRAC * len(grp)))
    test += rng.choice(grp.index.to_numpy(), k, replace=False).tolist()
test = sorted(int(t) for t in test)
json.dump({"seed": SEED, "test_blocks": test, "sliver_blocks_train_only": [int(s) for s in slivers]}, open(OUT, "w"))
t = tab.assign(test=tab.index.isin(test))
print(f"blocks in domain {len(tab)} (+{len(slivers)} slivers, train only); test blocks {len(test)} ({len(test) / len(tab):.0%})")
print(t.groupby("stratum").agg(blocks=("px", "size"), test=("test", "sum"), pos_px=("pos", "sum")).to_string())
print(f"positive pixels: train {int(t[~t.test].pos.sum()):,} | test {int(t[t.test].pos.sum()):,} "
      f"({t[t.test].pos.sum() / t.pos.sum():.0%} of all)")
