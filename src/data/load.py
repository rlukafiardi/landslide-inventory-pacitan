"""Loader for the train / test split. 'train' is the default; 'test' needs LANDSLIDE_FINAL_EVAL=1 and is meant for
the final evaluation script only (ds-workflow phase 6)."""
import json
import os

import numpy as np
import rasterio

TEST_IDS = "data/interim/test_blocks.json"


def block_mask(which="train"):
    """Boolean (H, W) mask of the domain pixels on the train or the test side."""
    if which == "test" and os.environ.get("LANDSLIDE_FINAL_EVAL") != "1":
        raise PermissionError("test blocks are unseen until phase 6; set LANDSLIDE_FINAL_EVAL=1 in the final evaluation only")
    test_blocks = np.array(json.load(open(TEST_IDS))["test_blocks"])
    with rasterio.open("data/interim/block.tif") as b:
        blk = b.read(1)
    in_test = np.isin(blk, test_blocks)
    return (blk >= 0) & (in_test if which == "test" else ~in_test)
