"""Block folds on the training part: whole 3 km blocks (or larger sensitivity blocks) go to 5 folds, balanced by positive pixels.

block_folds(seed) -> DataFrame group, fold, pos_px, px.  Repeats use different seeds; the test blocks never appear.
fold_of(rows, cols, folds, size_px) maps pixels to folds. Usage of the greedy rule: groups sorted by positives (ties shuffled
by the seed) go to the fold with the fewest positives so far, then the fewest pixels.
"""
import numpy as np
import pandas as pd
import rasterio

from src.data.constants import B
from src.data.load import block_mask

N_FOLDS = 5


def _group_ids(rows, cols, size_px):
    return (rows // size_px) * 1000 + cols // size_px


def block_folds(seed, size_px=B, n_folds=N_FOLDS):
    train = block_mask("train")
    with rasterio.open("data/interim/label.tif") as r:
        label = r.read(1)
    rows, cols = np.nonzero(train)
    g = _group_ids(rows, cols, size_px)
    pos = label[rows, cols] == 1
    tab = pd.DataFrame({"group": g, "pos": pos}).groupby("group").agg(pos_px=("pos", "sum"), px=("pos", "size")).reset_index()
    rng = np.random.default_rng(seed)
    tab = tab.iloc[rng.permutation(len(tab))].sort_values(["pos_px", "px"], ascending=False, kind="stable").reset_index(drop=True)
    load_pos, load_px = np.zeros(n_folds), np.zeros(n_folds)
    fold = np.empty(len(tab), int)
    for i, r in tab.iterrows():
        k = int(np.lexsort((load_px, load_pos))[0])
        fold[i] = k
        load_pos[k] += r.pos_px
        load_px[k] += r.px
    tab["fold"] = fold
    return tab


def fold_of(rows, cols, folds, size_px=B):
    """Fold index of each pixel (-1 if its group is not in the training folds)."""
    m = dict(zip(folds.group, folds.fold))
    return np.array([m.get(int(x), -1) for x in _group_ids(np.asarray(rows), np.asarray(cols), size_px)])
