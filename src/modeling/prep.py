"""Phase 5: feature sets, design matrices and model factories shared by tuning, out-of-fold prediction and evaluation.

Feature sets (groups from src/features/registry.py): site = terrain + site_cat; a = dNDVI + factors; b = 2018 embedding + factors;
c = everything. Linear models get one-hot land use / geology and training-median fills; boosting gets raw codes (categorical) and NaN.
"""

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cv_utils import SEED
from src.features.build import INTERIM
from src.features.registry import CATEGORICAL, GROUPS

SETS = {
    "site": GROUPS["terrain"] + GROUPS["site_cat"],
    "a": GROUPS["ndvi_pixel"] + GROUPS["ndvi_nbr"] + GROUPS["terrain"] + GROUPS["site_cat"] + GROUPS["flags"],
    "b": GROUPS["emb"] + GROUPS["terrain"] + GROUPS["site_cat"] + GROUPS["flags"],
    "c": GROUPS["ndvi_pixel"]
    + GROUPS["ndvi_nbr"]
    + GROUPS["terrain"]
    + GROUPS["site_cat"]
    + GROUPS["flags"]
    + GROUPS["emb"],
}
# (model kind, feature set); "simple_logreg" is the best simple baseline of docs/validation-design.md
CONFIGS = {
    "floor_logreg": ("logreg", "site"),
    "simple_logreg": ("logreg", "a"),
    "emb_logreg": ("logreg", "b"),
    "all_logreg": ("logreg", "c"),
    "a_hgb": ("hgb", "a"),
    "b_hgb": ("hgb", "b"),
    "c_hgb": ("hgb", "c"),
}
LEVELS = {"geologi": list(range(1, 34)), "landuse": list(range(1, 14))}
# monotone priors from the EDA: steeper and more vegetation loss / bare ground means more landslide
MONO = {
    "slope": 1,
    **{f"{b}_{s}{w}": -1 for b in ("dB", "postB") for s in ("mean", "min") for w in (3, 5)},
    "dB": -1,
    "postB": -1,
    "dB_anom11": -1,
    "postB_anom11": -1,
}
FILLS = json.load(open(f"{INTERIM}/fill_values.json"))


def design(df, cols, kind):
    """Numpy design matrix for a model kind. df has the registry columns (NaN where missing)."""
    num = [c for c in cols if c not in CATEGORICAL]
    cat = [c for c in cols if c in CATEGORICAL]
    if kind == "hgb":
        return df[num + cat].to_numpy(dtype="float32")
    X = df[num].copy()
    for c, v in FILLS.items():
        if c in X:
            X[c] = X[c].fillna(v)
    parts = [X.to_numpy(dtype="float32")]
    for c in cat:
        v = df[c].to_numpy()
        parts.append(np.stack([(v == lv) for lv in LEVELS[c]], axis=1).astype("float32"))
    return np.concatenate(parts, axis=1)


def make_model(kind, params, cols):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=300))
    num = [c for c in cols if c not in CATEGORICAL]
    names = num + [c for c in cols if c in CATEGORICAL]
    return HistGradientBoostingClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        monotonic_cst=[MONO.get(n, 0) for n in names],
        categorical_features=[n in CATEGORICAL for n in names],
        early_stopping=False,
        random_state=SEED,
    )


def default_params(kind):
    return (
        {"C": 0.1}
        if kind == "logreg"
        else {"max_depth": 4, "learning_rate": 0.06, "max_iter": 300, "min_samples_leaf": 200, "l2_regularization": 1.0}
    )


def load_params():
    try:
        return json.load(open(f"{INTERIM}/tuned_params.json"))
    except FileNotFoundError:
        return {k: default_params(v[0]) for k, v in CONFIGS.items()}
