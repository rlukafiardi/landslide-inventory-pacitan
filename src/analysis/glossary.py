"""Plain-language definitions of the terms the reports and the deck use. One home: reports/modeling.html, reports/eda.html and
docs/slidepack/build_deck.py all read these lists, so a definition is corrected once. No numbers from results belong here."""

METRICS = [
    (
        "Scar, polygon",
        "A mapped landslide outline in the inventory. The targets count scars of 500 m2 or more (5 pixels or more at 10 m).",
    ),
    (
        "Flagged object",
        "A connected group of at least 3 flagged pixels. A pixel is flagged when the model's score passes the threshold. Objects are what an expert reviews.",
    ),
    (
        "Object recall",
        "Of the scars of 500 m2 or more, the share that at least one flagged object touches. In plain words: out of every 100 real scars, how many did the model find?",
    ),
    (
        "Object precision",
        "Of the flagged objects, the share that touch a mapped scar. In plain words: out of every 100 flags, how many are real? It is a floor, because a flag on a real scar that nobody mapped counts as wrong.",
    ),
    (
        "Threshold, trade-off",
        "A lower score threshold flags more objects: recall goes up and precision goes down. The threshold (the operating point) was fixed on training blocks before the test.",
    ),
    (
        "Strict recall (IoU)",
        "Recall that needs real overlap, not just a touch. IoU (intersection over union) is the overlap area divided by the combined area of object and scar; here at least 0.3.",
    ),
    (
        "False objects per true scar",
        "Flagged objects that touch no scar, divided by the objects that touch one: how many flags an expert clears for each scar found.",
    ),
]

METHOD = [
    (
        "Baseline",
        "A simple method the model has to beat, such as a threshold on vegetation loss or a plain logistic regression.",
    ),
    (
        "Held-out test blocks",
        "3 km squares kept out of all training and tuning, and scored once at the end with the frozen model.",
    ),
    (
        "Block cross-validation",
        "Train on some 3 km blocks, check on the others, rotate and repeat. Whole blocks are held out so that neighbouring pixels cannot leak the answer.",
    ),
    (
        "95% interval",
        "The range the number could plausibly take given a small test set; here from resampling whole blocks (block bootstrap).",
    ),
    (
        "AUC",
        "The chance that a random scar pixel scores higher than a random non-scar pixel: 0.5 is a coin flip, 1.0 is perfect. It hides how many false alarms a rare event produces, so models are compared at a fixed recall instead.",
    ),
    ("Ablation", "Remove one group of features, refit and re-test, to see whether the group was needed."),
    (
        "Boosting, logistic regression",
        "Boosting adds many small decision trees, each correcting the errors of the last. Logistic regression is a linear model that turns weighted features into a probability.",
    ),
]

DATA = [
    (
        "NDVI",
        "A vegetation index from the satellite's red and near-infrared light: high is dense green, low is bare ground. dB is its change from before the storm to April-May 2018; a large drop means vegetation was lost.",
    ),
    (
        "Satellite Embedding",
        "A 64-number summary per pixel, produced by a Google model from what the satellite saw during 2018. The individual numbers have no single physical meaning.",
    ),
]

EDA = [
    ("Prevalence", "The share of pixels that are scars."),
    (
        "Pixel precision",
        "Of the pixels a detector flags, the share that fall inside a scar (the object versions above count connected groups instead).",
    ),
    (
        "VIF",
        "Variance inflation factor: how well a feature can be predicted from the others. Above about 10 it adds little new information and makes linear models unstable.",
    ),
    (
        "Moran's I",
        "A measure of spatial clustering from -1 to 1: near 0, values are scattered at random; positive, nearby pixels have similar values.",
    ),
]

DECK = [("Hillshade", "A shaded-relief image computed from the elevation model, used as a grey base for maps.")]
