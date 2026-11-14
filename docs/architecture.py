"""Source of docs/architecture.excalidraw and docs/architecture.svg. Edit the layout here, then run:

    python3 docs/architecture.py

Uses the architecture-diagram skill (excalidraw_lib, render_svg).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.join(HERE, "..", "..", "..", ".claude", "skills", "architecture-diagram", "scripts")
sys.path.insert(0, SKILL)
from excalidraw_lib import Diagram  # noqa: E402

d = Diagram(seed=7)
d.title(
    "Landslide inventory, Pacitan · architecture",
    "Samodra inventory + Earth Engine imagery → features → boosting → candidates for an expert",
)
d.lane(20, 0, 1470, 215, "BUILD · phases 1 to 3 (make data, make features)")
d.lane(20, 240, 1470, 150, "MODEL · phases 4 and 5 (flows right to left)")
d.lane(20, 420, 1470, 240, "DELIVER · phases 6 and 7")

# BUILD
d.box(50, 40, 200, 56, "service", "MENDELEY RELEASE", "inventory · site factors")
d.box(50, 130, 200, 56, "service", "EARTH ENGINE", "Sentinel-2 · Embedding 2018")
d.box(290, 40, 200, 56, "code", "LABELS + SPLIT", "shift · rings · test blocks")
d.box(290, 130, 200, 56, "code", "DOWNLOAD STACK", "tiled · resumable")
d.box(530, 85, 200, 56, "store", "INTERIM RASTERS", "labels · NDVI · embedding")
d.box(770, 85, 200, 56, "code", "FEATURES", "97 · one tile-wise path")
d.box(1010, 85, 200, 56, "store", "FEATURE TABLE", "16k positive · 400k negative")
d.arrow([(252, 68), (288, 68)])
d.arrow([(252, 158), (288, 158)])
d.arrow([(492, 68), (510, 68), (510, 113), (528, 113)])
d.arrow([(492, 158), (510, 158), (510, 113), (528, 113)])
d.arrow([(732, 113), (768, 113)])
d.arrow([(972, 113), (1008, 113)])

# MODEL (right to left)
d.box(1010, 305, 200, 56, "code", "TUNE", "boosting search · 6 km folds")
d.box(770, 305, 200, 56, "code", "OUT-OF-FOLD x5", "block CV · 5 repeats")
d.box(530, 305, 200, 56, "code", "EVALUATE", "80% recall · paired bootstrap")
d.box(290, 305, 200, 56, "code", "FINAL FIT", "all training rows")
d.box(50, 305, 200, 56, "ai", "c_hgb MODEL", "frozen threshold 0.1956")
d.arrow([(1110, 143), (1110, 303)], "training blocks only", at=(1118, 215))
d.arrow([(1008, 333), (972, 333)])
d.arrow([(768, 333), (732, 333)])
d.arrow([(528, 333), (492, 333)])
d.arrow([(288, 333), (252, 333)])

# DELIVER
d.box(50, 480, 200, 56, "code", "EVALUATE TEST", "34 held-out blocks, once")
d.box(290, 480, 200, 56, "store", "TEST RESULTS", "recall · precision · intervals")
d.box(530, 480, 200, 56, "code", "REPORT", "charts · tables · limits")
d.box(770, 480, 200, 56, "output", "modeling.html", "checked in a browser")
d.box(290, 575, 200, 56, "code", "CANDIDATES", "out-of-fold + final scores")
d.box(530, 575, 200, 56, "output", "candidates.gpkg", "ranked · two frames")
d.person(830, 567, "Expert")
d.arrow([(225, 363), (225, 478)], "frozen point", at=(233, 420))
d.arrow([(252, 508), (288, 508)])
d.arrow([(492, 508), (528, 508)])
d.arrow([(732, 508), (768, 508)])
d.arrow([(48, 333), (6, 333), (6, 603), (288, 603)], "model", at=(70, 585))
d.arrow([(492, 603), (528, 603)])
d.arrow([(732, 603), (828, 603)], "review", at=(738, 585))

scene = os.path.join(HERE, "architecture.excalidraw")
d.save(scene)
sys.exit(
    subprocess.run(
        [sys.executable, os.path.join(SKILL, "render_svg.py"), scene, os.path.join(HERE, "architecture.svg")]
        + sys.argv[1:]
    ).returncode
)
