# Phase 1 rebuild from raw (the 2017 embedding is no longer part of it; the POC scripts need `python -m src.ingestion.download_stack emb2017`): grid check, feature stack (resumable), range checks, labels, test carve (once).
FDIR = data/spatial/mendeley_vrtx3w6mjd
RUN = set -a && . ./.env && set +a && uv run python -W ignore -m

.PHONY: data
data:
	$(RUN) src.data.check_grid $(FDIR)
	for L in emb2018 ndvi ndvi_prev; do $(RUN) src.ingestion.download_stack $$L; done
	$(RUN) src.data.check_stack
	$(RUN) src.data.labels $(FDIR)
	test -f data/interim/test_blocks.json || ( $(RUN) src.data.split )

# Phase 3: training feature table (97 features, one code path with full-area prediction), then the leakage and correctness checks.
.PHONY: features
features:
	$(RUN) src.features.build
	$(RUN) src.features.check

# Phase 4: object-metric and fold checks on synthetic scenes and the real split; the random-vs-block leak measurement.
.PHONY: validation
validation:
	$(RUN) src.data.polygon_ids $(FDIR)
	$(RUN) src.modeling.check_metrics
	$(RUN) src.modeling.leak_check
