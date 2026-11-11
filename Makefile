# Phase 1 rebuild from raw: grid check, feature stack (resumable), range checks, labels, test carve (once).
FDIR = data/spatial/mendeley_vrtx3w6mjd
RUN = set -a && . ./.env && set +a && uv run python -W ignore -m

.PHONY: data
data:
	$(RUN) src.data.check_grid $(FDIR)
	for L in emb2018 emb2017 ndvi; do $(RUN) src.ingestion.download_stack $$L; done
	$(RUN) src.data.check_stack
	$(RUN) src.data.labels $(FDIR)
	test -f data/interim/test_blocks.json || ( $(RUN) src.data.split )
