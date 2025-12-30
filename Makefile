.PHONY: smoke audit preprocess train eval viz paper clean

PY ?= python
RUN = $(PY) scripts/run.py

SMOKE_CFG ?= configs/smoke.yaml
DEFAULT_CFG ?= configs/default.yaml
FULL_CFG ?= configs/full.yaml
PREPROCESS_CFG ?= $(FULL_CFG)

smoke:
	$(RUN) smoke --config $(SMOKE_CFG)

audit:
	$(RUN) audit --config $(DEFAULT_CFG)

preprocess:
	$(RUN) preprocess --config $(PREPROCESS_CFG)

train:
	$(RUN) train --config $(FULL_CFG)

eval:
	$(RUN) eval --config $(FULL_CFG)

viz:
	$(RUN) viz --config $(FULL_CFG)

paper:
	$(RUN) paper

clean:
	$(RUN) clean --yes
