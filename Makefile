PYTHON ?= python
RUNTIME ?= $(CURDIR)/runtime
MIIC_BUNDLE ?=

.PHONY: setup verify test lint development development-manual final postlock status

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/gpu-cu126.lock
	$(PYTHON) -m pip install -e .

verify:
	$(PYTHON) scripts/verify_environment.py --require-gpu

test:
	MPLBACKEND=Agg PYTHONPATH=src $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src scripts tests

development:
	$(PYTHON) scripts/run_native_pipeline.py --stage development --runtime-root $(RUNTIME) --accept-miic-license

development-manual:
	@test -n "$(MIIC_BUNDLE)" || (echo "Set MIIC_BUNDLE=/path/to/MIIC_verified_files"; exit 2)
	$(PYTHON) scripts/run_native_pipeline.py --stage development --runtime-root $(RUNTIME) --accept-miic-license --miic-manual-bundle $(MIIC_BUNDLE)

final:
	$(PYTHON) scripts/run_native_pipeline.py --stage final --runtime-root $(RUNTIME)

postlock:
	$(PYTHON) scripts/run_native_pipeline.py --stage postlock --runtime-root $(RUNTIME)

status:
	$(PYTHON) scripts/run_native_pipeline.py --stage status --runtime-root $(RUNTIME)
