# M2-LLM. Everything is config-driven; no notebook state.
PY ?= python3
EXP ?= configs/experiments/phase0_infra.yaml

.PHONY: help install install-gpu test phase0 endpoints analyze clean-results

help:
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

install: ## Install the analysis pipeline (CPU only, no model weights)
	$(PY) -m pip install -e ".[dev,analysis]"

install-gpu: ## Add the DGX Spark inference + LoRA stack
	$(PY) -m pip install -r requirements-gpu.txt && $(PY) -m pip install -e ".[dev,analysis]"

test: ## Run the test suite
	$(PY) -m pytest -q

phase0: ## Phase 0 infrastructure validation (protocol section 15 / section 19)
	$(PY) experiments/m2_llm/run_phase0.py --config $(EXP)

endpoints: ## Phase 1 endpoint comparison HS -> DD (protocol section 4)
	$(PY) experiments/m2_llm/run_endpoints.py \
	  --config configs/experiments/m2_llm_endpoints.yaml

analyze: ## Re-run analysis over saved prediction matrices, no model calls
	$(PY) experiments/m2_llm/analyze.py --run-dir $(RUN_DIR)

clean-results: ## Delete generated artifacts, keep the directory layout
	find data results -type f ! -name '.gitkeep' -delete
