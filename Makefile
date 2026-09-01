# M2-LLM. Everything is config-driven; no notebook state.
PY ?= python3
# The HG-v2 targets run through uv (the venv is uv-managed and the repository
# is not pip-installed system-wide). Kept separate from PY so the M2-LLM
# targets and the container image keep working unchanged.
UVPY ?= uv run python
EXP ?= configs/experiments/phase0_infra.yaml
HG_V2_EXP ?= configs/experiments/hg_v2/phase0_index.yaml
HG_V2_P1_EXP ?= configs/experiments/hg_v2/phase1_router.yaml
HG_V2_P2_EXP ?= configs/experiments/hg_v2/phase2_frozen_lm.yaml
HG_V2_E1_EXP ?= configs/experiments/hg_v2/e1_frontier.yaml
HG_V2_E2_EXP ?= configs/experiments/hg_v2/e2_integrated_sim.yaml

.PHONY: help install install-uv install-gpu test test-uv phase0 endpoints analyze \
        hg-v2-phase0 hg-v2-phase1 hg-v2-phase2 hg-v2-e1 hg-v2-e2 hg-v2-analyze \
        clean-results

help:
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

install: ## Install the analysis pipeline (CPU only, no model weights)
	$(PY) -m pip install -e ".[dev,analysis]"

install-uv: ## Same, into a uv-managed .venv (what the HG-v2 targets use)
	uv venv --python 3.12 && uv pip install -e ".[dev,analysis]"

install-gpu: ## Add the DGX Spark inference + LoRA stack
	$(PY) -m pip install -r requirements-gpu.txt && $(PY) -m pip install -e ".[dev,analysis]"

test: ## Run the test suite
	$(PY) -m pytest -q

test-uv: ## Run the test suite in the uv-managed venv
	$(UVPY) -m pytest -q

phase0: ## Phase 0 infrastructure validation (protocol section 15 / section 19)
	$(PY) experiments/m2_llm/run_phase0.py --config $(EXP)

endpoints: ## Phase 1 endpoint comparison HS -> DD (protocol section 4)
	$(PY) experiments/m2_llm/run_endpoints.py \
	  --config configs/experiments/m2_llm_endpoints.yaml

analyze: ## Re-run analysis over saved prediction matrices, no model calls
	$(PY) experiments/m2_llm/analyze.py --run-dir $(RUN_DIR)

hg-v2-phase0: ## HG-v2 Phase 0: exact synthetic complementarity, gate G0 (no GPU)
	$(UVPY) experiments/hg_v2/run_phase0.py --config $(HG_V2_EXP)

hg-v2-phase1: ## HG-v2 Phase 1: learned act-vs-ask routing, gate G1 (no GPU)
	$(UVPY) experiments/hg_v2/run_phase1.py --config $(HG_V2_P1_EXP)

hg-v2-phase2: ## HG-v2 Phase 2: query compression, gate G2 (simulated backend by default)
	$(UVPY) experiments/hg_v2/run_phase2.py --config $(HG_V2_P2_EXP)

hg-v2-e1: ## HG-v2 E1: the bounded-query frontier, gate H0 (no GPU)
	$(UVPY) experiments/hg_v2/run_e1.py --config $(HG_V2_E1_EXP)

hg-v2-e2: ## HG-v2 E2/E3: integrated routing + query synthesis, gates H1/H2
	$(UVPY) experiments/hg_v2/run_e2.py --config $(HG_V2_E2_EXP)

hg-v2-analyze: ## Recompute HG-v2 metrics from saved episodes, no model calls
	$(UVPY) experiments/hg_v2/analyze.py --run-dir $(RUN_DIR)

clean-results: ## Delete generated artifacts, keep the directory layout
	find data results -type f ! -name '.gitkeep' -delete
