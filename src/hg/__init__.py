"""M2-LLM: testing whether clean shared error survives strong internal model diversity.

Subsystems mirror the protocol's repository layout (section 14):

    hg.inference   run a model family over a dataset, deterministically
    hg.finetune    LoRA adaptation with a common effective token budget
    hg.anchors     human panel judgments -> reference label + anchor status
    hg.diversity   train-split construction, overlap O_jk, residual survival
    hg.metrics     C_E, C_E^clean, D_E, Q_contested, dependence, bootstrap
    hg.routing     localization strategies for the targeting race
    hg.report      decision-gate rendering and figures

The design rule throughout: item-level predictions are the artifact worth
keeping, so every stage reads and writes the schema in `hg.schema` and no
metric is computed anywhere the raw rows are unavailable.
"""

__version__ = "0.1.0"
