# Configuration

All pipeline notebooks load settings from this folder via `config/loader.py`.

| File | Used by | Purpose |
|------|---------|---------|
| `settings.yaml` | All notebooks | Paths, orchestrator, inference chain, Phase 1/2 parameters |
| `generator_config.json` | `aml_generator_complete_pipeline.ipynb`, `01__aml_typology_detector.ipynb` | Synthetic data + typology generation (`CONFIG` / `CREATOR_PARAMS`) |
| `detector.yaml` | Reference | Static detector notes; live `DETECT_CONFIG` is built in `loader.build_detect_config()` |
| `base_config.ipynb` | Optional alternate generator | Legacy design-doc parameters |

## Edit workflow

1. Change paths or timeouts in **`settings.yaml`**
2. Change fraud rates, typology weights, or generation params in **`generator_config.json`**
3. Re-run notebooks from `python_scripts/` (or set env vars from `.env.example`)

## Bootstrap in notebooks

```python
from config.loader import ensure_notebook_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
```

## Environment overrides

| Variable | Effect |
|----------|--------|
| `AML_PIPELINE_MODE` | `train` / `predict` for orchestrator |
| `AML_RUN_MODE` | `train` / `predict` for ML notebooks |
| `AML_PHASE1_DIR` | Phase 1 model output directory |
| `AML_PHASE2_DIR` | Phase 2 model output directory |
| `AML_INPUT_FILE` | Input parquet for rules/features/ML in inference |
| `AML_INFERENCE_INPUT` | Raw input for inference orchestrator |
| `AML_GENERATOR_CONFIG` | Override path to generator JSON |
