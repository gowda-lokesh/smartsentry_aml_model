#!/usr/bin/env python3
"""Patch python_scripts notebooks to load configuration from config/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "python_scripts"

SETUP_SNIPPET = '''import os
import sys

from config.loader import ensure_notebook_path, get_run_mode, get_artifact_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_MODE = get_run_mode(_SETTINGS)
'''

ORCH_SETUP = '''import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from config.loader import (
    ensure_notebook_path,
    build_training_pipeline,
    get_pipeline_mode,
)

_SETTINGS, _PATHS = ensure_notebook_path()
NOTEBOOK_DIR = str(_PATHS["notebook_dir"])
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

_orch = _SETTINGS["orchestrator"]
TIMEOUT_MINUTES = _orch["timeout_minutes"]
STOP_ON_FAILURE = _orch["stop_on_failure"]
SAVE_EXECUTED_NOTEBOOKS = _orch["save_executed_notebooks"]
NBCONVERT_TIMEOUT = _orch.get("nbconvert_timeout_seconds", 14400)
KERNEL_NAME = _orch.get("kernel_name", "python3")

PIPELINE = build_training_pipeline(_SETTINGS, _PATHS)
'''

INF_SETUP = '''import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from config.loader import (
    ensure_notebook_path,
    build_inference_pipeline,
)

_SETTINGS, _PATHS = ensure_notebook_path()
NOTEBOOK_DIR = str(_PATHS["notebook_dir"])
OUTPUT_DIR = str(_PATHS["outputs_dir"])
PHASE1_DIR = str(_PATHS["phase1_dir"])
PHASE2_DIR = str(_PATHS["phase2_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

_inf = _SETTINGS["inference"]
TIMEOUT_MINUTES = _inf["timeout_minutes"]
STOP_ON_FAILURE = _inf["stop_on_failure"]
SAVE_EXECUTED_NOTEBOOKS = _inf["save_executed_notebooks"]

DEFAULT_INPUT = os.environ.get(
    "AML_INFERENCE_INPUT",
    str(_PATHS["outputs_dir"] / Path(_inf["default_input"]).name),
)
INPUT_FILE = os.environ.get("AML_INFERENCE_INPUT", DEFAULT_INPUT)
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Inference input file not found: {INPUT_FILE}\\n"
        f"Set AML_INFERENCE_INPUT or place file at {DEFAULT_INPUT}"
    )

INFERENCE_PIPELINE = build_inference_pipeline(_SETTINGS, _PATHS, INPUT_FILE)
'''

GEN_OUTPUT_CELL = '''import os
from config.loader import ensure_notebook_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ["ml_outputs", "phase2_outputs", "executed_notebooks"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
print(f"Output directory ready: {os.path.abspath(OUTPUT_DIR)}")
'''

GEN_IMPORTS_APPEND = '''
from config.loader import load_generator_config, ensure_notebook_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = load_generator_config(_SETTINGS)
MASTER_SEED = CONFIG.get("master_seed", _SETTINGS.get("generator", {}).get("master_seed", 42))
random.seed(MASTER_SEED)
print(f"CONFIG loaded from {_PATHS['generator_config_json']}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"Master seed: {MASTER_SEED}")
'''

DETECTOR_CREATOR = '''from config.loader import (
    ensure_notebook_path,
    load_generator_config,
    build_detect_config,
)

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

_generator = load_generator_config(_SETTINGS)
CREATOR_PARAMS = _generator["typology_generation"]
DETECT_CONFIG = build_detect_config(CREATOR_PARAMS)

print("DETECT_CONFIG v7 (from config) loaded:")
for typ, params in DETECT_CONFIG.items():
    print(f"  {typ}: {len(params)} params")
'''


def set_cell_source(nb: dict, cell_idx: int, source: str) -> None:
    nb["cells"][cell_idx]["source"] = [line + "\n" for line in source.strip().split("\n")]


def find_cell_starting_with(nb: dict, prefix: str) -> int | None:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.strip().startswith(prefix):
            return i
    return None


def remove_config_dict_cell(nb: dict) -> None:
    """Remove aml_generator CONFIG = { ... } cell."""
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.strip().startswith("CONFIG = {"):
            nb["cells"].pop(i)
            return


def patch_notebook(path: Path) -> list[str]:
    changes: list[str] = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    name = path.name

    if name == "aml_generator_complete_pipeline.ipynb":
        idx = find_cell_starting_with(nb, "import os")
        if idx is not None and "OUTPUT_DIR = os.path.join" in "".join(nb["cells"][idx]["source"]):
            set_cell_source(nb, idx, GEN_OUTPUT_CELL)
            changes.append("generator output dir cell")
        idx2 = find_cell_starting_with(nb, "import random")
        if idx2 is not None:
            src = "".join(nb["cells"][idx2]["source"])
            if "CONFIG = {" not in src:
                # trim duplicate OUTPUT_DIR lines, append config load
                lines = [l for l in src.split("\n") if "OUTPUT_DIR" not in l and "Master seed" not in l]
                new_src = "\n".join(lines).rstrip() + GEN_IMPORTS_APPEND
                set_cell_source(nb, idx2, new_src)
                changes.append("generator imports+CONFIG")
        remove_config_dict_cell(nb)
        changes.append("removed inline CONFIG dict")

    elif name == "01__aml_typology_detector.ipynb":
        idx = find_cell_starting_with(nb, "import pandas")
        if idx is not None:
            base = SETUP_SNIPPET + 'print("Environment ready")\n'
            set_cell_source(nb, idx, base)
            changes.append("setup")
        # Remove CREATOR_PARAMS dict cell
        for i, cell in enumerate(list(nb["cells"])):
            src = "".join(cell.get("source", []))
            if "CREATOR_PARAMS = {" in src and "DETECT_CONFIG" not in src:
                nb["cells"].pop(i)
                changes.append("removed CREATOR_PARAMS dict")
                break
        # Replace DETECT_CONFIG cell
        idx_d = find_cell_starting_with(nb, "_s  = CREATOR_PARAMS")
        if idx_d is None:
            idx_d = find_cell_starting_with(nb, "DETECT_CONFIG = {")
        if idx_d is not None:
            set_cell_source(nb, idx_d, DETECTOR_CREATOR)
            changes.append("DETECT_CONFIG from config")

    elif name in ("02__aml_rules_engine.ipynb", "03__aml_feature_engineering.ipynb"):
        idx = find_cell_starting_with(nb, "import pandas")
        if idx is not None:
            extra = ""
            if name.startswith("03"):
                extra = '''
from config.loader import get_artifact_path
_default_rules = str(get_artifact_path(_PATHS, "flagged", _SETTINGS))
if RUN_MODE == "predict":
    INPUT_FILE = os.environ.get("AML_INPUT_FILE", str(_PATHS["outputs_dir"] / "inference_input.parquet"))
else:
    INPUT_FILE = os.environ.get("AML_INPUT_FILE", _default_rules)
'''
            elif name.startswith("02"):
                extra = '''
from config.loader import get_artifact_path
if RUN_MODE == "predict":
    _default_input = str(_PATHS["outputs_dir"] / "inference_input.parquet")
else:
    _default_input = str(get_artifact_path(_PATHS, "flagged", _SETTINGS))
INPUT_FILE = os.environ.get("AML_INPUT_FILE", _default_input)
'''
            set_cell_source(nb, idx, SETUP_SNIPPET + extra + 'print("Environment ready")\n')
            changes.append("setup+input")
        # Remove duplicate input resolution cells
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if "Input file resolution" in src and i > 2:
                nb["cells"].pop(i)
                changes.append("removed duplicate input cell")
                break

    elif name == "04__aml_ml_preparation.ipynb":
        idx = find_cell_starting_with(nb, "import pandas")
        if idx is not None:
            p1 = '''
PHASE1_OUTPUT_DIR = str(_PATHS["phase1_dir"])
OUTPUT_DIR = os.environ.get("AML_PHASE1_DIR", PHASE1_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_COLUMN = _SETTINGS["phase1"]["target_column"]
print(f"Phase 1 ML — RUN_MODE = {RUN_MODE.upper()}")
print(f"Output dir: {OUTPUT_DIR}")
'''
            set_cell_source(nb, idx, SETUP_SNIPPET.replace("RUN_MODE = get_run_mode", "RUN_MODE = get_run_mode") + p1)
            changes.append("setup phase1")
        # Remove second RUN_MODE cell if exists
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if "AML_RUN_MODE" in src and "RUN_MODE = os.environ" in src and i > 1:
                nb["cells"].pop(i)
                changes.append("removed duplicate RUN_MODE cell")
                break
        # Update predict input
        for cell in nb["cells"]:
            src = cell.get("source", [])
            text = "".join(src)
            if "AML_INPUT_FILE" in text and "_default_input" in text:
                new = text.replace(
                    '_default_input = os.path.join(os.path.dirname(os.getcwd()), "outputs_updated", "stg_transactions_features.parquet")',
                    'from config.loader import get_artifact_path\n    _default_input = str(get_artifact_path(_PATHS, "features", _SETTINGS))',
                )
                cell["source"] = [l + "\n" for l in new.split("\n")]

    elif name == "05__aml_phase2_typology_classifier.ipynb":
        idx = find_cell_starting_with(nb, "import pandas")
        if idx is not None:
            p2 = '''
PHASE1_DIR = str(_PATHS["phase1_dir"])
OUTPUT_DIR = os.environ.get("AML_PHASE2_DIR", str(_PATHS["phase2_dir"]))
os.makedirs(OUTPUT_DIR, exist_ok=True)
TYPOLOGY_THRESHOLD = float(_SETTINGS["phase2"]["typology_threshold"])
P2_TUNING = _SETTINGS["phase2"]["hyperparameter_tuning"]
print("Libraries loaded")
'''
            set_cell_source(nb, idx, SETUP_SNIPPET + p2)
            changes.append("setup phase2")
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if "RUN_MODE = os.environ.get(\"AML_RUN_MODE\"" in src:
                nb["cells"].pop(i)
                changes.append("removed RUN_MODE cell")
                break
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if "PHASE1_DIR = os.environ.get" in src and "AML_PHASE1_DIR" in src:
                nb["cells"].pop(i)
                changes.append("removed paths cell")
                break
        for cell in nb["cells"]:
            text = "".join(cell.get("source", []))
            if text.strip() == "TYPOLOGY_THRESHOLD = 0.30":
                cell["source"] = ["# TYPOLOGY_THRESHOLD set in setup from config/settings.yaml\n"]

    elif name == "00__aml_pipeline_orchestrator.ipynb":
        idx = find_cell_starting_with(nb, "import os")
        if idx is not None and "PIPELINE = [" in "".join(nb["cells"][idx]["source"]):
            set_cell_source(nb, idx, ORCH_SETUP + '''
print("=" * 70)
print("SmartSentry AML — Pipeline Orchestrator")
print("=" * 70)
print(f"  Notebook directory:  {NOTEBOOK_DIR}")
print(f"  Output directory:    {OUTPUT_DIR}")
print(f"  Timeout per module:  {TIMEOUT_MINUTES} minutes")
print(f"  Stop on failure:     {STOP_ON_FAILURE}")
print(f"  Modules to run:      {len(PIPELINE)}")
''')
            changes.append("orchestrator setup")

    elif name == "00b__aml_inference_orchestrator.ipynb":
        idx = find_cell_starting_with(nb, "NOTEBOOK_DIR = os.getcwd()")
        if idx is not None:
            set_cell_source(nb, idx, INF_SETUP + '''
print("=" * 70)
print("SmartSentry AML — Inference Orchestrator")
print("=" * 70)
print(f"  Input file:        {INPUT_FILE}")
print(f"  Phase 1 bundle:    {os.path.join(PHASE1_DIR, 'phase1_model_bundle.joblib')}")
print(f"  Phase 2 bundle:    {os.path.join(PHASE2_DIR, 'phase2_model_bundle.joblib')}")
print(f"  Working directory: {NOTEBOOK_DIR}")
print(f"  Output directory:  {OUTPUT_DIR}")
''')
            changes.append("inference setup")
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if "INFERENCE_PIPELINE = [" in src:
                nb["cells"].pop(i)
                changes.append("removed inline INFERENCE_PIPELINE")
                break

    if changes:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return changes


def main() -> None:
    for nb_path in sorted(NB_DIR.glob("*.ipynb")):
        ch = patch_notebook(nb_path)
        if ch:
            print(f"{nb_path.name}: {', '.join(ch)}")


if __name__ == "__main__":
    main()
