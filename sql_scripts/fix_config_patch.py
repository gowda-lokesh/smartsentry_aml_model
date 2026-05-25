#!/usr/bin/env python3
import json
from pathlib import Path

from patch_notebooks_use_config import INF_SETUP, set_cell_source

ROOT = Path(__file__).resolve().parents[1]

# Fix 01
p = ROOT / "python_scripts/01__aml_typology_detector.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "CREATOR_PARAMS = {" in src and "build_detect_config" not in src:
        nb["cells"].pop(i)
        print("Removed CREATOR_PARAMS cell")
        break
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

# Fix 00b
p2 = ROOT / "python_scripts/00b__aml_inference_orchestrator.ipynb"
nb2 = json.loads(p2.read_text(encoding="utf-8"))
for i, cell in enumerate(nb2["cells"]):
    if cell.get("cell_type") == "code" and "NOTEBOOK_DIR = os.getcwd()" in "".join(cell.get("source", [])):
        set_cell_source(
            nb2,
            i,
            INF_SETUP
            + """
required_bundles = [
    os.path.join(PHASE1_DIR, "phase1_model_bundle.joblib"),
    os.path.join(PHASE2_DIR, "phase2_model_bundle.joblib"),
]
print("=" * 70)
print("SmartSentry AML — Inference Orchestrator")
print("=" * 70)
print(f"  Input file:        {INPUT_FILE}")
print(f"  Phase 1 bundle:    {required_bundles[0]}")
print(f"  Phase 2 bundle:    {required_bundles[1]}")
print(f"  Working directory: {NOTEBOOK_DIR}")
print(f"  Output directory:  {OUTPUT_DIR}")
for b in required_bundles:
    if not os.path.exists(b):
        raise FileNotFoundError(f"Required model bundle missing: {b}")
    print(f"  ✓ {os.path.basename(b)}: {os.path.getsize(b)/(1024*1024):.2f} MB")
print()
""",
        )
        print("Patched 00b setup")
        break
p2.write_text(json.dumps(nb2, indent=1, ensure_ascii=False), encoding="utf-8")
