#!/usr/bin/env python
# coding: utf-8

# # SmartSentry AML — Training Pipeline Orchestrator
# 
# **Training-only.** Runs the full pipeline: synthetic data generation → typology detector → rules engine → feature engineering → Phase 1 model training → Phase 2 model training.
# 
# Do not run this notebook directly to choose a mode — use **`00_router.ipynb`**, which asks yes/no and dispatches to either this notebook or the inference orchestrator (`00b`).
# 

# ## 1 — Setup & Configuration
# 

# In[ ]:

import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Update these paths to match your environment
# ═══════════════════════════════════════════════════════════════

# Directory containing all notebooks/scripts
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
print(NOTEBOOK_DIR)

# Output directory (parent-level outputs_updated)
OUTPUT_DIR = os.path.join(os.path.dirname(NOTEBOOK_DIR), "outputs_updated")

# Pipeline python scripts in execution order
PIPELINE = [
    {
        "id": "00",
        "name": "Data Generator",
        "notebook": "aml_generator_complete_pipeline.py",
        "outputs": ["transactions_generated_typology.parquet"],
        "description": "Generate synthetic transactions with 10 AML typologies",
    },
    {
        "id": "01",
        "name": "Typology Detector",
        "notebook": "01__aml_typology_detector.py",
        "outputs": ["stg_transactions_flagged.parquet"],
        "description": "Graph-based detection of AML patterns, assign is_aml labels",
    },
    {
        "id": "02",
        "name": "Rules Engine",
        "notebook": "02__aml_rules_engine.py",
        "outputs": ["stg_transactions_rules.parquet"],
        "description": "Apply 126 regulatory compliance rules (RBI/PMLA/FIU-IND)",
    },
    {
        "id": "03",
        "name": "Feature Engineering",
        "notebook": "03__aml_feature_engineering.py",
        "outputs": ["stg_transactions_features.parquet"],
        "description": "Compute velocity, balance, IP risk, and volume features",
    },
    {
        "id": "04",
        "name": "Phase 1: AML Detection",
        "notebook": "04__aml_ml_preparation.py",
        "outputs": [
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "final_lgb_model.txt"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "phase1_model_bundle.joblib"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "model_metadata.json"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "X_train.parquet"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "X_test.parquet"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "y_train.parquet"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs", "y_test.parquet"),
        ],
        "description": "Train binary AML classifier with hyperparameter tuning",
    },
    {
        "id": "05",
        "name": "Phase 2: Typology Classifier",
        "notebook": "05__aml_phase2_typology_classifier.py",
        "outputs": [
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "phase2_outputs", "phase2_typology_model.txt"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "phase2_outputs", "phase2_model_bundle.joblib"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "phase2_outputs", "combined_aml_output.parquet"),
            os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "phase2_outputs", "model_parameters_full.json"),
        ],
        "description": "Train 10-class typology classifier with multi-label threshold",
    },
]

# Execution settings
TIMEOUT_MINUTES = 60
STOP_ON_FAILURE = True
SAVE_EXECUTED_NOTEBOOKS = True

print("=" * 70)
print("SmartSentry AML — Pipeline Orchestrator")
print("=" * 70)
print(f"  Notebook directory:  {NOTEBOOK_DIR}")
print(f"  Output directory:    {OUTPUT_DIR}")
print(f"  Timeout per module:  {TIMEOUT_MINUTES} minutes")
print(f"  Stop on failure:     {STOP_ON_FAILURE}")
print(f"  Modules to run:      {len(PIPELINE)}")
print()

# Verify all scripts exist
all_found = True
for stage in PIPELINE:
    nb_path = os.path.join(NOTEBOOK_DIR, stage["notebook"])
    exists = os.path.exists(nb_path)
    status = "✓" if exists else "⚠ NOT FOUND"
    print(f"  [{stage['id']}] {stage['notebook']:<55s} {status}")
    if not exists:
        all_found = False

if not all_found:
    raise FileNotFoundError("One or more python scripts are missing. Fix paths above.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n  ✓ All scripts found. Ready to execute.")


# In[8]:

# Add this in Cell 2 of the orchestrator, right after OUTPUT_DIR is defined:
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ["executed_notebooks"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


# In[3]:

# Shared run id — every script in this run reads it via AML_RUN_ID
import datetime as _dt
RUN_ID = os.environ.get("AML_RUN_ID", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.environ["AML_RUN_ID"] = RUN_ID
PIPELINE_MODE = "train"
print(f"Run ID: {RUN_ID}")


# ## 2 — Script Runner Engine

# In[4]:

def run_notebook(notebook_path, timeout_minutes=60, working_dir=None):
    """
    Execute a Python script and return status.
    The script runs in its own isolated process.
    """
    start_time = time.time()
    notebook_name = os.path.basename(notebook_path)

    # Output path for execution logs
    executed_dir = os.path.join(OUTPUT_DIR, "executed_notebooks")
    os.makedirs(executed_dir, exist_ok=True)

    log_path = os.path.join(
        executed_dir,
        notebook_name.replace(".py", "_execution.log")
    )

    print(f"  Executing: {notebook_name}")
    print(f"  Working dir: {working_dir or os.path.dirname(notebook_path)}")

    try:
        cmd = [
            sys.executable,
            notebook_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60 + 60,
            cwd=working_dir or os.path.dirname(notebook_path),
        )

        elapsed = time.time() - start_time

        # Save stdout/stderr logs
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("STDOUT\n")
            f.write("=" * 80 + "\n")
            f.write(result.stdout or "")
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("STDERR\n")
            f.write("=" * 80 + "\n")
            f.write(result.stderr or "")

        if result.returncode == 0:
            return {
                "status": "SUCCESS",
                "elapsed_seconds": elapsed,
                "elapsed_str": str(timedelta(seconds=int(elapsed))),
                "executed_path": log_path if SAVE_EXECUTED_NOTEBOOKS else None,
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": "",
            }
        else:
            error_msg = result.stderr[-1000:] if result.stderr else "Unknown error"
            return {
                "status": "FAILED",
                "elapsed_seconds": elapsed,
                "elapsed_str": str(timedelta(seconds=int(elapsed))),
                "error": error_msg,
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else "",
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "status": "TIMEOUT",
            "elapsed_seconds": elapsed,
            "elapsed_str": str(timedelta(seconds=int(elapsed))),
            "error": f"Script exceeded {timeout_minutes} minute timeout",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "ERROR",
            "elapsed_seconds": elapsed,
            "elapsed_str": str(timedelta(seconds=int(elapsed))),
            "error": str(e),
        }


def validate_outputs(stage, output_dir):
    """Check that expected output files were created by a stage."""

    results = []

    for output_file in stage["outputs"]:
        full_path = output_file if os.path.isabs(output_file) else os.path.join(output_dir, output_file)

        if os.path.isabs(output_file):
            display = os.path.relpath(output_file, os.path.dirname(output_dir))
        else:
            display = output_file

        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            results.append({"file": display, "status": "✓", "size_mb": size_mb})
        else:
            results.append({"file": display, "status": "⚠ MISSING", "size_mb": 0})

    return results


print("Runner engine loaded.")
print("  run_notebook()     — executes a python script")
print("  validate_outputs() — checks expected output files exist")


# ## 3 — Execute Full Pipeline

# In[9]:

print(f"{'=' * 75}")
print("PIPELINE EXECUTION STARTED")
print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

pipeline_start = time.time()
pipeline_results = []
pipeline_failed = False

for i, stage in enumerate(PIPELINE):

    print(f"\n{'─' * 70}")
    print(f"  STAGE [{stage['id']}] {stage['name']}")
    print(f"  {stage['description']}")
    print(f"{'─' * 70}")

    if pipeline_failed and STOP_ON_FAILURE:
        print(f"  ⊘ SKIPPED (previous stage failed)")
        pipeline_results.append({
            "stage": stage["id"],
            "name": stage["name"],
            "status": "SKIPPED",
            "elapsed_str": "—",
            "elapsed_seconds": 0,
        })
        continue

    # Execute script
    nb_path = os.path.join(NOTEBOOK_DIR, stage["notebook"])

    result = run_notebook(
        nb_path,
        timeout_minutes=TIMEOUT_MINUTES,
        working_dir=NOTEBOOK_DIR
    )

    # Status indicator
    status_icon = {
        "SUCCESS": "✓",
        "FAILED": "✗",
        "TIMEOUT": "⏱",
        "ERROR": "⚠"
    }.get(result["status"], "?")

    print(f"\n  {status_icon} Status: {result['status']} ({result['elapsed_str']})")

    if result["status"] != "SUCCESS":
        print(f"  Error: {result.get('error', 'Unknown')[:300]}")

        if result.get("stderr"):
            print(f"  Stderr: {result['stderr'][:300]}")

        pipeline_failed = True

    # Validate outputs
    if result["status"] == "SUCCESS":

        validations = validate_outputs(stage, OUTPUT_DIR)

        print(f"\n  Output Validation:")

        all_valid = True

        for v in validations:
            print(f"    {v['status']} {v['file']:<50s} {v['size_mb']:>8.2f} MB")

            if v["status"] != "✓":
                all_valid = False

        if not all_valid:
            print(f"\n  ⚠ WARNING: Some expected outputs are missing.")
            print(f"    Pipeline will continue but downstream modules may fail.")

    pipeline_results.append({
        "stage": stage["id"],
        "name": stage["name"],
        "notebook": stage["notebook"],
        "status": result["status"],
        "elapsed_str": result["elapsed_str"],
        "elapsed_seconds": result["elapsed_seconds"],
    })

pipeline_elapsed = time.time() - pipeline_start

print(f"\n{'=' * 70}")
print(f"PIPELINE EXECUTION COMPLETE")
print(f"  Total time: {str(timedelta(seconds=int(pipeline_elapsed)))}")
print(f"  End time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}")


# ## 4 — Execution Summary
# 

# In[11]:


print("\n" + "=" * 70)
print("PIPELINE SUMMARY REPORT")
print("=" * 70)

print(f"\n  {'Stage':<6s} {'Module':<35s} {'Status':<10s} {'Time':>10s}")
print(f"  {'─' * 65}")

total_success = 0
total_failed = 0
total_skipped = 0

for r in pipeline_results:
    icon = {"SUCCESS":"✓","FAILED":"✗","TIMEOUT":"⏱","SKIPPED":"⊘","ERROR":"⚠"}.get(r["status"],"?")
    print(f"  [{r['stage']}]  {r['name']:<35s} {icon} {r['status']:<8s} {r['elapsed_str']:>10s}")

    if r["status"] == "SUCCESS": total_success += 1
    elif r["status"] == "SKIPPED": total_skipped += 1
    else: total_failed += 1

print(f"\n  Total: {total_success} succeeded | {total_failed} failed | {total_skipped} skipped")
print(f"  Pipeline time: {str(timedelta(seconds=int(pipeline_elapsed)))}")

# Overall status
if total_failed == 0 and total_skipped == 0:
    print(f"\n  ✓ PIPELINE COMPLETED SUCCESSFULLY")
elif total_failed > 0:
    print(f"\n  ✗ PIPELINE FAILED — check error logs above")
    failed_stages = [r for r in pipeline_results if r["status"] in ("FAILED", "ERROR", "TIMEOUT")]
    for r in failed_stages:
        print(f"    → Stage [{r['stage']}] {r['name']}: {r['status']}")

# ═══ Output file inventory ═══
print(f"\n{'─' * 70}")
print(f"OUTPUT FILE INVENTORY")
print(f"{'─' * 70}")

total_size = 0
file_count = 0
# for root, dirs, files in os.walk(OUTPUT_DIR):
#     # Skip executed_notebooks directory for cleaner output
#     if "executed_notebooks" in root:
#         continue
#     for fname in sorted(files):
#         fpath = os.path.join(root, fname)
#         size_mb = os.path.getsize(fpath) / (1024 * 1024)
#         rel_path = os.path.relpath(fpath, OUTPUT_DIR)
#         print(f"  {rel_path:<60s} {size_mb:>8.2f} MB")
#         total_size += size_mb
#         file_count += 1

# print(f"\n  Total: {file_count} files, {total_size:.2f} MB")
# print(f"  Location: {OUTPUT_DIR}")


# ## 5 — Quick Output Validation
# 
# Loads key output files and prints summary statistics to confirm the pipeline produced valid results.
# 

# In[12]:


# import pandas as pd

# print("\n" + "=" * 70)
# print("QUICK VALIDATION")
# print("=" * 70)

# validation_checks = []

# # Check 1: Generator output
# gen_file = os.path.join(OUTPUT_DIR, "transactions_generated_typology.parquet")
# if os.path.exists(gen_file):
#     df_gen = pd.read_parquet(gen_file)
#     aml_count = (df_gen.get("is_aml", pd.Series()) == 1).sum()
#     total = len(df_gen)
#     fraud_rate = aml_count / total * 100 if total > 0 else 0
#     print(f"\n  [00] Generator: {total:,} transactions, {aml_count:,} AML ({fraud_rate:.1f}%)")
#     validation_checks.append(("Generator", total > 300000 and 15 < fraud_rate < 30))
#     del df_gen
# else:
#     print(f"\n  [00] Generator: ⚠ Output not found")
#     validation_checks.append(("Generator", False))

# # Check 2: Detector output
# det_file = os.path.join(OUTPUT_DIR, "stg_transactions_flagged.parquet")
# if os.path.exists(det_file):
#     df_det = pd.read_parquet(det_file)
#     flagged = (df_det.get("is_aml", pd.Series()) == 1).sum()
#     typs = df_det.get("aml_typology", pd.Series()).nunique()
#     multi = df_det.get("aml_typology", pd.Series()).astype(str).str.contains(";", na=False).sum()
#     print(f"  [01] Detector:  {flagged:,} flagged, {typs} typologies, {multi} multi-label (should be 0)")
#     validation_checks.append(("Detector", flagged > 50000 and multi == 0))
#     del df_det
# else:
#     print(f"  [01] Detector:  ⚠ Output not found")
#     validation_checks.append(("Detector", False))

# # Check 3: Rules output
# rules_file = os.path.join(OUTPUT_DIR, "stg_transactions_rules.parquet")
# if os.path.exists(rules_file):
#     df_rules = pd.read_parquet(rules_file)
#     rule_cols = [c for c in df_rules.columns if c.startswith("rule_") and c not in {"rule_score","rules_triggered","rules_triggered_count"}]
#     trigger_rate = (df_rules[rule_cols].sum(axis=1) > 0).mean() * 100 if rule_cols else 0
#     print(f"  [02] Rules:     {len(rule_cols)} rules, {trigger_rate:.1f}% trigger rate (target: 50-60%)")
#     validation_checks.append(("Rules", 40 < trigger_rate < 75))
#     del df_rules
# else:
#     print(f"  [02] Rules:     ⚠ Output not found")
#     validation_checks.append(("Rules", False))

# # Check 4: Features output
# feat_file = os.path.join(OUTPUT_DIR, "stg_transactions_features.parquet")
# if os.path.exists(feat_file):
#     df_feat = pd.read_parquet(feat_file)
#     n_features = len(df_feat.columns)
#     print(f"  [03] Features:  {len(df_feat):,} rows × {n_features} columns")
#     validation_checks.append(("Features", n_features > 150))
#     del df_feat
# else:
#     print(f"  [03] Features:  ⚠ Output not found")
#     validation_checks.append(("Features", False))

# # Check 5: Phase 1 model
# model_file = os.path.join(OUTPUT_DIR, "ml_outputs", "model_metadata.json")
# if os.path.exists(model_file):
#     with open(model_file) as f:
#         meta = json.load(f)
#     print(f"  [04] Phase 1:   AUC={meta.get('auc_roc','?'):.4f}, F1={meta.get('f1_score','?'):.4f}, "
#           f"Threshold={meta.get('optimal_threshold','?')}, Features={meta.get('n_features','?')}")
#     auc = meta.get("auc_roc", 0)
#     validation_checks.append(("Phase 1", auc > 0.90))
# else:
#     print(f"  [04] Phase 1:   ⚠ Model metadata not found")
#     validation_checks.append(("Phase 1", False))

# # Check 6: Phase 2 model
# p2_meta_file = os.path.join(OUTPUT_DIR, "phase2_outputs", "phase2_metadata.json")
# if os.path.exists(p2_meta_file):
#     with open(p2_meta_file) as f:
#         p2_meta = json.load(f)
#     print(f"  [05] Phase 2:   Accuracy={p2_meta.get('phase2_accuracy','?'):.4f}, "
#           f"Classes={p2_meta.get('n_classes','?')}, Features={p2_meta.get('n_features','?')}")
#     p2_acc = p2_meta.get("phase2_accuracy", 0)
#     validation_checks.append(("Phase 2", p2_acc > 0.70))
# else:
#     print(f"  [05] Phase 2:   ⚠ Model metadata not found")
#     validation_checks.append(("Phase 2", False))

# # Combined output check
# combined_file = os.path.join(OUTPUT_DIR, "phase2_outputs", "combined_aml_output.parquet")
# if os.path.exists(combined_file):
#     df_combined = pd.read_parquet(combined_file)
#     aml_alerts = (df_combined.get("predicted_typology", pd.Series()) != "None").sum()
#     multi_label = (df_combined.get("num_typologies_matched", pd.Series(0)) >= 2).sum()
#     print(f"\n  Combined Output: {len(df_combined):,} rows, {aml_alerts:,} AML alerts, {multi_label:,} multi-label")

#     # Priority breakdown
#     if "investigation_priority" in df_combined.columns:
#         print(f"  Priority: ", end="")
#         for pri in ["Critical", "High", "Medium", "Low"]:
#             cnt = (df_combined["investigation_priority"] == pri).sum()
#             print(f"{pri}={cnt:,} ", end="")
#         print()
#     del df_combined
# else:
#     print(f"\n  Combined Output: ⚠ Not found")

# # Final verdict
# print(f"\n{'─' * 70}")
# print(f"VALIDATION SUMMARY")
# print(f"{'─' * 70}")
# all_passed = True
# for name, passed in validation_checks:
#     icon = "✓" if passed else "✗"
#     print(f"  {icon} {name}")
#     if not passed: all_passed = False

# if all_passed:
#     print(f"\n  ✓ ALL VALIDATIONS PASSED — Pipeline output is ready for deployment")
# else:
#     print(f"\n  ⚠ SOME VALIDATIONS FAILED — Review output above for details")


# ## 6 — Run Individual Modules (Optional)
# 
# Use this cell to re-run a single module without executing the full pipeline. Useful for debugging or re-running a specific stage after fixing an issue.
# 
# **Change `MODULE_TO_RUN`** to the module ID you want to execute (00–05).
# 

# In[ ]:


# # ═══ Change this to run a specific module ═══
# MODULE_TO_RUN = "01"  # Options: "00", "01", "02", "03", "04", "05"

# # Find the module
# target = next((s for s in PIPELINE if s["id"] == MODULE_TO_RUN), None)
# if not target:
#     print(f"Module {MODULE_TO_RUN} not found. Valid IDs: {[s['id'] for s in PIPELINE]}")
# else:
#     print(f"Running single module: [{target['id']}] {target['name']}")
#     print(f"  {target['description']}")
#     print(f"  Notebook: {target['notebook']}")
#     print()

#     nb_path = os.path.join(NOTEBOOK_DIR, target["notebook"])
#     result = run_notebook(nb_path, timeout_minutes=TIMEOUT_MINUTES, working_dir=NOTEBOOK_DIR)

#     icon = {"SUCCESS":"✓","FAILED":"✗","TIMEOUT":"⏱","ERROR":"⚠"}.get(result["status"],"?")
#     print(f"\n  {icon} {result['status']} ({result['elapsed_str']})")

#     if result["status"] == "SUCCESS":
#         validations = validate_outputs(target, OUTPUT_DIR)
#         print(f"\n  Outputs:")
#         for v in validations:
#             print(f"    {v['status']} {v['file']:<50s} {v['size_mb']:>8.2f} MB")
#     else:
#         print(f"  Error: {result.get('error', 'Unknown')[:500]}")
#         if result.get("stderr"):
#             print(f"\n  Stderr (last 500 chars):")
#             print(f"  {result['stderr'][:500]}")


# ## 7 — Run Pipeline from a Specific Stage (Optional)
# 
# If a module failed and you've fixed it, use this to resume from that stage instead of re-running the entire pipeline. All subsequent modules will also be re-run.
# 

# In[ ]:


# # ═══ Change this to the stage to START from ═══
# START_FROM = "02"  # Will run 02, 03, 04, 05

# print(f"Running pipeline from stage [{START_FROM}] onwards...")
# print()

# start_idx = next((i for i, s in enumerate(PIPELINE) if s["id"] == START_FROM), None)
# if start_idx is None:
#     print(f"Stage {START_FROM} not found. Valid IDs: {[s['id'] for s in PIPELINE]}")
# else:
#     stages_to_run = PIPELINE[start_idx:]
#     print(f"  Stages to execute: {[s['id'] + ' ' + s['name'] for s in stages_to_run]}")
#     print()

#     partial_results = []
#     failed = False
#     partial_start = time.time()

#     for stage in stages_to_run:
#         print(f"{'─' * 50}")
#         print(f"  [{stage['id']}] {stage['name']}")

#         if failed and STOP_ON_FAILURE:
#             print(f"  ⊘ SKIPPED")
#             partial_results.append({"stage": stage["id"], "name": stage["name"], "status": "SKIPPED", "elapsed_str": "—"})
#             continue

#         nb_path = os.path.join(NOTEBOOK_DIR, stage["notebook"])
#         result = run_notebook(nb_path, timeout_minutes=TIMEOUT_MINUTES, working_dir=NOTEBOOK_DIR)

#         icon = {"SUCCESS":"✓","FAILED":"✗","TIMEOUT":"⏱","ERROR":"⚠"}.get(result["status"],"?")
#         print(f"  {icon} {result['status']} ({result['elapsed_str']})")

#         if result["status"] != "SUCCESS":
#             print(f"  Error: {result.get('error', 'Unknown')[:300]}")
#             failed = True
#         else:
#             validations = validate_outputs(stage, OUTPUT_DIR)
#             for v in validations:
#                 print(f"    {v['status']} {v['file']}")

#         partial_results.append({"stage": stage["id"], "name": stage["name"], "status": result["status"], "elapsed_str": result["elapsed_str"]})

#     partial_elapsed = time.time() - partial_start
#     print(f"\n{'─' * 50}")
#     print(f"  Partial pipeline complete: {str(timedelta(seconds=int(partial_elapsed)))}")
#     for r in partial_results:
#         icon = {"SUCCESS":"✓","FAILED":"✗","TIMEOUT":"⏱","SKIPPED":"⊘"}.get(r["status"],"?")
#         print(f"    {icon} [{r['stage']}] {r['name']}: {r['status']} ({r['elapsed_str']})")


# ## 8 — Save Execution Log
# 

# ## 9 — Persistent Run History Log

# In[13]:


# ════════════════════════════════════════════════════════════════
# PERSISTENT RUN LOG  — reads model metrics from PostgreSQL
# ════════════════════════════════════════════════════════════════
# Builds one run-record per pipeline run and writes it to the
# pipeline_execution_log table. Model metrics are read back from:
#   Phase 1 -> model_parameters_full         (flat key/value)
#   Phase 2 -> model_parameters_full_phase2  (flat key/value)
# Both tables are keyed on run_id. No JSON / CSV files are written.
# ════════════════════════════════════════════════════════════════

import os as _os
import json as _json
import subprocess as _subprocess
from datetime import datetime as _datetime, timedelta as _timedelta

import pandas as _pd
from sqlalchemy import text as _text
from db_utils import get_engine, write_table_fast


def _read_params_table(table_name, run_id):
    """Read a flat key/value model-parameters table -> {parameter: value}.

    Filters on run_id; if that run is not present, falls back to the most
    recent run in the table so the log still captures something useful.
    """
    eng = get_engine()
    try:
        df = _pd.read_sql(
            _text(f"SELECT parameter, value FROM {table_name} WHERE run_id = :r"),
            eng, params={"r": run_id})
        if df.empty:
            df = _pd.read_sql(
                _text(f"""SELECT parameter, value FROM {table_name}
                          WHERE run_id = (SELECT run_id FROM {table_name}
                                          ORDER BY loaded_at DESC LIMIT 1)"""),
                eng)
        return dict(zip(df["parameter"], df["value"]))
    except Exception as _e:
        print(f"  WARNING: could not read {table_name}: {_e}")
        return {}


def _num(d, key):
    """Fetch a flat-dict value and cast to float; None if missing/blank."""
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _git_hash():
    try:
        out = _subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=2)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def write_run_log(mode, pipeline_results, pipeline_elapsed_seconds, run_id):
    """Build the run record, write it to pipeline_execution_log, return it."""
    now = _datetime.now()

    # ── Phase 1 metrics — model_parameters_full ──
    p1f = _read_params_table("model_parameters_full", run_id)
    p1_metrics = {
        "best_config":     p1f.get("phase1.best_config"),
        "threshold":       _num(p1f, "phase1.threshold"),
        "auc_roc":         _num(p1f, "phase1.auc_roc"),
        "f1_score":        _num(p1f, "phase1.f1_score"),
        "precision":       _num(p1f, "phase1.precision"),
        "recall":          _num(p1f, "phase1.recall"),
        "tp":              _num(p1f, "phase1.tp"),
        "fp":              _num(p1f, "phase1.fp"),
        "fn":              _num(p1f, "phase1.fn"),
        "tn":              _num(p1f, "phase1.tn"),
        "best_iteration":  _num(p1f, "phase1.best_iteration"),
        "n_features":      _num(p1f, "phase1.n_features"),
        "n_train":         _num(p1f, "phase1.n_train"),
        "n_test":          _num(p1f, "phase1.n_test"),
        "imbalance_ratio": _num(p1f, "phase1.imbalance_ratio"),
    } if p1f else {}

    # ── Phase 2 metrics — model_parameters_full_phase2 ──
    p2f = _read_params_table("model_parameters_full_phase2", run_id)
    p2_metrics = {
        "best_config":           p2f.get("phase2.best_config"),
        "accuracy_primary":      _num(p2f, "phase2.accuracy_primary"),
        "accuracy_multi_label":  _num(p2f, "phase2.accuracy_multi_label"),
        "multi_label_threshold": _num(p2f, "phase2.multi_label_threshold"),
        "macro_f1":              _num(p2f, "phase2.macro_f1"),
        "weighted_f1":           _num(p2f, "phase2.weighted_f1"),
        "best_iteration":        _num(p2f, "phase2.best_iteration"),
        "n_classes":             _num(p2f, "phase2.n_classes"),
        "n_features":            _num(p2f, "phase2.n_features"),
        "n_train":               _num(p2f, "phase2.n_train"),
        "n_test":                _num(p2f, "phase2.n_test"),
    } if p2f else {}

    n_total  = len(pipeline_results)
    n_ok     = sum(1 for r in pipeline_results if r["status"] == "SUCCESS")
    n_failed = sum(1 for r in pipeline_results if r["status"] == "FAILED")
    overall  = "SUCCESS" if n_failed == 0 else "FAILED"

    entry = {
        "run_id":            run_id,
        "timestamp":         now.isoformat(timespec="seconds"),
        "mode":              mode,
        "duration_seconds":  round(pipeline_elapsed_seconds, 2),
        "duration_str":      str(_timedelta(seconds=int(pipeline_elapsed_seconds))),
        "git_hash":          _git_hash(),
        "stages":            [{"id": r["stage"], "name": r["name"],
                               "status": r["status"],
                               "duration": r.get("elapsed_str", "—")}
                              for r in pipeline_results],
        "stages_succeeded":  n_ok,
        "stages_failed":     n_failed,
        "stages_total":      n_total,
        "overall_status":    overall,
        "phase1_metrics":    p1_metrics,
        "phase2_metrics":    p2_metrics,
    }

    # ── Write one row to pipeline_execution_log ──
    log_row = _pd.DataFrame([{
        "run_id":            run_id,
        "pipeline_mode":     mode,
        "execution_date":    now,
        "overall_status":    overall,
        "total_elapsed_sec": round(pipeline_elapsed_seconds, 2),
        "total_elapsed_str": entry["duration_str"],
        "stages_total":      n_total,
        "stages_succeeded":  n_ok,
        "stages_failed":     n_failed,
        "stage_detail":      _json.dumps(entry["stages"], default=str),
        "notebook_dir":      NOTEBOOK_DIR,
    }])
    write_table_fast(log_row, "pipeline_execution_log", mode="append")

    # ── Summary ──
    print()
    print("=" * 70)
    print(f"RUN LOG — run {run_id}  ({mode.upper()})  ->  {overall}")
    print("=" * 70)
    print(f"  Duration:  {entry['duration_seconds']}s")
    print(f"  Stages:    {n_ok}/{n_total} succeeded")
    if p1_metrics and p1_metrics.get("auc_roc") is not None:
        print(f"  Phase 1:   AUC={p1_metrics['auc_roc']:.4f}  "
              f"Recall={p1_metrics['recall']:.4f}  F1={p1_metrics['f1_score']:.4f}")
    if p2_metrics and p2_metrics.get("accuracy_primary") is not None:
        ap, am = p2_metrics["accuracy_primary"], p2_metrics["accuracy_multi_label"]
        print(f"  Phase 2:   primary={ap*100:.2f}%  multi-label={am*100:.2f}%")
    print("=" * 70)
    return entry


# ── Run it — pipeline_results / pipeline_elapsed / RUN_ID in scope ──
entry = write_run_log(PIPELINE_MODE, pipeline_results, pipeline_elapsed, RUN_ID)


# ## 10 — Dashboard View *(monitoring summary)*

# In[16]:


# ════════════════════════════════════════════════════════════════
# DASHBOARD ROW — append this run's metrics to PostgreSQL
# ════════════════════════════════════════════════════════════════
# Writes ONE row per run to performance_metrics_dashboard (append).
# Metrics come from the 'entry' dict built by the previous cell.
# ════════════════════════════════════════════════════════════════

import pandas as _pd
from db_utils import write_table_fast


def update_run_dashboard(entry):
    """Flatten one run record into a dashboard row and append it."""
    p1 = entry.get("phase1_metrics") or {}
    p2 = entry.get("phase2_metrics") or {}

    row = {
        "run_id":            entry.get("run_id"),
        "timestamp":         entry.get("timestamp"),
        "mode":              entry.get("mode"),
        "duration_seconds":  entry.get("duration_seconds"),
        "stages_succeeded":  entry.get("stages_succeeded"),
        "stages_total":      entry.get("stages_total"),
        "overall_status":    entry.get("overall_status"),
        "git_hash":          entry.get("git_hash"),
        # Phase 1
        "p1_best_config":    p1.get("best_config"),
        "p1_threshold":      p1.get("threshold"),
        "p1_auc_roc":        p1.get("auc_roc"),
        "p1_f1_score":       p1.get("f1_score"),
        "p1_precision":      p1.get("precision"),
        "p1_recall":         p1.get("recall"),
        "p1_tp":             p1.get("tp"),
        "p1_fp":             p1.get("fp"),
        "p1_fn":             p1.get("fn"),
        "p1_tn":             p1.get("tn"),
        "p1_n_features":     p1.get("n_features"),
        "p1_n_train":        p1.get("n_train"),
        "p1_n_test":         p1.get("n_test"),
        "p1_imbalance":      p1.get("imbalance_ratio"),
        # Phase 2
        "p2_best_config":         p2.get("best_config"),
        "p2_accuracy_primary":    p2.get("accuracy_primary"),
        "p2_accuracy_multilabel": p2.get("accuracy_multi_label"),
        "p2_multilabel_thresh":   p2.get("multi_label_threshold"),
        "p2_macro_f1":            p2.get("macro_f1"),
        "p2_weighted_f1":         p2.get("weighted_f1"),
        "p2_n_classes":           p2.get("n_classes"),
        "p2_n_train":             p2.get("n_train"),
        "p2_n_test":              p2.get("n_test"),
    }

    df_row = _pd.DataFrame([row])
    write_table_fast(df_row, "perfomance_metrics_dashboard", mode="append")

    print()
    print("=" * 70)
    print("DASHBOARD ROW written -> perfomance_metrics_dashboard")
    print("=" * 70)
    print(f"  Run ID:   {row['run_id']}")
    print(f"  Mode:     {row['mode']}")
    print(f"  Status:   {row['overall_status']}")
    if row["p1_auc_roc"] is not None:
        print(f"  Phase 1:  AUC={row['p1_auc_roc']:.4f}  Recall={row['p1_recall']:.4f}")
    if row["p2_accuracy_primary"] is not None:
        print(f"  Phase 2:  primary={row['p2_accuracy_primary']*100:.2f}%")
    print("=" * 70)
    return df_row


_dashboard_df = update_run_dashboard(entry)

