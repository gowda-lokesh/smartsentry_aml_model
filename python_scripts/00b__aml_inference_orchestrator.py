#!/usr/bin/env python
# coding: utf-8

# # SmartSentry AML — Inference Orchestrator
# 
# Runs the pipeline against a **new input file** (same schema as the output of `01__aml_typology_detector`, but **without** `is_aml`, `aml_typology`, or `typology_group_id` columns).
# 
# Stages executed:
# 1. **02 Rules Engine** — apply 126 compliance rules
# 2. **03 Feature Engineering** — compute velocity, balance, graph features
# 3. **04 Phase 1 (predict)** — load saved Phase 1 model and score
# 4. **05 Phase 2 (predict)** — load saved Phase 2 model and classify typology
# 
# Final output: `predictions_output.parquet` in the Phase 2 output directory.
# 

# ## 1 — Configuration

# In[6]:



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
# CONFIG — update paths to match your environment
# ═══════════════════════════════════════════════════════════════

#NOTEBOOK_DIR = os.getcwd()
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
print(NOTEBOOK_DIR)
OUTPUT_DIR   = os.path.join(os.path.dirname(NOTEBOOK_DIR), "outputs_updated")
PHASE1_DIR   = os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "ml_outputs")
PHASE2_DIR   = os.path.join(os.path.dirname(NOTEBOOK_DIR), "python_scripts", "phase2_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Input file: the user-provided transaction file ───────────
# Override at runtime by setting AML_INFERENCE_INPUT env var.
#DEFAULT_INPUT = os.path.join(OUTPUT_DIR, "inference_input.parquet")
#INPUT_FILE = os.environ.get("AML_INFERENCE_INPUT", DEFAULT_INPUT)

# if not os.path.exists(INPUT_FILE):
#     raise FileNotFoundError(
#         f"Inference input file not found: {INPUT_FILE}\n"
#         f"Set AML_INFERENCE_INPUT env var or place file at default path."
#     )

TIMEOUT_MINUTES = 60
STOP_ON_FAILURE = True
SAVE_EXECUTED_NOTEBOOKS = True

# # Verify saved model bundles exist
# required_bundles = [
#     os.path.join(PHASE1_DIR, "phase1_model_bundle.joblib"),
#     os.path.join(PHASE2_DIR, "phase2_model_bundle.joblib"),
# ]

print("=" * 70)
print("SmartSentry AML — Inference Orchestrator")
print("=" * 70)
#print(f"  Input file:        {INPUT_FILE}")
# print(f"  Phase 1 bundle:    {required_bundles[0]}")
# print(f"  Phase 2 bundle:    {required_bundles[1]}")
print(f"  Working directory: {NOTEBOOK_DIR}")
print(f"  Output directory:  {OUTPUT_DIR}")

# for b in required_bundles:
#     if not os.path.exists(b):
#         raise FileNotFoundError(
#             f"Required model bundle missing: {b}\n"
#             f"Train the models first using 00__aml_pipeline_orchestrator with yes."
#         )

#     print(f"  ✓ {os.path.basename(b)}: {os.path.getsize(b)/(1024*1024):.2f} MB")

print()


# In[ ]:

import datetime as _dt

RUN_ID = os.environ.get(
    "AML_RUN_ID",
    _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
)

os.environ["AML_RUN_ID"] = RUN_ID
PIPELINE_MODE = "predict"

print(f"Run ID: {RUN_ID}")


# ## 2 — Inference Pipeline Definition

# In[ ]:

# Python scripts to run, in order.
# AML_RUN_MODE=predict, AML_INPUT_FILE chains through the pipeline.

INFERENCE_PIPELINE = [
    {
        "id":          "02",
        "name":        "Rules Engine",
        "notebook":    "02__aml_rules_engine.py",
        "description": "Apply 126 RBI/PMLA compliance rules",
        "input_env":   "transactions file",
        "output_file": os.path.join(OUTPUT_DIR, "stg_transactions_rules.parquet"),
    },
    {
        "id":          "03",
        "name":        "Feature Engineering",
        "notebook":    "03__aml_feature_engineering.py",
        "description": "Compute velocity, balance, graph features",
        "input_env":   os.path.join(OUTPUT_DIR, "stg_transactions_rules.parquet"),
        "output_file": os.path.join(OUTPUT_DIR, "stg_transactions_features.parquet"),
    },
    {
        "id":          "04",
        "name":        "Phase 1 (predict)",
        "notebook":    "04__aml_ml_preparation.py",
        "description": "Score transactions with Phase 1 binary AML model",
        "input_env":   os.path.join(OUTPUT_DIR, "stg_transactions_features.parquet"),
        "output_file": os.path.join(PHASE1_DIR, "df_ml_phase_1.parquet"),
    },
    {
        "id":          "05",
        "name":        "Phase 2 (predict)",
        "notebook":    "05__aml_phase2_typology_classifier.py",
        "description": "Classify into 10 typologies; multi-label output",
        # Phase 2 reads df_ml_phase_1.parquet from PHASE1_DIR via env var.
        # Set input_env to the same so AML_INPUT_FILE is always a valid path string.
        "input_env":   os.path.join(PHASE1_DIR, "df_ml_phase_1.parquet"),
        "output_file": os.path.join(PHASE2_DIR, "predictions_output.parquet"),
    },
]

# Defensive: every stage MUST have a string input_env.
for stage in INFERENCE_PIPELINE:
    if not stage["input_env"] or not isinstance(stage["input_env"], str):
        raise ValueError(
            f"Stage {stage['id']} has invalid input_env={stage['input_env']!r}. "
            f"Re-run this cell from scratch."
        )

print("Verifying scripts present...")

for stage in INFERENCE_PIPELINE:
    nb_path = os.path.join(NOTEBOOK_DIR, stage["notebook"])
    status = "✓" if os.path.exists(nb_path) else "⚠ NOT FOUND"

    print(f"  [{stage['id']}] {stage['notebook']:<55s} {status}")

    if not os.path.exists(nb_path):
        raise FileNotFoundError(stage["notebook"])

print("  All scripts present.\n")

# Sanity-print the chain
print("Pipeline chain (input → script → output):")

for stage in INFERENCE_PIPELINE:
    print(
        f"  [{stage['id']}] "
        f"{os.path.basename(stage['input_env']):<45s} "
        f"→ {stage['notebook']:<45s} "
        f"→ {os.path.basename(stage['output_file'])}"
    )


# ## 3 — Script Runner

# In[8]:

def run_notebook_with_env(
    notebook_path,
    env_overrides,
    timeout_minutes=60,
    working_dir=None
):
    """Execute a Python script in a fresh process, with env variables set."""

    start_time = time.time()
    notebook_name = os.path.basename(notebook_path)

    executed_dir = os.path.join(
        OUTPUT_DIR,
        "executed_inference_notebooks"
    )

    os.makedirs(executed_dir, exist_ok=True)

    executed_path = os.path.join(
        executed_dir,
        notebook_name.replace(".py", "_execution.log")
    )

    env = os.environ.copy()

    env.update({
        str(k): str(v)
        for k, v in env_overrides.items()
    })

    print(f"  Executing: {notebook_name}")

    for k, v in env_overrides.items():
        print(f"    {k}={v}")

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
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        elapsed = time.time() - start_time

        # Save logs
        with open(executed_path, "w", encoding="utf-8") as f:
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
                "stdout": result.stdout[-500:] if result.stdout else "",
            }

        else:
            return {
                "status": "FAILED",
                "elapsed_seconds": elapsed,
                "elapsed_str": str(timedelta(seconds=int(elapsed))),
                "error": (result.stderr or "")[-1500:],
                "stdout": (result.stdout or "")[-500:],
            }

    except subprocess.TimeoutExpired:

        elapsed = time.time() - start_time

        return {
            "status": "TIMEOUT",
            "elapsed_seconds": elapsed,
            "elapsed_str": str(timedelta(seconds=int(elapsed))),
            "error": f"exceeded {timeout_minutes} min",
        }

    except Exception as e:

        elapsed = time.time() - start_time

        return {
            "status": "ERROR",
            "elapsed_seconds": elapsed,
            "elapsed_str": str(timedelta(seconds=int(elapsed))),
            "error": str(e),
        }


print("Runner loaded.")


# ## 4 — Execute Inference Pipeline

# In[9]:

print("=" * 75)
print("INFERENCE PIPELINE EXECUTION STARTED")
print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

pipeline_start = time.time()
pipeline_results = []
pipeline_failed = False

for stage in INFERENCE_PIPELINE:

    print(f"\n{'─' * 70}")
    print(f"  STAGE [{stage['id']}] {stage['name']}")
    print(f"  {stage['description']}")
    print(f"{'─' * 70}")

    if pipeline_failed and STOP_ON_FAILURE:

        print("  ⊘ SKIPPED (previous stage failed)")

        pipeline_results.append({
            "stage": stage["id"],
            "name": stage["name"],
            "status": "SKIPPED",
            "elapsed_str": "—",
            "elapsed_seconds": 0,
        })

        continue

    # Defensive: input_env must be a real, existing string path.
    stage_input = stage.get("input_env")

    if (
        not stage_input
        or stage_input in ("None", "none", None)
        or not isinstance(stage_input, str)
    ):

        print(
            f"  ✗ Stage [{stage['id']}] "
            f"has invalid input_env={stage_input!r} — aborting"
        )

        pipeline_failed = True

        pipeline_results.append({
            "stage": stage["id"],
            "name": stage["name"],
            "status": "ERROR",
            "elapsed_str": "—",
            "elapsed_seconds": 0,
        })

        continue

    # if not os.path.exists(stage_input):

    #     print(
    #         f"  ✗ Stage [{stage['id']}] "
    #         f"input file does not exist: {stage_input}"
    #     )

    #     print(
    #         f"     Previous stage may have completed "
    #         f"without writing the expected output."
    #     )

    #     pipeline_failed = True

    #     pipeline_results.append({
    #         "stage": stage["id"],
    #         "name": stage["name"],
    #         "status": "ERROR",
    #         "elapsed_str": "—",
    #         "elapsed_seconds": 0,
    #     })

    #     continue

    env_overrides = {
        "AML_RUN_MODE":   "predict",
        "AML_INPUT_FILE": stage_input,
        "AML_PHASE1_DIR": PHASE1_DIR,
        "AML_PHASE2_DIR": PHASE2_DIR,
    }

    nb_path = os.path.join(
        NOTEBOOK_DIR,
        stage["notebook"]
    )

    result = run_notebook_with_env(
        nb_path,
        env_overrides,
        timeout_minutes=TIMEOUT_MINUTES,
        working_dir=NOTEBOOK_DIR
    )

    status_icon = {
        "SUCCESS": "✓",
        "FAILED": "✗",
        "TIMEOUT": "⏱",
        "ERROR": "⚠",
    }.get(result["status"], "?")

    print(
        f"\n  {status_icon} "
        f"Status: {result['status']} ({result['elapsed_str']})"
    )

    if result["status"] != "SUCCESS":

        print(f"  Error: {(result.get('error') or '')[:400]}")

        pipeline_failed = True

    else:

        out = stage["output_file"]

        # if os.path.exists(out):

        #     sz = os.path.getsize(out) / (1024 * 1024)

        #     print(
        #         f"  Output:  ✓ "
        #         f"{os.path.basename(out):<50s} "
        #         f"{sz:>8.2f} MB"
        #     )

        # else:

        #     print(f"  Output:  ⚠ MISSING: {out}")

        #     pipeline_failed = True

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
print("INFERENCE PIPELINE COMPLETE")
print(f"  Total time: {str(timedelta(seconds=int(pipeline_elapsed)))}")
print(f"  End time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}")


# ## 5 — Summary

# In[5]:


print("\n" + "=" * 70)
print("STAGE SUMMARY")
print("=" * 70)
print(f"  {'Stage':<6s} {'Name':<25s} {'Status':<10s} {'Duration':<12s}")
print(f"  {'─' * 60}")
total_ok = 0
for r in pipeline_results:
    icon = {"SUCCESS": "✓", "FAILED": "✗", "TIMEOUT": "⏱", "ERROR": "⚠", "SKIPPED": "⊘"}.get(r["status"], "?")
    print(f"  [{r['stage']}]  {r['name']:<25s} {icon} {r['status']:<8s} {r['elapsed_str']:<12s}")
    if r["status"] == "SUCCESS":
        total_ok += 1

print(f"\n  {total_ok}/{len(pipeline_results)} stages succeeded")
#print(f"\n  Final output: {os.path.join(PHASE2_DIR, 'predictions_output.parquet')}")
#print(f"                {os.path.join(PHASE2_DIR, 'predictions_output.csv')}")


# ## 6 — Persistent Run History Log

# In[ ]:


# ── Database connection (PostgreSQL) ──
from db_utils import read_table, write_table, save_model, load_model, test_connection
test_connection()      # prints a one-line OK on connect


# In[ ]:


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


# ## 7 — Dashboard View *(monitoring summary)*

# In[ ]:


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
    print("DASHBOARD ROW written -> performance_metrics_dashboard")
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

