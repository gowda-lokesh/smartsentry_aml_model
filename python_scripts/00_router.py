#!/usr/bin/env python
# coding: utf-8

# # SmartSentry AML — Router
# 
# **Single entry point for the whole framework.**
# 
# Run all cells. When prompted, type **`yes`** or **`no`**:
# 
# | Answer | Runs | Pipeline |
# |--------|------|----------|
# | **`yes`** | `00__aml_pipeline_orchestrator.py` | Full training — data generation → detector → rules → features → Phase 1 → Phase 2 |
# | **`no`**  | `00b__aml_inference_orchestrator.py` | Inference only — rules → features → Phase 1 predict → Phase 2 predict |

# ## 1 — Setup

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────
NOTEBOOK_DIR = os.getcwd()
print(NOTEBOOK_DIR)
OUTPUT_DIR   = os.path.join(os.path.dirname(NOTEBOOK_DIR), "outputs_updated")
EXEC_DIR     = os.path.join(OUTPUT_DIR, "executed_notebooks")
os.makedirs(EXEC_DIR, exist_ok=True)

# =========================================================
# CHANGED ONLY THESE FILES FROM .ipynb TO .py
# =========================================================
TRAIN_NOTEBOOK   = os.path.join(NOTEBOOK_DIR, "python_scripts", "00__aml_pipeline_orchestrator.py")
PREDICT_NOTEBOOK = os.path.join(NOTEBOOK_DIR, "python_scripts", "00b__aml_inference_orchestrator.py")

# Generous timeout — training can be long; inference is quick.
TIMEOUT_MINUTES = 240

print("=" * 70)
print("SmartSentry AML — Router")
print("=" * 70)
print(f"  Notebook directory: {NOTEBOOK_DIR}")
print(f"  Output directory:   {OUTPUT_DIR}")
print(f"  Training target:    {os.path.basename(TRAIN_NOTEBOOK)}",
      "  ✓" if os.path.exists(TRAIN_NOTEBOOK) else "  ⚠ MISSING")
print(f"  Inference target:   {os.path.basename(PREDICT_NOTEBOOK)}",
      " ✓" if os.path.exists(PREDICT_NOTEBOOK) else " ⚠ MISSING")

for _nb in (TRAIN_NOTEBOOK, PREDICT_NOTEBOOK):
    if not os.path.exists(_nb):
        raise FileNotFoundError(
            f"Required orchestrator not found:\n  {_nb}\n\n"
            f"Both 00__aml_pipeline_orchestrator.py and "
            f"00b__aml_inference_orchestrator.py must sit in the same "
            f"directory as this router script."
        )

print("\n  Both orchestrators found.")


# ## 2 — Main Function

def run_pipeline(choice):
    """Route to the correct orchestrator based on a yes/no answer."""

    answer = str(choice).strip().lower()

    yes_set = {"yes", "y", "true", "1", "train", "t"}
    no_set  = {"no", "n", "false", "0", "predict", "p"}

    if answer in yes_set:
        mode          = "train"
        target_nb     = TRAIN_NOTEBOOK
        description   = "FULL TRAINING pipeline (generation → detector → rules → features → Phase 1 → Phase 2)"

    elif answer in no_set:
        mode          = "predict"
        target_nb     = PREDICT_NOTEBOOK
        description   = "INFERENCE pipeline (rules → features → Phase 1 predict → Phase 2 predict)"

    else:
        raise ValueError(
            f"Unrecognised choice {choice!r}. "
            f"Reply 'yes' (training) or 'no' (inference)."
        )

    print("=" * 70)
    print(f"ROUTER → {mode.upper()} MODE")
    print("=" * 70)
    print(f"  Target notebook: {os.path.basename(target_nb)}")
    print(f"  Pipeline:        {description}")
    print(f"  Started:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timeout:         {TIMEOUT_MINUTES} minutes")
    print()

    # =========================================================
    # ONLY THIS OUTPUT FILE CHANGED TO .log
    # =========================================================
    executed_copy = os.path.join(
        EXEC_DIR,
        os.path.basename(target_nb).replace(".py", ".log")
    )

    # =========================================================
    # CHANGED COMMAND FROM nbconvert TO python execution
    # =========================================================
    cmd = [
        sys.executable, "-u"
        target_nb,
    ]

    print("─" * 70)
    print("Live output from the target orchestrator:")
    print("─" * 70)

    start = time.time()

    with open(executed_copy, "w", encoding="utf-8") as log_file:

        # Stream stdout/stderr line-by-line so progress is visible here.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=NOTEBOOK_DIR,
        )

        try:
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)

            proc.wait(timeout=TIMEOUT_MINUTES * 60 + 120)

        except subprocess.TimeoutExpired:
            proc.kill()
            elapsed = time.time() - start

            print(f"\n⚠ TIMEOUT after {elapsed/60:.1f} min — process killed.")
            raise

    elapsed = time.time() - start

    print("─" * 70)
    print(f"  Finished:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Elapsed:       {str(timedelta(seconds=int(elapsed)))}")
    print(f"  Exit code:     {proc.returncode}")
    print(f"  Executed copy: {executed_copy}")
    print("─" * 70)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{mode.upper()} orchestrator exited with code {proc.returncode}.\n"
            f"Open the executed log to see the failure:\n  {executed_copy}"
        )

    print(f"\n✓ {mode.upper()} pipeline completed successfully.")

    return {
        "mode":            mode,
        "notebook":        os.path.basename(target_nb),
        "exit_code":       proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "executed_copy":   executed_copy,
    }


print("Main function ready:  run_pipeline(choice)")


# ## 3 — Choose Mode & Run

# Resolve the choice: env var takes precedence, otherwise prompt interactively.
_env_mode = os.environ.get("AML_PIPELINE_MODE", "").strip().lower()

if _env_mode in ("train", "predict"):

    user_choice = "yes" if _env_mode == "train" else "no"

    print(f"AML_PIPELINE_MODE={_env_mode!r} (from environment) → choice = {user_choice!r}")

else:

    try:
        #user_choice = input("Run full training pipeline?  (yes / no): ")
        user_choice = print("Default is set to prediction mode, using the last saved model objects")

    except EOFError:

        # Non-interactive context
        user_choice = "no"

        print("Non-interactive context — defaulting to 'no' (inference).")

# Single call drives the whole framework.

### PRE-SET TO NO SO THAT IT ALWAYS PREDICTS WITH THE CURRENT MODEL OBJECTS
user_choice = "no"
result = run_pipeline(user_choice)

print()
print("=" * 70)
print("ROUTER COMPLETE")
print("=" * 70)

for k, v in result.items():
    print(f"  {k:<18s}: {v}")


# ## 4 — Where to Look Next
# 
# - **`outputs_updated/run_history.jsonl`** — append-only log; the latest run is the last line.
# - **`outputs_updated/run_dashboard.csv`** — refreshed monitoring view of every run.
# - **Training mode** — model bundles at `python_scripts/ml_outputs/phase1_model_bundle.joblib` and `python_scripts/phase2_outputs/phase2_model_bundle.joblib`.
# - **Inference mode** — final predictions at `python_scripts/phase2_outputs/predictions_output.parquet` (+ `.csv`).
# - **If a run failed** — open the executed copy under `outputs_updated/executed_notebooks/` and scroll to the cell with the traceback.

