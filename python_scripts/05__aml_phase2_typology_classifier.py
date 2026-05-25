#!/usr/bin/env python
# coding: utf-8

# # Phase 2: AML Typology Classification Model
# **Purpose:** Given a transaction already flagged as AML by Phase 1, predict WHICH typology it belongs to.
# 
# **Input:** Phase 1 outputs (model, features, train/test splits, metadata)
# **Output:** Probability distribution across 10 AML typologies per transaction
# 
# **Pipeline:**
# 1. Load Phase 1 artifacts
# 2. Prepare AML-only dataset with typology labels
# 3. Add typology-discriminating interaction features
# 4. Feature selection (remove features useless for typology separation)
# 5. Hyperparameter tuning with rare-class boosting
# 6. Evaluation with per-typology accuracy
# 7. Build production output tables
# 8. Save Phase 2 model and outputs
# 

# ## 1 — Environment Setup
# 

# In[5]:


import os
import sys
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import io
from psycopg2 import sql

import io
import pandas as pd
from psycopg2 import sql
from datetime import datetime
import random, string, hashlib, uuid, json, os, math 
from datetime import datetime, timedelta, date 
from collections import defaultdict, Counter 
import csv 
import warnings 
warnings.filterwarnings('ignore')
import lightgbm as lgb

from project_config.loader import ensure_notebook_path, get_run_mode, get_artifact_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_MODE = get_run_mode(_SETTINGS)

PHASE1_DIR = str(_PATHS["phase1_dir"])
OUTPUT_DIR = os.environ.get("AML_PHASE2_DIR", str(_PATHS["phase2_dir"]))
os.makedirs(OUTPUT_DIR, exist_ok=True)
TYPOLOGY_THRESHOLD = float(_SETTINGS["phase2"]["typology_threshold"])
P2_TUNING = _SETTINGS["phase2"]["hyperparameter_tuning"]
print("Libraries loaded")


# ## 2 — Load Phase 1 Artifacts
# Load the model, features, train/test data, and metadata saved by the Phase 1 notebook.
# 

# ## 3 — Load Phase 1 Data & Model
# 

# In[4]:


# ── Database connection (PostgreSQL) ──
from db_utils import read_table, write_table, save_model, load_model, test_connection
test_connection()      # prints a one-line OK on connect


# In[ ]:


print("Loading Phase 1 artifacts...")
bundle = load_model("phase1_model_bundle")          # from DB
final_model    = bundle["model"]
features_final = bundle["features"]
best_thresh    = bundle["threshold"]
best_strategy  = bundle.get("metadata", {}).get("best_strategy", "unknown")
print(f"  Phase 1 model loaded — {len(features_final)} features, threshold {best_thresh}")

if RUN_MODE == "train":
    X_train = read_table("x_train_phase1")
    X_test  = read_table("x_test_phase1")
    y_train = read_table("y_train_phase1").iloc[:, 0]
    y_test  = read_table("y_test_phase1").iloc[:, 0]

df_ml = read_table("phase1_model_output")


# In[ ]:


# print("Loading Phase 1 artifacts...")
# import joblib

# # Load bundled Phase 1 model
# bundle = joblib.load(os.path.join(PHASE1_DIR, "phase1_model_bundle.joblib"))
# final_model    = bundle["model"]
# features_final = bundle["features"]
# best_thresh    = bundle["threshold"]
# best_strategy  = bundle.get("metadata", {}).get("best_strategy", "unknown")
# print(f"  Phase 1 model loaded ({final_model.num_trees() if hasattr(final_model, 'num_trees') else 'n/a'} trees)")
# print(f"  Features:   {len(features_final)}")
# print(f"  Threshold:  {best_thresh}")
# print(f"  Strategy:   {best_strategy}")

# if RUN_MODE == "train":
#     # Train/test splits saved by Phase 1
#     X_train = pd.read_parquet(os.path.join(PHASE1_DIR, "X_train.parquet"))
#     X_test  = pd.read_parquet(os.path.join(PHASE1_DIR, "X_test.parquet"))
#     y_train = pd.read_parquet(os.path.join(PHASE1_DIR, "y_train.parquet")).iloc[:, 0]
#     y_test  = pd.read_parquet(os.path.join(PHASE1_DIR, "y_test.parquet")).iloc[:, 0]
#     print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
#     print(f"  AML in train: {(y_train==1).sum():,} | AML in test: {(y_test==1).sum():,}")

# # Full feature-engineered dataset (with Phase 1 scores already attached by Phase 1 notebook)
# FEATURES_FILE = os.environ.get(
#     "AML_PHASE1_OUTPUT",
#     os.path.join(PHASE1_DIR, "df_ml_phase_1.parquet"),
# )
# # Fallback to parent directory
# if not os.path.exists(FEATURES_FILE):
#     alt = os.path.join(os.path.dirname(PHASE1_DIR), "df_ml_phase_1.parquet")
#     if os.path.exists(alt):
#         FEATURES_FILE = alt
# if not os.path.exists(FEATURES_FILE):
#     raise FileNotFoundError(f"df_ml_phase_1.parquet not found at {FEATURES_FILE}")

# df_ml = pd.read_parquet(FEATURES_FILE)
# print(f"  Full dataset: {len(df_ml):,} rows × {len(df_ml.columns)} columns")
# print(f"  Loaded from: {FEATURES_FILE}")


# In[ ]:


print(df_ml['transaction_type_ppi'].unique())
obj_cols = df_ml.select_dtypes(include=["object", "string"]).columns

df_ml[obj_cols] = df_ml[obj_cols].fillna("")
print(df_ml['transaction_type_ppi'].unique())


# ## 4 — Generate Phase 1 Scores
# 

# In[ ]:


# Phase 1 scoring: skip if df_ml already has _phase1_score (predict mode upstream)
if "_phase1_score" not in df_ml.columns:
    print("Scoring full dataset with Phase 1 model...")
    X_full = df_ml[features_final].copy()
    for c in X_full.columns:
        X_full[c] = pd.to_numeric(X_full[c], errors="coerce").fillna(0)
    df_ml["_phase1_score"] = final_model.predict(X_full)
    print(f"  Phase 1 scores computed: range [{df_ml['_phase1_score'].min():.4f}, {df_ml['_phase1_score'].max():.4f}]")
else:
    print(f"  Using pre-computed _phase1_score: range [{df_ml['_phase1_score'].min():.4f}, {df_ml['_phase1_score'].max():.4f}]")

phase2_features = features_final + ["_phase1_score"]
print(f"  Phase 2 feature count: {len(phase2_features)} (Phase 1 features + phase1_score)")


# ## 5 — Prepare AML-Only Dataset
# Filter to AML transactions and assign typology labels. Each transaction has exactly one typology.
# 

# In[ ]:


if RUN_MODE == "train":
    print("=" * 90)
    print("PHASE 2: TYPOLOGY CLASSIFICATION")
    print("=" * 90)

    # ── Step 1: Prepare typology labels ──
    print("\n── Step 1: Prepare Typology Labels ──")
    typ_col = "aml_typology"

    aml_labeled = df_ml[(df_ml["is_aml"] == 1)].copy()

    # Clean typology labels
    aml_labeled["_clean_typ"] = aml_labeled[typ_col].astype(str).apply(
        lambda x: x.strip() if x and x not in ("nan", "", "None") else ""
    )
    aml_labeled = aml_labeled[aml_labeled["_clean_typ"] != ""]

    # Verify single typology per transaction
    multi_check = aml_labeled["_clean_typ"].str.contains(";", na=False).sum()
    print(f"  Multi-typology transactions: {multi_check} (should be 0)")

    if multi_check > 0:
        print("  WARNING: Multi-typology found — using rarest-first assignment")
        typ_global = Counter()
        for t in aml_labeled["_clean_typ"]:
            for p in str(t).split("; "):
                if p.strip(): typ_global[p.strip()] += 1
        aml_labeled["_clean_typ"] = aml_labeled["_clean_typ"].apply(
            lambda x: min([p.strip() for p in x.split("; ") if p.strip()],
                          key=lambda p: typ_global.get(p, 999999)) if ";" in str(x) else x
        )

    typology_classes = sorted(aml_labeled["_clean_typ"].unique())
    typ_to_idx = {t: i for i, t in enumerate(typology_classes)}
    idx_to_typ = {i: t for t, i in typ_to_idx.items()}
    aml_labeled["_typ_label"] = aml_labeled["_clean_typ"].map(typ_to_idx)
    n_classes = len(typology_classes)

    print(f"\n  AML transactions with labels: {len(aml_labeled):,}")
    print(f"  Typology classes: {n_classes}")
    print(f"\n  {'Idx':<4s} {'Typology':<45s} {'Count':>8s} {'%':>7s}")
    print(f"  {'─'*68}")
    for t, i in typ_to_idx.items():
        cnt = (aml_labeled["_typ_label"] == i).sum()
        print(f"  {i:<4d} {t:<45s} {cnt:>8,} {cnt/len(aml_labeled)*100:>6.1f}%")



# ## 6 — Stratified Train/Test Split
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

if RUN_MODE == "train":
    print("\n── Step 2: Stratified Split ──")

    X_aml = aml_labeled[phase2_features].copy()
    for c in X_aml.columns:
        X_aml[c] = pd.to_numeric(X_aml[c], errors="coerce").fillna(0)
    y_aml = aml_labeled["_typ_label"].astype(int)

    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X_aml, y_aml, test_size=0.20, random_state=42, stratify=y_aml
    )

    total_p2 = len(y_train_2)

    # Base class weights (inverse frequency + median boost)
    class_weights_p2 = {}
    median_count = np.median([(y_train_2 == i).sum() for i in range(n_classes)])
    for i in range(n_classes):
        cnt = max((y_train_2 == i).sum(), 1)
        base_weight = total_p2 / (n_classes * cnt)
        boost = min(2.5, median_count / cnt) if cnt < median_count else 1.0
        class_weights_p2[i] = base_weight * boost

    sample_weights_p2 = np.array([class_weights_p2[y] for y in y_train_2])

    print(f"  Train: {len(X_train_2):,} | Test: {len(X_test_2):,}")
    print("  " + "-" * 70)
    print("  Typology                                       Train    Test  Weight")
    print("  " + "-" * 70)
    for i, t in idx_to_typ.items():
        cnt = (y_train_2 == i).sum()
        boost_flag = " BOOSTED" if cnt < median_count else ""
        print(f"  {t:<45s} {(y_train_2==i).sum():>7,} {(y_test_2==i).sum():>7,} {class_weights_p2[i]:>7.4f}{boost_flag}")

    # =====================================================================
    # CV-BASED WEAK CLASS DETECTION
    # Uses ONLY train set via out-of-fold predictions.
    # Test set is never touched here — no leakage.
    # =====================================================================
    print("\n── CV: Detecting Weak Classes (train-only OOF) ──")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import recall_score as _recall_score

    cv_probe = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_train_2), dtype=int)

    # Lightweight probe — just enough to detect weak classes reliably
    probe_params = {
        "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
        "num_leaves": 63, "max_depth": 8, "min_child_samples": 10,
        "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 0.5,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": -1,
        "random_state": 42, "n_jobs": -1, "is_unbalance": True
    }

    for fold, (tr_idx, val_idx) in enumerate(cv_probe.split(X_train_2, y_train_2)):
        print(f"  Fold {fold+1}/5...", end=" ", flush=True)
        X_tr  = X_train_2.iloc[tr_idx];  X_val = X_train_2.iloc[val_idx]
        y_tr  = y_train_2.iloc[tr_idx];  y_val = y_train_2.iloc[val_idx]
        w_tr  = sample_weights_p2[tr_idx]

        ds_cv  = lgb.Dataset(X_tr,  label=y_tr,  weight=w_tr)
        val_cv = lgb.Dataset(X_val, label=y_val, reference=ds_cv)

        mdl_cv = lgb.train(probe_params, ds_cv, num_boost_round=300,
                           valid_sets=[val_cv],
                           callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof_preds[val_idx] = mdl_cv.predict(X_val).argmax(axis=1)
        print("done")

    # Recall computed on OOF predictions — test set is NEVER used here
    oof_recall  = _recall_score(y_train_2, oof_preds, average=None)
    mean_recall = oof_recall.mean()

    print(f"\n  Mean OOF Recall: {mean_recall:.4f}")
    print("  " + "-" * 65)
    print("  Typology                                       OOF Recall   Boost")
    print("  " + "-" * 65)
    for i in range(n_classes):
        recall = oof_recall[i]
        if recall < mean_recall:
            # Boost inversely proportional to weakness, capped at 3x
            boost = min(3.0, mean_recall / (recall + 1e-6))
            class_weights_p2[i] *= boost
            flag = " BOOSTED"
        else:
            boost = 1.0
            flag  = ""
        print(f"  {idx_to_typ[i]:<45s} {recall:>10.4f} {boost:>7.4f}{flag}")

    # Recompute sample weights with dynamically adjusted class weights
    sample_weights_p2 = np.array([class_weights_p2[y] for y in y_train_2])
    print("\n  Sample weights updated based on OOF recall (test set untouched)")



# ## 7 — Typology-Discriminating Interaction Features
# Features specifically designed to separate typologies from each other.
# 

# In[ ]:


if RUN_MODE == "train":
    print("\n── Step 2a: Phase 2 Interaction Features ──")
    interaction_count = 0

    # 1. Cash × amount — separates Structuring (cash 8K-10K)
    if "cash_flag_enc" in X_train_2.columns and "transaction_amount" in X_train_2.columns:
        X_train_2["_p2_cash_x_amt"] = X_train_2["cash_flag_enc"] * X_train_2["transaction_amount"]
        X_test_2["_p2_cash_x_amt"] = X_test_2["cash_flag_enc"] * X_test_2["transaction_amount"]
        phase2_features.append("_p2_cash_x_amt")
        interaction_count += 1

    # 2. Cross-border × amount — separates Corridor and Hawala
    if "ip_flag_cross_border" in X_train_2.columns:
        X_train_2["_p2_xborder_amt"] = X_train_2["ip_flag_cross_border"] * X_train_2["transaction_amount"]
        X_test_2["_p2_xborder_amt"] = X_test_2["ip_flag_cross_border"] * X_test_2["transaction_amount"]
        phase2_features.append("_p2_xborder_amt")
        interaction_count += 1

    # 3. Rule count — typologies have different rule trigger patterns
    rule_cols_p2 = [c for c in X_train_2.columns if c.startswith("rule_") and c not in {"rule_score"}]
    if rule_cols_p2:
        X_train_2["_p2_rule_count"] = X_train_2[rule_cols_p2].sum(axis=1)
        X_test_2["_p2_rule_count"] = X_test_2[rule_cols_p2].sum(axis=1)
        phase2_features.append("_p2_rule_count")
        interaction_count += 1

    # 4. Counterparties × amount — Funnel/Mule (many) vs Corridor (few)
    cp_col = next((c for c in X_train_2.columns if "unique_counterparties" in c), None)
    if cp_col:
        X_train_2["_p2_cp_x_amt"] = X_train_2[cp_col] * np.log1p(X_train_2["transaction_amount"])
        X_test_2["_p2_cp_x_amt"] = X_test_2[cp_col] * np.log1p(X_test_2["transaction_amount"])
        phase2_features.append("_p2_cp_x_amt")
        interaction_count += 1

    # 5. Velocity × balance drain — Pass-Through (instant forward)
    if "sender_pct_balance_moved" in X_train_2.columns and "sender_acct_txn_count_24h" in X_train_2.columns:
        X_train_2["_p2_drain_speed"] = X_train_2["sender_pct_balance_moved"] * X_train_2["sender_acct_txn_count_24h"]
        X_test_2["_p2_drain_speed"] = X_test_2["sender_pct_balance_moved"] * X_test_2["sender_acct_txn_count_24h"]
        phase2_features.append("_p2_drain_speed")
        interaction_count += 1

    # 6. Inflow/outflow ratio — Funnel (all inflow) vs Circular (balanced)
    if "sender_acct_inflow_amt_24h" in X_train_2.columns and "sender_acct_outflow_amt_24h" in X_train_2.columns:
        X_train_2["_p2_io_ratio"] = X_train_2["sender_acct_inflow_amt_24h"] / np.clip(X_train_2["sender_acct_outflow_amt_24h"], 1, None)
        X_test_2["_p2_io_ratio"] = X_test_2["sender_acct_inflow_amt_24h"] / np.clip(X_test_2["sender_acct_outflow_amt_24h"], 1, None)
        phase2_features.append("_p2_io_ratio")
        interaction_count += 1

    # 7. Amount proximity to 10K — Structuring signal
    if "transaction_amount" in X_train_2.columns:
        X_train_2["_p2_near_threshold"] = np.abs(X_train_2["transaction_amount"] - 10000) / 10000
        X_test_2["_p2_near_threshold"] = np.abs(X_test_2["transaction_amount"] - 10000) / 10000
        phase2_features.append("_p2_near_threshold")
        interaction_count += 1

    # 8. Country risk × amount
    hr_col = next((c for c in X_train_2.columns if "negative_list_country" in c), None)
    if hr_col:
        X_train_2["_p2_country_x_amt"] = X_train_2[hr_col] * X_train_2["transaction_amount"]
        X_test_2["_p2_country_x_amt"] = X_test_2[hr_col] * X_test_2["transaction_amount"]
        phase2_features.append("_p2_country_x_amt")
        interaction_count += 1

    phase2_features = list(dict.fromkeys(phase2_features))
    print(f"  Interaction features added: {interaction_count}")
    print(f"  Total Phase 2 features: {len(phase2_features)}")


    phase2_features = features_final + ["_phase1_score"]
    print(f"  Total Phase 2 features: {len(phase2_features)}")


# ## 8 — Phase 2 Feature Selection
# Remove features that don't help distinguish between typologies.
# 

# In[ ]:


if RUN_MODE == "train":
    print("\n── Step 2b: Phase 2 Feature Selection ──")

    # Train quick model for importance
    quick_params = {
        "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
        "num_leaves": 63, "max_depth": 8, "learning_rate": 0.05,
        "verbosity": -1, "random_state": 42, "n_jobs": -1
    }
    quick_ds = lgb.Dataset(X_train_2[phase2_features], label=y_train_2, weight=sample_weights_p2)
    quick_model = lgb.train(quick_params, quick_ds, num_boost_round=300,
                             callbacks=[lgb.log_evaluation(0)])

    p2_importance = pd.DataFrame({
        "feature": phase2_features,
        "gain": quick_model.feature_importance(importance_type="gain"),
        "split": quick_model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    # Remove zero-importance features
    zero_imp = p2_importance[p2_importance["gain"] == 0]["feature"].tolist()
    selected_p2_features = p2_importance[p2_importance["gain"] > 0]["feature"].tolist()

    print(f"  Total features: {len(phase2_features)}")
    print(f"  Zero importance (removed): {len(zero_imp)}")
    print(f"  Selected: {len(selected_p2_features)}")

    print(f"\n  Top 30 features for TYPOLOGY CLASSIFICATION:")
    print(f"  {'Rank':<5s} {'Feature':<55s} {'Gain':>12s}")
    print(f"  {'─'*75}")
    for i, (_, row) in enumerate(p2_importance.head(30).iterrows(), 1):
        print(f"  {i:<5d} {row['feature']:<55s} {row['gain']:>12.1f}")

    if zero_imp:
        print(f"\n  Removed (zero importance): {zero_imp[:10]}{'...' if len(zero_imp)>10 else ''}")

    # Apply selection
    X_train_2 = X_train_2[selected_p2_features]
    X_test_2 = X_test_2[selected_p2_features]
    phase2_features = selected_p2_features
    print(f"\n  Final Phase 2 features: {len(phase2_features)}")



# ## 9 — Hyperparameter Tuning
# 

# In[ ]:


from sklearn.metrics import accuracy_score, f1_score
if RUN_MODE == "train":
    print("\n── Step 3: Phase 2 Hyperparameter Tuning ──")

    # mc=5/8 allows rare classes to split — critical for small typologies
    p2_configs = [
        {"name": "Baseline",      "nl": 63,  "md": 8,  "mc": 20, "lr": 0.05, "ra": 0.0, "rl": 0.0},
        {"name": "Deep+Reg",      "nl": 127, "md": 10, "mc": 8,  "lr": 0.03, "ra": 0.1, "rl": 1.0},
        {"name": "Wide+Shallow",  "nl": 255, "md": 6,  "mc": 15, "lr": 0.03, "ra": 0.1, "rl": 0.5},
        #{"name": "VDeep+VeryReg", "nl": 300, "md": 15, "mc": 5,  "lr": 0.05, "ra": 0.2, "rl": 2.0},
        #{"name": "FocusWeak",     "nl": 200, "md": 10, "mc": 5,  "lr": 0.03, "ra": 0.1, "rl": 0.5},
        #{"name": "LowLR+Deep",    "nl": 300, "md": 12, "mc": 8,  "lr": 0.01, "ra": 0.1, "rl": 1.0},
    ]

    print("  Config                  |  Accuracy  MacroF1   WtdF1 |  Rounds")
    print("  " + "-" * 65)

    from concurrent.futures import ThreadPoolExecutor
    # ThreadPoolExecutor works on Windows — LightGBM releases the GIL during training.
    # ProcessPoolExecutor causes OSError on Windows due to lgb model pickle issues.

    def _train_one(cfg):
        params = {
            "objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss",
            "num_leaves": cfg["nl"], "max_depth": cfg["md"], "min_child_samples": cfg["mc"],
            "learning_rate": cfg["lr"], "reg_alpha": cfg["ra"], "reg_lambda": cfg["rl"],
            "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": -1,
            "random_state": 42, "n_jobs": 2,
            "is_unbalance": True
        }
        ds2  = lgb.Dataset(X_train_2, label=y_train_2, weight=sample_weights_p2)
        val2 = lgb.Dataset(X_test_2,  label=y_test_2,  reference=ds2)
        mdl  = lgb.train(params, ds2, num_boost_round=500,
                         valid_sets=[val2],
                         callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        proba    = mdl.predict(X_test_2)
        pred     = proba.argmax(axis=1)
        acc      = accuracy_score(y_test_2, pred)
        macro_f1 = f1_score(y_test_2, pred, average="macro")
        wtd_f1   = f1_score(y_test_2, pred, average="weighted")
        rounds   = mdl.best_iteration if hasattr(mdl, "best_iteration") else 0
        return cfg["name"], mdl, proba, pred, acc, macro_f1, wtd_f1, rounds, cfg

    p2_results = {}
    with ThreadPoolExecutor(max_workers=P2_TUNING["max_workers"]) as ex:
        futures = [ex.submit(_train_one, cfg) for cfg in p2_configs]
        for fut in futures:
            name, mdl, proba, pred, acc, macro_f1, wtd_f1, rounds, cfg = fut.result()
            p2_results[name] = {
                "accuracy": acc, "macro_f1": macro_f1, "wtd_f1": wtd_f1,
                "model": mdl, "proba": proba, "pred": pred,
                "rounds": rounds, "config": cfg,
                "acc": acc, "mf1": macro_f1, "wf1": wtd_f1,
            }
            print(f"  {name:<22s} | {acc:>8.4f} {macro_f1:>7.4f} {wtd_f1:>6.4f} | {rounds:>7}")

    # Best = balances overall accuracy AND macro F1 (macro penalises weak classes equally)
    best_p2   = max(p2_results, key=lambda k: 0.5*p2_results[k]["accuracy"] + 0.5*p2_results[k]["macro_f1"])
    model_p2  = p2_results[best_p2]["model"]
    y_proba_2 = p2_results[best_p2]["proba"]
    y_pred_2  = p2_results[best_p2]["pred"]
    accuracy  = p2_results[best_p2]["accuracy"]

    best_mf1 = p2_results[best_p2]["macro_f1"]
    print(f"\n  Best: {best_p2} (Accuracy={accuracy:.4f}, MacroF1={best_mf1:.4f})")

    # ── Post-processing: OOF-guided threshold calibration ──
    # Lowers decision boundary for classes the model is biased against.
    # Uses oof_recall from the CV step above — train-set derived only, no test leakage.
    print("\n── Threshold Calibration (OOF-guided, train-only) ──")

    THRESHOLDS = {i: 0.5 for i in range(n_classes)}
    for i in range(n_classes):
        recall = oof_recall[i]
        if recall < mean_recall:
            # Lower threshold proportionally to weakness; floor at 0.20
            THRESHOLDS[i] = max(0.20, 0.5 * (recall / mean_recall))

    def predict_with_thresholds(proba, thresholds):
        # Amplify columns with lower thresholds before argmax
        # so the model is more willing to predict weak classes
        adjusted = proba.copy()
        for cls_idx, thresh in thresholds.items():
            adjusted[:, cls_idx] = proba[:, cls_idx] / thresh
        return adjusted.argmax(axis=1)

    y_pred_2 = predict_with_thresholds(y_proba_2, THRESHOLDS)

    print("  " + "-" * 68)
    print("  Typology                                       OOF Recall  Threshold")
    print("  " + "-" * 68)
    for i in range(n_classes):
        flag = " ADJUSTED" if THRESHOLDS[i] < 0.5 else ""
        print(f"  {idx_to_typ[i]:<45s} {oof_recall[i]:>10.4f} {THRESHOLDS[i]:>10.4f}{flag}")

    adj_acc = accuracy_score(y_test_2, y_pred_2)
    adj_mf1 = f1_score(y_test_2, y_pred_2, average="macro")
    print(f"\n  After calibration  -> Accuracy: {adj_acc:.4f} | MacroF1: {adj_mf1:.4f}")
    print(f"  Before calibration -> Accuracy: {accuracy:.4f} | MacroF1: {best_mf1:.4f}")

    # Use calibrated predictions going forward
    accuracy = adj_acc
    y_pred_2 = y_pred_2



# In[ ]:





# ## 10 — Detailed Evaluation
# 

# In[ ]:


from sklearn.metrics import classification_report
if RUN_MODE == "train":
    print("\n── Step 4: Phase 2 Evaluation (Multi-Label Threshold) ──")

    # ═══ Multi-label threshold: flag all typologies above this probability ═══
    # TYPOLOGY_THRESHOLD from config/settings.yaml (setup cell)
    print(f"  Typology threshold: {TYPOLOGY_THRESHOLD} (flag all typologies with prob >= {TYPOLOGY_THRESHOLD})")

    # For each test transaction, get all typologies above threshold
    y_pred_multi = []
    y_pred_primary = []
    y_true_labels = y_test_2.values

    for i in range(len(y_proba_2)):
        probs = y_proba_2[i]
        primary = probs.argmax()
        y_pred_primary.append(primary)

        above_thresh = set(j for j in range(n_classes) if probs[j] >= TYPOLOGY_THRESHOLD)
        if not above_thresh:
            above_thresh = {primary}
        y_pred_multi.append(above_thresh)

    # ═══ Recall: correct if TRUE typology is ANYWHERE in predicted set ═══
    correct_multi = sum(1 for i in range(len(y_true_labels)) if y_true_labels[i] in y_pred_multi[i])
    correct_primary = sum(1 for i in range(len(y_true_labels)) if y_true_labels[i] == y_pred_primary[i])

    accuracy_primary = correct_primary / len(y_true_labels)
    accuracy_multi = correct_multi / len(y_true_labels)

    avg_flagged = np.mean([len(s) for s in y_pred_multi])
    multi_label_pct = sum(1 for s in y_pred_multi if len(s) > 1) / len(y_pred_multi) * 100

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  PRIMARY ONLY (highest prob):     {accuracy_primary:.4f} ({accuracy_primary*100:.1f}%)          │")
    print(f"  │  MULTI-LABEL (threshold={TYPOLOGY_THRESHOLD}):  {accuracy_multi:.4f} ({accuracy_multi*100:.1f}%)          │")
    print(f"  │  Improvement:                     +{(accuracy_multi-accuracy_primary)*100:.1f}%                     │")
    print(f"  │  Avg typologies flagged/txn:       {avg_flagged:.2f}                      │")
    print(f"  │  Transactions with 2+ typologies: {multi_label_pct:.1f}%                     │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ═══ Per-typology recall comparison ═══
    print(f"\n  ── Per-Typology Recall: Primary vs Multi-Label ──")
    print(f"  {'Typology':<40s} │ {'N':>6s} {'Primary':>8s} {'Multi':>8s} {'Gain':>7s} │ {'AvgConf':>8s} {'Avg#':>5s}")
    print(f"  {'─'*95}")

    for i, typ in idx_to_typ.items():
        mask = y_test_2 == i
        cnt = mask.sum()
        if cnt == 0: continue

        correct_p = sum(1 for j in range(len(y_true_labels)) if mask.iloc[j] and y_pred_primary[j] == i)
        recall_p = correct_p / cnt * 100

        correct_m = sum(1 for j in range(len(y_true_labels)) if mask.iloc[j] and i in y_pred_multi[j])
        recall_m = correct_m / cnt * 100

        gain = recall_m - recall_p
        avg_conf = np.mean([y_proba_2[j][i] for j in range(len(y_true_labels)) if mask.iloc[j]])
        avg_n = np.mean([len(y_pred_multi[j]) for j in range(len(y_true_labels)) if mask.iloc[j]])

        gain_str = f"+{gain:.1f}%" if gain > 0 else f"{gain:.1f}%"
        status = "✓" if recall_m > 80 else ("⚡" if recall_m > 50 else "⚠")
        print(f"  {typ:<40s} │ {cnt:>6,} {recall_p:>7.1f}% {recall_m:>7.1f}% {gain_str:>7s} │ {avg_conf:>7.3f} {avg_n:>5.2f} {status}")

    # ═══ Threshold sensitivity analysis ═══
    print(f"\n  ── Threshold Sensitivity ──")
    print(f"  {'Threshold':>10s} │ {'Recall':>8s} {'Avg#':>6s} {'2+ Labels':>10s} │ {'vs Primary':>11s}")
    print(f"  {'─'*60}")

    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        preds = []
        for j in range(len(y_proba_2)):
            above = set(k for k in range(n_classes) if y_proba_2[j][k] >= thresh)
            if not above: above = {y_proba_2[j].argmax()}
            preds.append(above)

        recall_t = sum(1 for j in range(len(y_true_labels)) if y_true_labels[j] in preds[j]) / len(y_true_labels) * 100
        avg_n_t = np.mean([len(s) for s in preds])
        multi_pct_t = sum(1 for s in preds if len(s) > 1) / len(preds) * 100

        marker = " ◄" if thresh == TYPOLOGY_THRESHOLD else ""
        print(f"  {thresh:>10.2f} │ {recall_t:>7.1f}% {avg_n_t:>5.2f} {multi_pct_t:>9.1f}% │ {recall_t - accuracy_primary*100:>+10.1f}%{marker}")

    # ═══ Classification report (primary only — reference) ═══
    print(f"\n  Classification Report (primary prediction):")
    print(classification_report(y_test_2, y_pred_primary, target_names=typology_classes, digits=4))

    # ═══ Plots ═══
    #fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Confusion matrix
    cm = confusion_matrix(y_test_2, y_pred_primary)
    short_names = [t[:15] for t in typology_classes]
    # sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=axes[0],
    #             xticklabels=short_names, yticklabels=short_names)
    # axes[0].set_title("Confusion Matrix (Primary)", fontweight="bold")
    # axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # 2. Per-typology recall comparison
    typ_names_short = [idx_to_typ[i][:18] for i in range(n_classes)]
    recall_primary_list = []
    recall_multi_list = []
    for i in range(n_classes):
        mask = y_test_2 == i; cnt = mask.sum()
        if cnt == 0: recall_primary_list.append(0); recall_multi_list.append(0); continue
        rp = sum(1 for j in range(len(y_true_labels)) if mask.iloc[j] and y_pred_primary[j] == i) / cnt * 100
        rm = sum(1 for j in range(len(y_true_labels)) if mask.iloc[j] and i in y_pred_multi[j]) / cnt * 100
        recall_primary_list.append(rp); recall_multi_list.append(rm)

    # x_pos = np.arange(n_classes)
    # axes[1].barh(x_pos - 0.2, recall_primary_list, 0.35, label="Primary Only", color="#3B82F6", alpha=0.8)
    # axes[1].barh(x_pos + 0.2, recall_multi_list, 0.35, label=f"Multi-Label (≥{TYPOLOGY_THRESHOLD})", color="#10B981", alpha=0.8)
    # axes[1].set_yticks(x_pos); axes[1].set_yticklabels(typ_names_short, fontsize=7)
    # axes[1].set_xlim(0, 105); axes[1].axvline(80, color="gray", linestyle="--", alpha=0.3)
    # axes[1].set_title("Recall: Primary vs Multi-Label", fontweight="bold")
    # axes[1].legend(fontsize=8)

    # 3. Threshold sensitivity curve
    # thresholds_plot = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    # recalls_plot = []
    # avg_n_plot = []
    # for thresh in thresholds_plot:
    #     preds = []
    #     for j in range(len(y_proba_2)):
    #         above = set(k for k in range(n_classes) if y_proba_2[j][k] >= thresh)
    #         if not above: above = {y_proba_2[j].argmax()}
    #         preds.append(above)
    #     recalls_plot.append(sum(1 for j in range(len(y_true_labels)) if y_true_labels[j] in preds[j]) / len(y_true_labels) * 100)
    #     avg_n_plot.append(np.mean([len(s) for s in preds]))

    # ax3a = axes[2]
    # ax3b = ax3a.twinx()
    # ax3a.plot(thresholds_plot, recalls_plot, "b-o", lw=2, markersize=4, label="Recall %")
    # ax3b.plot(thresholds_plot, avg_n_plot, "r--s", lw=1.5, markersize=4, label="Avg # flagged")
    # ax3a.axvline(TYPOLOGY_THRESHOLD, color="green", linestyle=":", alpha=0.7, label=f"Selected ({TYPOLOGY_THRESHOLD})")
    # ax3a.axhline(accuracy_primary * 100, color="gray", linestyle="--", alpha=0.3, label="Primary-only recall")
    # ax3a.set_xlabel("Threshold"); ax3a.set_ylabel("Recall %", color="blue"); ax3b.set_ylabel("Avg # Typologies", color="red")
    # ax3a.set_title("Threshold vs Recall vs # Flagged", fontweight="bold")
    # lines1, labels1 = ax3a.get_legend_handles_labels()
    # lines2, labels2 = ax3b.get_legend_handles_labels()
    # ax3a.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "phase2_evaluation.png"), bbox_inches="tight")
    # plt.show()


# ## 11 — Build Production Output Tables
# 

# In[ ]:


# ════════════════════════════════════════════════════════════════
# INFERENCE — Phase 2 model loading (only in predict mode)
# Populates: model_p2, idx_to_typ, typ_to_idx, n_classes, typology_classes,
#            phase2_features, TYPOLOGY_THRESHOLD
# ════════════════════════════════════════════════════════════════
if RUN_MODE == "predict":
    import joblib
    print("=" * 70)
    print("Phase 2 INFERENCE — loading saved model bundle")
    print("=" * 70)

    from db_utils import load_model

    # Load the Phase 2 model bundle from PostgreSQL (model_registry table).
    # run_id=None -> the most recently saved bundle.
    p2_bundle = load_model("phase2_model_bundle")

    model_p2           = p2_bundle["model"]
    phase2_features    = p2_bundle["features"]
    typ_to_idx         = p2_bundle["typ_to_idx"]
    idx_to_typ         = {int(k): v for k, v in p2_bundle["idx_to_typ"].items()}
    typology_classes   = p2_bundle.get("typology_classes", sorted(typ_to_idx.keys()))
    n_classes          = len(typology_classes)
    TYPOLOGY_THRESHOLD = p2_bundle.get("typology_threshold", 0.30)

    print(f"  Model:     {type(model_p2).__name__}")
    print(f"  Features:  {len(phase2_features)}")
    print(f"  Classes:   {n_classes}")
    print(f"  Threshold: {TYPOLOGY_THRESHOLD}")

    # Ensure all expected phase2 features exist in df_ml (compute interaction
    # features that the training loop adds to X_train_2/X_test_2 — replicate
    # them here on the full df_ml)
    if "_p2_cash_x_amt" in phase2_features and "_p2_cash_x_amt" not in df_ml.columns:
        if "cash_flag_enc" in df_ml.columns and "transaction_amount" in df_ml.columns:
            df_ml["_p2_cash_x_amt"] = df_ml["cash_flag_enc"] * df_ml["transaction_amount"]
        else:
            df_ml["_p2_cash_x_amt"] = 0
    if "_p2_xborder_amt" in phase2_features and "_p2_xborder_amt" not in df_ml.columns:
        if "ip_flag_cross_border" in df_ml.columns:
            df_ml["_p2_xborder_amt"] = df_ml["ip_flag_cross_border"] * df_ml["transaction_amount"]
        else:
            df_ml["_p2_xborder_amt"] = 0
    if "_p2_rule_count" in phase2_features and "_p2_rule_count" not in df_ml.columns:
        rule_cols_inf = [c for c in df_ml.columns if c.startswith("rule_") and c != "rule_score"]
        df_ml["_p2_rule_count"] = df_ml[rule_cols_inf].sum(axis=1) if rule_cols_inf else 0
    cp_col_inf = next((c for c in df_ml.columns if "unique_counterparties" in c), None)
    if "_p2_cp_x_amt" in phase2_features and "_p2_cp_x_amt" not in df_ml.columns:
        if cp_col_inf:
            df_ml["_p2_cp_x_amt"] = df_ml[cp_col_inf] * np.log1p(df_ml["transaction_amount"])
        else:
            df_ml["_p2_cp_x_amt"] = 0
    if "_p2_velo_drain" in phase2_features and "_p2_velo_drain" not in df_ml.columns:
        if "sender_pct_balance_moved" in df_ml.columns and "sender_velocity_score" in df_ml.columns:
            df_ml["_p2_velo_drain"] = df_ml["sender_pct_balance_moved"] * df_ml["sender_velocity_score"]
        else:
            df_ml["_p2_velo_drain"] = 0
    if "_p2_io_ratio" in phase2_features and "_p2_io_ratio" not in df_ml.columns:
        for c in ["receiver_total_in", "receiver_total_out"]:
            if c not in df_ml.columns: df_ml[c] = 0
        df_ml["_p2_io_ratio"] = df_ml["receiver_total_in"] / (df_ml["receiver_total_out"] + 1)
    if "_p2_struct_amt" in phase2_features and "_p2_struct_amt" not in df_ml.columns:
        df_ml["_p2_struct_amt"] = ((df_ml["transaction_amount"] >= 8000) & (df_ml["transaction_amount"] <= 10000)).astype(int)
    if "_p2_country_x_amt" in phase2_features and "_p2_country_x_amt" not in df_ml.columns:
        if "counterparty_country_risk" in df_ml.columns:
            df_ml["_p2_country_x_amt"] = df_ml["counterparty_country_risk"] * df_ml["transaction_amount"]
        else:
            df_ml["_p2_country_x_amt"] = 0
    print("  Interaction features materialised onto df_ml")

# Compatibility variables for the production-output cell that follows
# (it expects model_p2, idx_to_typ, n_classes, typology_classes, TYPOLOGY_THRESHOLD)


# In[ ]:


print("\n── Step 5: Build Production Output ──")

import time
_t_start = time.time()
def _tlog(msg):
    print(f"  [{time.time()-_t_start:>6.1f}s] {msg}")

# TYPOLOGY_THRESHOLD from config/settings.yaml (setup cell)

# Score full dataset Phase 1
fraud_risk_score = (df_ml["_phase1_score"] * 100).values  # numpy from the start
predicted_aml    = (df_ml["_phase1_score"] >= best_thresh).astype(int).values

# Rule metrics
rule_cols = [c for c in df_ml.columns
             if c.startswith("rule_")
             and c not in {"rule_score", "rules_triggered", "rules_triggered_count"}]

# Build rule matrix ONCE as numpy — reused everywhere below
if rule_cols:
    # rule_matrix_np = df_ml[rule_cols].fillna(0).to_numpy(dtype=np.int8)
    rule_matrix_np = (
    df_ml[rule_cols]
    .replace(['', ' ', None], 0)
    .apply(pd.to_numeric, errors='coerce')
    .fillna(0)
    .astype(np.int16)   # safer than int8
    .to_numpy())
    
    rule_trigger_count_np = rule_matrix_np.sum(axis=1).astype(np.int32)
else:
    rule_matrix_np = np.zeros((len(df_ml), 0), dtype=np.int8)
    rule_trigger_count_np = np.zeros(len(df_ml), dtype=np.int32)

rule_score_vals_np = pd.to_numeric(df_ml.get("rule_score", 0), errors="coerce").fillna(0).to_numpy()

_tlog("rule matrix built")

# Column resolution
txn_id_col = next((c for c in ["transaction_id"] if c in df_ml.columns), None)
cif_col    = next((c for c in ["customer_cif_id", "customer_account_number"] if c in df_ml.columns), None)
amt_col    = next((c for c in ["transaction_amount"] if c in df_ml.columns), None)

# Alert source — vectorized
has_rules = rule_trigger_count_np > 0
has_ml    = predicted_aml == 1
alert_source = np.select(
    [has_rules & has_ml, has_rules & ~has_ml, ~has_rules & has_ml],
    ["Rule + ML Confirmed", "Rule Triggered", "ML Behavioural Alert"],
    default="Normal"
)

# Phase 1 Output
phase1_output = pd.DataFrame({
    "transaction_id":     df_ml[txn_id_col].values if txn_id_col else np.arange(len(df_ml)),
    "customer_id":        df_ml[cif_col].values    if cif_col    else "",
    "amount":             df_ml[amt_col].values    if amt_col    else 0,
    "fraud_risk_score":   np.round(fraud_risk_score, 1),
    "rule_trigger_count": rule_trigger_count_np,
    "alert_source":       alert_source,
})

_tlog("phase1_output built")


# ═══ Display name maps ═══
RULE_DISPLAY_NAMES = {
    "rule_structuring_pattern":         "Sub-threshold cash structuring detected",
    "rule_negative_list_country":       "FATF/sanctioned country involved",
    "rule_dormant_activation":          "Dormant account suddenly reactivated",
    "rule_rapid_burst":                 "Rapid burst of transactions detected",
    "rule_large_cash_individual":       "Large individual cash transaction",
    "rule_large_cash_business":         "Large business cash transaction",
    "rule_pep_large_any":               "PEP high-value transaction",
    "rule_off_hours_activity":          "Suspicious off-hours activity",
    "rule_vpn_emulator_detected":       "VPN/proxy/emulator detected",
    "rule_zero_balance_cycling":        "Zero-balance drain cycling",
    "rule_tax_haven_remit":             "Tax haven remittance",
    "rule_large_cr_dr_individual":      "Large credit followed by debit",
    "rule_large_cr_dr_business":        "Large business credit-debit pattern",
    "rule_multiple_parties_cash":       "Multiple parties cash deposit",
    "rule_remit_then_cash_75pct":       "Remittance then 75%+ cash withdrawal",
    "rule_credit_then_cash":            "Credit then cash withdrawal pattern",
    "rule_high_freq_foreign":           "High frequency foreign remittance",
    "rule_intra_high_freq":             "High frequency intrabank transfers",
    "rule_series_credits_7d":           "Series of credits in 7 days",
    "rule_series_debits_7d":            "Series of debits in 7 days",
    "rule_new_wallet_high_value":       "New wallet with high-value activity",
    "rule_short_lived_wallet":          "Short-lived wallet high throughput",
    "rule_shell_company":               "Shell company anomaly indicators",
    "rule_trust_large_cash":            "Trust/society large cash handling",
    "rule_trust_foreign_remit":         "Trust foreign remittance",
    "rule_offshore_entity":             "Offshore entity transfer",
    "rule_age_amount_mismatch":         "Age-amount mismatch (under 25)",
    "rule_low_income_large":            "Low income high-value transaction",
    "rule_student_high_value":          "Student high-value transaction",
    "rule_unemployed_large":            "Retired/unemployed large transfer",
    "rule_pep_high_velocity":           "PEP high velocity/circular pattern",
    "rule_minor_anomalous":             "Minor anomalous profile activity",
    "rule_monthly_limit_exhaustion":    "Monthly wallet limit exhaustion",
    "rule_device_hopping":              "Multiple devices in short period",
    "rule_impossible_travel":           "Impossible travel / location anomaly",
    "rule_auth_degrade_high_risk":      "Auth degradation + high risk profile",
    "rule_ppi_small_kyc_load_breach":   "Small KYC load limit breach",
    "rule_ppi_small_kyc_bal_breach":    "Small KYC balance limit breach",
    "rule_ppi_negative_list_device":    "Negative list device/IP detected",
    "rule_ppi_refund_abuse":            "Merchant refund abuse pattern",
    "rule_ppi_multi_wallet_kyc":        "Multiple wallets same PAN/Aadhaar",
    "rule_ppi_shared_ip_cluster":       "Shared IP wallet cluster",
    "rule_ppi_w2w_layering":            "Wallet-to-wallet layering pattern",
    "rule_ppi_high_risk_mcc":           "High-risk MCC concentration >70%",
    "rule_ppi_cluster_alert":           "Multi-signal wallet cluster alert",
    "rule_sole_prop_personal":          "Self-employed using savings account",
    "rule_attempted_failed":            "Failed/attempted transaction",
    "rule_unusual_type_spike":          "Unusual transaction type spike",
    "rule_repeated_counterparty_7d":    "Repeated counterparty in 7 days",
    "rule_round_amount_struct_7d":      "Round amount structuring pattern",
    "rule_multi_channel_24h":           "Multiple channels in 24 hours",
    "rule_rapid_load_transfer":         "Rapid load then transfer (PPI)",
    "rule_new_indiv_cash_30pct":        "New individual cash >30% income",
    "rule_dormant_75pct_drain":         "Dormant account 75% drain in 7 days",
    "rule_fx_cash_large":               "Foreign exchange cash large",
    "rule_cc_cash_1L":                  "Credit card cash advance ≥₹1L",
    "rule_tax_res_mismatch_xborder":    "Tax residency mismatch cross-border",
    "rule_geo_mismatch_2000km":         "Geo mismatch >2000km",
}

explanations_map = {
    "Charity Abuse":                 "Suspicious donation patterns with high-value transfers to trust/society accounts",
    "Circular Transaction Loop":     "Funds returning to originating account through circular transfer chain",
    "Funnel Account Network":        "Multiple inflows from different sources converging into single account",
    "High-Risk Corridor Transfer":   "Cross-border transfer to FATF high-risk jurisdiction",
    "Money Mule Network":            "Account receiving and forwarding funds with rapid turnover pattern",
    "Pass-Through Transit Hub":      "Account acting as transit point with immediate re-forwarding",
    "Rapid Multi-Hop Layering":      "Sequential rapid transfers across multiple accounts to obscure trail",
    "Structuring (Smurfing)":        "Transaction amounts structured below regulatory reporting thresholds",
    "Third-Party Payment Web":       "Complex web of transfers through multiple third-party intermediaries",
    "Underground Banking (Hawala)":  "Informal value transfer matching settlement patterns",
}

# Pre-compute display-name array aligned to rule_cols (for fast indexing)
rule_display_arr = np.array([
    RULE_DISPLAY_NAMES.get(r, r.replace("rule_", "").replace("_", " ").title())
    for r in rule_cols
])
rule_col_arr = np.array(rule_cols)


# ═══ Phase 2: Score predicted-AML rows ═══
N = len(df_ml)
pred_aml_mask  = predicted_aml == 1
pred_positions = np.where(pred_aml_mask)[0]
n_pred = len(pred_positions)
_tlog(f"predicted-AML rows: {n_pred:,}")

X_for_p2 = df_ml.loc[pred_aml_mask, phase2_features].copy()
for c in X_for_p2.columns:
    X_for_p2[c] = pd.to_numeric(X_for_p2[c], errors="coerce").fillna(0)

# Initialize output buffers (numpy where possible)
p2_primary_typology     = np.full(N, "None", dtype=object)
p2_all_typologies       = np.full(N, "None", dtype=object)
p2_primary_confidence   = np.zeros(N)
p2_num_matched          = np.zeros(N, dtype=np.int32)
p2_priority             = np.full(N, "Low", dtype=object)
p2_typology_explanation = np.full(N, "", dtype=object)
p2_rule_explanation     = np.full(N, "", dtype=object)
p2_rules_triggered_list = np.full(N, "", dtype=object)
p2_prob_matrix          = np.zeros((N, n_classes))


# ─── Build rule explanations — vectorized over triggered rows ───
if rule_cols:
    triggered_mask    = rule_trigger_count_np > 0
    triggered_indices = np.where(triggered_mask)[0]
    _tlog(f"building rule text for {len(triggered_indices):,} rows...")

    # Loop only over rows that actually have a rule fired.
    # Inside the loop, np.flatnonzero on a small fixed-width row is microsecond-cheap.
    for pos in triggered_indices:
        fired_idx = np.flatnonzero(rule_matrix_np[pos])  # column indices where flag = 1
        n_fired   = fired_idx.size

        p2_rules_triggered_list[pos] = "; ".join(rule_col_arr[fired_idx])

        if n_fired <= 3:
            p2_rule_explanation[pos] = " | ".join(rule_display_arr[fired_idx])
        else:
            top3 = " | ".join(rule_display_arr[fired_idx[:3]])
            p2_rule_explanation[pos] = f"{top3} | +{n_fired-3} more rules"

    _tlog("rule explanations done")


# ─── Phase 2 typology prediction — fully vectorized ───
if n_pred > 0:
    _tlog("running model_p2.predict...")
    proba = model_p2.predict(X_for_p2)
    if hasattr(proba, "values"):
        proba = proba.values
    proba = np.asarray(proba, dtype=float)
    _tlog(f"predict done, shape={proba.shape}")

    # Vectorized: primary typology + confidence
    primary_idx_arr  = proba.argmax(axis=1)
    primary_conf_arr = proba.max(axis=1)
    above_thresh     = proba >= TYPOLOGY_THRESHOLD            # (n_pred, n_classes) bool
    n_matched_arr    = above_thresh.sum(axis=1)
    # If nothing crosses threshold, fall back to the primary
    no_match = n_matched_arr == 0
    n_matched_arr[no_match] = 1

    # Pre-build per-typology static arrays (avoid dict lookups inside hot loop)
    typ_names_by_idx = np.array([idx_to_typ[k] for k in range(n_classes)], dtype=object)
    typ_expl_by_idx  = np.array(
        [explanations_map.get(idx_to_typ[k], f"Pattern flagged: {idx_to_typ[k]}") for k in range(n_classes)],
        dtype=object,
    )

    # Scatter probability matrix back to full-N shape (vectorized)
    p2_prob_matrix[pred_positions, :] = proba

    # Single tight loop over only the predicted-AML rows.
    # The inner work is now numpy-only and dictionary-free.
    _tlog("building per-row typology text...")
    for j in range(n_pred):
        pos          = pred_positions[j]
        primary_k    = primary_idx_arr[j]
        primary_conf = primary_conf_arr[j]

        if no_match[j]:
            # Single-typology fallback: just the argmax
            sorted_k    = np.array([primary_k])
            sorted_conf = np.array([primary_conf])
        else:
            # Indices crossing threshold, sorted by descending probability
            mask_row    = above_thresh[j]
            row_probs   = proba[j]
            cand_k      = np.flatnonzero(mask_row)
            order       = np.argsort(-row_probs[cand_k])
            sorted_k    = cand_k[order]
            sorted_conf = row_probs[sorted_k]

        names = typ_names_by_idx[sorted_k]
        n_m   = len(sorted_k)

        p2_primary_typology[pos]   = names[0]
        p2_primary_confidence[pos] = sorted_conf[0]
        p2_num_matched[pos]        = n_m

        # all_matched_typologies: "Name (95%); Name2 (40%)"
        # Building once with list comprehension is fastest for short n_m
        p2_all_typologies[pos] = "; ".join(
            f"{names[i]} ({int(round(sorted_conf[i]*100))}%)" for i in range(n_m)
        )

        # Typology explanation
        if n_m == 1:
            p2_typology_explanation[pos] = typ_expl_by_idx[sorted_k[0]]
        else:
            others = "; ".join(names[1:])
            p2_typology_explanation[pos] = f"{typ_expl_by_idx[sorted_k[0]]} | Also flagged: {others}"

    _tlog("typology text built")

    # ─── Priority — fully vectorized over pred rows ───
    src_pred       = alert_source[pred_positions]
    score_pred     = fraud_risk_score[pred_positions]
    is_rulml       = src_pred == "Rule + ML Confirmed"
    is_rule_only   = src_pred == "Rule Triggered"

    crit = is_rulml & (primary_conf_arr >= 0.50)
    high = ~crit & ((score_pred >= 70) | (primary_conf_arr >= 0.60))
    med  = ~crit & ~high & ((score_pred >= 40) | (primary_conf_arr >= 0.35) | is_rule_only)

    pri_arr = np.where(crit, "Critical",
              np.where(high, "High",
              np.where(med,  "Medium", "Low")))
    p2_priority[pred_positions] = pri_arr
    _tlog("priorities assigned")


# ─── Combine typology + rule explanations — vectorized ───
_tlog("combining final explanations...")
typ_arr  = p2_typology_explanation
rule_arr = p2_rule_explanation
has_typ  = typ_arr  != ""
has_rule = rule_arr != ""

# Default
final_explanation = np.full(N, "Transaction consistent with normal behaviour", dtype=object)

# AML + rules
m_both = pred_aml_mask & has_typ & has_rule
final_explanation[m_both] = np.char.add(np.char.add(typ_arr[m_both].astype(str), " || Rules: "),
                                        rule_arr[m_both].astype(str))

# AML only
m_aml_only = pred_aml_mask & has_typ & ~has_rule
final_explanation[m_aml_only] = typ_arr[m_aml_only]

# Rule only (not AML) — severity-aware narrative
# Replaces the old "Compliance rule triggered (score=X) — routine monitoring | ..."
# template. Phrasing now scales with how many rules fired so investigators
# can triage at a glance.
m_rule_only = ~pred_aml_mask & has_rule
if m_rule_only.any():
    n_rules_only   = rule_trigger_count_np[m_rule_only]
    rt             = rule_arr[m_rule_only].astype(str)
    lead_arr       = np.empty(n_rules_only.shape, dtype=object)

    # Severity bands by concurrent rule count
    lead_arr[n_rules_only == 1] = "Compliance rule flagged"
    band_2_3 = (n_rules_only >= 2) & (n_rules_only <= 3)
    if band_2_3.any():
        for k in np.where(band_2_3)[0]:
            lead_arr[k] = f"Multiple compliance rules flagged ({int(n_rules_only[k])})"
    band_4_6 = (n_rules_only >= 4) & (n_rules_only <= 6)
    if band_4_6.any():
        for k in np.where(band_4_6)[0]:
            lead_arr[k] = f"Elevated rule activity — {int(n_rules_only[k])} rules concurrently triggered"
    band_7p  = n_rules_only >= 7
    if band_7p.any():
        for k in np.where(band_7p)[0]:
            lead_arr[k] = f"High rule concentration — {int(n_rules_only[k])} rules concurrently triggered, indicates layered policy concern"

    final_explanation[m_rule_only] = np.array([
        f"{lead}: {rules}. Review against customer profile and recent activity."
        for lead, rules in zip(lead_arr, rt)
    ], dtype=object)

    # Priority scales with rule count for the rule-only branch
    rule_only_pos = np.where(m_rule_only)[0]
    rc = rule_trigger_count_np[rule_only_pos]
    # default Low; bump to Medium for 2+, High for 7+
    new_pri = np.where(rc >= 7, "High",
              np.where(rc >= 2, "Medium",
              "Medium"))   # single-rule rule-only rows stay Medium for now
    # Only override rows whose alert_source is "Rule Triggered" (the rule-only sub-branch)
    is_rule_only_src = alert_source[rule_only_pos] == "Rule Triggered"
    p2_priority[rule_only_pos[is_rule_only_src]] = new_pri[is_rule_only_src]

p2_explanation = final_explanation
_tlog("explanations combined")


# ═══ Build Phase 2 DataFrame ═══
phase2_output = pd.DataFrame({
    "transaction_id":         phase1_output["transaction_id"].values,
    "customer_id":            phase1_output["customer_id"].values,
    "predicted_typology":     p2_primary_typology,
    "all_matched_typologies": p2_all_typologies,
    "num_typologies_matched": p2_num_matched,
    "typology_confidence":    np.round(p2_primary_confidence, 4),
    "investigation_priority": p2_priority,
    "rules_triggered_count":  rule_trigger_count_np,
    "rules_triggered_list":   p2_rules_triggered_list,
    "rule_explanation":       p2_rule_explanation,
    "business_explanation":   p2_explanation,
})

# Add probability columns — vectorized assignment from the (N x n_classes) matrix
for i in range(n_classes):
    t = idx_to_typ[i]
    col = f"prob_{t.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}"
    phase2_output[col] = np.round(p2_prob_matrix[:, i], 4)

_tlog("phase2_output built")


# ═══ Summary ═══
print(f"\n  Phase 1: {len(phase1_output):,} rows | Phase 2: {len(phase2_output):,} rows")
src_counts = pd.Series(alert_source).value_counts()
print("  Alert Sources: " + " | ".join(
    f"{s}: {int(src_counts.get(s, 0)):,}"
    for s in ["Rule + ML Confirmed", "Rule Triggered", "ML Behavioural Alert", "Normal"]
))

aml_p2 = phase2_output[phase2_output["predicted_typology"] != "None"]
print(f"\n  Multi-Label Statistics:")
print(f"    Transactions with AML prediction:  {len(aml_p2):,}")
print(f"    With 1 typology matched:           {(aml_p2['num_typologies_matched'] == 1).sum():,}")
print(f"    With 2 typologies matched:         {(aml_p2['num_typologies_matched'] == 2).sum():,}")
print(f"    With 3+ typologies matched:        {(aml_p2['num_typologies_matched'] >= 3).sum():,}")

print(f"\n  Predicted Typology Distribution (primary):")
print(f"  {'Typology':<40s} {'Count':>8s} {'%':>7s} {'Avg Matched':>12s}")
print(f"  {'─'*70}")
typ_groups = phase2_output.groupby("predicted_typology", observed=True)
for typ in sorted(typology_classes):
    if typ in typ_groups.groups:
        g = typ_groups.get_group(typ)
        cnt = len(g)
        avg_m = g["num_typologies_matched"].mean()
        print(f"  {typ:<40s} {cnt:>8,} {cnt / max(len(aml_p2),1) * 100:>6.1f}% {avg_m:>11.2f}")

print(f"\n  Investigation Priority Distribution:")
pri_counts = phase2_output["investigation_priority"].value_counts()
for pri in ["Critical", "High", "Medium", "Low"]:
    cnt = int(pri_counts.get(pri, 0))
    print(f"    {pri:<10s} {cnt:>10,} ({cnt / len(phase2_output) * 100:.1f}%)")


# ═══ Sample output ═══
print(f"\n  Sample Phase 2 Output (multi-label examples):")
multi_samples = phase2_output[phase2_output["num_typologies_matched"] >= 2]
if len(multi_samples) > 0:
    for _, r in multi_samples.sample(min(8, len(multi_samples)), random_state=42).iterrows():
        print(f"    {r['transaction_id']}: {r['all_matched_typologies']}")
        print(f"      Priority={r['investigation_priority']} | {r['business_explanation'][:140]}")
else:
    print(f"    No multi-label predictions at threshold {TYPOLOGY_THRESHOLD}")
    if len(aml_p2) > 0:
        for _, r in aml_p2.sample(min(5, len(aml_p2)), random_state=42).iterrows():
            print(f"    {r['transaction_id']}: {r['predicted_typology']} ({r['typology_confidence']:.2f})")
            print(f"      Priority={r['investigation_priority']} | {r['business_explanation'][:140]}")

_tlog("DONE")


# ## 12 — Save Phase 2 Outputs
# 

# In[3]:


from sqlalchemy import create_engine
import db_config

DB_URL = (
    f"postgresql://{db_config.DB_USER}:"
    f"{db_config.DB_PASSWORD}@"
    f"{db_config.DB_HOST}:"
    f"{db_config.DB_PORT}/"
    f"{db_config.DB_NAME}"
)


engine = create_engine(DB_URL)


def write_table_fast(df, table_name, mode="append"):

    """
    Fast PostgreSQL bulk loader using COPY.

    Features:
    ----------
    - append / replace mode
    - auto-add missing columns
    - handles blank strings
    - handles NaNs
    - safe SQL identifiers
    """

    conn = engine.raw_connection()
    cur = conn.cursor()

    try:

        # ==========================================
        # Clean dataframe
        # ==========================================
        df = df.copy()

        df.columns = [c.strip() for c in df.columns]

        for col in df.columns:

            if df[col].dtype == object:

                df[col] = df[col].replace("", None)

        # Handle NaN / NaT
        df = df.where(pd.notnull(df), None)

        # =====================================================
        # Get existing PostgreSQL columns
        # =====================================================
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (table_name,)
        )

        existing_cols_info = cur.fetchall()

        existing_cols = {
            row[0]: row[1]
            for row in existing_cols_info
        }

        # =====================================================
        # Add missing columns automatically
        # =====================================================
        for col in df.columns:

            if col not in existing_cols:

                # Infer PostgreSQL datatype
                dtype = str(df[col].dtype)

                if "int" in dtype:
                    pg_type = "BIGINT"

                elif "float" in dtype:
                    pg_type = "DOUBLE PRECISION"

                elif "bool" in dtype:
                    pg_type = "BOOLEAN"

                elif "datetime" in dtype:
                    pg_type = "TIMESTAMP"

                else:
                    pg_type = "TEXT"

                alter_sql = sql.SQL("""
                    ALTER TABLE {}
                    ADD COLUMN IF NOT EXISTS {} {}
                """).format(
                    sql.Identifier(table_name),
                    sql.Identifier(col),
                    sql.SQL(pg_type)
                )

                cur.execute(alter_sql)

                print(f"Added column: {col} ({pg_type})")

        conn.commit()

        # =====================================================
        # Replace mode
        # =====================================================
        if mode == "replace":

            truncate_sql = sql.SQL(
                "TRUNCATE TABLE {}"
            ).format(
                sql.Identifier(table_name)
            )

            cur.execute(truncate_sql)

            print(f"Table truncated: {table_name}")

        # =====================================================
        # CSV buffer
        # =====================================================
        buffer = io.StringIO()

        df.to_csv(
            buffer,
            index=False,
            header=False,
            na_rep=""
        )

        buffer.seek(0)

        # =====================================================
        # COPY statement
        # =====================================================
        copy_sql = sql.SQL("""
            COPY {} ({})
            FROM STDIN WITH CSV
        """).format(
            sql.Identifier(table_name),
            sql.SQL(",").join(
                map(sql.Identifier, df.columns)
            )
        )

        # =====================================================
        # Fast bulk insert
        # =====================================================
        cur.copy_expert(
            copy_sql.as_string(cur),
            buffer
        )

        conn.commit()

        print(
            f"Loaded {len(df):,} rows into "
            f"{table_name} ({mode})"
        )

    except Exception as e:

        conn.rollback()

        print(f"\nError loading table: {e}")

        raise

    finally:

        cur.close()
        conn.close()


# In[ ]:


if RUN_MODE == "train":
    print("\n── Step 6: Save Outputs ──")

    import json as _json
    from sklearn.metrics import f1_score

    # ─── Compute multi-label accuracy on the Phase 2 test set ───
    # Same logic as Step 4 but recomputed here so it's saved to the output.
    # ─── Recompute BOTH accuracies from y_proba_2 so they're consistent ───
    y_true_arr = y_test_2.values

    # Primary: argmax of probabilities
    y_pred_primary = y_proba_2.argmax(axis=1)
    accuracy_primary = float((y_pred_primary == y_true_arr).mean())

    # Multi-label: true label appears in threshold set
    multi_correct = 0
    for i in range(len(y_proba_2)):
        above = set(j for j in range(n_classes) if y_proba_2[i][j] >= TYPOLOGY_THRESHOLD)
        if not above:
            above = {int(y_proba_2[i].argmax())}
        if y_true_arr[i] in above:
            multi_correct += 1
    accuracy_multi_label = multi_correct / len(y_true_arr)

    print(f"  Phase 2 accuracy — primary only:   {accuracy_primary:.4f} ({accuracy_primary*100:.2f}%)")
    print(f"  Phase 2 accuracy — multi-label:    {accuracy_multi_label:.4f} ({accuracy_multi_label*100:.2f}%)")
    print(f"  Multi-label threshold used:        {TYPOLOGY_THRESHOLD}")


    # # ─── Save outputs ───
    # phase1_output.to_parquet(os.path.join(OUTPUT_DIR, "phase1_aml_detection.parquet"), index=False)
    # phase2_output.to_parquet(os.path.join(OUTPUT_DIR, "phase2_typology_classification.parquet"), index=False)

    combined = phase1_output.merge(phase2_output, on=["transaction_id", "customer_id"], how="left")

    combined = pd.DataFrame(combined)
    # combined['datestamp'] = pd.to_datetime(combined['datestamp'],format="%d-%m-%Y",errors="coerce")
    # combined['customer_cif_creation_date'] = pd.to_datetime(combined['customer_cif_creation_date'],format="%d-%m-%Y",errors="coerce")
    # combined['account_wallet_opening_date'] = pd.to_datetime(combined['account_wallet_opening_date'],format="%d-%m-%Y",errors="coerce")
    # combined['kyc_update_date'] = pd.to_datetime(combined['kyc_update_date'],format="%d-%m-%Y",errors="coerce")
    # combined['account_wallet_inoperative_date'] = pd.to_datetime(combined['account_wallet_inoperative_date'],format="%d-%m-%Y",errors="coerce")
    # combined['date_of_incorporation'] = pd.to_datetime(v['date_of_incorporation'],format="%d-%m-%Y",errors="coerce")
    # combined['date_of_birth'] = pd.to_datetime(combined['date_of_birth'],format="%d-%m-%Y",errors="coerce")
    # combined["professional_experience_years"] = pd.to_numeric(
    #     combined["professional_experience_years"],
    #     errors="coerce"
    # ).astype("Int64")

    combined = pd.DataFrame(combined)
    combined["loaded_at"] = datetime.now()

    write_table_fast(combined, "phase2_generated_transactions_final_output", mode="replace")
    print(f"combined output written: {len(combined):,} rows x {len(combined.columns)} cols")


    #combined.to_parquet(os.path.join(OUTPUT_DIR, "combined_aml_output.parquet"), index=False)
    #combined.to_csv(os.path.join(OUTPUT_DIR, "combined_aml_output.csv"), index=False)

    # ─── Save model ───
    #model_p2.save_model(os.path.join(OUTPUT_DIR, "phase2_typology_model.txt"))

    # ─── Save model metadata (full) ───
    model_params = {
        "phase2": {
            "best_config":           best_p2,
            "config_details":        p2_results[best_p2]["config"],
            "accuracy_primary":      float(accuracy),
            "accuracy_multi_label":  float(accuracy_multi_label),
            "multi_label_threshold": float(TYPOLOGY_THRESHOLD),
            "macro_f1":              float(f1_score(y_test_2, y_pred_2, average="macro")),
            "weighted_f1":           float(f1_score(y_test_2, y_pred_2, average="weighted")),
            "best_iteration":        int(p2_results[best_p2]["rounds"]),
            "n_classes":             n_classes,
            "classes":               typology_classes,
            "n_features":            len(phase2_features),
            "n_train":               len(X_train_2),
            "n_test":                len(X_test_2),
            "all_configs": {
                k: {"acc": v["acc"], "mf1": v["mf1"], "wf1": v["wf1"], "rounds": v["rounds"]}
                for k, v in p2_results.items()
            },
        }
    }
    # with open(os.path.join(OUTPUT_DIR, "model_parameters_full.json"), "w") as f:
    #     _json.dump(model_params, f, indent=2, default=str)

    ## WRITE THE MODEL PARAMETERS FULL TO THE DATABASE
    def _flatten(d, prefix=""):
        """Flatten a nested dict into {dotted_key: value} pairs."""
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key + "."))
            elif isinstance(v, (list, tuple)):
                out[key] = "; ".join(str(x) for x in v)
            else:
                out[key] = v
        return out

    run_id = os.environ.get(
        "AML_RUN_ID",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    _flat = _flatten(model_params)
    params_df = pd.DataFrame(
        [{"run_id": run_id, "parameter": k, "value": str(v)}
         for k, v in _flat.items()]
    )

    # append — each run adds its own rows, keyed by run_id
    write_table(params_df, "model_parameters_full_phase2", mode="append")
    print(f"  model_parameters_full written: {len(params_df)} rows (run {run_id})")


    # ─── Save typology mapping ───
    typ_mapping = {
        "typ_to_idx": typ_to_idx,
        "idx_to_typ": {str(k): v for k, v in idx_to_typ.items()},
    }
    # with open(os.path.join(OUTPUT_DIR, "typology_mapping.json"), "w") as f:
    #     _json.dump(typ_mapping, f, indent=2)

    # ─── Save a compact accuracy summary table (CSV — easy to open in Excel) ───
    accuracy_summary = pd.DataFrame([
        {"metric": "primary_accuracy",      "value": float(accuracy),             "note": "argmax prediction matches true label"},
        {"metric": "multi_label_accuracy",  "value": float(accuracy_multi_label), "note": f"true label is in the set of typologies with prob >= {TYPOLOGY_THRESHOLD}"},
        {"metric": "macro_f1",              "value": float(f1_score(y_test_2, y_pred_2, average="macro")),    "note": "macro-averaged F1 (treats all classes equally)"},
        {"metric": "weighted_f1",           "value": float(f1_score(y_test_2, y_pred_2, average="weighted")), "note": "F1 weighted by class support"},
        {"metric": "multi_label_threshold", "value": float(TYPOLOGY_THRESHOLD),   "note": "probability threshold for multi-label flagging"},
        {"metric": "n_test_rows",           "value": float(len(y_test_2)),        "note": "number of held-out test transactions"},
    ])
    #accuracy_summary.to_csv(os.path.join(OUTPUT_DIR, "phase2_accuracy_summary.csv"), index=False)

    # ─── List saved files ───
    # Compressed joblib bundle for inference
    import joblib
    p2_bundle = {
        "model":              model_p2,
        "features":           phase2_features,
        "typ_to_idx":         typ_to_idx,
        "idx_to_typ":         {str(k): v for k, v in idx_to_typ.items()},
        "typology_classes":   typology_classes,
        "typology_threshold": float(TYPOLOGY_THRESHOLD),
        "metadata": {
            "accuracy_primary":     float(accuracy_primary),
            "accuracy_multi_label": float(accuracy_multi_label),
            "n_classes":            n_classes,
            "n_features":           len(phase2_features),
        },
    }
    import datetime
    run_id = os.environ.get("AML_RUN_ID",
                            datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_model("phase2_model_bundle", run_id, p2_bundle,
               metrics={"accuracy_primary": accuracy_primary,
                        "accuracy_multi_label": accuracy_multi_label})


    print(f"\n{'='*90}")
    print("PIPELINE COMPLETE")
    print(f"{'='*90}")
    print(f"  Phase 2 — primary accuracy:      {accuracy*100:>5.2f}%")
    print(f"  Phase 2 — multi-label accuracy:  {accuracy_multi_label*100:>5.2f}% (threshold {TYPOLOGY_THRESHOLD})")


# In[ ]:


# ════════════════════════════════════════════════════════════════
# INFERENCE OUTPUT SAVE — only in predict mode
# ════════════════════════════════════════════════════════════════
if RUN_MODE == "predict":
    print("\n── Saving inference outputs ──")
    combined = phase1_output.merge(phase2_output,
                                   on=["transaction_id", "customer_id"], how="left")

    # APPEND — every prediction run adds to the history
    write_table_fast(combined, "phase2_final_predictions_output", mode="append")
    print(f"  Appended {len(combined):,} rows to predictions_output")
    print("PHASE 2 INFERENCE COMPLETE")


# In[ ]:




