#!/usr/bin/env python
# coding: utf-8

# # AML ML Preparation — EDA, Feature Selection & Baseline Model
# ---
# **Target**: `is_aml` (binary)
# 
# **Feature Strategy:**
# - **RETAIN AS-IS**: All 126 rule flags + encoded categoricals (no selection applied)
# - **SELECT FROM**: Graph/velocity/balance features only (correlation + VIF + importance)
# - **EXCLUDE**: Typology signal, convergence risk, temporal risk (leakage from typology labels)
# - **EXCLUDE**: FIS, alert_level, fis_band (post-hoc derived scores)
# 
# **Class Imbalance**: SMOTE, class weights, and threshold tuning compared
# 

# ## 1 — Environment Setup
# 

# In[42]:


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

from project_config.loader import ensure_notebook_path, get_run_mode, get_artifact_path
from project_config.loader import get_artifact_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_MODE = get_run_mode(_SETTINGS)

PHASE1_OUTPUT_DIR = str(_PATHS["phase1_dir"])
OUTPUT_DIR = os.environ.get("AML_PHASE1_DIR", PHASE1_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_COLUMN = _SETTINGS["phase1"]["target_column"]
print(f"Phase 1 ML — RUN_MODE = {RUN_MODE.upper()}")
print(f"Output dir: {OUTPUT_DIR}")


# ## 2 — Load Feature-Engineered Data
# 

# In[2]:


# ── Database connection (PostgreSQL) ──
from db_utils import read_table, write_table, save_model, load_model, test_connection
test_connection()      # prints a one-line OK on connect


# In[ ]:


df = read_table("stg_transactions_features")
df.head()


# In[ ]:


#df = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_features.parquet")


# In[7]:


df.head()


# In[9]:


print(df['transaction_type_ppi'].unique())
obj_cols = df.select_dtypes(include=["object", "string"]).columns

df[obj_cols] = df[obj_cols].fillna("")
print(df['transaction_type_ppi'].unique())


# In[10]:


# INPUT_FILE = os.environ.get(
#     "AML_INPUT_FILE",
#     str(get_artifact_path(_PATHS, "features", _SETTINGS)),
# )

# if not os.path.exists(INPUT_FILE):
#     for alt in ["stg_transactions_features.parquet",
#                 "../aml_features_output/stg_transactions_features.parquet",
#                 "../stg_transactions_features.parquet"]:
#         if os.path.exists(alt):
#             INPUT_FILE = alt
#             break

# df = pd.read_parquet(INPUT_FILE)
# print(f"Loaded: {INPUT_FILE}")
# print(f"  {len(df):,} rows x {len(df.columns)} columns")

if "is_aml" in df.columns:
    print(f"  is_aml distribution: 0={(df['is_aml']==0).sum():,}  1={(df['is_aml']==1).sum():,}  ({(df['is_aml']==1).mean()*100:.1f}%)")
else:
    print(f"  No is_aml column — inference mode")


# In[11]:


if RUN_MODE == "train":


    flagged = df[df["is_aml"] == 1]

    if len(flagged) > 0:
        all_typ = []
        grp_sets = defaultdict(set)
        for _, row in flagged.iterrows():
            typs = str(row.get("aml_typology", ""))
            if typs and typs != "nan":
                for t in typs.split("; "):
                    t = t.strip()
                    if t:
                        all_typ.append(t)
                        gids = str(row.get("typology_group_id", ""))
                        if gids and gids != "nan":
                            for g in gids.split("; "):
                                grp_sets[t].add(g.strip())

        typ_counts = defaultdict(int)
        for t in all_typ:
            typ_counts[t] += 1

        print(f"\n{'='*75}")
        print(f"{'Typology':<40s} {'Txns':>8s} {'Scenarios':>10s} {'% Flagged':>10s}")
        print(f"{'-'*75}")
        for typ in sorted(typ_counts.keys()):
            txns = typ_counts[typ]
            scen = len(grp_sets.get(typ, set()))
            pct = txns / len(flagged) * 100
            print(f"  {typ:<38s} {txns:>8,} {scen:>10,} {pct:>9.1f}%")

        print(f"\n  TOTAL FLAGGED        {len(flagged):>10,}")
        total_scen = sum(len(s) for s in grp_sets.values())
        print(f"  TOTAL SCENARIOS      {total_scen:>10,}")

        # # Source breakdown
        # print(f"\n  -- Label Source --")
        # for src in ["Creator + Detector", "Creator Only", "Detector Only"]:
        #     cnt = (flagged["aml_flag_source"] == src).sum()
        #     print(f"    {src:<25s} {cnt:>8,} ({cnt/len(flagged)*100:.1f}%)")

    #     # Typology Distribution
    #     print(f"\n  -- Typology Distribution --")
    #     for typ in sorted(typ_counts.keys()):
    #         txns = typ_counts[typ]
    #         scen = len(grp_sets.get(typ, set()))
    #         print(f"    {typ:<38s} {txns:>8,} txns ({txns/len(flagged)*100:>5.1f}% of AML) [{scen:>5,} scenarios]")
    # else:
    #     print("No AML transactions detected")




# In[12]:


if RUN_MODE == "train":
    df.aml_typology.unique()


# In[13]:


if RUN_MODE == "train":
    test_df = df.copy()
    typ_col = "aml_typology"
    df_ml = df.copy()
    # ═══ DEBUG: Multi-typology analysis ═══
    print(f"\n  ── DEBUG: Multi-Typology Analysis ──")

    # 1. Check all typologies in test set
    print(f"\n  All typologies found in test set:")
    all_typs_debug = set()
    for t in test_df[typ_col].dropna():
        for part in str(t).split("; "):
            part = part.strip()
            if part and part not in ("nan", "None", ""):
                all_typs_debug.add(part)
    for typ in sorted(all_typs_debug):
        mask = test_df[typ_col] == typ
        print(f"    {typ:<45s} {mask.sum():>6,} transactions")

    missing = {"Structuring (Smurfing)", "Underground Banking (Hawala)"} - all_typs_debug
    if missing:
        print(f"\n  ⚠ MISSING from test set: {missing}")

    # 2. Count multi-typology transactions
    print(f"\n  Multi-typology transactions:")
    aml_test = test_df[test_df["is_aml"] == 1]
    multi_count = 0
    multi_combos = {}
    single_count = 0
    no_typ_count = 0

    for _, row in aml_test.iterrows():
        typ_str = str(row.get(typ_col, ""))
        if typ_str in ("", "nan", "None", "NaN"):
            no_typ_count += 1
            continue
        parts = [p.strip() for p in typ_str.split("; ") if p.strip() and p.strip() not in ("nan", "None", "")]
        if len(parts) == 0:
            no_typ_count += 1
        elif len(parts) == 1:
            single_count += 1
        else:
            multi_count += 1
            combo = " + ".join(sorted(parts))
            multi_combos[combo] = multi_combos.get(combo, 0) + 1

    print(f"    Single typology:     {single_count:>8,}")
    print(f"    Multi-typology:      {multi_count:>8,}")
    print(f"    No typology (empty): {no_typ_count:>8,}")
    print(f"    Total AML in test:   {len(aml_test):>8,}")

    if multi_combos:
        print(f"\n  Multi-typology combinations (top 20):")
        print(f"    {'Combination':<70s} {'Count':>8s}")
        print(f"    {'─'*80}")
        for combo, cnt in sorted(multi_combos.items(), key=lambda x: -x[1])[:20]:
            print(f"    {combo:<70s} {cnt:>8,}")

    # 3. Check FULL dataframe (not just test)
    print(f"\n  ── Full Dataset (df_ml) Multi-Typology Check ──")
    aml_full = df_ml[df_ml["is_aml"] == 1]
    multi_full = 0
    multi_combos_full = {}
    single_full = 0
    no_typ_full = 0
    typ_counts_full = {}

    for _, row in aml_full.iterrows():
        typ_str = str(row.get(typ_col, ""))
        if typ_str in ("", "nan", "None", "NaN"):
            no_typ_full += 1
            continue
        parts = [p.strip() for p in typ_str.split("; ") if p.strip() and p.strip() not in ("nan", "None", "")]
        if len(parts) == 0:
            no_typ_full += 1
        elif len(parts) == 1:
            single_full += 1
        else:
            multi_full += 1
            combo = " + ".join(sorted(parts))
            multi_combos_full[combo] = multi_combos_full.get(combo, 0) + 1
        for p in parts:
            typ_counts_full[p] = typ_counts_full.get(p, 0) + 1

    print(f"    Single typology:     {single_full:>8,}")
    print(f"    Multi-typology:      {multi_full:>8,}")
    print(f"    No typology (empty): {no_typ_full:>8,}")
    print(f"    Total AML:           {len(aml_full):>8,}")

    print(f"\n  Per-typology counts (full dataset, counting multi-labels):")
    for typ, cnt in sorted(typ_counts_full.items(), key=lambda x: -x[1]):
        print(f"    {typ:<45s} {cnt:>8,}")

    if multi_combos_full:
        print(f"\n  Multi-typology combinations in full dataset (top 20):")
        print(f"    {'Combination':<70s} {'Count':>8s}")
        print(f"    {'─'*80}")
        for combo, cnt in sorted(multi_combos_full.items(), key=lambda x: -x[1])[:20]:
            print(f"    {combo:<70s} {cnt:>8,}")

    # 4. Check if Structuring/Hawala exist ANYWHERE
    print(f"\n  ── Specific Check for Missing Typologies ──")
    for check_typ in ["Structuring", "Smurfing", "Hawala", "Underground"]:
        mask_full = df_ml["aml_typology"].astype(str).str.contains(check_typ, case=False, na=False)
        mask_test = test_df[typ_col].astype(str).str.contains(check_typ, case=False, na=False)
        print(f"    '{check_typ}' in full df_ml: {mask_full.sum():>8,} | in test_df: {mask_test.sum():>8,}")


# ## 3 — Initial EDA: Target Distribution & Data Quality
# 

# In[15]:


if RUN_MODE == "train":
    print("=" * 90)
    print("INITIAL EDA")
    print("=" * 90)

    # ── 3.1: Target distribution ──
    print("\n── 3.1: Target Variable (is_aml) ──")
    target_counts = df["is_aml"].value_counts().sort_index()
    for val, cnt in target_counts.items():
        print(f"  is_aml={val}: {cnt:>10,} ({cnt/len(df)*100:.1f}%)")
    imbalance_ratio = target_counts[0] / max(target_counts[1], 1)
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1 (Clean:AML)")

    # ── 3.2: Column type breakdown ──
    print("\n── 3.2: Column Types ──")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    print(f"  Numeric: {len(numeric_cols)} | Object/String: {len(object_cols)} | Boolean: {len(bool_cols)}")

    # ── 3.3: Missing values ──
    print("\n── 3.3: Missing Values (top 20) ──")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_top = missing_pct[missing_pct > 0].head(20)
    if len(missing_top) > 0:
        for col, pct in missing_top.items():
            print(f"  {col:<50s} {pct:>6.2f}%")
    else:
        print("  No missing values found")

    # ── 3.4: Typology distribution within AML ──
    print("\n── 3.4: Typology Distribution (within is_aml=1) ──")
    if "aml_typology" in df.columns:
        aml_df = df[df["is_aml"] == 1]
        all_typs = {}
        for t in aml_df["aml_typology"].dropna():
            for part in str(t).split("; "):
                part = part.strip()
                if part: all_typs[part] = all_typs.get(part, 0) + 1
        for typ, cnt in sorted(all_typs.items(), key=lambda x: -x[1]):
            print(f"  {typ:<40s} {cnt:>8,} ({cnt/len(aml_df)*100:.1f}%)")

    # ── 3.5: Key numeric feature statistics ──
    print("\n── 3.5: Key Feature Statistics (AML vs Clean) ──")
    key_features = ["transaction_amount", "rule_score", "fraud_intensity_score",
                    "sender_acct_txn_count_24h", "sender_acct_outflow_amt_24h",
                    "sender_acct_unique_counterparties_7d", "ip_risk_score"]

    existing_keys = [f for f in key_features if f in df.columns]
    print(f"\n  {'Feature':<45s} │ {'AML Mean':>10s} {'Clean Mean':>10s} {'Ratio':>7s} │ {'AML Med':>10s} {'Clean Med':>10s}")
    print("  " + "─" * 100)
    for feat in existing_keys:
        am = df.loc[df["is_aml"]==1, feat].mean()
        cm = df.loc[df["is_aml"]==0, feat].mean()
        amed = df.loc[df["is_aml"]==1, feat].median()
        cmed = df.loc[df["is_aml"]==0, feat].median()
        ratio = am / max(cm, 0.0001)
        print(f"  {feat:<45s} │ {am:>10.2f} {cm:>10.2f} {ratio:>6.2f}x │ {amed:>10.2f} {cmed:>10.2f}")

    # ── 3.6: Plots ──
    # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # fig.suptitle("Initial EDA — Target & Key Features", fontsize=14, fontweight="bold")

    # # Target bar
    # ax = axes[0, 0]
    # colors = ["#2ecc71", "#e74c3c"]
    # target_counts.plot(kind="bar", ax=ax, color=colors)
    # ax.set_title("Target Distribution (is_aml)")
    # ax.set_xticklabels(["Clean (0)", "AML (1)"], rotation=0)
    # for i, v in enumerate(target_counts):
    #     ax.text(i, v + len(df)*0.01, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

    # FIS by AML status
    # ax = axes[0, 1]
    # if "fraud_intensity_score" in df.columns:
    #     df.loc[df["is_aml"]==0, "fraud_intensity_score"].hist(bins=50, alpha=0.6, ax=ax, label="Clean", color="#2ecc71", density=True)
    #     df.loc[df["is_aml"]==1, "fraud_intensity_score"].hist(bins=50, alpha=0.6, ax=ax, label="AML", color="#e74c3c", density=True)
    #     ax.set_title("FIS Distribution: Clean vs AML")
    #     ax.legend()

    # # Rule score by AML
    # ax = axes[1, 0]
    # if "rule_score" in df.columns:
    #     df.loc[df["is_aml"]==0, "rule_score"].hist(bins=50, alpha=0.6, ax=ax, label="Clean", color="#2ecc71", density=True)
    #     df.loc[df["is_aml"]==1, "rule_score"].hist(bins=50, alpha=0.6, ax=ax, label="AML", color="#e74c3c", density=True)
    #     ax.set_title("Rule Score Distribution: Clean vs AML")
    #     ax.legend()

    # # Alert level by AML
    # ax = axes[1, 1]
    # if "alert_level" in df.columns:
    #     ct = pd.crosstab(df["alert_level"], df["is_aml"], normalize="index") * 100
    #     ct = ct.reindex(["Critical", "High", "Medium", "Low", "None"])
    #     ct.plot(kind="barh", stacked=True, ax=ax, color=colors)
    #     ax.set_title("AML Rate by Alert Level")
    #     ax.set_xlabel("Percentage")
    #     ax.legend(["Clean", "AML"])

    # # plt.tight_layout()
    # # plt.savefig(os.path.join(OUTPUT_DIR, "01_initial_eda.png"), bbox_inches="tight")
    # # plt.show()
    # # print(f"\n  Saved: {OUTPUT_DIR}/01_initial_eda.png")




# ## 4 — Feature Classification
# Features are split into 3 tiers:
# 1. **Protected** (always included): Rule flags + encoded categoricals
# 2. **Selectable** (subject to feature selection): Graph, velocity, balance, IP features
# 3. **Excluded**: Labels, IDs, post-hoc scores, typology/convergence/temporal signals
# 

# In[16]:


print("=" * 90)
print("FEATURE CLASSIFICATION — 3-Tier Strategy")
print("=" * 90)

# ═══ TIER 3: EXCLUDED — Never used as features ═══

LABEL_COLS = {
    "is_aml", "is_aml_typology", "aml_typology", "typology_group_id", "aml_flag_source"
}

ID_COLS = {
    "transaction_id", "timestamp", "datestamp", "customer_account_number",
    "customer_cif_id", "counterparty_account_number", "customer_name",
    "counterparty_name", "merchant_id", "merchant_name", "merchant_location",
    "session_id", "device_id_fingerprint", "ip_address", "pan", "aadhaar_number",
    "mobile_number", "email_id", "wallet_account_id", "beneficiary_wallet_id_vpa",
    "load_source_account_card_details", "customer_branch_ifsc_code",
    "counterparty_branch_ifsc_swift", "customer_cif_creation_date",
    "kyc_update_date", "account_wallet_opening_date", "account_wallet_inoperative_date",
    "date_of_birth", "date_of_incorporation",
    "father_spouse_name", "identification_proof_doc_no", "entity_identification_proof_doc_no",
    "cif_beneficial_owners", "name_beneficial_owners",
    "address_registered_office", "address_place_of_business",
    "address_beneficial_owners", "address_individual_customer",
    "place_of_incorporation", "browser_app_information",
    "geo_location_city_country", "escrow_account_linked",
    "gps_coordinates_lat", "gps_coordinates_lon",
    "customer_address_lat", "customer_address_lon"
}

# Post-hoc scores derived from is_aml — data leakage
POSTHOC_COLS = {
    "fraud_intensity_score", "fraud_intensity_score_raw", "fis_band",
    "alert_level", "rules_triggered", "rules_triggered_count",
    "predicted_aml", "predicted_typology", "typology_confidence"
}
POSTHOC_COLS.update({c for c in df.columns if c.startswith("prob_")})

# Typology-derived signals — leakage because they're computed using typology labels
TYPOLOGY_LEAKAGE_COLS = {c for c in df.columns if any(c.startswith(p) for p in
    ["typology_signal", "ts_", "convergence_risk", "cr_", "temporal_risk", "tr_"])}
# Also catch exact column names
TYPOLOGY_LEAKAGE_COLS.update({"typology_signal", "convergence_risk", "temporal_risk"})

INTERNAL_COLS = {c for c in df.columns if c.startswith("_")}

all_exclude = LABEL_COLS | ID_COLS | POSTHOC_COLS | TYPOLOGY_LEAKAGE_COLS | INTERNAL_COLS

# ═══ TIER 1: PROTECTED — Always included, no selection applied ═══

# Rule flag columns (binary 0/1 from 126-rule engine)
rule_flags = sorted([c for c in df.columns if c.startswith("rule_") and c not in
              {"rule_score", "rules_triggered", "rules_triggered_count"}
              and c not in all_exclude])

# rule_score itself (composite)
core_scores = [c for c in ["rule_score", "transaction_amount", "annual_income",
               "credit_summation_period", "debit_summation_period",
               "professional_experience_years"] if c in df.columns and c not in all_exclude]

# Categorical features to encode
STRING_FEATURES = sorted([c for c in [
    "transaction_type_dr_cr", "transaction_mode_channel_bank", "cash_flag",
    "transaction_type_ppi", "transaction_mode_channel_ppi", "transaction_status",
    "account_wallet_status", "pep_flag", "hni_flag", "minor_flag",
    "customer_type", "customer_entity_type", "account_category", "account_type",
    "customer_occupation_industry", "vkyc_flag", "wallet_kyc_category",
    "vpn_flag", "emulator_flag", "refund_chargeback_flag",
    "customer_current_risk_score", "tax_residency", "residency",
    "nationality", "citizenship", "non_face_to_face_flag",
    "merchant_category_code", "load_instrument_type", "authentication_method",
    "beneficial_owner_types", "passive_nfe", "source_of_funds",
    "source_of_funds_wallet", "currency"
] if c in df.columns and c not in all_exclude])

PROTECTED_NUMERIC = rule_flags + core_scores
# (encoded categoricals will be added after encoding step)

# ═══ TIER 2: SELECTABLE — Subject to feature selection ═══

SELECTABLE_PREFIXES = [
    "sender_acct_", "sender_cust_", "sender_running", "sender_daily",
    "sender_balance", "sender_pct_", "sender_cumulative",
    "receiver_acct_", "receiver_running", "receiver_balance",
    "inflow_outflow_", "ip_risk", "ip_flag", "ip_txn", "ip_unique", "ip_cross"
]

selectable_features = sorted([c for c in df.select_dtypes(include=[np.number]).columns
                               if c not in all_exclude
                               and c not in PROTECTED_NUMERIC
                               and any(c.startswith(p) for p in SELECTABLE_PREFIXES)])

# ═══ SUMMARY ═══
print(f"\n  ┌─────────────────────────────────────────────────────────────────────────┐")
print(f"  │  TIER 1 — PROTECTED (always included, no selection)                    │")
print(f"  │    Rule flags (binary):          {len(rule_flags):>4d}                                  │")
print(f"  │    Core numeric scores:          {len(core_scores):>4d}                                  │")
print(f"  │    Categorical (to encode):      {len(STRING_FEATURES):>4d}                                  │")
print(f"  ├─────────────────────────────────────────────────────────────────────────┤")
print(f"  │  TIER 2 — SELECTABLE (feature selection applied)                       │")
print(f"  │    Graph/velocity/balance/IP:    {len(selectable_features):>4d}                                  │")
print(f"  ├─────────────────────────────────────────────────────────────────────────┤")
print(f"  │  TIER 3 — EXCLUDED                                                     │")
print(f"  │    Labels:                       {len(LABEL_COLS):>4d}                                  │")
print(f"  │    Identifiers:                  {len(ID_COLS):>4d}                                  │")
print(f"  │    Post-hoc / leakage:           {len(POSTHOC_COLS):>4d}                                  │")
print(f"  │    Typology/convergence/temporal: {len(TYPOLOGY_LEAKAGE_COLS):>4d}                                  │")
print(f"  │    Internal working:             {len(INTERNAL_COLS):>4d}                                  │")
print(f"  └─────────────────────────────────────────────────────────────────────────┘")

# Show excluded typology columns explicitly
print(f"\n  Typology-derived columns EXCLUDED (would cause leakage):")
for c in sorted(TYPOLOGY_LEAKAGE_COLS):
    if c in df.columns:
        print(f"    ✗ {c}")



# ## 5 — Encode Categorical Features & Save Mappings
# 

# In[17]:


print("Encoding categorical features...")
import pickle

df_ml = df.copy()
encoded_cols = []
if RUN_MODE == "train":

    # ─── TRAIN: fit encoders from data, save them ───
    encoding_maps = {}

    for col in STRING_FEATURES:

        if col not in df_ml.columns:
            continue

        vals = df_ml[col].astype(str).str.strip().str.upper()

        vals = vals.replace({
            "NAN": "",
            "NONE": "",
            "": "MISSING"
        })

        categories = sorted(vals.unique())

        cat_map = {
            cat: i for i, cat in enumerate(categories)
        }

        encoded_col = f"{col}_enc"

        df_ml[encoded_col] = (
            vals.map(cat_map)
                .fillna(-1)
                .astype(int)
        )

        encoded_cols.append(encoded_col)

        encoding_maps[col] = cat_map

        print(
            f"  {col:<45s} → "
            f"{encoded_col:<50s} "
            f"({len(categories)} categories)"
        )

    print(f"\n  Total encoded columns: {len(encoded_cols)}")

    # =====================================================
    # SAVE ENCODERS TO POSTGRES
    # =====================================================
    import base64
    import pickle
    import pandas as pd
    from sqlalchemy import text
    from db_utils import get_engine, write_table_fast

    engine = get_engine()

    encoder_blob = pickle.dumps(encoding_maps)
    # convert to base64 string (SAFE for CSV COPY)
    encoder_blob_b64 = base64.b64encode(encoder_blob).decode("utf-8")

    encoder_df = pd.DataFrame([{
        "run_id": "RUN_ID_1",
        "encoder_name": "label_encoding_maps",
        "encoder_blob": encoder_blob
    }])

    # Replace previous encoders
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM model_encoders
            WHERE encoder_name = 'label_encoding_maps'
        """))

    write_table_fast(
        encoder_df,
        "model_encoders",
        mode="replace"
    )

    print("\n  ✓ Encoders saved to PostgreSQL")

    # Build PROTECTED features
    PROTECTED_FEATURES = PROTECTED_NUMERIC + encoded_cols

    print(
        f"\n  PROTECTED features (always in model): "
        f"{len(PROTECTED_FEATURES)}"
    )

else:

    # ─── PREDICT: load saved encoders, apply to input ───

    import pickle
    import base64
    import pandas as pd
    from db_utils import get_engine

    engine = get_engine()

    query = """
        SELECT encoder_blob
        FROM model_encoders
        WHERE encoder_name = 'label_encoding_maps'
        ORDER BY run_id DESC
        LIMIT 1
    """

    encoder_row = pd.read_sql(query, engine)

    if encoder_row.empty:
        raise ValueError(
            "No encoders found in PostgreSQL table: model_encoders"
        )
    
    

    encoder_blob_b64 = encoder_row.iloc[0]["encoder_blob"]
    encoder_bytes = base64.b64decode(encoder_blob_b64)
    encoding_maps = pickle.loads(encoder_bytes)

    # encoding_maps = pickle.loads(
    #     encoder_blob_b64= encoder_row.iloc[0]["encoder_blob"]
    #     encoder_bytes = base64.b64decode(encoder_blob_b64)
    #     encoding_maps = pickle.loads(encoder_bytes)
    # )

    print(
        f"  Loaded encoders from PostgreSQL "
        f"({len(encoding_maps)} columns)"
    )

    for col, cat_map in encoding_maps.items():

        if col not in df_ml.columns:

            # Column missing — fill with -1
            df_ml[f"{col}_enc"] = -1

            encoded_cols.append(f"{col}_enc")

            print(
                f"  ⚠  {col!r} missing in input; "
                f"encoded col filled with -1"
            )

            continue

        vals = (
            df_ml[col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        vals = vals.replace({
            "NAN": "",
            "NONE": "",
            "": "MISSING"
        })

        # Categories not seen during training → -1
        df_ml[f"{col}_enc"] = (
            vals.map(cat_map)
                .fillna(-1)
                .astype(int)
        )

        encoded_cols.append(f"{col}_enc")


# if RUN_MODE == "train":
#     # ─── TRAIN: fit encoders from data, save them ───
#     encoding_maps = {}
#     for col in STRING_FEATURES:
#         if col not in df_ml.columns:
#             continue
#         vals = df_ml[col].astype(str).str.strip().str.upper()
#         vals = vals.replace({"NAN": "", "NONE": "", "": "MISSING"})
#         categories = sorted(vals.unique())
#         cat_map = {cat: i for i, cat in enumerate(categories)}
#         encoded_col = f"{col}_enc"
#         df_ml[encoded_col] = vals.map(cat_map).fillna(-1).astype(int)
#         encoded_cols.append(encoded_col)
#         encoding_maps[col] = cat_map
#         print(f"  {col:<45s} → {encoded_col:<50s} ({len(categories)} categories)")
#     print(f"\n  Total encoded columns: {len(encoded_cols)}")

#     # Save encoding maps
#     import json as _json
#     json_path = os.path.join(OUTPUT_DIR, "label_encoding_maps.json")
#     with open(json_path, "w") as f:
#        _json.dump({col: {str(k): int(v) for k, v in m.items()} for col, m in encoding_maps.items()}, f, indent=2)
#     pkl_path = os.path.join(OUTPUT_DIR, "label_encoding_maps.pkl")
#     with open(pkl_path, "wb") as f:
#        pickle.dump(encoding_maps, f)
#     print(f"  Saved encoders: {json_path}")

#     # Build PROTECTED features
#     PROTECTED_FEATURES = PROTECTED_NUMERIC + encoded_cols
#     print(f"\n  PROTECTED features (always in model): {len(PROTECTED_FEATURES)}")

# else:
#     # ─── PREDICT: load saved encoders, apply to input ───
#     pkl_path = os.path.join(OUTPUT_DIR, "label_encoding_maps.pkl")
#     with open(pkl_path, "rb") as f:
#        encoding_maps = pickle.load(f)
#     print(f"  Loaded encoders: {pkl_path}  ({len(encoding_maps)} columns)")

#     for col, cat_map in encoding_maps.items():
#         if col not in df_ml.columns:
#             # Column missing — fill with -1
#             df_ml[f"{col}_enc"] = -1
#             encoded_cols.append(f"{col}_enc")
#             print(f"  ⚠  {col!r} missing in input; encoded col filled with -1")
#             continue
#         vals = df_ml[col].astype(str).str.strip().str.upper()
#         vals = vals.replace({"NAN": "", "NONE": "", "": "MISSING"})
#         # Categories not seen during training → -1
#         df_ml[f"{col}_enc"] = vals.map(cat_map).fillna(-1).astype(int)
#         encoded_cols.append(f"{col}_enc")


# ## 6 — Correlation Analysis (Selectable Features Only)
# 

# In[18]:


if RUN_MODE == "train":
    print("=" * 90)
    print("CORRELATION ANALYSIS — SELECTABLE Features vs is_aml")
    print("(Rule flags + categoricals are PROTECTED and skip this step)")
    print("=" * 90)

    target = df_ml["is_aml"].astype(float)

    # Correlations for SELECTABLE features only
    sel_correlations = {}
    for feat in selectable_features:
        if feat not in df_ml.columns: continue
        vals = pd.to_numeric(df_ml[feat], errors="coerce").fillna(0)
        if vals.std() == 0:
            sel_correlations[feat] = 0.0
            continue
        sel_correlations[feat] = vals.corr(target)

    corr_df = pd.DataFrame([
        {"feature": k, "correlation": v, "abs_correlation": abs(v)}
        for k, v in sel_correlations.items()
    ]).sort_values("abs_correlation", ascending=False)

    # Also compute correlations for PROTECTED features (for reporting only, no selection)
    prot_correlations = {}
    for feat in PROTECTED_FEATURES:
        if feat not in df_ml.columns: continue
        vals = pd.to_numeric(df_ml[feat], errors="coerce").fillna(0)
        if vals.std() == 0: prot_correlations[feat] = 0.0; continue
        prot_correlations[feat] = vals.corr(target)

    prot_corr_df = pd.DataFrame([
        {"feature": k, "correlation": v, "abs_correlation": abs(v)}
        for k, v in prot_correlations.items()
    ]).sort_values("abs_correlation", ascending=False)

    print(f"\n── 6.1: Top 30 SELECTABLE Features by Correlation with is_aml ──")
    print(f"  (These are the features subject to selection)\n")
    print(f"  {'Rank':<5s} {'Feature':<55s} {'Correlation':>12s} {'Signal':>8s}")
    print("  " + "─" * 83)
    for i, (_, row) in enumerate(corr_df.head(30).iterrows(), 1):
        strength = "STRONG" if row["abs_correlation"] > 0.1 else ("MEDIUM" if row["abs_correlation"] > 0.05 else "WEAK")
        bar = "█" * int(row["abs_correlation"] * 200)
        print(f"  {i:<5d} {row['feature']:<55s} {row['correlation']:>+11.6f} {strength:<8s} {bar}")

    print(f"\n── 6.2: Top 20 PROTECTED Features by Correlation (for reference, NOT selected out) ──\n")
    print(f"  {'Rank':<5s} {'Feature':<55s} {'Correlation':>12s} {'Status':>10s}")
    print("  " + "─" * 85)
    for i, (_, row) in enumerate(prot_corr_df.head(20).iterrows(), 1):
        print(f"  {i:<5d} {row['feature']:<55s} {row['correlation']:>+11.6f} {'PROTECTED':>10s}")

    # Heatmap (top selectable + top protected)
    top_sel = corr_df.head(15)["feature"].tolist()
    top_prot = prot_corr_df.head(5)["feature"].tolist()
    heatmap_feats = top_sel + top_prot + ["is_aml"]
    heatmap_feats = [c for c in heatmap_feats if c in df_ml.columns]

    # fig, ax = plt.subplots(figsize=(16, 14))
    # hm = df_ml[heatmap_feats].corr()
    # mask = np.triu(np.ones_like(hm, dtype=bool))
    # sns.heatmap(hm, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    #             ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    # ax.set_title("Top Selectable + Protected Features — Correlation Heatmap", fontsize=13, fontweight="bold")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png"), bbox_inches="tight")
    # plt.show()
    # print(f"\n  Saved: {OUTPUT_DIR}/02_correlation_heatmap.png")
    # corr_df.to_csv(os.path.join(OUTPUT_DIR, "selectable_feature_correlations.csv"), index=False)




# ## 7 — Multicollinearity & VIF (Selectable Features Only)
# 

# In[19]:


if RUN_MODE == "train":
    print("=" * 90)
    print("MULTICOLLINEARITY & VIF — Selectable Features Only")
    print("(Rule flags + categoricals are PROTECTED, not checked here)")
    print("=" * 90)

    from sklearn.linear_model import LinearRegression

    THRESHOLD = 0.90
    valid_sel = [f for f in selectable_features if f in df_ml.columns
                 and pd.to_numeric(df_ml[f], errors="coerce").std() > 0]

    print(f"  Computing pairwise correlations for {len(valid_sel)} selectable features...")
    if len(df_ml) > 50000:
        sample_df = df_ml[valid_sel].sample(50000, random_state=42)
    else:
        sample_df = df_ml[valid_sel].copy()
    for c in sample_df.columns:
        sample_df[c] = pd.to_numeric(sample_df[c], errors="coerce").fillna(0)

    corr_all = sample_df.corr()

    # Find pairs
    high_corr_pairs = []
    for i in range(len(corr_all.columns)):
        for j in range(i+1, len(corr_all.columns)):
            r = corr_all.iloc[i, j]
            if abs(r) >= THRESHOLD:
                high_corr_pairs.append((corr_all.columns[i], corr_all.columns[j], r))
    high_corr_pairs.sort(key=lambda x: -abs(x[2]))

    target_corr = {row["feature"]: row["abs_correlation"] for _, row in corr_df.iterrows()}

    print(f"\n── 7.1: Highly Correlated Selectable Pairs (|r| >= {THRESHOLD}) ──")
    print(f"  Found: {len(high_corr_pairs)} pairs\n")
    print(f"  {'Feature A':<45s} {'Feature B':<45s} {'Corr':>8s} {'TgtCorr A':>10s} {'TgtCorr B':>10s} {'Recommendation':>20s}")
    print("  " + "─" * 142)
    for f1, f2, r in high_corr_pairs[:40]:
        c1 = target_corr.get(f1, 0); c2 = target_corr.get(f2, 0)
        rec = f"Drop {f2[:18]}" if c1 >= c2 else f"Drop {f1[:18]}"
        print(f"  {f1:<45s} {f2:<45s} {r:>+7.4f} {c1:>9.6f} {c2:>9.6f}   {rec}")

    # VIF
    print(f"\n── 7.2: VIF Scores (Selectable Features) ──")
    top_vif = corr_df.head(min(50, len(valid_sel)))["feature"].tolist()
    top_vif = [f for f in top_vif if f in sample_df.columns]
    print(f"  Computing VIF for {len(top_vif)} features...")

    X_vif = sample_df[top_vif].copy()
    X_vif = (X_vif - X_vif.mean()) / X_vif.std().replace(0, 1)

    vif_results = []
    lr = LinearRegression()
    for i, feat in enumerate(top_vif):
        y_vif = X_vif[feat].values; X_others = X_vif.drop(columns=[feat]).values
        try:
            lr.fit(X_others, y_vif); r2 = lr.score(X_others, y_vif)
            vif = 1 / max(1 - r2, 0.0001)
        except: vif = float("inf"); r2 = 0
        vif_results.append({"feature": feat, "vif": vif, "r_squared": r2, "target_corr": target_corr.get(feat, 0)})

    vif_df = pd.DataFrame(vif_results).sort_values("vif", ascending=False)

    print(f"\n  {'Rank':<5s} {'Feature':<50s} {'VIF':>10s} {'R²':>8s} {'Target Corr':>12s} {'Severity':>12s}")
    print("  " + "─" * 102)
    for i, (_, row) in enumerate(vif_df.iterrows(), 1):
        v = row["vif"]
        sev = "⚠ CRITICAL" if v>=50 else ("⚡ SEVERE" if v>=10 else ("● MODERATE" if v>=5 else "✓ OK"))
        vd = f"{v:>10.2f}" if v < 10000 else f"{v:>10.0f}"
        print(f"  {i:<5d} {row['feature']:<50s} {vd} {row['r_squared']:>7.4f} {row['target_corr']:>11.6f} {sev}")

    # Summary
    crit=len(vif_df[vif_df["vif"]>=50]); sev=len(vif_df[(vif_df["vif"]>=10)&(vif_df["vif"]<50)])
    mod=len(vif_df[(vif_df["vif"]>=5)&(vif_df["vif"]<10)]); ok=len(vif_df[vif_df["vif"]<5])
    print(f"\n  VIF Summary: ✓ OK={ok} | ● Moderate={mod} | ⚡ Severe={sev} | ⚠ Critical={crit}")

    #vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif_scores_selectable.csv"), index=False)

    # Greedy removal (on selectable only)
    to_remove = set()
    for f1, f2, r in high_corr_pairs:
        if f1 in to_remove or f2 in to_remove: continue
        c1 = target_corr.get(f1, 0); c2 = target_corr.get(f2, 0)
        to_remove.add(f2 if c1 >= c2 else f1)

    selectable_after_multicollinearity = [f for f in selectable_features if f not in to_remove]
    print(f"\n  Selectable before: {len(selectable_features)} → after multicollinearity: {len(selectable_after_multicollinearity)} (removed {len(to_remove)})")

    # Plot
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # top30_vif = vif_df.head(30).sort_values("vif")
    # colors = ["#e74c3c" if v>=50 else "#f39c12" if v>=10 else "#3498db" if v>=5 else "#2ecc71" for v in top30_vif["vif"]]
    # axes[0].barh(top30_vif["feature"], top30_vif["vif"], color=colors)
    # axes[0].axvline(x=5, color="orange", linestyle="--", alpha=0.7); axes[0].axvline(x=10, color="red", linestyle="--", alpha=0.7)
    # axes[0].set_title("VIF Scores (Selectable Features)", fontsize=12, fontweight="bold")
    # sizes = [ok, mod, sev, crit]; labels = [f"OK<5 ({ok})", f"Mod 5-10 ({mod})", f"Sev 10-50 ({sev})", f"Crit 50+ ({crit})"]
    # axes[1].pie([s for s in sizes if s>0], labels=[l for l,s in zip(labels,sizes) if s>0],
    #             colors=["#2ecc71","#3498db","#f39c12","#e74c3c"][:sum(1 for s in sizes if s>0)], autopct="%1.0f%%")
    # axes[1].set_title("VIF Severity Distribution", fontsize=12, fontweight="bold")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "05_vif_analysis.png"), bbox_inches="tight")
    # plt.show()




# ## 8 — Feature Importance (Selectable Features — LightGBM Scan)
# 

# In[20]:


if RUN_MODE == "train":
    print("=" * 90)
    print("FEATURE IMPORTANCE — Quick LightGBM on SELECTABLE Features Only")
    print("=" * 90)

    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        print("  LightGBM not installed. Run: pip install lightgbm")
        HAS_LGB = False

    if HAS_LGB:
        X_sel = df_ml[selectable_after_multicollinearity].copy()
        for c in X_sel.columns: X_sel[c] = pd.to_numeric(X_sel[c], errors="coerce").fillna(0)
        y_sel = df_ml["is_aml"].astype(int)

        train_ds = lgb.Dataset(X_sel, label=y_sel)
        params = {"objective":"binary","metric":"auc","learning_rate":0.1,"num_leaves":31,
                  "max_depth":6,"min_child_samples":50,"subsample":0.8,"colsample_bytree":0.8,
                  "scale_pos_weight":(y_sel==0).sum()/max((y_sel==1).sum(),1),
                  "verbosity":-1,"random_state":42,"n_jobs":-1}

        print("  Training quick LightGBM for feature importance...")
        model_imp = lgb.train(params, train_ds, num_boost_round=200)

        importance_gain = pd.DataFrame({
            "feature": selectable_after_multicollinearity,
            "gain": model_imp.feature_importance(importance_type="gain"),
            "split": model_imp.feature_importance(importance_type="split")
        }).sort_values("gain", ascending=False)

        print(f"\n── Top 30 Selectable Features by Gain ──")
        print(f"  {'Rank':<5s} {'Feature':<55s} {'Gain':>12s} {'Splits':>8s} {'Cum%':>7s}")
        print("  " + "─" * 90)
        total_gain = importance_gain["gain"].sum()
        cum = 0
        for i, (_, row) in enumerate(importance_gain.head(30).iterrows(), 1):
            pct = row["gain"]/max(total_gain,1)*100; cum += pct
            print(f"  {i:<5d} {row['feature']:<55s} {row['gain']:>12.1f} {row['split']:>8.0f} {cum:>6.1f}%")

        zero_imp = importance_gain[importance_gain["gain"] == 0]
        selected_graph_features = importance_gain[importance_gain["gain"] > 0]["feature"].tolist()
        print(f"\n  Selectable features with importance > 0: {len(selected_graph_features)}")
        print(f"  Removed (zero importance): {len(zero_imp)}")

        # fig, ax = plt.subplots(figsize=(14, 10))
        # top30 = importance_gain.head(30).sort_values("gain")
        # ax.barh(top30["feature"], top30["gain"], color="#3498db")
        # ax.set_title("Top 30 Selectable Features by Gain", fontsize=12, fontweight="bold")
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, "03_feature_importance_selectable.png"), bbox_inches="tight")
        # plt.show()

        # importance_gain.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_selectable.csv"), index=False)
    else:
        selected_graph_features = selectable_after_multicollinearity




# ## 9 — Feature Selection Summary & Final Feature Set
# 

# In[21]:


if RUN_MODE == "train":
    print("=" * 90)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 90)

    # Final features = PROTECTED (all retained) + SELECTED graph features
    features_final = PROTECTED_FEATURES + selected_graph_features
    features_final = list(dict.fromkeys(features_final))  # deduplicate

    print(f"""
      ┌──────────────────────────────────────────────────────────────────────────────┐
      │  FINAL FEATURE SET                                                          │
      ├──────────────────────────────────────────────────────────────────────────────┤
      │  PROTECTED (no selection, always included):                                 │
      │    Rule flags:                   {len(rule_flags):>4d}  (126 binary rule columns)          │
      │    Encoded categoricals:         {len(encoded_cols):>4d}  (occupation, channel, risk, etc)   │
      │    Core numeric scores:          {len(core_scores):>4d}  (rule_score, amount, income, etc)   │
      │    Subtotal PROTECTED:           {len(PROTECTED_FEATURES):>4d}                                       │
      ├──────────────────────────────────────────────────────────────────────────────┤
      │  SELECTED (after correlation + VIF + importance):                           │
      │    Started with:                 {len(selectable_features):>4d}  graph/velocity/balance/IP features│
      │    After multicollinearity:      {len(selectable_after_multicollinearity):>4d}  (removed {len(to_remove)} correlated pairs)      │
      │    After zero-importance:        {len(selected_graph_features):>4d}  (final selected)                  │
      ├──────────────────────────────────────────────────────────────────────────────┤
      │  EXCLUDED:                                                                  │
      │    Typology/convergence/temporal: {len(TYPOLOGY_LEAKAGE_COLS):>4d}  (leakage from typology labels)    │
      │    FIS/alert_level/fis_band:     {len(POSTHOC_COLS):>4d}  (post-hoc derived scores)          │
      │    Labels + IDs:                 {len(LABEL_COLS)+len(ID_COLS):>4d}  (target + identifiers)             │
      ├──────────────────────────────────────────────────────────────────────────────┤
      │  TOTAL FEATURES IN MODEL:       {len(features_final):>4d}                                       │
      └──────────────────────────────────────────────────────────────────────────────┘
    """)

    # Category breakdown
    categories = {
        "Rule flags (126 binary)": [f for f in features_final if f.startswith("rule_") and f != "rule_score"],
        "Core scores (rule_score, amount, income)": [f for f in features_final if f in core_scores],
        "Encoded categoricals": [f for f in features_final if f.endswith("_enc")],
        "Sender account velocity": [f for f in features_final if f.startswith("sender_acct_")],
        "Sender customer velocity": [f for f in features_final if f.startswith("sender_cust_")],
        "Sender balance": [f for f in features_final if "sender" in f and any(k in f for k in ["balance","running","cumulative","pct_balance","daily"])],
        "Receiver features": [f for f in features_final if f.startswith("receiver_")],
        "IP risk features": [f for f in features_final if f.startswith("ip_")],
        "Volume ratios": [f for f in features_final if "volume_balance_ratio" in f],
    }
    print(f"  Breakdown:")
    for cat, cols in categories.items():
        if cols: print(f"    {cat:<45s} {len(cols):>4d}")

    # with open(os.path.join(OUTPUT_DIR, "selected_features.txt"), "w") as f:
    #     for feat in features_final: f.write(feat + "\n")
    # print(f"\n  Saved: {OUTPUT_DIR}/selected_features.txt ({len(features_final)} features)")




# ## 10 — Class Imbalance Handling
# Three strategies compared:
# 1. **Class weights** (`scale_pos_weight` in LightGBM) — penalizes misclassifying minority class
# 2. **SMOTE** (Synthetic Minority Oversampling) — generates synthetic AML examples
# 3. **Threshold tuning** — adjusts decision boundary instead of 0.5
# 

# In[23]:


if RUN_MODE == "train":
    print("=" * 90)
    print("CLASS IMBALANCE + HYPERPARAMETER TUNING + TYPOLOGY-AWARE SPLIT")
    print("=" * 90)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

    # ═══ Prepare data ═══
    X = df_ml[features_final].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    y = df_ml["is_aml"].astype(int)

    # # ═══ Typology-aware stratification ═══
    # print("\n── Typology-Aware Stratified Split ──")
    # typ_col = "aml_typology"
    # df_ml["_primary_typology"] = df_ml[typ_col].astype(str).apply(
    #     lambda x: x.split("; ")[0].strip() if x and x not in ("nan", "", "None") else "Clean"
    # )
    # df_ml["_strat_key"] = df_ml["_primary_typology"]
    # typ_counts = df_ml["_strat_key"].value_counts()
    # rare = typ_counts[typ_counts < 50].index.tolist()
    # if rare:
    #     df_ml.loc[df_ml["_strat_key"].isin(rare), "_strat_key"] = "Rare_Typology"
    #     print(f"  Merged {len(rare)} rare typologies for stratification: {rare}")

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.20, random_state=42, stratify=df_ml["_strat_key"]
    # )
    # train_indices = X_train.index; test_indices = X_test.index

    # n_pos = (y_train == 1).sum(); n_neg = (y_train == 0).sum()
    # imbalance_ratio = n_neg / max(n_pos, 1)
    # print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Imbalance: {imbalance_ratio:.2f}:1")

    # # Verify ALL typologies in both sets
    # print(f"\n  {'Typology':<40s} {'Train':>8s} {'Test':>8s} {'Status':>8s}")
    # print(f"  {'─'*68}")
    # train_typs = df_ml.loc[train_indices, "_primary_typology"]
    # test_typs = df_ml.loc[test_indices, "_primary_typology"]
    # for typ in sorted(set(train_typs.unique()) | set(test_typs.unique())):
    #     tr = (train_typs == typ).sum(); te = (test_typs == typ).sum()
    #     status = "✓" if tr > 0 and te > 0 else "⚠ MISSING"
    #     print(f"  {typ:<40s} {tr:>8,} {te:>8,} {status}")


    ##########################################################################
    # ═══ Typology-aware stratification — FIXED ═══
    print("\n── Typology-Aware Stratified Split ──")
    typ_col = "aml_typology"

    # For multi-typology transactions, assign to the RAREST typology
    # This prevents small typologies (Hawala, Layering) from being absorbed by large ones

    # Step 1: Count total transactions per typology
    from collections import Counter
    typ_global_counts = Counter()
    for t in df_ml[typ_col].dropna():
        for part in str(t).split("; "):
            part = part.strip()
            if part and part not in ("nan", "", "None"):
                typ_global_counts[part] += 1

    print(f"  Typology transaction counts (including multi-label):")
    for typ, cnt in typ_global_counts.most_common():
        print(f"    {typ:<40s} {cnt:>8,}")

    # Step 2: For each transaction, pick the RAREST typology as primary
    def pick_rarest_typology(typ_str):
        if not typ_str or typ_str in ("nan", "", "None"):
            return "Clean"
        parts = [p.strip() for p in str(typ_str).split("; ") if p.strip() and p.strip() not in ("nan", "", "None")]
        if not parts:
            return "Clean"
        # Pick the typology with the FEWEST total transactions globally
        return min(parts, key=lambda p: typ_global_counts.get(p, 999999))

    df_ml["_primary_typology"] = df_ml[typ_col].apply(pick_rarest_typology)

    # Verify the assignment
    print(f"\n  Primary typology assignment (rarest-first):")
    for typ in sorted(df_ml["_primary_typology"].unique()):
        cnt = (df_ml["_primary_typology"] == typ).sum()
        print(f"    {typ:<40s} {cnt:>8,}")

    df_ml["_strat_key"] = df_ml["_primary_typology"]
    typ_counts = df_ml["_strat_key"].value_counts()
    rare = typ_counts[typ_counts < 50].index.tolist()
    if rare:
        df_ml.loc[df_ml["_strat_key"].isin(rare), "_strat_key"] = "Rare_Typology"
        print(f"\n  Merged {len(rare)} rare typologies for stratification: {rare}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=df_ml["_strat_key"]
    )
    train_indices = X_train.index; test_indices = X_test.index

    n_pos = (y_train == 1).sum(); n_neg = (y_train == 0).sum()
    imbalance_ratio = n_neg / max(n_pos, 1)
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,} | Imbalance: {imbalance_ratio:.2f}:1")

    print(f"\n  {'Typology':<40s} {'Train':>8s} {'Test':>8s} {'Status':>8s}")
    print(f"  {'─'*68}")
    train_typs = df_ml.loc[train_indices, "_primary_typology"]
    test_typs = df_ml.loc[test_indices, "_primary_typology"]
    for typ in sorted(set(train_typs.unique()) | set(test_typs.unique())):
        tr = (train_typs == typ).sum(); te = (test_typs == typ).sum()
        status = "✓" if tr > 0 and te > 0 else "⚠ MISSING"
        print(f"  {typ:<40s} {tr:>8,} {te:>8,} {status}")


    ###########################################################################

    # ═══ SMOTE ═══
    try:
        from imblearn.over_sampling import SMOTE
        HAS_SMOTE = True
        smote = SMOTE(random_state=42, k_neighbors=5)
        # =====================================================
        # Fix nullable integer columns before SMOTE
        # =====================================================

        for col in X_train.columns:
            # Convert pandas nullable Int64/Int32/etc to float
            if str(X_train[col].dtype).startswith("Int"):
                X_train[col] = X_train[col].astype("float64")

# Optional but safer:
        X_train = X_train.astype("float32")
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"\n  SMOTE: {len(X_train):,} → {len(X_train_smote):,} (added {(y_train_smote==1).sum()-n_pos:,} synthetic AML)")
    except ImportError:
        HAS_SMOTE = False; X_train_smote = X_train; y_train_smote = y_train

    # ═══ Hyperparameter Tuning (8 configs) ═══
    print(f"\n── Hyperparameter Tuning (8 configurations) ──\n")

    # tuning_configs = [
    #     {"name":"Baseline",          "nl":63,  "md":8,  "mc":50,  "lr":0.05, "ra":0,   "rl":0,   "sub":0.8,"col":0.8,"data":"orig","spw":imbalance_ratio},
    #     {"name":"Deep+Reg",          "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1, "rl":1.0, "sub":0.8,"col":0.8,"data":"orig","spw":imbalance_ratio},
    #     {"name":"Slow+Deep+L2",      "nl":127, "md":10, "mc":20,  "lr":0.01, "ra":0.1, "rl":2.0, "sub":0.7,"col":0.7,"data":"orig","spw":imbalance_ratio},
    #     {"name":"Wide+Shallow",      "nl":255, "md":6,  "mc":100, "lr":0.05, "ra":0.1, "rl":1.0, "sub":0.7,"col":0.8,"data":"orig","spw":imbalance_ratio},
    #     {"name":"SMOTE+Balanced",    "nl":63,  "md":8,  "mc":50,  "lr":0.05, "ra":0,   "rl":0,   "sub":0.8,"col":0.8,"data":"smote","spw":1.0},
    #     {"name":"SMOTE+Deep+Reg",    "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1, "rl":1.0, "sub":0.8,"col":0.8,"data":"smote","spw":1.0},
    #     {"name":"SMOTE+HalfWt",      "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1, "rl":1.0, "sub":0.8,"col":0.8,"data":"smote","spw":imbalance_ratio*0.5},
    #     {"name":"HeavyReg+Slow",     "nl":31,  "md":6,  "mc":100, "lr":0.01, "ra":0.5, "rl":5.0, "sub":0.6,"col":0.7,"data":"orig","spw":imbalance_ratio},
    # ]

    ############## UPDATED TUNING CONFIGS ####################
    tuning_configs = [
        # ── Original configs with more rounds ──
        {"name":"Baseline",          "nl":63,  "md":8,  "mc":50,  "lr":0.05, "ra":0,    "rl":0,   "sub":0.8, "col":0.8, "data":"orig",  "spw":imbalance_ratio},
        {"name":"Deep+Reg",          "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1,  "rl":1.0, "sub":0.8, "col":0.8, "data":"orig",  "spw":imbalance_ratio},
        {"name":"SMOTE+Balanced",    "nl":63,  "md":8,  "mc":50,  "lr":0.05, "ra":0,    "rl":0,   "sub":0.8, "col":0.8, "data":"smote", "spw":1.0},
        {"name":"SMOTE+Deep+Reg",    "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1,  "rl":1.0, "sub":0.8, "col":0.8, "data":"smote", "spw":1.0},

        # ── NEW: Higher class weight to push recall ──
        {"name":"HighWeight",        "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1,  "rl":1.0, "sub":0.8, "col":0.8, "data":"orig",  "spw":imbalance_ratio * 1.5},
        {"name":"VHighWeight",       "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1,  "rl":1.0, "sub":0.8, "col":0.8, "data":"orig",  "spw":imbalance_ratio * 2.0},

        # ── NEW: Deeper trees to learn complex AML patterns ──
        {"name":"XDeep+Slow",        "nl":255, "md":12, "mc":20,  "lr":0.01, "ra":0.05, "rl":0.5, "sub":0.75,"col":0.75,"data":"orig",  "spw":imbalance_ratio},
        {"name":"SMOTE+XDeep",       "nl":255, "md":12, "mc":20,  "lr":0.01, "ra":0.05, "rl":0.5, "sub":0.75,"col":0.75,"data":"smote", "spw":1.0},

        # ── NEW: SMOTE + higher weight (double boost for minority class) ──
        {"name":"SMOTE+1.5xWt",      "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1,  "rl":1.0, "sub":0.8, "col":0.8, "data":"smote", "spw":imbalance_ratio * 0.75},
        {"name":"SMOTE+2xWt",        "nl":127, "md":10, "mc":30,  "lr":0.03, "ra":0.1,  "rl":1.0, "sub":0.8, "col":0.8, "data":"smote", "spw":imbalance_ratio},

        # ── NEW: More leaves + low learning rate (more capacity) ──
        {"name":"HighCap+Slow",      "nl":400, "md":8,  "mc":50,  "lr":0.008,"ra":0.1,  "rl":1.0, "sub":0.7, "col":0.7, "data":"orig",  "spw":imbalance_ratio},
        {"name":"SMOTE+HighCap",     "nl":400, "md":8,  "mc":50,  "lr":0.008,"ra":0.1,  "rl":1.0, "sub":0.7, "col":0.7, "data":"smote", "spw":1.0},
    ]

    print(f"  {'Config':<22s} │ {'AUC':>7s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s} │ {'Rounds':>6s} {'Thresh':>6s} {'F1@Thr':>7s}")
    print("  " + "─" * 85)

    results = {}

    for cfg in tuning_configs:
        params = {"objective":"binary","metric":"auc","num_leaves":cfg["nl"],"max_depth":cfg["md"],
                  "min_child_samples":cfg["mc"],"learning_rate":cfg["lr"],"reg_alpha":cfg["ra"],
                  "reg_lambda":cfg["rl"],"subsample":cfg["sub"],"colsample_bytree":cfg["col"],
                  "scale_pos_weight":cfg["spw"],"verbosity":-1,"random_state":42,"n_jobs":-1}

        Xtr = X_train_smote if cfg["data"]=="smote" else X_train
        ytr = y_train_smote if cfg["data"]=="smote" else y_train

        ds = lgb.Dataset(Xtr, label=ytr)
        val = lgb.Dataset(X_test, label=y_test, reference=ds)
        mdl = lgb.train(params, ds, num_boost_round=1000,
                        valid_sets=[val], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        prob = mdl.predict(X_test); auc_v = roc_auc_score(y_test, prob)
        bt=0.5; bf1=0
        for t in np.arange(0.25,0.75,0.02):
            f = f1_score(y_test,(prob>=t).astype(int))
            if f > bf1: bf1=f; bt=round(t,2)

        pred = (prob>=bt).astype(int)
        f1_v=f1_score(y_test,pred); p_v=precision_score(y_test,pred); r_v=recall_score(y_test,pred)
        cm=confusion_matrix(y_test,pred)

        results[cfg["name"]] = {"auc":auc_v,"f1":f1_v,"precision":p_v,"recall":r_v,
                                "model":mdl,"y_prob":prob,"threshold":bt,
                                "tp":cm[1,1],"fp":cm[0,1],"fn":cm[1,0],"tn":cm[0,0],
                                "config":cfg,"rounds":mdl.best_iteration if hasattr(mdl,"best_iteration") else 0}

        print(f"  {cfg['name']:<22s} │ {auc_v:>6.4f} {f1_v:>6.4f} {p_v:>6.4f} {r_v:>6.4f} │ {results[cfg['name']]['rounds']:>6} {bt:>5.2f} {bf1:>6.4f}")

    # ═══ Config Selection: Best recall among configs with F1 within 95% of peak ═══
    peak_f1_configs = max(v["f1"] for v in results.values())
    f1_floor_configs = peak_f1_configs * 0.99

    eligible_configs = {k: v for k, v in results.items() if v["f1"] >= f1_floor_configs}
    best_strategy = max(eligible_configs, key=lambda k: eligible_configs[k]["recall"])
    best_thresh = results[best_strategy]["threshold"]

    print(f"\n  Config Selection Logic:")
    print(f"    Peak F1 across all configs:   {peak_f1_configs:.4f}")
    print(f"    F1 floor (95% of peak):       {f1_floor_configs:.4f}")
    print(f"    Configs above F1 floor:       {len(eligible_configs)} of {len(results)}")
    for k in eligible_configs:
        marker = " ◄ SELECTED" if k == best_strategy else ""
        print(f"      {k:<22s} F1={results[k]['f1']:.4f}  Recall={results[k]['recall']:.4f}{marker}")
    print(f"\n  ► Best config: {best_strategy}")
    print(f"    AUC={results[best_strategy]['auc']:.4f} F1={results[best_strategy]['f1']:.4f} Recall={results[best_strategy]['recall']:.4f}")

    # ═══ Threshold Tuning: Best recall among thresholds with F1 within 95% of peak ═══
    print(f"\n── Fine-Grained Threshold Tuning ──")
    best_prob = results[best_strategy]["y_prob"]

    # Collect all threshold results
    threshold_results = []
    for t in np.arange(0.15, 0.70, 0.025):
        yp = (best_prob >= t).astype(int)
        f1 = f1_score(y_test, yp)
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp)
        cm = confusion_matrix(y_test, yp)
        threshold_results.append({"thresh": round(t, 3), "f1": f1, "prec": p, "recall": r,
                                   "tp": cm[1,1], "fp": cm[0,1], "fn": cm[1,0]})

    # Find peak F1, set floor at 95%, pick highest recall within floor
    peak_f1_thresh = max(t["f1"] for t in threshold_results)
    f1_floor_thresh = peak_f1_thresh * 0.99

    eligible_thresholds = [t for t in threshold_results if t["f1"] >= f1_floor_thresh]
    best_entry = max(eligible_thresholds, key=lambda t: t["recall"])
    best_thresh = best_entry["thresh"]
    best_f1 = best_entry["f1"]

    print(f"\n  {'Thresh':>7s} │ {'F1':>7s} {'Prec':>7s} {'Recall':>7s} │ {'TP':>8s} {'FP':>8s} {'FN':>8s} │ {'Status':>12s}")
    print("  " + "─" * 75)
    for t in threshold_results:
        in_range = t["f1"] >= f1_floor_thresh
        marker = " ◄ SELECTED" if t["thresh"] == best_thresh else ""
        status = "✓ Eligible" if in_range else ""
        print(f"  {t['thresh']:>7.3f} │ {t['f1']:>6.4f} {t['prec']:>6.4f} {t['recall']:>6.4f} │ {t['tp']:>8,} {t['fp']:>8,} {t['fn']:>8,} │ {status:>12s}{marker}")

    print(f"\n  Threshold Selection Logic:")
    print(f"    Peak F1 across all thresholds: {peak_f1_thresh:.4f}")
    print(f"    F1 floor (95% of peak):        {f1_floor_thresh:.4f}")
    print(f"    Eligible thresholds:            {len(eligible_thresholds)} of {len(threshold_results)}")
    print(f"    ► Selected: threshold={best_thresh} (F1={best_f1:.4f}, Recall={best_entry['recall']:.4f})")
    print(f"    ► F1 drop from peak:           {peak_f1_thresh - best_f1:.4f} ({(peak_f1_thresh - best_f1)/peak_f1_thresh*100:.1f}%)")

    # Plots
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # strats = list(results.keys()); metrics_list = ["auc","f1","precision","recall"]
    # x = np.arange(len(strats)); width = 0.2
    # for i,m in enumerate(metrics_list):
    #     axes[0].bar(x+i*width, [results[s][m] for s in strats], width, label=m.upper())
    # axes[0].set_xticks(x+width*1.5); axes[0].set_xticklabels(strats, rotation=30, ha="right", fontsize=7)
    # axes[0].set_title("Config Comparison", fontweight="bold"); axes[0].legend(fontsize=8); axes[0].set_ylim(0,1.1)

    # thresholds_plot = np.arange(0.1, 0.9, 0.02)
    # f1s_plot = [f1_score(y_test, (best_prob>=t).astype(int)) for t in thresholds_plot]
    # recs_plot = [recall_score(y_test, (best_prob>=t).astype(int)) for t in thresholds_plot]
    # precs_plot = [precision_score(y_test, (best_prob>=t).astype(int), zero_division=0) for t in thresholds_plot]
    # axes[1].plot(thresholds_plot, f1s_plot, "b-", lw=2, label="F1")
    # axes[1].plot(thresholds_plot, recs_plot, "r--", label="Recall")
    # axes[1].plot(thresholds_plot, precs_plot, "g--", label="Precision")
    # axes[1].axhline(f1_floor_thresh, color="blue", linestyle=":", alpha=0.5, label=f"F1 floor ({f1_floor_thresh:.3f})")
    # axes[1].axvline(best_thresh, color="k", linestyle=":", alpha=0.5, label=f"Selected ({best_thresh})")
    # axes[1].set_title("Threshold Tuning", fontweight="bold"); axes[1].set_xlabel("Threshold"); axes[1].legend(fontsize=8)

    # axes[2].hist(best_prob[y_test==0], bins=50, alpha=0.6, label="Clean", color="#2ecc71", density=True)
    # axes[2].hist(best_prob[y_test==1], bins=50, alpha=0.6, label="AML", color="#e74c3c", density=True)
    # axes[2].axvline(best_thresh, color="k", linestyle="--", label=f"Threshold={best_thresh}")
    # axes[2].set_title("Score Distribution", fontweight="bold"); axes[2].legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "06_phase1_tuning.png"), bbox_inches="tight"); plt.show()


# ## 11 — Final Baseline Model (Best Imbalance Strategy)
# 

# In[24]:


if RUN_MODE == "train":
    print(X_test.shape)
    print(X_train.shape)


# In[26]:


if RUN_MODE == "train":
    best_strategy,best_thresh
    print(best_thresh,",",best_strategy)

    #best_thresh = 0.3


# In[27]:


if RUN_MODE == "train":
    # DEBUG: Check what typologies exist in test set
    print(f"\n  DEBUG: All typologies in test set:")
    all_typs_debug = set()
    for t in test_df[typ_col].dropna():
        for part in str(t).split("; "):
            if part.strip() and part.strip() not in ("nan", "None", ""):
                all_typs_debug.add(part.strip())
    for typ in sorted(all_typs_debug):
        mask = test_df[typ_col].astype(str).str.contains(typ, na=False)
        print(f"    {typ:<45s} {mask.sum():>6,} transactions")

    if "Structuring" not in str(all_typs_debug) and "Smurfing" not in str(all_typs_debug):
        print(f"\n  ⚠ Structuring/Smurfing NOT FOUND in test set typology labels")
        print(f"    Checking is_aml=1 transactions with no typology:")
        aml_no_typ = test_df[(test_df["is_aml"]==1) & (test_df[typ_col].isna() | (test_df[typ_col].astype(str).isin(["","nan","None"])))]
        print(f"    AML with empty typology: {len(aml_no_typ):,}")


# In[28]:


# ═══ DEBUG: Multi-typology analysis (training-only diagnostic) ═══
if RUN_MODE != "train":
    print("  Skipping multi-typology debug (predict mode — no test split exists).")
else:
    print(f"\n  ── DEBUG: Multi-Typology Analysis ──")

    # 1. Check all typologies in test set
    print(f"\n  All typologies found in test set:")
    all_typs_debug = set()
    for t in test_df[typ_col].dropna():
        for part in str(t).split("; "):
            part = part.strip()
            if part and part not in ("nan", "None", ""):
                all_typs_debug.add(part)
    for typ in sorted(all_typs_debug):
        mask = test_df[typ_col].astype(str).str.contains(typ, na=False)
        print(f"    {typ:<45s} {mask.sum():>6,} transactions")

    missing = {"Structuring (Smurfing)", "Underground Banking (Hawala)"} - all_typs_debug
    if missing:
        print(f"\n  ⚠ MISSING from test set: {missing}")

    # 2. Count multi-typology transactions
    print(f"\n  Multi-typology transactions:")
    aml_test = test_df[test_df["is_aml"] == 1]
    multi_count = 0
    multi_combos = {}
    single_count = 0
    no_typ_count = 0

    for _, row in aml_test.iterrows():
        typ_str = str(row.get(typ_col, ""))
        if typ_str in ("", "nan", "None", "NaN"):
            no_typ_count += 1
            continue
        parts = [p.strip() for p in typ_str.split("; ") if p.strip() and p.strip() not in ("nan", "None", "")]
        if len(parts) == 0:
            no_typ_count += 1
        elif len(parts) == 1:
            single_count += 1
        else:
            multi_count += 1
            combo = " + ".join(sorted(parts))
            multi_combos[combo] = multi_combos.get(combo, 0) + 1

    print(f"    Single typology:     {single_count:>8,}")
    print(f"    Multi-typology:      {multi_count:>8,}")
    print(f"    No typology (empty): {no_typ_count:>8,}")
    print(f"    Total AML in test:   {len(aml_test):>8,}")

    if multi_combos:
        print(f"\n  Multi-typology combinations (top 20):")
        print(f"    {'Combination':<70s} {'Count':>8s}")
        print(f"    {'─'*80}")
        for combo, cnt in sorted(multi_combos.items(), key=lambda x: -x[1])[:20]:
            print(f"    {combo:<70s} {cnt:>8,}")

    # 3. Check FULL dataframe (not just test)
    print(f"\n  ── Full Dataset (df_ml) Multi-Typology Check ──")
    aml_full = df_ml[df_ml["is_aml"] == 1]
    multi_full = 0
    multi_combos_full = {}
    single_full = 0
    no_typ_full = 0
    typ_counts_full = {}

    for _, row in aml_full.iterrows():
        typ_str = str(row.get(typ_col, ""))
        if typ_str in ("", "nan", "None", "NaN"):
            no_typ_full += 1
            continue
        parts = [p.strip() for p in typ_str.split("; ") if p.strip() and p.strip() not in ("nan", "None", "")]
        if len(parts) == 0:
            no_typ_full += 1
        elif len(parts) == 1:
            single_full += 1
        else:
            multi_full += 1
            combo = " + ".join(sorted(parts))
            multi_combos_full[combo] = multi_combos_full.get(combo, 0) + 1
        for p in parts:
            typ_counts_full[p] = typ_counts_full.get(p, 0) + 1

    print(f"    Single typology:     {single_full:>8,}")
    print(f"    Multi-typology:      {multi_full:>8,}")
    print(f"    No typology (empty): {no_typ_full:>8,}")
    print(f"    Total AML:           {len(aml_full):>8,}")

    print(f"\n  Per-typology counts (full dataset, counting multi-labels):")
    for typ, cnt in sorted(typ_counts_full.items(), key=lambda x: -x[1]):
        print(f"    {typ:<45s} {cnt:>8,}")

    if multi_combos_full:
        print(f"\n  Multi-typology combinations in full dataset (top 20):")
        print(f"    {'Combination':<70s} {'Count':>8s}")
        print(f"    {'─'*80}")
        for combo, cnt in sorted(multi_combos_full.items(), key=lambda x: -x[1])[:20]:
            print(f"    {combo:<70s} {cnt:>8,}")

    # 4. Check if Structuring/Hawala exist ANYWHERE
    print(f"\n  ── Specific Check for Missing Typologies ──")
    for check_typ in ["Structuring", "Smurfing", "Hawala", "Underground"]:
        mask_full = df_ml["aml_typology"].astype(str).str.contains(check_typ, case=False, na=False)
        mask_test = test_df[typ_col].astype(str).str.contains(check_typ, case=False, na=False)
        print(f"    '{check_typ}' in full df_ml: {mask_full.sum():>8,} | in test_df: {mask_test.sum():>8,}")


# In[29]:


if RUN_MODE == "train":
    print("=" * 90)
    print(f"FINAL BASELINE MODEL — {best_strategy} + Threshold={best_thresh}")
    print("=" * 90)

    if HAS_LGB:
        from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, average_precision_score

        final_model = results[best_strategy]["model"]
        y_prob_final = results[best_strategy]["y_prob"]
        y_pred_final = (y_prob_final >= best_thresh).astype(int)

        auc = roc_auc_score(y_test, y_prob_final)
        avg_prec = average_precision_score(y_test, y_prob_final)

        print(f"\n  AUC-ROC:           {auc:.4f}")
        print(f"  Average Precision: {avg_prec:.4f}")
        print(f"  Threshold:         {best_thresh}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred_final, target_names=["Clean", "AML"], digits=4))

        cm = confusion_matrix(y_test, y_pred_final)
        print(f"  Confusion Matrix:")
        print(f"    {'':>15s} {'Pred Clean':>12s} {'Pred AML':>12s}")
        print(f"    {'Actual Clean':<15s} {cm[0,0]:>12,} {cm[0,1]:>12,}")
        print(f"    {'Actual AML':<15s} {cm[1,0]:>12,} {cm[1,1]:>12,}")

        # Per-typology recall
        if "aml_typology" in df.columns:
            print(f"\n  ── Per-Typology Detection Rate ──")
            test_df = df_ml.loc[X_test.index].copy()
            test_df["_pred_prob"] = y_prob_final
            test_df["_pred"] = y_pred_final
            typ_col = "aml_typology"
            all_typs = sorted(test_df.loc[test_df["is_aml"]==1, typ_col].unique())
            all_typs = [t for t in all_typs if t and str(t) not in ("", "nan", "None")]


            # for t in test_df[typ_col].dropna():
            #     for part in str(t).split("; "):
            #         if part.strip(): all_typs.add(part.strip())

            print(f"    {'Typology':<40s} {'Total':>7s} {'Caught':>7s} {'Recall':>7s} {'Avg Prob':>9s} {'Status':>8s}")
            print(f"    {'─'*83}")
            for typ in sorted(all_typs):
                mask = test_df[typ_col] == typ
                cnt = mask.sum()
                if cnt == 0: continue
                caught = test_df.loc[mask, "_pred"].sum()
                recall = caught / cnt * 100
                avg_p = test_df.loc[mask, "_pred_prob"].mean()
                status = "✓ GOOD" if recall > 80 else ("⚡ CHECK" if recall > 50 else "⚠ LOW")
                print(f"    {typ:<40s} {cnt:>7,} {caught:>7,} {recall:>6.1f}% {avg_p:>8.3f} {status}")

        # Feature importance
        imp = pd.DataFrame({"feature": features_final,
                             "gain": final_model.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False)

        print(f"\n  ── Top 20 Features by Importance ──")
        print(f"    {'Rank':<5s} {'Feature':<55s} {'Gain':>12s} {'Category':>15s}")
        print(f"    {'─'*90}")
        for i, (_, row) in enumerate(imp.head(20).iterrows(), 1):
            cat = "RULE" if row["feature"].startswith("rule_") else ("ENCODED" if row["feature"].endswith("_enc") else "GRAPH/VEL")
            print(f"    {i:<5d} {row['feature']:<55s} {row['gain']:>12.1f} {cat:>15s}")

        # Plots
        # fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # fig.suptitle(f"Final Model Results — {best_strategy} (threshold={best_thresh})", fontsize=14, fontweight="bold")

        # # ROC
        # fpr, tpr, _ = roc_curve(y_test, y_prob_final)
        # axes[0,0].plot(fpr, tpr, "b-", lw=2, label=f"AUC={auc:.4f}")
        # axes[0,0].plot([0,1],[0,1],"k--",alpha=0.3); axes[0,0].set_title("ROC Curve"); axes[0,0].legend()

        # # PR
        # prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob_final)
        # axes[0,1].plot(rec_c, prec_c, "r-", lw=2, label=f"AP={avg_prec:.4f}")
        # axes[0,1].set_title("Precision-Recall Curve"); axes[0,1].legend()

        # # Confusion matrix heatmap
        # sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=axes[1,0],
        #             xticklabels=["Pred Clean","Pred AML"], yticklabels=["Actual Clean","Actual AML"])
        # axes[1,0].set_title("Confusion Matrix")

        # # Top 15 importance
        # top15 = imp.head(15).sort_values("gain")
        # colors = ["#e74c3c" if f.startswith("rule_") else "#3498db" if f.endswith("_enc") else "#2ecc71" for f in top15["feature"]]
        # axes[1,1].barh(top15["feature"], top15["gain"], color=colors)
        # axes[1,1].set_title("Top 15 Feature Importance (Red=Rules, Blue=Categorical, Green=Graph)")

        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, "07_final_model_results.png"), bbox_inches="tight")
        # plt.show()

        # Save model + metadata
        # final_model.save_model(os.path.join(OUTPUT_DIR, "final_lgb_model.txt"))




# ## SAVING X TRAIN

# In[30]:


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
        # =====================================================
        # Check columns — do NOT alter the table.
        # Report what the table has vs. what the DataFrame has,
        # and load only the columns that exist in both.
        # =====================================================
        df_cols    = list(df.columns)
        table_cols = list(existing_cols.keys())

        matched      = [c for c in df_cols if c in existing_cols]
        missing_in_db = [c for c in df_cols if c not in existing_cols]   # in df, not in table
        unused_in_db  = [c for c in table_cols if c not in df_cols]      # in table, not in df

        print(f"Table '{table_name}' has {len(table_cols)} columns:")
        for c in table_cols:
            print(f"    {c}  ({existing_cols[c]})")

        print(f"\nDataFrame has {len(df_cols)} columns.")
        print(f"  Matched (will be loaded): {len(matched)}")
        if missing_in_db:
            print(f"  In DataFrame but NOT in table (will be SKIPPED): {missing_in_db}")
        if unused_in_db:
            print(f"  In table but NOT in DataFrame (left empty/default): {unused_in_db}")

        if not matched:
            raise ValueError(
                f"None of the DataFrame columns exist in table '{table_name}'. "
                f"Nothing to load."
            )

        # Keep only the columns that exist in the table.
        df = df[matched]

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



# In[46]:


if RUN_MODE == "train":

    ############### WRITING X TRAIN DATA TO POSTGRESQL    
    from datetime import datetime
    X_train = pd.DataFrame(X_train)
    X_train["loaded_at"] = datetime.now()
    X_train["professional_experience_years"] = pd.to_numeric(
    X_train["professional_experience_years"],
    errors="coerce").astype("Int64")

    write_table_fast(X_train, "x_train_phase1", mode="replace")
    print(f"Rules output written: {len(X_train):,} rows x {len(X_train.columns)} cols")


    ############### WRITING X TEST DATA TO POSTGRESQL
    X_test = pd.DataFrame(X_test)
    X_test["loaded_at"] = datetime.now()
    X_test["professional_experience_years"] = pd.to_numeric(
    X_test["professional_experience_years"],
    errors="coerce").astype("Int64")

    write_table_fast(X_test, "x_test_phase1", mode="replace")
    print(f"Rules output written: {len(X_test):,} rows x {len(X_test.columns)} cols")

    ############## WRITING Y TEST DATA TO POSTGRESQL
    y_test = pd.DataFrame(y_test)
    y_test["loaded_at"] = datetime.now()
    write_table_fast(y_test, "y_test_phase1", mode="replace")
    print(f"Rules output written: {len(y_test):,} rows x {len(y_test.columns)} cols")  

    ############# WRITING Y TRAIN DATA TO POSTGRESQL
    y_train = pd.DataFrame(y_train)
    y_train["loaded_at"] = datetime.now()
    write_table_fast(y_train, "y_train_phase1", mode="replace")
    print(f"Rules output written: {len(y_train):,} rows x {len(y_train.columns)} cols") 


# In[31]:


if RUN_MODE == "train":

        import json as _json
        metadata = {
            "features": features_final,
            "n_features": len(features_final),
            "protected_count": len(PROTECTED_FEATURES),
            "selected_graph_count": len(selected_graph_features),
            "best_strategy": best_strategy,
            "optimal_threshold": best_thresh,
            "auc_roc": auc,
            "f1_score": float(f1_score(y_test, y_pred_final)),
            "imbalance_ratio": imbalance_ratio,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        # with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
        #     _json.dump(metadata, f, indent=2)

        # print(f"\n  Saved outputs:")
        # for fn in sorted(os.listdir(OUTPUT_DIR)):
        #     if not os.path.isdir(os.path.join(OUTPUT_DIR, fn)):
        #         size = os.path.getsize(os.path.join(OUTPUT_DIR, fn)) / (1024*1024)




# In[32]:


OUTPUT_DIR


# In[25]:


# import pandas as pd
# df_ml = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\python_scripts\ml_outputs\df_ml_phase_1.parquet")
# df_ml.head()


# In[33]:


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

################################### CHANGING THE DATA TYPES #########################################

df = df.copy()

for col in df.columns:

    dtype = str(df[col].dtype)

    # =====================================================
    # OBJECT COLUMNS
    # =====================================================
    if dtype == "object":

        df[col] = (
            df[col]
            .replace(
                ["nan", "NaN", "None", "<NA>"],
                None
            )
        )

    # =====================================================
    # NULLABLE INTEGER COLUMNS
    # =====================================================
    elif dtype == "Int64":

        # Replace blanks with NA
        df[col] = (
            df[col]
            .replace("", pd.NA)
        )

        # Convert safely
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("Int64")

    # =====================================================
    # NORMAL INTEGER
    # =====================================================
    elif dtype == "int64":

        pass

    # =====================================================
    # FLOAT
    # =====================================================
    elif dtype == "float64":

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

print(df.dtypes)


# In[34]:


df_ml = pd.DataFrame(df_ml)
df_ml['datestamp'] = pd.to_datetime(df_ml['datestamp'],format="%d-%m-%Y",errors="coerce")
df_ml['customer_cif_creation_date'] = pd.to_datetime(df_ml['customer_cif_creation_date'],format="%d-%m-%Y",errors="coerce")
df_ml['account_wallet_opening_date'] = pd.to_datetime(df_ml['account_wallet_opening_date'],format="%d-%m-%Y",errors="coerce")
df_ml['kyc_update_date'] = pd.to_datetime(df_ml['kyc_update_date'],format="%d-%m-%Y",errors="coerce")
df_ml['account_wallet_inoperative_date'] = pd.to_datetime(df_ml['account_wallet_inoperative_date'],format="%d-%m-%Y",errors="coerce")
df_ml['date_of_incorporation'] = pd.to_datetime(df_ml['date_of_incorporation'],format="%d-%m-%Y",errors="coerce")
df_ml['date_of_birth'] = pd.to_datetime(df_ml['date_of_birth'],format="%d-%m-%Y",errors="coerce")
df_ml["professional_experience_years"] = pd.to_numeric(
    df_ml["professional_experience_years"],
    errors="coerce"
).astype("Int64")

df_ml.head()
df_ml['cif_beneficial_owners'].unique()


# ## WRITING df_ml phase 1 

# In[37]:


df_ml.drop(["_primary_typology", "_strat_key"], axis=1, inplace=True, errors="ignore")


# In[38]:


df_ml.head()


# In[39]:


if RUN_MODE == "train":
    from datetime import datetime
    df_ml = pd.DataFrame(df_ml)
    df_ml["loaded_at"] = datetime.now()

    write_table_fast(df_ml, "phase1_model_output", mode="replace")
    print(f"Rules output written: {len(df_ml):,} rows x {len(df_ml.columns)} cols")


# In[40]:


if RUN_MODE == "train":
    # ─── Recompute Phase 1 metrics from the final threshold-tuned predictions ───
    # This ensures the saved metrics match what was printed in the classification report.
    from sklearn.metrics import precision_score, recall_score

    y_pred_final_arr = (y_prob_final >= best_thresh).astype(int)

    phase1_recall    = float(recall_score(y_test, y_pred_final_arr))
    phase1_precision = float(precision_score(y_test, y_pred_final_arr))
    phase1_f1        = float(f1_score(y_test, y_pred_final_arr))
    phase1_auc       = float(roc_auc_score(y_test, y_prob_final))

    cm_final = confusion_matrix(y_test, y_pred_final_arr)
    tn, fp, fn, tp = cm_final.ravel()

    print(f"\n  Phase 1 metrics at threshold {best_thresh}:")
    print(f"    AUC-ROC:    {phase1_auc:.4f}")
    print(f"    Recall:     {phase1_recall:.4f}")
    print(f"    Precision:  {phase1_precision:.4f}")
    print(f"    F1:         {phase1_f1:.4f}")


# In[44]:


if RUN_MODE == "train":
    # Save model parameters for Excel summary
    import json as _json
    model_params = {
        "phase1": {
        "best_config":     best_strategy,
        "config_details":  results[best_strategy]["config"],
        "threshold":       best_thresh,
        "auc_roc":         phase1_auc,                    # recomputed
        "f1_score":        phase1_f1,                     # recomputed at threshold 0.65
        "precision":       phase1_precision,              # recomputed at threshold 0.65
        "recall":          phase1_recall,                 # recomputed at threshold 0.65
        "tp":              int(tp),
        "fp":              int(fp),
        "fn":              int(fn),
        "tn":              int(tn),
        "best_iteration":  int(results[best_strategy]["rounds"]),
        "n_features":      len(features_final),
        "n_graph_selected": len(selected_graph_features),
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "imbalance_ratio": imbalance_ratio,
        # all_configs still uses results[k] — those are at threshold 0.5, which is OK
        # for cross-strategy comparison since they were all measured the same way
        "all_configs": {
            k: {"auc": v["auc"], "f1": v["f1"], "precision": v["precision"], "recall": v["recall"], "rounds": v["rounds"]}
            for k, v in results.items()
        },
    },

        "features": {
            "final_features": features_final,
            #"selected_rules": selected_rules,
            #"removed_rules": removed_rules,
            "selected_graph": selected_graph_features,
            "protected_features": list(PROTECTED_FEATURES),
            "encoded_categoricals": encoded_cols,
        }
    }

    import csv as _csv

    # def _flatten(d, prefix=""):
    #     """Flatten a nested dict into {dotted_key: value} pairs."""
    #     out = {}
    #     for k, v in d.items():
    #         key = f"{prefix}{k}"
    #         if isinstance(v, dict):
    #             out.update(_flatten(v, key + "."))
    #         elif isinstance(v, (list, tuple)):
    #             out[key] = "; ".join(str(x) for x in v)
    #         else:
    #             out[key] = v
    #     return out

    # _flat = _flatten(model_params)
    # _csv_path = os.path.join(OUTPUT_DIR, "model_parameters_full.csv")
    # with open(_csv_path, "w", newline="", encoding="utf-8") as f:
    #     w = _csv.writer(f)
    #     w.writerow(["parameter", "value"])
    #     for k, v in _flat.items():
    #         w.writerow([k, v])
    # print(f"  Saved model parameters: {_csv_path}  ({len(_flat)} parameters)")


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
    write_table(params_df, "model_parameters_full", mode="append")
    print(f"  model_parameters_full written: {len(params_df)} rows (run {run_id})")

    print(f"\n  Saved files:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp=os.path.join(OUTPUT_DIR,fn)
        if not os.path.isdir(fp):
            print(f"    {fn:<50s} {os.path.getsize(fp)/(1024*1024):>8.2f} MB")

    print(f"\n{'='*90}")
    print("PIPELINE COMPLETE")
    print(f"{'='*90}")
    print(f"  Phase 1: AUC={phase1_auc:.4f} | F1={phase1_f1:.4f} | Recall={phase1_recall:.4f}")
    #print(f"  Phase 1: AUC={results[best_strategy]['auc']:.4f} | F1={results[best_strategy]['f1']:.4f} | Recall={results[best_strategy]['recall']:.4f}")
    #print(f"  Phase 2: Accuracy={accuracy:.4f} | Config={best_p2}")
    #print(f"  Rules: {len(selected_rules)} kept / {len(removed_rules)} removed (from 126)")
    print(f"  Total features: {len(features_final)}")
    #print(f"  Model params saved to: model_parameters_full.json")




# In[24]:


# # Save model parameters as a CSV (flattened key/value pairs)
#     import csv as _csv

#     def _flatten(d, prefix=""):
#         """Flatten a nested dict into {dotted_key: value} pairs."""
#         out = {}
#         for k, v in d.items():
#             key = f"{prefix}{k}"
#             if isinstance(v, dict):
#                 out.update(_flatten(v, key + "."))
#             elif isinstance(v, (list, tuple)):
#                 out[key] = "; ".join(str(x) for x in v)
#             else:
#                 out[key] = v
#         return out

#     _flat = _flatten(model_params)
#     _csv_path = os.path.join(OUTPUT_DIR, "model_parameters_full.csv")
#     with open(_csv_path, "w", newline="", encoding="utf-8") as f:
#         w = _csv.writer(f)
#         w.writerow(["parameter", "value"])
#         for k, v in _flat.items():
#             w.writerow([k, v])
#     print(f"  Saved model parameters: {_csv_path}  ({len(_flat)} parameters)")


# In[47]:


RUN_MODE = "predict"


# In[45]:


# ════════════════════════════════════════════════════════════════
# INFERENCE — Phase 1 prediction on the input dataset
# Only runs when RUN_MODE == "predict"
# ════════════════════════════════════════════════════════════════
if RUN_MODE == "predict":
    import joblib
    print("=" * 70)
    print("Phase 1 INFERENCE")
    print("=" * 70)

    bundle_path = os.path.join(OUTPUT_DIR, "phase1_model_bundle.joblib")
    print(f"Loading bundle: {bundle_path}")
    bundle = load_model("phase1_model_bundle")     # latest model from DB

    p1_model    = bundle["model"]
    p1_features = bundle["features"]
    p1_thresh   = bundle["threshold"]
    p1_meta     = bundle.get("metadata", {})

    print(f"  Model:       {type(p1_model).__name__}")
    print(f"  Features:    {len(p1_features)}")
    print(f"  Threshold:   {p1_thresh}")

    # Build X with model's exact feature order; missing → 0
    X_inf = pd.DataFrame(index=df_ml.index)
    for col in p1_features:
        if col in df_ml.columns:
            X_inf[col] = pd.to_numeric(df_ml[col], errors="coerce").fillna(0)
        else:
            X_inf[col] = 0
            print(f"  ⚠  feature {col!r} missing in input; filled with 0")
    print(f"  Inference matrix: {X_inf.shape}")

    # Predict
    print("Predicting...")
    y_prob = p1_model.predict(X_inf)
    y_pred = (y_prob >= p1_thresh).astype(int)

    # Persist score onto df_ml so downstream Phase 2 cell can use it
    df_ml["_phase1_score"] = y_prob
    df_ml["predicted_aml"] = y_pred
    best_thresh = p1_thresh  # keep variable name consistent with Phase 2 expectations

    n_aml = int(y_pred.sum())
    print(f"  Predicted AML:  {n_aml:,} ({n_aml/len(df_ml)*100:.2f}%)")
    print(f"  Predicted clean: {len(df_ml)-n_aml:,}")

    # # Save scored output
    # scored_path = os.path.join(OUTPUT_DIR, "df_ml_phase_1.parquet")
    # df_ml.to_parquet(scored_path, index=False)
    # print(f"  Saved: {scored_path}")

    from datetime import datetime
    df_ml = pd.DataFrame(df_ml)
    df_ml["loaded_at"] = datetime.now()

    write_table_fast(df_ml, "phase1_model_output", mode="replace")
    print(f"Rules output written: {len(df_ml):,} rows x {len(df_ml.columns)} cols")

    # Compatibility variables for Phase 2 notebook (which expects these globals)
    features_final = p1_features
    print("\nPhase 1 inference complete.")


# In[46]:


# ════════════════════════════════════════════════════════════════
# BUNDLE SAVE — compressed joblib model bundle for inference
# Replaces the bulky .txt model dump with a single ~5-10 MB .joblib file
# ════════════════════════════════════════════════════════════════
if RUN_MODE == "train":
    import joblib
    bundle = {
        "model":     final_model,
        "features":  features_final,
        "threshold": float(best_thresh),
        "metadata": {
            "best_strategy":   best_strategy,
            "auc_roc":         float(phase1_auc),
            "f1":              float(phase1_f1),
            "recall":          float(phase1_recall),
            "precision":       float(phase1_precision),
            "imbalance_ratio": float(imbalance_ratio),
            "n_features":      len(features_final),
            "n_train":         int(len(X_train)),
            "n_test":          int(len(X_test)),
            "encoded_cols":    encoded_cols,
        },
    }

    import datetime
    run_id = os.environ.get("AML_RUN_ID",
                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_model("phase1_model_bundle", run_id, bundle,
            metrics=bundle.get("metadata", {}))
    print(f"Phase 1 model saved to DB (run {run_id})")

