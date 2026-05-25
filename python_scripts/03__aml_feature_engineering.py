#!/usr/bin/env python
# coding: utf-8

# # AML Feature Engineering -- Graph, Velocity, Balance & Fraud Intensity Score
# ---
# Reads the rules-engine output and computes **90+ new features** across 5 categories:
# 
# | Category | Features | Description |
# |----------|----------|-------------|
# | Sender Account Velocity | 20 | Inflow/outflow counts & amounts at 1h, 24h, 7d, 30d |
# | Sender Customer Velocity | 20 | Same metrics rolled up across all customer accounts |
# | Sender Balance Tracking | 6 | Running balance, daily change, balance ratios |
# | Receiver Account Velocity + Balance | 26 | Mirror of sender metrics for counterparty |
# | IP Risk Score | 9 | VPN, emulator, night, FATF country, shared IP, geo-mismatch |
# | Fraud Intensity Score (FIS) | 3 | Composite raw score, normalized 0-100, and band |
# 
# ### Output
# `stg_transactions_features.parquet` -- original data + all new columns
# 
# 

# ## 1 -- Environment Setup
# 

# In[ ]:


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

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_MODE = get_run_mode(_SETTINGS)




# ## 2 -- Load Data
# 

# ## TEST DB CONNECTION

# In[ ]:


# ── Database connection (PostgreSQL) ──
from db_utils import read_table, write_table, save_model, load_model, test_connection
test_connection()      # prints a one-line OK on connect


# ## READ THE DATA TABLE UPDATED WITH RULES

# In[ ]:


# from project_config.loader import get_artifact_path
# _default_rules = str(get_artifact_path(_PATHS, "flagged", _SETTINGS))
# if RUN_MODE == "predict":
#     INPUT_FILE = os.environ.get("AML_INPUT_FILE", str(_PATHS["outputs_dir"] / "inference_input.parquet"))
# else:
#     INPUT_FILE = os.environ.get("AML_INPUT_FILE", _default_rules)
# print("Environment ready")


# In[ ]:


df = read_table("stg_transactions_rules")
df.head()


# In[ ]:


# df = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_rules.parquet")
# print(df.shape)
# df.head()


# In[ ]:


print(df['transaction_type_ppi'].unique())
df = df.fillna("")
print(df['transaction_type_ppi'].unique())


# In[ ]:


# RUN_MODE flag — read from env var. Used to skip diagnostic/scoring
# cells in predict mode (when the input has no real is_aml labels).
RUN_MODE = os.environ.get("AML_RUN_MODE", "train").lower()
assert RUN_MODE in ("train", "predict")
print(f"Run mode: {RUN_MODE.upper()}")

# # Input file — honour the orchestrator's AML_INPUT_FILE env var.
# # Falls back through several plausible default paths if not set.
# INPUT_FILE = os.environ.get(
#     "AML_INPUT_FILE",
#     "../outputs_updated/stg_transactions_rules.parquet",
# )

# if not os.path.exists(INPUT_FILE):
#     candidates = [
#         "../outputs_updated/stg_transactions_rules.parquet",
#         "../outputs_updated/stg_transactions_rules.parquet",
#         "../outputs_updated/stg_transactions_rules.csv",
#         "stg_transactions_rules.parquet",
#         "stg_transactions_rules.parquet",
#         "stg_transactions_rules.csv",
#     ]
#     for alt in candidates:
#         if os.path.exists(alt):
#             INPUT_FILE = alt
#             break

# ext = INPUT_FILE.rsplit('.', 1)[-1].lower()
# df = pd.read_parquet(INPUT_FILE) if ext == 'parquet' else pd.read_csv(INPUT_FILE, low_memory=False)
# print(f"Loaded: {INPUT_FILE}")
# print(f"  {len(df):,} rows x {len(df.columns)} columns")

# Schema guard: training-mode data has is_aml/aml_typology/typology_group_id.
# Inference data does not. The rules engine already creates dummies, but if
# this notebook is run standalone with raw input, do the same defensive guard.
for col, default in [("is_aml", 0), ("aml_typology", ""), ("typology_group_id", "")]:
    if col not in df.columns:
        df[col] = default

print(df.columns)


# In[ ]:


# Show existing typology labels if present
if "is_aml" in df.columns:
    aml_count = (df["is_aml"] == 1).sum()
    print(f"  Pre-labeled AML transactions: {aml_count:,} ({aml_count/len(df)*100:.2f}%)")
if "aml_typology" in df.columns:
    typs = df[df["aml_typology"].notna() & (df["aml_typology"] != "")]
    if len(typs) > 0:
        print(f"  Typologies present: {typs['aml_typology'].nunique()}")


# In[ ]:


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



# ## 3 -- Resolve Columns & Parse Datetime
# 

# In[ ]:


def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL_ACCT     = find_col(["customer_account_number","account_number"])
COL_CP_ACCT  = find_col(["counterparty_account_number","cp_account_number"])
COL_CIF      = find_col(["customer_cif_id","cif_id"])
COL_AMOUNT   = find_col(["transaction_amount","amount"])
COL_TYPE     = find_col(["transaction_type_dr_cr","txn_type"])
COL_TS       = find_col(["timestamp","Timestamp"])
COL_DS       = find_col(["datestamp","Datestamp"])
COL_STATUS   = find_col(["transaction_status","txn_status"])
COL_RULE_SCORE = find_col(["rule_score"])
COL_VPN      = find_col(["vpn_flag"])
COL_EMULATOR = find_col(["emulator_flag"])
COL_IP       = find_col(["ip_address"])
COL_DEVICE   = find_col(["device_id_fingerprint","device_id"])

print("Key columns:")
for k, v in [("Account", COL_ACCT), ("Counterparty", COL_CP_ACCT), ("CIF", COL_CIF),
             ("Amount", COL_AMOUNT), ("Type", COL_TYPE), ("Rule Score", COL_RULE_SCORE)]:
    print(f"  {k:<15s} -> {str(v)}")

# Parse datetime
def parse_dt(row):
    ds = str(row.get(COL_DS, ""))
    ts = str(row.get(COL_TS, "00:00:00"))
    for fmt in ["%d-%m-%Y %H:%M:%S","%d/%m/%Y %H:%M:%S","%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(f"{ds} {ts}".strip(), fmt)
        except:
            continue
    return pd.NaT

df["_dt"] = df.apply(parse_dt, axis=1)
df["_amt"] = pd.to_numeric(df[COL_AMOUNT], errors="coerce").fillna(0)
df["_acct"] = df[COL_ACCT].astype(str).str.strip()
df["_cp"] = df[COL_CP_ACCT].astype(str).str.strip()
df["_cif"] = df[COL_CIF].astype(str).str.strip() if COL_CIF else df["_acct"]
df["_type"] = df[COL_TYPE].astype(str).str.strip().str.upper() if COL_TYPE else "DR"
df["_stat"] = df[COL_STATUS].astype(str).str.strip().str.upper() if COL_STATUS else "SUCCESS"
df["_date_only"] = df["_dt"].dt.date

# Determine direction: is this a debit (outflow) or credit (inflow) for the customer account
df["_is_debit"] = df["_type"].isin(["DR","DEBIT","D"])
df["_is_credit"] = df["_type"].isin(["CR","CREDIT","C"])

# Signed amount from sender perspective
df["_signed_amt"] = np.where(df["_is_debit"], -df["_amt"], df["_amt"])

print(f"Datetime parsed. Valid: {df['_dt'].notna().sum():,}")



# ## 4 -- Sender Account Velocity Features (20 columns)
# Rolling 1h, 24h, 7d, 30d windows: inflow/outflow counts and amounts.
# 

# In[ ]:


print("Computing sender account velocity features...")
df = df.sort_values(["_acct", "_dt"]).reset_index(drop=True)

WINDOWS = {
    "1h":  timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d":  timedelta(days=7),
    "30d": timedelta(days=30),
}

# Initialize all sender account velocity columns
for w in WINDOWS:
    df[f"sender_acct_txn_count_{w}"] = 0
    df[f"sender_acct_inflow_amt_{w}"] = 0.0
    df[f"sender_acct_outflow_amt_{w}"] = 0.0
    df[f"sender_acct_inflow_count_{w}"] = 0
    df[f"sender_acct_outflow_count_{w}"] = 0

groups = df.groupby("_acct")
total_groups = len(groups)
processed = 0

for acct, gdf in groups:
    processed += 1
    if processed % 3000 == 0:
        print(f"  Sender acct: {processed:,}/{total_groups:,}...")

    idxs = gdf.index.values
    dts = gdf["_dt"].values
    amts = gdf["_amt"].values
    is_deb = gdf["_is_debit"].values
    is_cre = gdf["_is_credit"].values
    n = len(idxs)

    for wi, (wname, wdelta) in enumerate(WINDOWS.items()):
        wdelta_ns = wdelta / timedelta(seconds=1) * 1e9  # nanoseconds
        txn_cnt = np.zeros(n, dtype=int)
        in_amt = np.zeros(n, dtype=float)
        out_amt = np.zeros(n, dtype=float)
        in_cnt = np.zeros(n, dtype=int)
        out_cnt = np.zeros(n, dtype=int)

        left = 0
        for i in range(n):
            t = dts[i]
            if pd.isna(t):
                continue
            # Move left pointer
            while left < i and (t - dts[left]) / np.timedelta64(1, 'ns') > wdelta_ns:
                left += 1
            # Sum from left to i-1
            for j in range(left, i):
                if pd.isna(dts[j]):
                    continue
                txn_cnt[i] += 1
                if is_cre[j]:
                    in_amt[i] += amts[j]
                    in_cnt[i] += 1
                if is_deb[j]:
                    out_amt[i] += amts[j]
                    out_cnt[i] += 1

        df.loc[idxs, f"sender_acct_txn_count_{wname}"] = txn_cnt
        df.loc[idxs, f"sender_acct_inflow_amt_{wname}"] = np.round(in_amt, 2)
        df.loc[idxs, f"sender_acct_outflow_amt_{wname}"] = np.round(out_amt, 2)
        df.loc[idxs, f"sender_acct_inflow_count_{wname}"] = in_cnt
        df.loc[idxs, f"sender_acct_outflow_count_{wname}"] = out_cnt

print("  Sender account velocity: 20 columns done")



# ## 5 -- Sender Customer Velocity Features (20 columns)
# Same windows rolled up by customer CIF across all their accounts.
# 

# In[ ]:


print("Computing sender customer velocity features...")
df = df.sort_values(["_cif", "_dt"]).reset_index(drop=True)

for w in WINDOWS:
    df[f"sender_cust_txn_count_{w}"] = 0
    df[f"sender_cust_inflow_amt_{w}"] = 0.0
    df[f"sender_cust_outflow_amt_{w}"] = 0.0
    df[f"sender_cust_inflow_count_{w}"] = 0
    df[f"sender_cust_outflow_count_{w}"] = 0

df["sender_cust_id_for_rollup"] = df["_cif"]

groups = df.groupby("_cif")
total_groups = len(groups)
processed = 0

for cif, gdf in groups:
    processed += 1
    if processed % 2000 == 0:
        print(f"  Sender cust: {processed:,}/{total_groups:,}...")

    idxs = gdf.index.values
    dts = gdf["_dt"].values
    amts = gdf["_amt"].values
    is_deb = gdf["_is_debit"].values
    is_cre = gdf["_is_credit"].values
    n = len(idxs)

    for wname, wdelta in WINDOWS.items():
        wdelta_ns = wdelta / timedelta(seconds=1) * 1e9
        txn_cnt = np.zeros(n, dtype=int)
        in_amt = np.zeros(n, dtype=float)
        out_amt = np.zeros(n, dtype=float)
        in_cnt = np.zeros(n, dtype=int)
        out_cnt = np.zeros(n, dtype=int)

        left = 0
        for i in range(n):
            t = dts[i]
            if pd.isna(t):
                continue
            while left < i and (t - dts[left]) / np.timedelta64(1, 'ns') > wdelta_ns:
                left += 1
            for j in range(left, i):
                if pd.isna(dts[j]):
                    continue
                txn_cnt[i] += 1
                if is_cre[j]:
                    in_amt[i] += amts[j]
                    in_cnt[i] += 1
                if is_deb[j]:
                    out_amt[i] += amts[j]
                    out_cnt[i] += 1

        df.loc[idxs, f"sender_cust_txn_count_{wname}"] = txn_cnt
        df.loc[idxs, f"sender_cust_inflow_amt_{wname}"] = np.round(in_amt, 2)
        df.loc[idxs, f"sender_cust_outflow_amt_{wname}"] = np.round(out_amt, 2)
        df.loc[idxs, f"sender_cust_inflow_count_{wname}"] = in_cnt
        df.loc[idxs, f"sender_cust_outflow_count_{wname}"] = out_cnt

print("  Sender customer velocity: 20 columns done")



# ## 6 -- Sender Balance Tracking (6 columns)
# Running balance, daily cumulative change, and balance ratios.
# 

# In[ ]:


print("Computing sender balance tracking...")
df = df.sort_values(["_acct", "_dt"]).reset_index(drop=True)

df["sender_balance_before_txn"] = 0.0
df["sender_running_balance_txn_amount"] = df["_signed_amt"]
df["sender_balance_after_txn"] = 0.0
df["sender_cumulative_daily_balance_change"] = 0.0
df["sender_current_balance"] = 0.0
df["sender_bal_ratio_after_to_current"] = 0.0

for acct, gdf in df.groupby("_acct"):
    idxs = gdf.index.values
    signed = gdf["_signed_amt"].values
    dates = gdf["_date_only"].values
    n = len(idxs)

    bal = 0.0
    daily_change = 0.0
    prev_date = None
    bal_before = np.zeros(n)
    bal_after = np.zeros(n)
    daily_cum = np.zeros(n)

    for i in range(n):
        curr_date = dates[i]
        if prev_date is not None and curr_date != prev_date:
            daily_change = 0.0
        prev_date = curr_date

        bal_before[i] = bal
        bal += signed[i]
        bal_after[i] = bal
        daily_change += signed[i]
        daily_cum[i] = daily_change

    df.loc[idxs, "sender_balance_before_txn"] = np.round(bal_before, 2)
    df.loc[idxs, "sender_balance_after_txn"] = np.round(bal_after, 2)
    df.loc[idxs, "sender_cumulative_daily_balance_change"] = np.round(daily_cum, 2)
    df.loc[idxs, "sender_current_balance"] = round(bal, 2)

# Balance ratio
df["sender_bal_ratio_after_to_current"] = np.where(
    df["sender_current_balance"] != 0,
    np.round(df["sender_balance_after_txn"] / df["sender_current_balance"], 4),
    0.0
)

print("  Sender balance: 6 columns done")



# ## 7 -- Receiver Account Velocity + Balance (26 columns)
# Mirror of sender metrics computed from the counterparty's perspective.
# For external/unknown counterparties, values default to 0.
# 

# In[ ]:


print("Computing receiver account velocity + balance features...")

# Build receiver-centric lookup: for each account, pre-compute its velocity at each timestamp
# We already computed sender metrics. For receiver, we join sender metrics on counterparty account.

# First, build a lookup: acct -> sorted list of (dt, inflow_amt, outflow_amt, inflow_cnt, outflow_cnt, balance)
# from the sender computation (which covers ALL accounts as senders)

# Efficient approach: create a mapping from acct -> its row indices, then for each transaction,
# look up the counterparty's metrics at the closest preceding timestamp.

print("  Building receiver lookup tables...")

# Pre-build per-account sorted arrays for binary search
acct_data = {}
for acct, gdf in df.groupby("_acct"):
    gdf_sorted = gdf.sort_values("_dt")
    acct_data[acct] = {
        "dts": gdf_sorted["_dt"].values,
        "idxs": gdf_sorted.index.values,
    }

# Initialize receiver columns
for w in WINDOWS:
    df[f"receiver_acct_txn_count_{w}"] = 0
    df[f"receiver_acct_inflow_amt_{w}"] = 0.0
    df[f"receiver_acct_outflow_amt_{w}"] = 0.0
    df[f"receiver_acct_inflow_count_{w}"] = 0
    df[f"receiver_acct_outflow_count_{w}"] = 0

df["receiver_balance_before_txn"] = 0.0
df["receiver_running_balance_txn_amount"] = 0.0
df["receiver_balance_after_txn"] = 0.0
df["receiver_cumulative_daily_balance_change"] = 0.0
df["receiver_current_balance"] = 0.0
df["receiver_bal_ratio_after_to_current"] = 0.0

# For each transaction, look up the counterparty's sender metrics
# (since counterparty is also a sender in some other transaction)
print("  Mapping receiver metrics from counterparty's sender data...")

# Build fast lookup: for each account, its latest computed sender metrics
# We map receiver columns from the sender columns of the counterparty account
# at the closest prior timestamp using the pre-computed sender_acct_* columns

# Group by counterparty, and for each row, find the counterparty's own metrics
cp_groups = df.groupby("_cp")
total_cp = len(cp_groups)
processed = 0

for cp_acct, gdf_main in cp_groups:
    processed += 1
    if processed % 5000 == 0:
        print(f"    Receiver mapping: {processed:,}/{total_cp:,}...")

    if cp_acct in ("", "nan", "None"):
        continue

    # Get the counterparty's own transactions (where it is the sender/_acct)
    if cp_acct not in acct_data:
        continue

    cp_own = df.loc[df["_acct"] == cp_acct].sort_values("_dt")
    if len(cp_own) == 0:
        continue

    cp_dts = cp_own["_dt"].values
    cp_idxs_arr = cp_own.index.values

    # For each row where this account is the counterparty, find nearest prior row in cp_own
    main_idxs = gdf_main.index.values
    main_dts = gdf_main["_dt"].values

    for i in range(len(main_idxs)):
        t = main_dts[i]
        if pd.isna(t):
            continue
        # Binary search for latest cp transaction before t
        pos = np.searchsorted(cp_dts, t, side='right') - 1
        if pos < 0:
            continue
        cp_idx = cp_idxs_arr[pos]

        # Copy sender metrics from counterparty's row as receiver metrics
        row_idx = main_idxs[i]
        for w in WINDOWS:
            df.at[row_idx, f"receiver_acct_txn_count_{w}"] = df.at[cp_idx, f"sender_acct_txn_count_{w}"]
            df.at[row_idx, f"receiver_acct_inflow_amt_{w}"] = df.at[cp_idx, f"sender_acct_inflow_amt_{w}"]
            df.at[row_idx, f"receiver_acct_outflow_amt_{w}"] = df.at[cp_idx, f"sender_acct_outflow_amt_{w}"]
            df.at[row_idx, f"receiver_acct_inflow_count_{w}"] = df.at[cp_idx, f"sender_acct_inflow_count_{w}"]
            df.at[row_idx, f"receiver_acct_outflow_count_{w}"] = df.at[cp_idx, f"sender_acct_outflow_count_{w}"]

        df.at[row_idx, "receiver_balance_before_txn"] = df.at[cp_idx, "sender_balance_before_txn"]
        df.at[row_idx, "receiver_running_balance_txn_amount"] = df.at[cp_idx, "sender_running_balance_txn_amount"]
        df.at[row_idx, "receiver_balance_after_txn"] = df.at[cp_idx, "sender_balance_after_txn"]
        df.at[row_idx, "receiver_cumulative_daily_balance_change"] = df.at[cp_idx, "sender_cumulative_daily_balance_change"]
        df.at[row_idx, "receiver_current_balance"] = df.at[cp_idx, "sender_current_balance"]

# Receiver balance ratio
df["receiver_bal_ratio_after_to_current"] = np.where(
    df["receiver_current_balance"] != 0,
    np.round(df["receiver_balance_after_txn"] / df["receiver_current_balance"], 4),
    0.0
)

# receiver_account_outflow_30d (reuse)
df["receiver_account_outflow_30d"] = df["receiver_acct_outflow_count_30d"]

print("  Receiver velocity + balance: 26 columns done")



# ## 8 -- Volume Balance Ratios (2 columns)
# Measures how much of inflows are immediately re-forwarded.
# 

# In[ ]:


print("Computing volume balance ratios...")

# 24h ratio
s_in_24 = df["sender_acct_inflow_amt_24h"]
s_out_24 = df["sender_acct_outflow_amt_24h"]
df["inflow_outflow_volume_balance_ratio_24h"] = np.where(
    s_in_24 > 0,
    np.round(np.minimum(s_in_24, s_out_24) / s_in_24, 4),
    0.0
)

# 7d ratio
s_in_7d = df["sender_acct_inflow_amt_7d"]
s_out_7d = df["sender_acct_outflow_amt_7d"]
df["inflow_outflow_volume_balance_ratio_7d"] = np.where(
    s_in_7d > 0,
    np.round(np.minimum(s_in_7d, s_out_7d) / s_in_7d, 4),
    0.0
)

print("  Volume balance ratios: 2 columns done")



# ## 9 -- IP Risk Score (9 columns)
# Composite IP-level risk signal combining VPN, emulator, night hours, FATF
# high-risk country, cross-border, shared IP across accounts, geo-mismatch
# between IP location and registered address, minus KYC verification credit.
# 
# **Formula**: `score = 0.10 (base) + vpn(0.15) + emulator(0.10) + night(0.10)`
# `+ country_high(0.10) + cross_border(0.05) + shared_ip(0.10) + geo_mismatch(0.10)`
# `- kyc_verified(0.10)` clipped to [0.0, 1.0]
# 

# In[ ]:


print("Computing IP risk score...")

# -- Resolve flag columns --
def flag_series(col_name):
    if col_name and col_name in df.columns:
        return df[col_name].astype(str).str.strip().str.upper().isin(["Y","1","TRUE","YES"])
    return pd.Series(False, index=df.index)

COL_VPN      = find_col(["vpn_flag", "VPN Flag"])
COL_EMULATOR = find_col(["emulator_flag", "Emulator Flag"])
COL_IP       = find_col(["ip_address", "IP Address of Originating Device"])
COL_GEO      = find_col(["geo_location_city_country", "geo_location", "Geo-Location (City/Country)"])
COL_SENDER   = find_col(["sender_country_code", "sender_country", "Sender Country Code*"])
COL_RECEIVER = find_col(["receiver_country_code", "receiver_country", "Receiver Country Code*"])
COL_RISK     = find_col(["customer_current_risk_score", "risk_score", "Customer Current Risk Score"])
COL_WALKYC   = find_col(["wallet_kyc_category", "wallet_kyc", "Wallet KYC Category"])
COL_VKYC     = find_col(["vkyc_flag", "VKYC Flag"])

is_vpn      = flag_series(COL_VPN)
is_emulator = flag_series(COL_EMULATOR)
is_vkyc     = flag_series(COL_VKYC)

# -- Night transaction (22:00 - 06:00) --
hour = df["_dt"].dt.hour if "_dt" in df.columns else pd.Series(12, index=df.index)
is_night = (hour >= 22) | (hour < 6)

# -- High-risk country (FATF grey/blacklist) --
FATF_HIGH_RISK = {"KP","IR","MM","AF","SY","YE","HT","SD","ML","BF","MZ","TZ",
                   "CD","SS","LY","PK","NI","JM","TR","PH"}

sender_high = df[COL_SENDER].astype(str).str.strip().str.upper().isin(FATF_HIGH_RISK) if COL_SENDER and COL_SENDER in df.columns else pd.Series(False, index=df.index)
receiver_high = df[COL_RECEIVER].astype(str).str.strip().str.upper().isin(FATF_HIGH_RISK) if COL_RECEIVER and COL_RECEIVER in df.columns else pd.Series(False, index=df.index)
country_high = sender_high | receiver_high

# -- Cross-border (sender != receiver country) --
is_cross_border = pd.Series(False, index=df.index)
if COL_SENDER and COL_RECEIVER and COL_SENDER in df.columns and COL_RECEIVER in df.columns:
    s_cc = df[COL_SENDER].astype(str).str.strip().str.upper()
    r_cc = df[COL_RECEIVER].astype(str).str.strip().str.upper()
    is_cross_border = (s_cc != r_cc) & (s_cc != "") & (r_cc != "") & (s_cc != "NAN") & (r_cc != "NAN")

# -- KYC strength (verified KYC = lower risk) --
kyc_verified = is_vkyc.copy()
if COL_WALKYC and COL_WALKYC in df.columns:
    full_kyc = df[COL_WALKYC].astype(str).str.lower().str.contains("full", na=False)
    kyc_verified = kyc_verified | full_kyc
if COL_RISK and COL_RISK in df.columns:
    low_risk = df[COL_RISK].astype(str).str.strip().str.upper().isin(["LOW","1"])
    kyc_verified = kyc_verified | low_risk

# -- IP frequency anomaly (same IP used by multiple distinct accounts) --
ip_shared_score = pd.Series(0.0, index=df.index)
if COL_IP and COL_IP in df.columns:
    ip_unique_accts = df.groupby(df[COL_IP].astype(str).str.strip())[COL_ACCT].transform("nunique")
    ip_acct_count = ip_unique_accts.fillna(1)
    # Normalize: 1 account = 0, 2 = 0.25, 3 = 0.50, 5+ = 1.0
    ip_shared_score = np.clip((ip_acct_count - 1) / 4, 0, 1)
else:
    ip_acct_count = pd.Series(1, index=df.index)

# -- Geo mismatch (IP GPS location far from registered address) --
geo_mismatch = pd.Series(False, index=df.index)
gps_lat_col = find_col(["gps_coordinates_lat"])
cust_lat_col = find_col(["customer_address_lat"])
gps_lon_col = find_col(["gps_coordinates_lon"])
cust_lon_col = find_col(["customer_address_lon"])

if (gps_lat_col and cust_lat_col and gps_lon_col and cust_lon_col
    and gps_lat_col in df.columns and cust_lat_col in df.columns
    and gps_lon_col in df.columns and cust_lon_col in df.columns):
    glat = pd.to_numeric(df[gps_lat_col], errors="coerce").fillna(0)
    clat = pd.to_numeric(df[cust_lat_col], errors="coerce").fillna(0)
    glon = pd.to_numeric(df[gps_lon_col], errors="coerce").fillna(0)
    clon = pd.to_numeric(df[cust_lon_col], errors="coerce").fillna(0)
    # Distance in degrees (1 degree ~ 111 km); flag if > 2 degrees (~220 km)
    dist_deg = np.sqrt((glat - clat)**2 + (glon - clon)**2)
    geo_mismatch = (dist_deg > 2.0) & (glat != 0) & (clat != 0)

# ── Compute IP Score ──
ip_score = (
    0.10                                        # base score
    + is_vpn.astype(float)          * 0.15      # VPN active
    + is_emulator.astype(float)     * 0.10      # emulator detected
    + is_night.astype(float)        * 0.10      # night transaction
    + country_high.astype(float)    * 0.10      # FATF high-risk country
    + is_cross_border.astype(float) * 0.05      # cross-border transaction
    + ip_shared_score               * 0.10      # shared IP across accounts
    + geo_mismatch.astype(float)    * 0.10      # IP location far from address
    - kyc_verified.astype(float)    * 0.10      # KYC verified reduces risk
)

df["ip_risk_score"] = np.clip(np.round(ip_score, 4), 0.0, 1.0)

# Component flags for explainability
df["ip_flag_vpn"] = is_vpn.astype(int)
df["ip_flag_emulator"] = is_emulator.astype(int)
df["ip_flag_night"] = is_night.astype(int)
df["ip_flag_country_high_risk"] = country_high.astype(int)
df["ip_flag_cross_border"] = is_cross_border.astype(int)
df["ip_flag_shared_ip"] = (ip_acct_count > 1).astype(int) if COL_IP and COL_IP in df.columns else 0
df["ip_flag_geo_mismatch"] = geo_mismatch.astype(int)
df["ip_flag_kyc_verified"] = kyc_verified.astype(int)

# Summary
print(f"\nIP Risk Score Distribution:")
print(f"  Min:    {df['ip_risk_score'].min():.3f}")
print(f"  Median: {df['ip_risk_score'].median():.3f}")
print(f"  Mean:   {df['ip_risk_score'].mean():.3f}")
print(f"  P95:    {df['ip_risk_score'].quantile(0.95):.3f}")
print(f"  Max:    {df['ip_risk_score'].max():.3f}")

print(f"\nFlag trigger rates:")
for flag_col in [c for c in df.columns if c.startswith("ip_flag_")]:
    rate = df[flag_col].mean() * 100
    print(f"  {flag_col:<30s} {rate:>6.2f}%")

print(f"\n  IP risk score: 9 columns done (1 score + 8 component flags)")



# ## 10 -- Typology Signal (10 sub-signals)
# Behavioural fingerprint detection for each typology using only observable features.
# Each sub-signal fires 0.08-0.12 when the transaction matches a typology's pattern.
# Sum clipped to [0, 1]. No labels or typology names used.
# 

# In[ ]:


print("Computing typology signal (10 sub-signals)...")
COL_CASH     = find_col(["cash_flag", "Cash Flag"])
COL_CHANNEL  = find_col(["transaction_mode_channel_bank", "channel_bank"])
COL_SENDER   = find_col(["sender_country_code", "sender_country", "Sender Country Code*"])
COL_RECEIVER = find_col(["receiver_country_code", "receiver_country", "Receiver Country Code*"])

# Resolve columns needed
_amt = df["_amt"] if "_amt" in df.columns else pd.to_numeric(df.get(COL_AMOUNT, 0), errors="coerce").fillna(0)
_cash = df.get(COL_CASH, pd.Series("N", index=df.index)).astype(str).str.upper() == "Y" if COL_CASH else pd.Series(False, index=df.index)
_channel = df.get(COL_CHANNEL, pd.Series("", index=df.index)).astype(str).str.upper() if COL_CHANNEL else pd.Series("", index=df.index)

# Sender country / receiver country
_s_cc = df[COL_SENDER].astype(str).str.strip().str.upper() if COL_SENDER and COL_SENDER in df.columns else pd.Series("", index=df.index)
_r_cc = df[COL_RECEIVER].astype(str).str.strip().str.upper() if COL_RECEIVER and COL_RECEIVER in df.columns else pd.Series("", index=df.index)

CORRIDOR_COUNTRIES = {"AE","PK","BD","NP","LK","MM","AF","KP","IR","SY"}

# Helper: safe column access with default 0
def sc(col, default=0):
    return pd.to_numeric(df[col], errors="coerce").fillna(default) if col in df.columns else pd.Series(default, index=df.index)

# ── Sub-signal 1: Structuring ──
structuring_signal = np.where(
    _cash
    & (_amt >= 8000) & (_amt <= 9999)
    & (sc("sender_acct_txn_count_7d") >= 3)
    & (_channel.isin(["BRANCH CASH", "ATM"])),
    0.12, 0.0
)

# ── Sub-signal 2: Circular ──
_in24 = sc("sender_acct_inflow_amt_24h")
_out24 = sc("sender_acct_outflow_amt_24h")
circular_signal = np.where(
    (_out24 > 0) & (_in24 > 0)
    & (np.abs(_in24 - _out24) / np.maximum(_in24, 1) < 0.05)
    & (sc("sender_acct_txn_count_24h") >= 2)
    & (_amt >= 50000),
    0.10, 0.0
)

# ── Sub-signal 3: Funnel ──
funnel_signal = np.where(
    (sc("receiver_acct_inflow_count_24h") >= 10)
    & (sc("receiver_acct_txn_count_24h") >= 10)
    & (sc("inflow_outflow_volume_balance_ratio_24h") >= 0.80)
    & (_amt >= 5000),
    0.12, 0.0
)

# ── Sub-signal 4: Pass-Through ──
_in1h = sc("sender_acct_inflow_amt_1h")
_out1h = sc("sender_acct_outflow_amt_1h")
passthrough_signal = np.where(
    (sc("inflow_outflow_volume_balance_ratio_24h") >= 0.90)
    & (_out1h > 0) & (_in1h > 0)
    & (np.abs(_in1h - _out1h) / np.maximum(_in1h, 1) < 0.06)
    & (_amt >= 200000),
    0.12, 0.0
)

# ── Sub-signal 5: Layering ──
_bal_before = sc("sender_balance_before_txn")
_bal_after = sc("sender_balance_after_txn")
layering_signal = np.where(
    (sc("sender_acct_txn_count_1h") >= 3)
    & (sc("sender_acct_outflow_amt_1h") >= 100000)
    & (_bal_after < _bal_before * 0.3)
    & (_channel.isin(["IMPS","NEFT","UPI","RTGS"])),
    0.12, 0.0
)

# ── Sub-signal 6: Third-Party Web ──
third_party_signal = np.where(
    (sc("receiver_acct_inflow_count_7d") >= 5)
    & (sc("receiver_acct_inflow_amt_7d") >= 50000)
    & (_amt >= 10000) & (_amt <= 100000)
    & (_channel.isin(["NEFT","IMPS","UPI"])),
    0.08, 0.0
)

# ── Sub-signal 7: Money Mule ──
mule_signal = np.where(
    (sc("sender_acct_outflow_count_24h") >= 3)
    & (_in24 > 0)
    & (_out24 >= _in24 * 0.80)
    & (_out24 <= _in24 * 0.98)
    & (_amt >= 20000) & (_amt <= 200000),
    0.10, 0.0
)

# ── Sub-signal 8: High-Risk Corridor ──
corridor_signal = np.where(
    (_s_cc != _r_cc)
    & (_r_cc.isin(CORRIDOR_COUNTRIES))
    & (_amt >= 50000)
    & (sc("sender_acct_outflow_count_30d") >= 3),
    0.10, 0.0
)

# ── Sub-signal 9: Hawala ──
_in7d = sc("sender_acct_inflow_amt_7d")
_out7d = sc("sender_acct_outflow_amt_7d")
hawala_signal = np.where(
    (_in7d > 0) & (_out7d > 0)
    & (np.abs(_in7d - _out7d) / np.maximum(_in7d, 1) < 0.08)
    & (_amt >= 100000)
    & (sc("sender_acct_txn_count_7d") <= 6)
    & (_channel.isin(["NEFT","RTGS","BRANCH CASH"])),
    0.08, 0.0
)

# ── Sub-signal 10: Charity Abuse ──
charity_signal = np.where(
    (sc("receiver_acct_inflow_count_7d") >= 8)
    & (_amt >= 1000) & (_amt <= 50000)
    & (sc("inflow_outflow_volume_balance_ratio_7d") >= 0.70)
    & (sc("receiver_acct_outflow_count_30d") >= 2),
    0.08, 0.0
)

# ── Combined typology signal ──
df["typology_signal"] = np.clip(
    structuring_signal + circular_signal + funnel_signal +
    passthrough_signal + layering_signal + third_party_signal +
    mule_signal + corridor_signal + hawala_signal + charity_signal,
    0.0, 1.0
).round(4)

# Store sub-signals for explainability
df["ts_structuring"] = structuring_signal
df["ts_circular"] = circular_signal
df["ts_funnel"] = funnel_signal
df["ts_passthrough"] = passthrough_signal
df["ts_layering"] = layering_signal
df["ts_third_party"] = third_party_signal
df["ts_mule"] = mule_signal
df["ts_corridor"] = corridor_signal
df["ts_hawala"] = hawala_signal
df["ts_charity"] = charity_signal

# Summary
print(f"\nTypology signal distribution:")
print(f"  Zero (no pattern):  {(df['typology_signal'] == 0).sum():,} ({(df['typology_signal'] == 0).mean()*100:.1f}%)")
print(f"  > 0 (some pattern): {(df['typology_signal'] > 0).sum():,} ({(df['typology_signal'] > 0).mean()*100:.1f}%)")
print(f"  > 0.2 (multi):      {(df['typology_signal'] > 0.2).sum():,}")
print(f"  Mean: {df['typology_signal'].mean():.4f}  Max: {df['typology_signal'].max():.4f}")

print(f"\nSub-signal trigger rates:")
for name in ["structuring","circular","funnel","passthrough","layering",
             "third_party","mule","corridor","hawala","charity"]:
    col = f"ts_{name}"
    rate = (df[col] > 0).mean() * 100
    print(f"  {name:<16s}: {rate:>6.2f}%")

print(f"\n  Typology signal: 11 columns done (1 composite + 10 sub-signals)")



# ## 11 -- Convergence Risk
# Measures how much the account's counterparty network looks like
# a concentration/relay node. Combines fan-in, fan-out, pass-through ratio,
# balance drain, and receiver relay score.
# 

# In[ ]:


print("Computing convergence risk...")

# Fan-in: how many senders feed the receiver
fan_in_score = np.clip(sc("receiver_acct_inflow_count_30d") / 20, 0, 1)

# Fan-out: how many receivers this sender pays
fan_out_score = np.clip(sc("sender_acct_outflow_count_30d") / 15, 0, 1)

# Pass-through ratio: how close inflow = outflow
pt_24h = sc("inflow_outflow_volume_balance_ratio_24h")
pt_7d  = sc("inflow_outflow_volume_balance_ratio_7d")
pass_through_combined = np.maximum(pt_24h, pt_7d)

# Balance drain: is the account being emptied
_bal_after = sc("sender_balance_after_txn")
_bal_current = sc("sender_current_balance")
balance_drain = np.where(
    _bal_current > 0,
    np.clip(1.0 - (_bal_after / np.maximum(_bal_current, 1)), 0, 1),
    0
)

# Receiver relay: is the receiver also forwarding money
_recv_out_30 = sc("receiver_account_outflow_30d")
_recv_txn_30 = sc("receiver_acct_txn_count_30d")
receiver_relay = np.clip(_recv_out_30 / np.maximum(_recv_txn_30, 1), 0, 1)

df["convergence_risk"] = np.clip(
    fan_in_score       * 0.25 +
    fan_out_score      * 0.20 +
    pass_through_combined * 0.25 +
    balance_drain      * 0.15 +
    receiver_relay     * 0.15,
    0.0, 1.0
).round(4)

# Component columns
df["cr_fan_in"] = fan_in_score.round(4)
df["cr_fan_out"] = fan_out_score.round(4)
df["cr_passthrough"] = pass_through_combined.round(4)
df["cr_balance_drain"] = np.round(balance_drain, 4)
df["cr_receiver_relay"] = receiver_relay.round(4)

print(f"  Mean: {df['convergence_risk'].mean():.4f}  Median: {df['convergence_risk'].median():.4f}  Max: {df['convergence_risk'].max():.4f}")
print(f"  > 0.5: {(df['convergence_risk'] > 0.5).sum():,} txns")
print(f"  Convergence risk: 6 columns done (1 composite + 5 components)")



# ## 12 -- Temporal Risk
# Captures timing anomalies beyond night/weekend — velocity spikes,
# burst patterns, and customer-level timing deviations.
# 

# In[ ]:


print("Computing temporal risk...")

# Night score (from IP risk cell)
night_score = df["ip_flag_night"].astype(float) if "ip_flag_night" in df.columns else pd.Series(0, index=df.index)

# Weekend + high value
_hour = df["_dt"].dt.hour if "_dt" in df.columns else pd.Series(12, index=df.index)
_dow = df["_dt"].dt.dayofweek if "_dt" in df.columns else pd.Series(0, index=df.index)
weekend_hv = np.where((_dow >= 5) & (_amt > 20000), 1.0, 0.0)

# Velocity spike: current hour vs daily average
hourly_avg = sc("sender_acct_txn_count_24h") / 24
velocity_spike = np.clip(
    (sc("sender_acct_txn_count_1h") - hourly_avg) / np.maximum(hourly_avg, 0.5),
    0, 1
)

# Burst detection
burst_flag = np.where(sc("sender_acct_txn_count_1h") >= 4, 1.0, 0.0)

# Customer-level velocity anomaly
cust_hourly_avg = sc("sender_cust_txn_count_24h") / 24
cust_velocity_spike = np.clip(
    (sc("sender_cust_txn_count_1h") - cust_hourly_avg) / np.maximum(cust_hourly_avg, 0.5),
    0, 1
)

df["temporal_risk"] = np.clip(
    night_score        * 0.20 +
    weekend_hv         * 0.15 +
    velocity_spike     * 0.25 +
    burst_flag         * 0.20 +
    cust_velocity_spike * 0.20,
    0.0, 1.0
).round(4)

# Component columns
df["tr_night"] = night_score.astype(float)
df["tr_weekend_hv"] = weekend_hv
df["tr_velocity_spike"] = np.round(velocity_spike, 4)
df["tr_burst"] = burst_flag
df["tr_cust_velocity_spike"] = np.round(cust_velocity_spike, 4)

print(f"  Mean: {df['temporal_risk'].mean():.4f}  Max: {df['temporal_risk'].max():.4f}")
print(f"  Night: {(night_score > 0).sum():,}  Bursts: {(burst_flag > 0).sum():,}")
print(f"  Temporal risk: 6 columns done (1 composite + 5 components)")



# ## 13 -- Fraud Intensity Score (4 components, no normalization)
# Original composite risk score: `raw = rule_risk×35 + behaviour_risk×30 + ip_risk×18 + device_risk×17`
# 
# **No normalization applied** — raw weighted sum IS the final score.
# Each component is [0,1], weights sum to 100, so raw score range is 0-100.
# **Bands**: very_low (0-20), low (20-40), medium (40-60), high (60-80), critical (80-100)
# 

# In[ ]:


print("Computing Fraud Intensity Score (4 components, no normalization)...")

# ── Component 1: Rule Risk (weight 35) ──
rule_score_col = find_col(["rule_score"])
rule_score_vals = pd.to_numeric(df.get(rule_score_col, 0), errors="coerce").fillna(0) if rule_score_col else pd.Series(0, index=df.index)
rule_p99 = max(rule_score_vals.quantile(0.99), 1)
rule_risk = rule_score_vals #np.clip(rule_score_vals / rule_p99, 0, 1)

# ── Component 2: Behaviour Risk (weight 30) ──
# behaviour_risk = np.clip(
#     np.clip(df["sender_acct_txn_count_1h"] / 5, 0, 1) * 0.3 +
#     np.clip(df["sender_acct_txn_count_24h"] / 20, 0, 1) * 0.2 +
#     np.clip(df["inflow_outflow_volume_balance_ratio_24h"], 0, 1) * 0.3 +
#     np.clip(df["sender_acct_outflow_amt_24h"] / 500000, 0, 1) * 0.2,
#     0, 1
# )
## OLD ONE ################

# behaviour_risk = np.clip(np.clip(df["sender_acct_inflow_amt_24h"] / 128751.90, 0, 1) * 0.20 +
#     np.clip(df["sender_acct_txn_count_1h"] / 1.00, 0, 1) * 0.20 +
#     np.clip(df["sender_cust_inflow_amt_24h"] / 171442.22, 0, 1) * 0.20 +
#     np.clip(df["sender_cust_txn_count_1h"] / 1.00, 0, 1) * 0.20 +
#     np.clip(df["sender_acct_outflow_amt_24h"] / 187280.37, 0, 1) * 0.20, 0, 1)

behaviour_risk = np.clip(
    np.clip(df["sender_acct_txn_count_1h"] / 1.00, 0, 1) * 0.1626 +
    np.clip(df["sender_acct_outflow_amt_24h"] / 187280.37, 0, 1) * 0.1404 +
    np.clip(df["sender_cust_txn_count_1h"] / 1.00, 0, 1) * 0.1279 +
    np.clip(df["sender_acct_outflow_count_24h"] / 2.00, 0, 1) * 0.1142 +
    np.clip(df["sender_cust_outflow_count_24h"] / 3.00, 0, 1) * 0.0944 +
    np.clip(df["sender_acct_txn_count_24h"] / 3.00, 0, 1) * 0.0900 +
    np.clip(df["sender_cust_txn_count_24h"] / 4.00, 0, 1) * 0.0783 +
    np.clip(df["receiver_acct_inflow_amt_24h"] / 111907.27, 0, 1) * 0.0665 +
    np.clip(df["inflow_outflow_volume_balance_ratio_24h"] / 1.00, 0, 1) * 0.0659 +
    np.clip(df["sender_acct_outflow_count_7d"] / 8.00, 0, 1) * 0.0597 +
    0, 0, 1)

# ── Component 3: IP Risk (weight 18) ──
ip_risk = df["ip_risk_score"] if "ip_risk_score" in df.columns else pd.Series(0, index=df.index)

# ── Component 4: Device Risk (weight 17) ──
emu_flag = df["ip_flag_emulator"].astype(float) if "ip_flag_emulator" in df.columns else 0
vpn_flag = df["ip_flag_vpn"].astype(float) if "ip_flag_vpn" in df.columns else 0
device_risk = np.clip(
    emu_flag * 0.7 +
    vpn_flag * 0.3,
    0, 1
)

# ── Raw FIS = Final Score (no normalization) ──
# Weights sum to 100, each component is [0,1], so score is already 0-100
fis_raw = (
    rule_risk       * 20 +
    behaviour_risk  * 55 +
    ip_risk         * 15 +
    device_risk     * 10
)

df["fraud_intensity_score_raw"] = np.round(fis_raw, 4)

# No normalization — raw IS the score
df["fraud_intensity_score"] = np.round(np.clip(fis_raw, 0, 100), 2)

# Bands
def fis_band(score):
    if score >= 80: return "critical"
    if score >= 60: return "high"
    if score >= 40: return "medium"
    if score >= 20: return "low"
    return "very_low"

df["fis_band"] = df["fraud_intensity_score"].apply(fis_band)

# Summary
print(f"\nFIS Distribution:")
print(f"  {'Band':<12s} {'Count':>10s} {'%':>7s}")
print(f"  {'-'*32}")
for band in ["critical","high","medium","low","very_low"]:
    cnt = (df["fis_band"] == band).sum()
    print(f"  {band:<12s} {cnt:>10,} {cnt/len(df)*100:>6.2f}%")

print(f"\n  Component weights: rule=35, behaviour=30, ip=18, device=17")
print(f"  Total weights: 100 (raw score IS the 0-100 score, no normalization)")

print(f"\n  Score stats:")
print(f"    min={fis_raw.min():.2f}  median={fis_raw.median():.2f}  mean={fis_raw.mean():.2f}  max={fis_raw.max():.2f}")

if "is_aml" in df.columns:
    aml_fis = df.loc[df["is_aml"] == 1, "fraud_intensity_score"]
    clean_fis = df.loc[df["is_aml"] != 1, "fraud_intensity_score"]
    print(f"\n  AML   FIS: mean={aml_fis.mean():.1f}  median={aml_fis.median():.1f}  p75={aml_fis.quantile(0.75):.1f}")
    print(f"  Clean FIS: mean={clean_fis.mean():.1f}  median={clean_fis.median():.1f}  p75={clean_fis.quantile(0.75):.1f}")
    print(f"  Separation: {aml_fis.mean()/max(clean_fis.mean(),0.01):.1f}x")

print(f"\n  Fraud Intensity Score: 3 columns done (raw, score, band)")



# ## CHECK THE COMPOSITION OF FIS

# In[ ]:


# Skipped in predict mode — this cell is training-only diagnostics/scoring
if RUN_MODE == "train":
    print("=" * 90)
    print("FIS SCORE ANALYSIS — Clean vs Typology Transactions")
    print("=" * 90)

    fis_col = "fraud_intensity_score_raw"
    typ_col = "aml_typology" if "aml_typology" in df.columns else None
    aml_col = "is_aml" if "is_aml" in df.columns else None

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: Clean vs All AML — FIS Summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("SECTION 1: Clean vs AML — FIS Score Summary")
    print("─" * 90)

    if aml_col:
        aml = df[df[aml_col] == 1]
        clean = df[df[aml_col] != 1]

        print(f"\n  {'Segment':<25s} │ {'Count':>10s} │ {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s} │ {'P25':>8s} {'P75':>8s} {'P90':>8s} {'P95':>8s} {'P99':>8s}")
        print("  " + "─" * 125)

        for label, subset in [("Clean (is_aml=0)", clean), ("AML (is_aml=1)", aml), ("ALL Transactions", df)]:
            s = subset[fis_col]
            print(f"  {label:<25s} │ {len(subset):>10,} │ {s.mean():>8.2f} {s.median():>8.2f} {s.min():>8.2f} {s.max():>8.2f} {s.std():>8.2f} │ {s.quantile(0.25):>8.2f} {s.quantile(0.75):>8.2f} {s.quantile(0.90):>8.2f} {s.quantile(0.95):>8.2f} {s.quantile(0.99):>8.2f}")

        gap = aml[fis_col].mean() - clean[fis_col].mean()
        ratio = aml[fis_col].mean() / max(clean[fis_col].mean(), 0.01)
        print(f"\n  Gap (AML mean − Clean mean): {gap:.2f} points")
        print(f"  Ratio (AML mean / Clean mean): {ratio:.1f}x")

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: FIS per Typology — Mean, Min, Max
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("SECTION 2: FIS Score per Typology — Mean, Min, Max, Percentiles")
    print("─" * 90)

    if typ_col and typ_col in df.columns:
        all_typs = set()
        for t in df[typ_col].dropna():
            for part in str(t).split("; "):
                if part.strip():
                    all_typs.add(part.strip())

        print(f"\n  {'Typology':<35s} │ {'Count':>8s} │ {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s} │ {'P25':>7s} {'P75':>7s} {'P95':>7s}")
        print("  " + "─" * 115)

        # Clean baseline
        cs = clean[fis_col]
        print(f"  {'[Clean Baseline]':<35s} │ {len(clean):>8,} │ {cs.mean():>8.2f} {cs.median():>8.2f} {cs.min():>8.2f} {cs.max():>8.2f} {cs.std():>8.2f} │ {cs.quantile(0.25):>7.2f} {cs.quantile(0.75):>7.2f} {cs.quantile(0.95):>7.2f}")
        print("  " + "─" * 115)

        typ_results = []
        for typ in sorted(all_typs):
            mask = df[typ_col].astype(str).str.contains(typ, na=False)
            s = df.loc[mask, fis_col]
            cnt = mask.sum()
            if cnt == 0:
                continue
            typ_results.append((typ, cnt, s.mean(), s.median(), s.min(), s.max(), s.std(),
                                s.quantile(0.25), s.quantile(0.75), s.quantile(0.95)))
            print(f"  {typ:<35s} │ {cnt:>8,} │ {s.mean():>8.2f} {s.median():>8.2f} {s.min():>8.2f} {s.max():>8.2f} {s.std():>8.2f} │ {s.quantile(0.25):>7.2f} {s.quantile(0.75):>7.2f} {s.quantile(0.95):>7.2f}")

        # All AML combined
        print("  " + "─" * 115)
        as_ = aml[fis_col]
        print(f"  {'[All AML Combined]':<35s} │ {len(aml):>8,} │ {as_.mean():>8.2f} {as_.median():>8.2f} {as_.min():>8.2f} {as_.max():>8.2f} {as_.std():>8.2f} │ {as_.quantile(0.25):>7.2f} {as_.quantile(0.75):>7.2f} {as_.quantile(0.95):>7.2f}")

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: FIS Band Distribution — Clean vs Each Typology
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("SECTION 3: FIS Band Distribution — Clean vs Each Typology")
    print("─" * 90)

    bands = ["critical", "high", "medium", "low", "very_low"]

    if typ_col:
        print(f"\n  {'Segment':<35s} │ {'Critical':>10s} {'High':>10s} {'Medium':>10s} {'Low':>10s} {'Very Low':>10s} │ {'Med+ %':>8s} {'High+ %':>8s}")
        print("  " + "─" * 115)

        # Clean
        ct = len(clean)
        cb = {b: (clean["fis_band"] == b).sum() for b in bands}
        cm_pct = sum(cb[b] for b in ["critical","high","medium"]) / max(ct,1) * 100
        ch_pct = sum(cb[b] for b in ["critical","high"]) / max(ct,1) * 100
        print(f"  {'[Clean]':<35s} │ {cb['critical']:>8,}  {cb['high']:>8,}  {cb['medium']:>8,}  {cb['low']:>8,}  {cb['very_low']:>8,}  │ {cm_pct:>7.1f}% {ch_pct:>7.1f}%")
        print("  " + "─" * 115)

        for typ in sorted(all_typs):
            mask = df[typ_col].astype(str).str.contains(typ, na=False)
            subset = df[mask]
            cnt = len(subset)
            if cnt == 0:
                continue
            tb = {b: (subset["fis_band"] == b).sum() for b in bands}
            tm_pct = sum(tb[b] for b in ["critical","high","medium"]) / max(cnt,1) * 100
            th_pct = sum(tb[b] for b in ["critical","high"]) / max(cnt,1) * 100
            print(f"  {typ:<35s} │ {tb['critical']:>8,}  {tb['high']:>8,}  {tb['medium']:>8,}  {tb['low']:>8,}  {tb['very_low']:>8,}  │ {tm_pct:>7.1f}% {th_pct:>7.1f}%")

        # All AML
        print("  " + "─" * 115)
        at = len(aml)
        ab = {b: (aml["fis_band"] == b).sum() for b in bands}
        am_pct = sum(ab[b] for b in ["critical","high","medium"]) / max(at,1) * 100
        ah_pct = sum(ab[b] for b in ["critical","high"]) / max(at,1) * 100
        print(f"  {'[All AML]':<35s} │ {ab['critical']:>8,}  {ab['high']:>8,}  {ab['medium']:>8,}  {ab['low']:>8,}  {ab['very_low']:>8,}  │ {am_pct:>7.1f}% {ah_pct:>7.1f}%")

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: Overlap Detection — Typology txns in very_low/low
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("SECTION 4: Problem Detection — Typology Transactions Scoring LOW")
    print("These are AML transactions the FIS is MISSING (score < 20 = very_low)")
    print("─" * 90)

    if typ_col:
        print(f"\n  {'Typology':<35s} │ {'Total':>8s} │ {'Very Low':>10s} {'% V.Low':>8s} │ {'Low':>8s} {'% Low':>8s} │ {'Below 40':>10s} {'% <40':>8s}")
        print("  " + "─" * 110)

        problem_typs = []
        for typ in sorted(all_typs):
            mask = df[typ_col].astype(str).str.contains(typ, na=False)
            subset = df[mask]
            cnt = len(subset)
            if cnt == 0:
                continue
            vlow = (subset[fis_col] < 20).sum()
            low = ((subset[fis_col] >= 20) & (subset[fis_col] < 40)).sum()
            below40 = (subset[fis_col] < 40).sum()
            vlow_pct = vlow / cnt * 100
            below40_pct = below40 / cnt * 100

            flag = " ⚠ NEEDS ATTENTION" if below40_pct > 50 else (" ⚡ CHECK" if below40_pct > 30 else "")
            print(f"  {typ:<35s} │ {cnt:>8,} │ {vlow:>10,} {vlow_pct:>7.1f}% │ {low:>8,} {low/cnt*100:>7.1f}% │ {below40:>10,} {below40_pct:>7.1f}%{flag}")

            if below40_pct > 30:
                problem_typs.append((typ, cnt, below40_pct))

        if problem_typs:
            print(f"\n  ⚠ Typologies with >30% of transactions scoring below 40:")
            for typ, cnt, pct in sorted(problem_typs, key=lambda x: -x[2]):
                print(f"    → {typ} ({cnt:,} txns, {pct:.1f}% below 40)")
            print(f"    Action: Increase rule_risk or behaviour_risk weight, or lower component thresholds")
        else:
            print(f"\n  ✓ All typologies have >70% of transactions scoring 40+ — FIS weights are well-calibrated")

    # ══════════════════════════════════════════════════════════════
    # SECTION 5: Score Overlap Zone — Where Clean and AML Scores Mix
    # ══════════════════════════════════════════════════════════════
    print("\n" + "─" * 90)
    print("SECTION 5: Score Overlap Zone — Where Clean and AML Mix")
    print("─" * 90)

    num_buckets = 10

    quantiles = np.linspace(0, 1, num_buckets + 1)
    bucket_edges = df["fraud_intensity_score_raw"].quantile(quantiles).values

    score_ranges = [
        (round(bucket_edges[i], 2), round(bucket_edges[i+1], 2))
        for i in range(len(bucket_edges) - 1)
    ]

    print(score_ranges)

    #print(df["score_bucket"].value_counts().sort_index())

    if aml_col:
        score_ranges = score_ranges

        print(f"\n  {'FIS Range':<15s} │ {'Clean Count':>12s} {'Clean %':>8s} │ {'AML Count':>10s} {'AML %':>8s} │ {'AML Rate':>9s} │ {'Bar':>20s}")
        print("  " + "─" * 95)

        for lo, hi in score_ranges:
            c_cnt = ((clean[fis_col] >= lo) & (clean[fis_col] < hi)).sum()
            a_cnt = ((aml[fis_col] >= lo) & (aml[fis_col] < hi)).sum()
            total_in_range = c_cnt + a_cnt
            aml_rate = a_cnt / max(total_in_range, 1) * 100
            c_pct = c_cnt / max(len(clean),1) * 100
            a_pct = a_cnt / max(len(aml),1) * 100

            bar_c = "░" * min(int(c_pct / 2), 20)
            bar_a = "█" * min(int(a_pct / 2), 20)
            bar = bar_a + bar_c
            print(f"  {lo:>6.2f} - {hi:<6.2f} │ {c_cnt:>12,} {c_pct:>7.1f}% │ {a_cnt:>10,} {a_pct:>7.1f}% │ {aml_rate:>8.1f}% │ {bar}")
            #print(f"  {lo:>3d} - {hi:<3d}       │ {c_cnt:>12,} {c_pct:>7.1f}% │ {a_cnt:>10,} {a_pct:>7.1f}% │ {aml_rate:>8.1f}% │ {bar}")

        # Separation threshold analysis
        print(f"\n  Threshold Analysis:")
        for thresh in [20, 30, 40, 50, 60]:
            aml_above = (aml[fis_col] >= thresh).sum()
            clean_above = (clean[fis_col] >= thresh).sum()
            aml_recall = aml_above / max(len(aml),1) * 100
            precision = aml_above / max(aml_above + clean_above, 1) * 100
            print(f"    FIS >= {thresh}: AML recall = {aml_recall:.1f}% ({aml_above:,}/{len(aml):,}) | Precision = {precision:.1f}% | Clean false positives = {clean_above:,}")

    print(f"\n{'='*90}")
    print("KEY INSIGHT: If AML mean >> Clean mean AND overlap zone is narrow → weights are good")
    print("If many AML txns fall in 0-20 range → those typologies need stronger signal in FIS")
    print(f"{'='*90}")


# ## CHECK THE RIGHT SET OF GRAPH FEATURES

# In[ ]:


# # Skipped in predict mode — this cell is training-only diagnostics/scoring
# if RUN_MODE == "train":
#     print("=" * 100)
#     print("DIAGNOSTIC: What Makes the 4 Problem Typologies Different?")
#     print("=" * 100)

#     aml_mask = df["is_aml"] == 1
#     clean_mask = df["is_aml"] != 1
#     typ_col = "aml_typology"

#     # The 4 problem typologies vs 4 well-detected ones
#     problem_typs = ["Charity Abuse", "Funnel Account Network", "Money Mule Network", "Third-Party Payment Web"]
#     good_typs = ["Circular Transaction Loop", "High-Risk Corridor Transfer", "Pass-Through Transit Hub", "Rapid Multi-Hop Layering"]

#     # ── Features to compare ──
#     features = [
#         # Current behaviour sub-signals
#         "sender_acct_inflow_amt_24h", "sender_acct_txn_count_1h", "sender_cust_inflow_amt_24h",
#         "sender_cust_txn_count_1h", "sender_acct_outflow_amt_24h",
#         # Counterparty / spread features (THESE are what we're missing)
#         "sender_acct_unique_counterparties_24h", "sender_acct_unique_counterparties_7d",
#         "sender_cust_unique_counterparties_24h", "sender_cust_unique_counterparties_7d",
#         # Volume counts (longer windows)
#         "sender_acct_txn_count_24h", "sender_acct_txn_count_7d", "sender_acct_txn_count_30d",
#         "sender_cust_txn_count_24h", "sender_cust_txn_count_7d",
#         # Outflow counts (fan-out)
#         "sender_acct_outflow_count_24h", "sender_acct_outflow_count_7d",
#         "sender_cust_outflow_count_24h", "sender_cust_outflow_count_7d",
#         # Inflow counts (funnel-in)
#         "sender_acct_inflow_count_24h", "sender_acct_inflow_count_7d",
#         # Receiver-side (concentration detection)
#         "receiver_acct_unique_senders_24h", "receiver_acct_unique_senders_7d",
#         "receiver_acct_inflow_amt_24h", "receiver_acct_txn_count_24h",
#         # Balance movement
#         "sender_pct_balance_moved", "sender_balance_velocity_7d",
#         "inflow_outflow_volume_balance_ratio_24h", "inflow_outflow_volume_balance_ratio_7d",
#         # IP spread
#         "ip_unique_accounts_24h", "ip_unique_cifs_24h", "ip_cross_acct_flag",
#     ]

#     existing = [f for f in features if f in df.columns]

#     print(f"\n{'Feature':<48s} │ {'Clean':>8s} │ {'Charity':>8s} {'Funnel':>8s} {'Mule':>8s} {'3rdPty':>8s} │ {'Circ':>8s} {'Corr':>8s} {'Pass':>8s} {'Layer':>8s}")
#     print("─" * 145)

#     feature_scores = []

#     for feat in existing:
#         vals = pd.to_numeric(df[feat], errors="coerce").fillna(0)
#         cm = vals[clean_mask].mean()

#         typ_means = {}
#         for typ in problem_typs + good_typs:
#             mask = df[typ_col].astype(str).str.contains(typ, na=False)
#             typ_means[typ] = vals[mask].mean() if mask.sum() > 0 else 0

#         # Score: how much do problem typologies exceed clean?
#         prob_avg_ratio = np.mean([typ_means[t] / max(cm, 0.001) for t in problem_typs])
#         good_avg_ratio = np.mean([typ_means[t] / max(cm, 0.001) for t in good_typs])

#         # We want features where problem typologies score HIGH relative to clean
#         feature_scores.append((feat, prob_avg_ratio, cm, typ_means))

#         # Highlight if problem typology mean > 1.5x clean
#         vals_str = []
#         for typ in problem_typs:
#             r = typ_means[typ] / max(cm, 0.001)
#             marker = "★" if r > 2 else ("●" if r > 1.3 else " ")
#             vals_str.append(f"{typ_means[typ]:>7.1f}{marker}")
#         for typ in good_typs:
#             vals_str.append(f"{typ_means[typ]:>8.1f}")

#         print(f"  {feat:<46s} │ {cm:>8.1f} │ {' '.join(vals_str[:4])} │ {' '.join(vals_str[4:])}")

#     # Sort by problem typology separation
#     feature_scores.sort(key=lambda x: -x[1])

#     print(f"\n\n{'─'*100}")
#     print("TOP 10 FEATURES THAT SEPARATE PROBLEM TYPOLOGIES FROM CLEAN")
#     print("These are what the behaviour_risk component SHOULD include")
#     print(f"{'─'*100}")

#     print(f"\n  {'Rank':<5s} {'Feature':<48s} {'Problem Typ / Clean Ratio':>25s}")
#     print("  " + "─" * 80)
#     for i, (feat, ratio, cm, _) in enumerate(feature_scores[:10], 1):
#         print(f"  {i:<5d} {feat:<48s} {ratio:>24.2f}x")

#     top_features = [f[0] for f in feature_scores[:10]]

#     # ══════════════════════════════════════════════════════════════
#     # BUILD THE NEW BEHAVIOUR RISK
#     # ══════════════════════════════════════════════════════════════
#     print(f"\n\n{'═'*100}")
#     print("BUILDING ENHANCED BEHAVIOUR RISK — Targeting Problem Typologies")
#     print(f"{'═'*100}")

#     # Normalize each feature using P95 of clean
#     norm = {}
#     for feat in top_features:
#         vals = pd.to_numeric(df[feat], errors="coerce").fillna(0)
#         p95 = vals[clean_mask].quantile(0.95)
#         if p95 <= 0: p95 = max(vals.quantile(0.95), 1)
#         norm[feat] = np.clip(vals / p95, 0, 1)

#     # Weight by problem typology separation ratio (higher ratio = more weight)
#     total_ratio = sum(f[1] for f in feature_scores[:10])
#     auto_weights = {f[0]: f[1] / total_ratio for f in feature_scores[:10]}

#     # Compute enhanced behaviour risk
#     behav_enhanced = pd.Series(0.0, index=df.index)
#     for feat, w in auto_weights.items():
#         behav_enhanced += norm[feat] * w
#     behav_enhanced = np.clip(behav_enhanced, 0, 1)

#     print(f"\n  Enhanced Behaviour Risk sub-signals:")
#     print(f"  {'Feature':<48s} {'P95 Clean':>12s} {'Weight':>8s} {'AML Mean':>10s} {'Cln Mean':>10s} {'Ratio':>7s}")
#     print("  " + "─" * 100)
#     for feat in top_features:
#         vals = pd.to_numeric(df[feat], errors="coerce").fillna(0)
#         p95 = vals[clean_mask].quantile(0.95)
#         if p95 <= 0: p95 = max(vals.quantile(0.95), 1)
#         w = auto_weights[feat]
#         am = norm[feat][aml_mask].mean()
#         cm_n = norm[feat][clean_mask].mean()
#         print(f"  {feat:<48s} {p95:>12.2f} {w:>7.4f} {am:>10.4f} {cm_n:>10.4f} {am/max(cm_n,0.0001):>6.1f}x")

#     print(f"\n  Enhanced behaviour AML mean: {behav_enhanced[aml_mask].mean():.4f}")
#     print(f"  Enhanced behaviour Clean mean: {behav_enhanced[clean_mask].mean():.4f}")
#     print(f"  Ratio: {behav_enhanced[aml_mask].mean()/max(behav_enhanced[clean_mask].mean(),0.0001):.1f}x")

#     # ══════════════════════════════════════════════════════════════
#     # TEST FIS WITH ENHANCED BEHAVIOUR
#     # ══════════════════════════════════════════════════════════════
#     print(f"\n\n{'═'*100}")
#     print("FIS WEIGHT SCENARIOS WITH ENHANCED BEHAVIOUR RISK")
#     print(f"{'═'*100}")

#     rule_score_vals_raw = pd.to_numeric(df.get(rule_score_col, 0), errors="coerce").fillna(0)
# #     rule_score_vals_raw = pd.to_numeric(
# #     df.get(rule_score_col, pd.Series(0, index=df.index)),
# #     errors="coerce"
# # ).fillna(0)
#     comp_rule = np.clip(rule_score_vals_raw / max(rule_score_vals_raw.quantile(0.99), 1), 0, 1)
#     comp_ip = df["ip_risk_score"].clip(0, 1) if "ip_risk_score" in df.columns else pd.Series(0, index=df.index)
#     emu = df["ip_flag_emulator"].astype(float) if "ip_flag_emulator" in df.columns else 0
#     vpn = df["ip_flag_vpn"].astype(float) if "ip_flag_vpn" in df.columns else 0
#     comp_device = np.clip(emu * 0.7 + vpn * 0.3, 0, 1)

#     # Old behaviour for comparison
#     comp_behav_old = np.clip(
#         np.clip(df["sender_acct_inflow_amt_24h"] / 128751.90, 0, 1) * 0.20 +
#         np.clip(df["sender_acct_txn_count_1h"] / 1.00, 0, 1) * 0.20 +
#         np.clip(df["sender_cust_inflow_amt_24h"] / 171442.22, 0, 1) * 0.20 +
#         np.clip(df["sender_cust_txn_count_1h"] / 1.00, 0, 1) * 0.20 +
#         np.clip(df["sender_acct_outflow_amt_24h"] / 187280.37, 0, 1) * 0.20,
#         0, 1
#     )

#     scenarios = {
#         "OLD: rule*20+oldBehav*55+ip*15+dev*10": (comp_rule, comp_behav_old, comp_ip, comp_device, 20, 55, 15, 10),
#         "NEW1: rule*20+newBehav*55+ip*15+dev*10": (comp_rule, behav_enhanced, comp_ip, comp_device, 20, 55, 15, 10),
#         "NEW2: rule*25+newBehav*50+ip*15+dev*10": (comp_rule, behav_enhanced, comp_ip, comp_device, 25, 50, 15, 10),
#         "NEW3: rule*15+newBehav*60+ip*15+dev*10": (comp_rule, behav_enhanced, comp_ip, comp_device, 15, 60, 15, 10),
#         "NEW4: rule*25+newBehav*45+ip*18+dev*12": (comp_rule, behav_enhanced, comp_ip, comp_device, 25, 45, 18, 12),
#         "NEW5: rule*20+newBehav*50+ip*18+dev*12": (comp_rule, behav_enhanced, comp_ip, comp_device, 20, 50, 18, 12),
#         "NEW6: rule*30+newBehav*45+ip*15+dev*10": (comp_rule, behav_enhanced, comp_ip, comp_device, 30, 45, 15, 10),
#     }

#     print(f"\n  {'Scenario':<50s} │ {'AML':>6s} {'Cln':>6s} {'Ratio':>6s} │ {'AML≥60':>6s} {'AML≥40':>6s} {'Cln<40':>6s}")
#     print("  " + "─" * 95)

#     best_name = ""; best_ratio = 0
#     fis_cache = {}

#     for name, (cr, cb, ci, cd, wr, wb, wi, wd) in scenarios.items():
#         fis = np.clip(cr*wr + cb*wb + ci*wi + cd*wd, 0, 100)
#         fis_cache[name] = fis
#         am = fis[aml_mask].mean(); cm = fis[clean_mask].mean()
#         ratio = am / max(cm, 0.01)
#         marker = ""
#         if ratio > best_ratio: best_ratio = ratio; best_name = name; marker = " ◄"
#         print(f"  {name:<50s} │ {am:>6.1f} {cm:>6.1f} {ratio:>5.1f}x │ {(fis[aml_mask]>=60).mean()*100:>5.1f}% {(fis[aml_mask]>=40).mean()*100:>5.1f}% {(fis[clean_mask]<40).mean()*100:>5.1f}%{marker}")

#     # ══════════════════════════════════════════════════════════════
#     # PER-TYPOLOGY WITH BEST FIS — Focus on Problem Typologies
#     # ══════════════════════════════════════════════════════════════
#     print(f"\n\n{'═'*100}")
#     print(f"PER-TYPOLOGY RESULTS: {best_name}")
#     print(f"{'═'*100}")

#     fis_best = fis_cache[best_name]
#     fis_old = fis_cache["OLD: rule*20+oldBehav*55+ip*15+dev*10"]

#     def band(s):
#         if s >= 80: return "critical"
#         if s >= 60: return "high"
#         if s >= 40: return "medium"
#         if s >= 20: return "low"
#         return "very_low"

#     band_best = fis_best.apply(band)

#     all_typs = set()
#     for t in df[typ_col].dropna():
#         for part in str(t).split("; "):
#             if part.strip(): all_typs.add(part.strip())

#     print(f"\n  {'Typology':<35s} │ {'N':>7s} │ {'OLD Mean':>8s} {'NEW Mean':>8s} {'Δ':>7s} │ {'Crit%':>6s} {'Hi%':>5s} {'Med+%':>6s} {'<40%':>6s} │ {'Status':>10s}")
#     print("  " + "─" * 115)

#     # Clean
#     s_new = fis_best[clean_mask]; s_old = fis_old[clean_mask]; b = band_best[clean_mask]
#     print(f"  {'[Clean]':<35s} │ {clean_mask.sum():>7,} │ {s_old.mean():>8.2f} {s_new.mean():>8.2f} {s_new.mean()-s_old.mean():>+6.1f} │ {(b=='critical').mean()*100:>5.1f}% {(b=='high').mean()*100:>4.1f}% {((b=='critical')|(b=='high')|(b=='medium')).mean()*100:>5.1f}% {'—':>6s} │")
#     print("  " + "─" * 115)

#     for typ in sorted(all_typs):
#         mask = df[typ_col].astype(str).str.contains(typ, na=False)
#         cnt = mask.sum()
#         if cnt == 0: continue
#         s_new = fis_best[mask]; s_old = fis_old[mask]; b = band_best[mask]
#         delta = s_new.mean() - s_old.mean()
#         crit = (b=="critical").mean()*100
#         high = (b=="high").mean()*100
#         medp = ((b=="critical")|(b=="high")|(b=="medium")).mean()*100
#         below40 = (s_new<40).mean()*100

#         if below40 > 50: status = "⚠ STILL LOW"
#         elif below40 > 30: status = "⚡ IMPROVED"
#         elif below40 > 15: status = "✓ GOOD"
#         else: status = "★ EXCELLENT"

#         is_problem = typ in problem_typs
#         prefix = "→ " if is_problem else "  "
#         print(f"{prefix}{typ:<35s} │ {cnt:>7,} │ {s_old.mean():>8.2f} {s_new.mean():>8.2f} {delta:>+6.1f} │ {crit:>5.1f}% {high:>4.1f}% {medp:>5.1f}% {below40:>5.1f}% │ {status:>10s}")

#     # All AML
#     s_new = fis_best[aml_mask]; s_old = fis_old[aml_mask]; b = band_best[aml_mask]
#     print("  " + "─" * 115)
#     print(f"  {'[All AML]':<35s} │ {aml_mask.sum():>7,} │ {s_old.mean():>8.2f} {s_new.mean():>8.2f} {s_new.mean()-s_old.mean():>+6.1f} │ {(b=='critical').mean()*100:>5.1f}% {(b=='high').mean()*100:>4.1f}% {((b=='critical')|(b=='high')|(b=='medium')).mean()*100:>5.1f}% {(s_new<40).mean()*100:>5.1f}% │")

#     # ══════════════════════════════════════════════════════════════
#     # PRINT FINAL CODE
#     # ══════════════════════════════════════════════════════════════
#     print(f"\n\n{'═'*100}")
#     print("PASTE THIS INTO CELL 26:")
#     print(f"{'═'*100}")

#     # Extract weights from best scenario
#     _, _, _, _, wr, wb, wi, wd = scenarios[best_name]

#     print(f'''
#     # ── Component 2: Enhanced Behaviour Risk (weight {wb}) ──
#     behaviour_risk = np.clip(''')

#     for feat in top_features:
#         vals = pd.to_numeric(df[feat], errors="coerce").fillna(0)
#         p95 = vals[clean_mask].quantile(0.95)
#         if p95 <= 0: p95 = max(vals.quantile(0.95), 1)
#         w = auto_weights[feat]
#         print(f'    np.clip(df["{feat}"] / {p95:.2f}, 0, 1) * {w:.4f} +')

#     print(f'    0, 0, 1)')
#     print(f'''
#     # ── FIS Formula ──
#     fis_raw = (
#         rule_risk       * {wr} +
#         behaviour_risk  * {wb} +
#         ip_risk         * {wi} +
#         device_risk     * {wd}
#     )''')

#     print(f"\n{'═'*100}")


# ## 14 -- Summary & Export
# 

# In[ ]:


print("DONEEE")


# In[ ]:


# import pandas as pd
# df = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_features.parquet")
# df.head()


# ## DEFINING TO WRITE TABLE FAST

# In[ ]:


from sqlalchemy import create_engine
import db_config
import io

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


import pandas as pd
import numpy as np

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


# In[ ]:


# Drop internal working columns + FIS intermediate component columns
internal = [c for c in df.columns if c.startswith("_")]

# FIS sub-signal and component columns (used during computation, not needed in output)
fis_intermediate = [c for c in df.columns if (
    c.startswith("ts_")          # typology sub-signals (ts_structuring, ts_circular, etc.)
    or c.startswith("cr_")       # convergence risk components (cr_fan_in, cr_fan_out, etc.)
    or c.startswith("tr_")       # temporal risk components (tr_night, tr_burst, etc.)
    or c.startswith("fis_w_")    # FIS weighted contributions (fis_w_rule, fis_w_typology, etc.)
)]

drop_cols = list(set(internal + fis_intermediate))
drop_cols = [c for c in drop_cols if c in df.columns]
df_out = df.drop(columns=drop_cols)

print(f"Dropped {len(internal)} internal columns + {len(fis_intermediate)} FIS intermediate columns = {len(drop_cols)} total")

# # Count new features
# original_cols = set()
# for alt in ["stg_transactions_rules.parquet","stg_transactions_rules.csv"]:
#     try:
#         if alt.endswith('.parquet'):
#             sample = pd.read_parquet(INPUT_FILE, columns=None)
#         else:
#             sample = pd.read_csv(INPUT_FILE, nrows=1)
#         original_cols = set(sample.columns)
#         break
#     except:
#         pass

# new_cols = [c for c in df_out.columns if c not in original_cols and not c.startswith("_")]
# print(f"New features added: {len(new_cols)}")
# print(f"Total columns: {len(df_out.columns)}")

# # Feature category breakdown
# categories = {
#     "Sender Acct Velocity": [c for c in new_cols if c.startswith("sender_acct_")],
#     "Sender Cust Velocity": [c for c in new_cols if c.startswith("sender_cust_")],
#     "Sender Balance":       [c for c in new_cols if c.startswith("sender_bal") or c.startswith("sender_balance") or c.startswith("sender_running") or c.startswith("sender_cumulative") or c.startswith("sender_current")],
#     "Receiver Features":    [c for c in new_cols if c.startswith("receiver_")],
#     "Volume Ratios":        [c for c in new_cols if "volume_balance_ratio" in c],
#     "IP Risk":              [c for c in new_cols if c.startswith("ip_risk") or c.startswith("ip_flag")],
#     "Typology Signal":      [c for c in new_cols if c == "typology_signal"],
#     "Convergence Risk":     [c for c in new_cols if c == "convergence_risk"],
#     "Temporal Risk":        [c for c in new_cols if c == "temporal_risk"],
#     "Fraud Intensity":      [c for c in new_cols if c.startswith("fraud_intensity") or c.startswith("fis_")],
# }

# print(f"\nFeature breakdown:")
# for cat, cols in categories.items():
#     print(f"  {cat:<25s} {len(cols):>3} columns")


# In[ ]:


df_out = pd.DataFrame(df_out)
df_out['datestamp'] = pd.to_datetime(df_out['datestamp'],format="%d-%m-%Y",errors="coerce")
df_out['customer_cif_creation_date'] = pd.to_datetime(df_out['customer_cif_creation_date'],format="%d-%m-%Y",errors="coerce")
df_out['account_wallet_opening_date'] = pd.to_datetime(df_out['account_wallet_opening_date'],format="%d-%m-%Y",errors="coerce")
df_out['kyc_update_date'] = pd.to_datetime(df_out['kyc_update_date'],format="%d-%m-%Y",errors="coerce")
df_out['account_wallet_inoperative_date'] = pd.to_datetime(df_out['account_wallet_inoperative_date'],format="%d-%m-%Y",errors="coerce")
df_out['date_of_incorporation'] = pd.to_datetime(df_out['date_of_incorporation'],format="%d-%m-%Y",errors="coerce")
df_out['date_of_birth'] = pd.to_datetime(df_out['date_of_birth'],format="%d-%m-%Y",errors="coerce")
df_out["professional_experience_years"] = pd.to_numeric(
    df_out["professional_experience_years"],
    errors="coerce"
).astype("Int64")

df_out.head()
df_out['cif_beneficial_owners'].unique()


# In[ ]:


print(df_out.shape)
df_out.head()


# In[ ]:


# Write to PostgreSQL (full refresh each run)
from datetime import datetime
df_out = pd.DataFrame(df_out)
df_out["loaded_at"] = datetime.now()

write_table_fast(df_out, "stg_transactions_features", mode="replace")
print(f"Rules output written: {len(df_out):,} rows x {len(df_out.columns)} cols")


# In[ ]:


# df_out.to_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_features.parquet", index=False)


# In[ ]:


print("DONEEE")

