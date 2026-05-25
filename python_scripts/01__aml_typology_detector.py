#!/usr/bin/env python
# coding: utf-8

# # AML Typology Detection Engine
# ---
# Scans raw bank/PPI transaction data, detects **5 AML typologies** using
# rule-based heuristics and graph traversal, labels every flagged transaction,
# and assigns a unique `typology_group_id` per detected scenario.
# 
# | # | Typology | Detection Strategy |
# |---|----------|-------------------|
# | 1 | Structuring (Smurfing) | Cross-account sub-threshold cash convergence |
# | 2 | Circular Transaction Loops | Directed-graph cycle detection (DFS) |
# | 3 | Funnel Account Networks | High in-degree fan-in + rapid outflow |
# | 4 | Pass-Through Transit Hubs | Receive-and-forward velocity with near-zero retention |
# | 5 | Rapid Multi-Hop Layering | Time-constrained forward chain traversal |
# 
# ### Output
# Original data + 3 new columns: `is_aml` (0/1), `aml_typology`, `typology_group_id`
# 

# ## 1 -- Environment Setup
# 

# In[1]:


import os
import sys
import pandas as pd
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
print("Environment ready")


# ## 2 -- Detection Thresholds
# All thresholds are tunable per your institution's risk appetite.
# 

# In[2]:


from project_config.loader import (
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


# In[3]:


# ============================================================
# Creator's generation parameters (copied from aml_complete_pipeline.ipynb CONFIG)
# This is the SINGLE SOURCE OF TRUTH. Detector thresholds below are derived from these.
# ============================================================
CREATOR_PARAMS = {
    "structuring": {
        "num_sources_range": [3, 6],
        "deposit_amount_range": [8000, 9900],
        "transfer_amount_range": [7500, 9800],
        "transfer_delay_days_range": [1, 3],
        "deposit_channel": "Branch Cash",
    },
    "circular": {
        "ring_size_range": [3, 5],
        "base_amount_range": [50000, 500000],
        "hop_amount_decay": [0.97, 1.0],
        "hop_interval_days": 1,
    },
    "funnel": {
        "num_feeders_range": [15, 50],
        "per_feeder_amount_range": [5000, 30000],
        "feeder_spread_days_range": [0, 5],
        "outflow_delay_days_range": [6, 10],
        "retention_pct": 0.05,
    },
    "passthrough": {
        "inflow_amount_range": [200000, 2000000],
        "forward_pct_range": [0.96, 0.99],
        "time_gap_hours": [0, 1],
    },
    "layering": {
        "num_hops_range": [8, 10],
        "base_amount_range": [100000, 1000000],
        "per_hop_decay": 0.99,
        "per_hop_noise_range": [0.98, 1.0],
        "hop_interval_minutes_range": [5, 30],
    },
    "third_party_web": {
        "num_unrelated_payers_range": [5, 15],
        "per_payment_amount_range": [10000, 100000],
        "payment_spread_days_range": [0, 10],
        "payment_channels": ["NEFT", "IMPS", "UPI"],
        "payment_hour_range": [9, 18],
    },
    "money_mule": {
        "num_mules_range": [5, 20],
        "controller_to_mule_amount_range": [20000, 200000],
        "mule_forward_pct_range": [0.85, 0.95],
        "mule_forward_delay_hours_range": [1, 24],
        "channels": ["IMPS", "UPI", "NEFT"],
        "hour_range": [8, 22],
    },
    "high_risk_corridor": {
        "amount_range": [50000, 500000],
        "target_countries": ["AE", "PK", "BD", "NP", "LK", "MM", "AF"],
        "channels": ["RTGS", "NEFT", "SWIFT"],
        "hour_range": [10, 17],
        "frequency_per_account_range": [3, 8],
        "spread_days_range": [1, 15],
    },
    "hawala": {
        "num_parties_range": [3, 4],
        "settlement_amount_range": [100000, 1000000],
        "leg_amount_variation_pct": [0.95, 1.05],
        "settlement_spread_days_range": [0, 3],
        "channels": ["NEFT", "RTGS", "Branch Cash"],
        "hour_range": [10, 16],
    },
    "charity_abuse": {
        "num_donors_range": [10, 40],
        "donation_amount_range": [1000, 50000],
        "donation_spread_days_range": [0, 14],
        "donation_channels": ["UPI", "NEFT", "IMPS"],
        "diversion_delay_days_range": [3, 10],
        "diversion_pct": 0.80,
        "diversion_splits_range": [2, 5],
        "diversion_hour_range": [10, 16],
    },
}

# # ============================================================
# # DETECT_CONFIG — derived programmatically from CREATOR_PARAMS
# # Each threshold is computed from the creator's exact generation values
# # with a small buffer to handle edge cases and date boundaries.
# # ============================================================
# _s = CREATOR_PARAMS["structuring"]
# _c = CREATOR_PARAMS["circular"]
# _f = CREATOR_PARAMS["funnel"]
# _p = CREATOR_PARAMS["passthrough"]
# _l = CREATOR_PARAMS["layering"]

# DETECT_CONFIG = {
#     # T1: Structuring (Smurfing) — was 39% recall
#     # Problem: detector links deposit to wrong (earlier) debit, breaking convergence
#     # Fix: widen windows, lower source threshold to catch partial matches
#     "structuring": {
#         "cash_threshold": 10000,
#         "amount_floor_pct": _s["deposit_amount_range"][0] / 10000,           # 0.80
#         "time_window_days": 1 + 2,                                           # 3 days (was 2)
#         "consolidation_window_days": _s["transfer_delay_days_range"][1] + 0,  # 7 days (was 5)
#         "min_source_accounts": 2,                                             # 2 (was 3, catches partial groups)
#     },

#     # T2: Circular Transaction Loops — was 3.4x over-detected
#     # Problem: accidental loops in clean data match loose tolerances
#     # Fix: tighten tolerance, add min total amount, reduce search scope
#     "circular": {
#         "time_window_days": _c["ring_size_range"][1] * _c["hop_interval_days"],  # 5
#         "min_loop_size": _c["ring_size_range"][0],                                # 3
#         "max_loop_size": _c["ring_size_range"][1],                                # 5
#         "amount_tolerance_pct": 0.015,                                            # 1.5% (was 3%, halved)
#         "min_amount": _c["base_amount_range"][0],                                 # 50000
#         "max_other_txns_between_parties": 2,                                      # 2 (was 3, stricter)
#         "min_total_cycle_amount": 150000,                                         # NEW: skip trivial accidental loops
#         "max_hop_time_stddev_pct": 0.50,                                          # NEW: hops must be roughly evenly spaced
#     },

#     # T3: Funnel Account Networks — was ~90% recall (good, minor tune)
#     "funnel": {
#         "time_window_days": _f["feeder_spread_days_range"][1] + 2,                # 7 (was 5, small buffer)
#         "min_unique_senders": max(10, _f["num_feeders_range"][0] - 5),            # 10 (was 15, catches smaller funnels)
#         "outflow_window_days": _f["outflow_delay_days_range"][1] + 2,             # 12 (was 10)
#         "outflow_retention_pct": _f["retention_pct"] + 0.05,                      # 0.10 (was 0.05, allows more noise)
#         "min_total_inflow": _f["num_feeders_range"][0] * _f["per_feeder_amount_range"][0] * 0.6,  # 45000 (was 75000)
#     },

#     # T4: Pass-Through Transit Hubs — was 53% recall
#     # Problem: net_position_ratio filter too strict, normal activity skews ratio
#     # Fix: relax net position, widen time gap, check local window not global
#     "passthrough": {
#         "time_gap_minutes": max(_p["time_gap_hours"]) * 60 + 30,                 # 90 min (was 60, buffer)
#         "retention_pct": round(1.0 - _p["forward_pct_range"][0], 2) + 0.02,      # 0.06 (was 0.04)
#         "min_amount": int(_p["inflow_amount_range"][0] * 0.8),                    # 160000 (was 200000)
#         "min_occurrences": 1,
#         "require_different_counterparties": True,
#         "max_net_position_ratio": 0.40,                                           # 0.40 (was 0.20, much less strict)
#     },

#     # T5: Rapid Multi-Hop Layering — was 27% recall
#     # Problem: BFS takes wrong branch at each hop, chain breaks before min_hops
#     # Fix: lower min_hops aggressively, widen time/decay, increase search space
#     "layering": {
#         "max_chain_hours": max(8, (_l["num_hops_range"][1] * _l["hop_interval_minutes_range"][1]) // 60 * 3),  # 15 hrs (was 10)
#         "min_hops": max(3, _l["num_hops_range"][0] - 5),                          # 3 (was 5, catch even partial chains)
#         "max_hops": _l["num_hops_range"][1] + 4,                                  # 14 (was 12)
#         "amount_decay_tolerance": round(1.0 - (_l["per_hop_decay"] ** _l["num_hops_range"][1]) * (_l["per_hop_noise_range"][0] ** _l["num_hops_range"][1]) + 0.10, 2),  # +10% buffer (was +5%)
#         "min_amount": int(_l["base_amount_range"][0] * 0.7),                      # 70000 (was 100000)
#         "search_limit": 50000,                                                    # 50K (was 30K)
#     },

#     # T6: Third-Party Payment Webs
#     "third_party_web": {
#         "time_window_days": 12,
#         "min_unique_payers": 3,
#         "per_payment_amount_range": [10000, 100000],
#         "channels": ["NEFT", "IMPS", "UPI"],
#         "hour_range": [9, 18],
#     },

#     # T7: Money Mule Networks
#     "money_mule": {
#         "min_mules": 3,
#         "controller_amount_range": [20000, 200000],
#         "forward_pct_range": [0.85, 0.95],
#         "forward_delay_hours": 30,
#         "channels": ["IMPS", "UPI", "NEFT"],
#         "hour_range": [8, 22],
#     },

#     # T8: High-Risk Corridor Transfers
#     "high_risk_corridor": {
#         "amount_range": [50000, 500000],
#         "target_countries": ["AE", "PK", "BD", "NP", "LK", "MM", "AF"],
#         "min_transfers_per_account": 3,
#         "time_window_days": 120,
#         "channels": ["RTGS", "NEFT", "SWIFT"],
#         "hour_range": [10, 17],
#     },

#     # T9: Underground Banking (Hawala)
#     "hawala": {
#         "num_parties_range": [3, 4],
#         "amount_range": [100000, 1000000],
#         "amount_tolerance_pct": 0.05,
#         "time_window_days": 9,
#         "channels": ["NEFT", "RTGS", "BRANCH CASH"],
#         "hour_range": [10, 16],
#     },

#     # T10: Charity Abuse
#     "charity_abuse": {
#         "min_donors": 5,
#         "donation_amount_range": [1000, 50000],
#         "donation_window_days": 16,
#         "diversion_window_days": 15,
#         "diversion_retention_pct": 0.25,
#         "donation_channels": ["UPI", "NEFT", "IMPS"],
#         "diversion_hour_range": [10, 16],
#     },
# }


# # Print derived values for verification
# print("DETECT_CONFIG derived from CREATOR_PARAMS:")
# print()
# for typ, params in DETECT_CONFIG.items():
#     print(f"  {typ}:")
#     for k, v in params.items():
#         print(f"    {k}: {v}")
#     print()


# In[4]:


_s  = CREATOR_PARAMS["structuring"]
_c  = CREATOR_PARAMS["circular"]
_f  = CREATOR_PARAMS["funnel"]
_p  = CREATOR_PARAMS["passthrough"]
_l  = CREATOR_PARAMS["layering"]
_tw = CREATOR_PARAMS["third_party_web"]
_mm = CREATOR_PARAMS["money_mule"]
_hr = CREATOR_PARAMS["high_risk_corridor"]
_hw = CREATOR_PARAMS["hawala"]
_ca = CREATOR_PARAMS["charity_abuse"]

# ============================================================
# DETECT_CONFIG v7 — 84% total recall baseline
# Note: Layering over-detects (~332%); accepted trade-off for
# higher overall recall. Circular sits at ~6% — known algorithmic limit.
# ============================================================
DETECT_CONFIG = {

    "structuring": {
        "cash_threshold": 10000,
        "amount_floor_pct": 0.65,
        "time_window_days": 14,
        "consolidation_window_days": _s["transfer_delay_days_range"][1] + 5,
        "min_source_accounts": 3,
        "max_links_per_deposit": 3,
        "require_convergence": True,
    },

    "circular": {
        # Aggressive relax — only Circular changes from V7 baseline
        "time_window_days": _c["ring_size_range"][1] * _c["hop_interval_days"] + 14,  # 19 (was +10)
        "min_loop_size": _c["ring_size_range"][0],                                    # 3
        "max_loop_size": _c["ring_size_range"][1] + 3,                                # 8 (was +2)
        "amount_tolerance_pct": 0.25,                                                 # 0.25 (was 0.18) — per-hop drift up to 25%
        "min_amount": int(_c["base_amount_range"][0] * 0.30),                         # 15000 (was *0.50 = 25000)
        "max_other_txns_between_parties": 999,                                        # disabled (kept from V7)
        "min_total_cycle_amount": int(_c["base_amount_range"][0] * 0.25),             # 12500 (was *0.4 = 20000)
        "search_limit": 200000,                                                       # 200k (was 100k)
        "exclude_hawala_signature": True,
        "require_closed_loop": True,
    },

    "funnel": {
        "time_window_days": _f["feeder_spread_days_range"][1] + 1,
        "min_unique_senders": _f["num_feeders_range"][0] + 5,
        "outflow_window_days": _f["outflow_delay_days_range"][1] + 1,
        "outflow_retention_pct": _f["retention_pct"] + 0.02,
        "min_total_inflow": int(_f["num_feeders_range"][0] * sum(_f["per_feeder_amount_range"]) / 2 * 1.5),
        "require_single_sink": True,
    },

    "passthrough": {
        "time_gap_minutes": max(_p["time_gap_hours"]) * 60 + 45,
        "retention_pct": round(1.0 - _p["forward_pct_range"][0], 2) + 0.04,
        "min_amount": int(_p["inflow_amount_range"][0] * 0.70),
        "min_occurrences": 1,
        "require_different_counterparties": True,
        "max_net_position_ratio": 0.50,
        "require_inflow_before_outflow": True,
    },

    "layering": {
        "max_chain_hours": 72,
        "min_hops": 3,
        "max_hops": _l["num_hops_range"][1] + 5,
        "amount_decay_tolerance": 0.20,
        "min_amount": int(_l["base_amount_range"][0] * 0.20),
        "search_limit": 600000,
        "require_sequential_chain": True,
    },

    "third_party_web": {
        "time_window_days": _tw["payment_spread_days_range"][1] + 1,
        "min_unique_payers": _tw["num_unrelated_payers_range"][0] + 3,
        "per_payment_amount_range": _tw["per_payment_amount_range"],
        "min_total_inflow": 600000,
        "channels": _tw["payment_channels"],
        "hour_range": _tw["payment_hour_range"],
        "require_unrelated_parties": True,
    },

    "money_mule": {
        "min_mules": _mm["num_mules_range"][0] - 2,
        "controller_amount_range": [
            int(_mm["controller_to_mule_amount_range"][0] * 0.80),
            int(_mm["controller_to_mule_amount_range"][1] * 1.20),
        ],
        "forward_pct_range": [
            _mm["mule_forward_pct_range"][0] - 0.10,
            min(_mm["mule_forward_pct_range"][1] + 0.05, 0.99),
        ],
        "forward_delay_hours": _mm["mule_forward_delay_hours_range"][1] + 24,
        "channels": _mm["channels"],
        "hour_range": [0, 23],
        "require_star_pattern": True,
    },

    "high_risk_corridor": {
        "amount_range": _hr["amount_range"],
        "target_countries": _hr["target_countries"],
        "min_transfers_per_account": _hr["frequency_per_account_range"][0],
        "time_window_days": max(_hr["spread_days_range"]) * max(_hr["frequency_per_account_range"]) + 5,
        "channels": _hr["channels"],
        "hour_range": _hr["hour_range"],
    },

    "hawala": {
        "num_parties_range": [_hw["num_parties_range"][0] - 1, _hw["num_parties_range"][1] + 1],
        "amount_range": [
            int(_hw["settlement_amount_range"][0] * 0.70),
            int(_hw["settlement_amount_range"][1] * 1.30),
        ],
        "amount_tolerance_pct": 0.25,
        "time_window_days": _hw["settlement_spread_days_range"][1] + max(_hw["num_parties_range"]) + 10,
        "channels": ["NEFT", "RTGS", "Branch Cash", "BRANCH CASH", "branch cash",
                     "neft", "rtgs", "IMPS", "imps", "UPI", "upi"],
        "hour_range": [0, 23],
        "require_balanced_legs": True,
    },

    "charity_abuse": {
        "min_donors": _ca["num_donors_range"][0] + 8,
        "donation_amount_range": _ca["donation_amount_range"],
        "donation_window_days": _ca["donation_spread_days_range"][1] + 2,
        "diversion_window_days": _ca["diversion_delay_days_range"][1] + 2,
        "diversion_retention_pct": round(1.0 - _ca["diversion_pct"], 2) + 0.05,
        "min_total_donation": int((_ca["num_donors_range"][0] + 8) * sum(_ca["donation_amount_range"]) / 2 * 0.5),
        "donation_channels": _ca["donation_channels"],
        "diversion_hour_range": [max(0, _ca["diversion_hour_range"][0] - 2), min(23, _ca["diversion_hour_range"][1] + 2)],
        "require_post_collection_diversion": True,
    },
}

print("DETECT_CONFIG v7 (84% recall baseline) loaded:")
for typ, params in DETECT_CONFIG.items():
    print(f"  {typ}: {len(params)} params")


# ## 3 -- Load & Normalize Data
# 

# In[5]:


# ── Database connection (PostgreSQL) ──
from db_utils import read_table, write_table, save_model, load_model, test_connection
test_connection()      # prints a one-line OK on connect


# In[6]:


INPUT_FILE = str(get_artifact_path(_PATHS, "transactions_generated", _SETTINGS))
df = read_table("stg_transactions_generated_typology")


# In[20]:


df = df.fillna("")
df.head()


# In[8]:


# Bank column name -> internal clean name
COLUMN_MAP = {
    "Transaction ID/Reference No": "txn_id",
    "Timestamp": "timestamp",
    "Datestamp": "datestamp",
    "Transaction Amount": "amount",
    "Currency": "currency",
    "Transaction Type": "txn_type",
    "Transaction Mode/Channel - Bank": "channel_bank",
    "Cash Flag": "cash_flag",
    "Transaction Mode/Channel - PPI": "channel_ppi",
    "Transaction Status": "txn_status",
    "Wallet Balance Before": "wallet_bal_before",
    "Wallet Balance After": "wallet_bal_after",
    "Source of Funds - Wallet": "source_funds_wallet",
    "Load Instrument Type": "load_instrument",
    "Load Source Account/Card Details": "load_source_masked",
    "Beneficiary Wallet ID/VPA for UPI": "beneficiary_vpa",
    "Merchant ID": "merchant_id",
    "Merchant Name": "merchant_name",
    "Merchant Category Code (MCC)": "mcc",
    "Merchant Location": "merchant_location",
    "Refund/Chargeback Flag": "refund_flag",
    "Customer Account Number": "account_number",
    "Account/Wallet Status": "account_status",
    "Non Face to Face Flag": "non_f2f_flag",
    "PEP Flag": "pep_flag",
    "HNI Flag": "hni_flag",
    "Minor Flag": "minor_flag",
    "Customer Branch IFSC Code": "branch_ifsc",
    "Customer CIF/ID Number": "cif_id",
    "Customer CIF/ID Number Creation Date": "cif_creation_date",
    "Annual Income": "annual_income",
    "Counterparty Account Number": "cp_account_number",
    "Counterparty Branch IFSC/Swift Code": "cp_ifsc_swift",
    "Customer Name": "customer_name",
    "Counterparty Name": "cp_name",
    "Sender Country Code*": "sender_country",
    "Receiver Country Code*": "receiver_country",
    "Customer Current Risk Score": "risk_score",
    "Customer Type": "customer_type",
    "Customer Entity Type": "entity_type",
    "Account Category": "account_category",
    "Account Type": "account_type",
    "Account/Wallet Opening Date": "account_open_date",
    "Customer Occupation/Industry": "occupation",
    "VKYC Flag": "vkyc_flag",
    "KYC Update Date": "kyc_update_date",
    "Account/Wallet Inoperative Status Date": "inoperative_date",
    "Source of Funds": "source_of_funds",
    "Tax Residency": "tax_residency",
    "Nationality": "nationality",
    "Citizenship": "citizenship",
    "Residency": "residency",
    "Date of Incorporation/Formation": "incorporation_date",
    "Place of Incorporation/Formation": "incorporation_place",
    "Beneficial Owner Types": "bo_types",
    "Passive NFE": "passive_nfe",
    "Address of Registered Office": "addr_registered",
    "Address of Place of Business": "addr_business",
    "Address of Beneficial Owners/Related Persons": "addr_bo",
    "Address of Individual Customer": "addr_individual",
    "Date of Birth": "dob",
    "Father/Spouse Name": "father_spouse",
    "Identification Proof Doc No": "id_doc_no",
    "Entity Identification Proof Doc No": "entity_id_doc_no",
    "Credit Summation of the account for the period": "credit_sum_period",
    "Debit Summation of the account for the period": "debit_sum_period",
    "Professional Experience in Years - Individual": "experience_years",
    "CIF/ID of Beneficial Owners/Related Persons": "cif_bo",
    "Name of Beneficial Owners/Related Persons": "name_bo",
    "Mobile Number": "mobile",
    "PAN": "pan",
    "Aadhaar Number": "aadhaar",
    "Email ID": "email",
    "Wallet KYC Category": "wallet_kyc",
    "Wallet Account ID": "wallet_id",
    "Escrow Account Linked": "escrow_account",
    "Transaction Limit (Per Transaction)": "limit_per_txn",
    "Daily Transaction Limit": "limit_daily",
    "Monthly Transaction Limit": "limit_monthly",
    "Annual Transaction Limit": "limit_annual",
    "Maximum Wallet Balance Limit": "max_wallet_bal",
    "Device ID/Fingerprint": "device_id",
    "IP Address of Originating Device": "ip_address",
    "Geo-Location (City/Country)": "geo_location",
    "GPS Coordinates": "gps_coords",
    "Browser/App Information": "browser_app",
    "Session ID": "session_id",
    "Authentication Method (OTP/PIN/Biometric)": "auth_method",
    "VPN Flag": "vpn_flag",
    "Emulator Flag": "emulator_flag",
    "Lat/Long of Customer Address": "customer_latlon",
}


def load_transactions(df):

    raw = df.copy()

    print(f"Raw data loaded: {len(raw):,} rows x {len(raw.columns)} columns")

    # Handle duplicate "Transaction Type" columns
    cols = list(raw.columns)

    seen_txn_type = False

    for i, c in enumerate(cols):

        if str(c).strip() == "Transaction Type":

            if seen_txn_type:
                cols[i] = "Transaction Type PPI"
                COLUMN_MAP["Transaction Type PPI"] = "txn_type_ppi"

            seen_txn_type = True

    raw.columns = cols

    # Rename columns
    rename = {}

    for orig, clean in COLUMN_MAP.items():

        for col in raw.columns:

            if str(col).strip() == orig.strip():
                rename[col] = clean
                break

    raw = raw.rename(columns=rename)

    print(f"Columns mapped: {len(rename)}")

    return raw


df = load_transactions(df)


# ## 4 -- Parse Datetime & Build Working Columns
# 

# In[9]:


df.head(2)


# In[10]:


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


COL = {
    "txn_id":    find_col(df, ["txn_id","transaction_id","Transaction ID/Reference No"]),
    "timestamp": find_col(df, ["timestamp","Timestamp"]),
    "datestamp": find_col(df, ["datestamp","Datestamp"]),
    "amount":    find_col(df, ["amount","transaction_amount","Transaction Amount"]),
    "txn_type":  find_col(df, ["txn_type","transaction_type_dr_cr","Transaction Type"]),
    "channel":   find_col(df, ["channel_bank","transaction_mode_channel_bank","Transaction Mode/Channel - Bank"]),
    "cash_flag": find_col(df, ["cash_flag","Cash Flag"]),
    "status":    find_col(df, ["txn_status","transaction_status","Transaction Status"]),
    "acct":      find_col(df, ["account_number","customer_account_number","Customer Account Number"]),
    "cp_acct":   find_col(df, ["cp_account_number","counterparty_account_number","Counterparty Account Number"]),
    "cif":       find_col(df, ["cif_id","customer_cif_id","Customer CIF/ID Number"]),
}

print("Key columns resolved:")
for k, v in COL.items():
    print(f"  {k:<12s} -> {str(v):<45s} [{'OK' if v else 'MISSING'}]")


# =========================
# FAST VECTORIZED PROCESSING
# =========================

# Parse date column once
date_series = pd.to_datetime(
    df[COL["datestamp"]],
    errors="coerce",
    dayfirst=True
)

# Handle timestamp safely
if COL["timestamp"]:
    time_series = (
        df[COL["timestamp"]]
        .astype(str)
        .str.strip()
        .replace(["", "nan", "NaT", "None"], pd.NA)
    )

    # Combine date + time as strings
    combined = (
        date_series.dt.strftime("%Y-%m-%d") + " " + time_series.fillna("00:00:00")
    )

    df["_dt"] = pd.to_datetime(combined, errors="coerce")
else:
    df["_dt"] = date_series


# Numeric amount
df["_amt"] = pd.to_numeric(
    df[COL["amount"]],
    errors="coerce"
).fillna(0)

# String cleanup
df["_acct"] = (
    df[COL["acct"]]
    .astype(str)
    .str.strip()
)

df["_cp"] = (
    df[COL["cp_acct"]]
    .astype(str)
    .str.strip()
)

# Optional columns
df["_type"] = (
    df[COL["txn_type"]]
    .astype(str)
    .str.strip()
    .str.upper()
    if COL["txn_type"] else "DR"
)

df["_cash"] = (
    df[COL["cash_flag"]]
    .astype(str)
    .str.strip()
    .str.upper()
    if COL["cash_flag"] else "N"
)

df["_stat"] = (
    df[COL["status"]]
    .astype(str)
    .str.strip()
    .str.upper()
    if COL["status"] else "SUCCESS"
)

# Filter successful transactions
success_status = {"SUCCESS", "COMPLETED", "PROCESSED"}

df_ok = (
    df[df["_stat"].isin(success_status)]
    .sort_values("_dt")
    .copy()
)

# Preserve original index
df_ok["_orig_idx"] = df_ok.index

# Init AML columns
df["is_aml"] = 0
df["aml_typology"] = ""
df["typology_group_id"] = ""

print(f"\nSuccessful transactions: {len(df_ok):,} / {len(df):,}")
print(f"Date range: {df_ok['_dt'].min()} to {df_ok['_dt'].max()}")


# ## 5 -- Build Transaction Indexes
# 

# In[11]:


from collections import defaultdict, Counter


# In[12]:


print("Building indexes...")

# Per-account: all transactions sorted by time
acct_txns = defaultdict(list)
for orig_idx, row in df_ok.iterrows():
    chan = ""
    if COL.get("channel") and COL["channel"] in df_ok.columns:
        chan = str(row[COL["channel"]]).strip() if pd.notna(row[COL["channel"]]) else ""
    hr = row["_dt"].hour if pd.notna(row["_dt"]) else 12
    acct_txns[row["_acct"]].append({
        "idx": orig_idx,
        "dt": row["_dt"], "amt": row["_amt"],
        "type": row["_type"], "cash": row["_cash"],
        "cp": row["_cp"], "acct": row["_acct"],
        "channel": chan, "hour": hr,
    })

for k in acct_txns:
    acct_txns[k].sort(key=lambda x: x["dt"])

# Directed edges: determine sender/receiver from Dr/Cr
edges_out = defaultdict(list)
edges_in  = defaultdict(list)

for orig_idx, row in df_ok.iterrows():
    if row["_type"] in ("DR","DEBIT","D"):
        sender, receiver = row["_acct"], row["_cp"]
    else:
        sender, receiver = row["_cp"], row["_acct"]

    chan = ""
    if COL.get("channel") and COL["channel"] in df_ok.columns:
        chan = str(row[COL["channel"]]).strip() if pd.notna(row[COL["channel"]]) else ""
    hr = row["_dt"].hour if pd.notna(row["_dt"]) else 12

    if sender and receiver and sender != receiver and sender != "nan" and receiver != "nan":
        e = {"sender":sender, "receiver":receiver, "amt":row["_amt"], "dt":row["_dt"],
             "idx":orig_idx, "channel":chan, "hour":hr}
        edges_out[sender].append(e)
        edges_in[receiver].append(e)

for k in edges_out: edges_out[k].sort(key=lambda x: x["dt"])
for k in edges_in:  edges_in[k].sort(key=lambda x: x["dt"])

print(f"  Unique accounts:  {len(acct_txns):,}")
print(f"  Directed edges:   {sum(len(v) for v in edges_out.values()):,}")

# Group-ID counter and result accumulator
_gcnt = defaultdict(int)
def next_gid(prefix):
    _gcnt[prefix] += 1
    return f"{prefix}_{_gcnt[prefix]:05d}"

# results: (df_index, typology, group_id, score)
# score is 0..1 confidence; higher means stronger match.
results = []
def flag(indices, typology, gid, score=0.5):
    for i in indices:
        results.append((i, typology, gid, score))

print("Indexes ready (with channel/hour, score-aware flag())")


# ## 6 -- T1: Structuring (Smurfing)
# **Pattern**: Multiple different accounts each make a sub-threshold cash deposit,
# then all transfer funds to the **same target** account within a short window.
# Detection is target-centric: find convergence of sub-threshold cash sources.
# 

# In[13]:


print("T1: Detecting Structuring (Smurfing)...")
cfg = DETECT_CONFIG["structuring"]
threshold   = cfg["cash_threshold"]
floor       = threshold * cfg["amount_floor_pct"]
win_days    = cfg["time_window_days"]
consol_days = cfg["consolidation_window_days"]
min_sources = cfg["min_source_accounts"]
t1_count = 0

# Step 1: For each account with sub-threshold cash deposits,
#         find subsequent outflows and record (source_acct, target_acct, deposit_idx, transfer_idx, dt)
links = []  # each: {src, tgt, dep_idx, xfer_idx, dep_dt}

for acct, txns in acct_txns.items():
    # Cash credits to this account that are sub-threshold
    cash_crs = [t for t in txns
                if t["cash"] == "Y"
                and t["type"] in ("CR","CREDIT","C")
                and floor <= t["amt"] < threshold]
    if not cash_crs:
        continue

    # Debits FROM this account (transfers out)
    debits = [t for t in txns
              if t["type"] in ("DR","DEBIT","D")
              and t["cp"] not in ("", "nan", acct)]

    for dep in cash_crs:
        deadline = dep["dt"] + timedelta(days=consol_days)
        for dr in debits:
            if dr["dt"] < dep["dt"]:
                continue
            if dr["dt"] > deadline:
                break
            links.append({
                "src": acct, "tgt": dr["cp"],
                "dep_idx": dep["idx"], "xfer_idx": dr["idx"],
                "dep_dt": dep["dt"],
            })
            # Removed `break` — let a deposit link to ALL outflows in the
            # consolidation window, not just the first. This recovers
            # scenarios where the first matching debit goes to an
            # unrelated counterparty before the real consolidation target.

            # Cap links per deposit to avoid runaway over-detection
            if sum(1 for lk in links if lk["dep_idx"] == dep["idx"]) >= cfg.get("max_links_per_deposit", 3):
                break

print(f"  Deposit-to-target links: {len(links):,}")

# Step 2: Group by target; find targets receiving from N+ distinct sources in a time window
tgt_links = defaultdict(list)
for lk in links:
    tgt_links[lk["tgt"]].append(lk)

for tgt, lks in tgt_links.items():
    if len(set(l["src"] for l in lks)) < min_sources:
        continue
    lks.sort(key=lambda x: x["dep_dt"])

    # Sliding window
    for i in range(len(lks)):
        w_end = lks[i]["dep_dt"] + timedelta(days=win_days)
        cluster = [lks[i]]
        for j in range(i+1, len(lks)):
            if lks[j]["dep_dt"] <= w_end:
                cluster.append(lks[j])
            else:
                break

        srcs = set(c["src"] for c in cluster)
        if len(srcs) >= min_sources:
            gid = next_gid("STRUCT")
            idxs = list(set(c["dep_idx"] for c in cluster) | set(c["xfer_idx"] for c in cluster))
            # Score: structuring is highly specific (cash deposits, sub-threshold amount band)
            # Base 0.85 + bonus for many sources
            score = min(1.0, 0.85 + (len(srcs) - min_sources) * 0.03)
            flag(idxs, "Structuring (Smurfing)", gid, score)
            t1_count += 1
            break  # one detection per target

print(f"  Structuring scenarios: {t1_count}")



# ## 7 -- T2: Circular Transaction Loops
# **Pattern**: A -> B -> C -> A within time window, similar amounts. DFS cycle detection.
# 

# In[21]:


# print("T2: Detecting Circular Transaction Loops...")
# cfg = DETECT_CONFIG["circular"]
# win_days  = cfg["time_window_days"]
# min_loop  = cfg["min_loop_size"]
# max_loop  = cfg["max_loop_size"]
# amt_tol   = cfg["amount_tolerance_pct"]
# min_amt   = cfg["min_amount"]
# max_other = cfg.get("max_other_txns_between_parties", 999)  # commercial filter
# t2_count  = 0
# seen_cycles = set()

# # Collect high-value starting edges
# starters = [e for elist in edges_out.values() for e in elist if e["amt"] >= min_amt]
# starters.sort(key=lambda x: -x["amt"])
# search_lim = min(len(starters), cfg.get("search_limit", 20000))
# print(f"  Searching from {search_lim:,} high-value edges...")

# for si, start in enumerate(starters[:search_lim]):
#     if si % 5000 == 0 and si > 0:
#         print(f"    edge {si:,}/{search_lim:,}...")

#     origin = start["sender"]
#     stack = [(start, [start])]

#     while stack:
#         cur, path = stack.pop()
#         if len(path) > max_loop:
#             continue
#         nxt_acct = cur["receiver"]
#         deadline = start["dt"] + timedelta(days=win_days)

#         if nxt_acct == origin and len(path) >= min_loop:
#             cyc_key = tuple(sorted(e["idx"] for e in path))
#             if cyc_key not in seen_cycles:
#                 # Commercial filter: check if cycle parties transact outside the cycle
#                 cycle_accts = list(set(e["sender"] for e in path))
#                 is_commercial = False
#                 if max_other < 999:
#                     for ca in cycle_accts:
#                         other_partners = set(e["receiver"] for e in edges_out.get(ca, [])
#                                            if e["idx"] not in set(e2["idx"] for e2 in path))
#                         cycle_partner_overlap = other_partners & set(cycle_accts)
#                         if len(cycle_partner_overlap) > 0:
#                             # Count non-cycle txns between these parties
#                             other_count = sum(1 for e in edges_out.get(ca, [])
#                                             if e["receiver"] in cycle_partner_overlap
#                                             and e["idx"] not in set(e2["idx"] for e2 in path))
#                             if other_count > max_other:
#                                 is_commercial = True
#                                 break

#                 if not is_commercial:
#                     seen_cycles.add(cyc_key)
#                     # Skip if the ring matches Hawala's signature
#                     # (3-4 parties, balanced legs in 80k-1.2M range, hawala channels).
#                     # Hawala detector will pick these up.
#                     skip_hawala = False
#                     if cfg.get("exclude_hawala_signature", False):
#                         hw_cfg = DETECT_CONFIG.get("hawala", {})
#                         hw_min, hw_max = hw_cfg.get("num_parties_range", [3, 4])
#                         hw_amt_lo, hw_amt_hi = hw_cfg.get("amount_range", [80000, 1200000])
#                         hw_channels = set(c.lower() for c in hw_cfg.get("channels", []))
#                         path_amts = [e["amt"] for e in path]
#                         path_channels = set(str(e.get("channel", "")).lower() for e in path)
#                         if (hw_min <= len(path) <= hw_max
#                             and all(hw_amt_lo <= a <= hw_amt_hi for a in path_amts)
#                             and (not hw_channels or path_channels & hw_channels)):
#                             skip_hawala = True
#                     if skip_hawala:
#                         continue
#                     gid = next_gid("CIRC")
#                     # Score: closed loops are highly specific.
#                     amts = [e["amt"] for e in path]
#                     drift = (max(amts) - min(amts)) / max(amts) if amts else 0
#                     score = max(0.85, min(1.0, 0.98 - drift * 0.8))
#                     flag([e["idx"] for e in path], "Circular Transaction Loop", gid, score)
#                     t2_count += 1

#             continue

#         for e in edges_out.get(nxt_acct, []):
#             if e["dt"] > deadline:
#                 break
#             # Allow same-day non-monotonic timestamps (generator's
#             # unique_timestamp can place ring hops out of order within a day)
#             if e["dt"].date() < cur["dt"].date():
#                 continue

#             # Compare to PREVIOUS hop, not start. Generator decays 0-3% per
#             # hop so by hop 4-5 the cumulative drift can exceed any reasonable
#             # tolerance from start. This matches the layering detector's logic.
#             prev_amt = cur["amt"]
#             if abs(e["amt"] - prev_amt) / max(prev_amt, 1) > amt_tol:
#                 continue
#             visited = {ed["sender"] for ed in path}
#             if e["receiver"] in visited and e["receiver"] != origin:
#                 continue
#             stack.append((e, path + [e]))

# print(f"  Circular loop scenarios: {t2_count}")


print("T2: Detecting Circular Transaction Loops...")

cfg = DETECT_CONFIG["circular"]

win_days  = cfg["time_window_days"]
min_loop  = cfg["min_loop_size"]
max_loop  = cfg["max_loop_size"]
amt_tol   = cfg["amount_tolerance_pct"]
min_amt   = cfg["min_amount"]

max_other = cfg.get("max_other_txns_between_parties", 999)

t2_count = 0
seen_cycles = set()

edges_out_local = edges_out
seen_cycles_add = seen_cycles.add

# ---------------------------------------------------
# PRECOMPUTE day values
# ---------------------------------------------------

for elist in edges_out_local.values():
    for e in elist:
        e["_day"] = e["dt"].date()

# ---------------------------------------------------
# PRECOMPUTE COMMERCIAL RELATIONSHIPS
# ---------------------------------------------------

party_links = {}

if max_other < 999:

    for sender, elist in edges_out_local.items():

        rel = {}

        for e in elist:

            r = e["receiver"]

            rel[r] = rel.get(r, 0) + 1

        party_links[sender] = rel

# ---------------------------------------------------
# STARTERS
# ---------------------------------------------------

starters = []

for elist in edges_out_local.values():

    for e in elist:

        if e["amt"] >= min_amt:
            starters.append(e)

starters.sort(key=lambda x: -x["amt"])

search_lim = min(
    len(starters),
    cfg.get("search_limit", 20000)
)

print(f"  Searching from {search_lim:,} high-value edges...")

# ---------------------------------------------------
# DFS
# ---------------------------------------------------

for si in range(search_lim):

    if si % 5000 == 0 and si > 0:
        print(f"    edge {si:,}/{search_lim:,}...")

    start = starters[si]

    origin = start["sender"]

    deadline = start["dt"] + timedelta(days=win_days)

    stack = [(
        start,
        (start,),
        frozenset((start["sender"],)),
        frozenset((start["idx"],))
    )]

    while stack:

        cur, path, visited, idx_set = stack.pop()

        path_len = len(path)

        if path_len > max_loop:
            continue

        nxt_acct = cur["receiver"]

        # ---------------------------------------------------
        # CYCLE FOUND
        # ---------------------------------------------------

        if nxt_acct == origin and path_len >= min_loop:

            cyc_key = tuple(sorted(idx_set))

            if cyc_key not in seen_cycles:

                is_commercial = False

                if max_other < 999:

                    cycle_accts = visited

                    for ca in cycle_accts:

                        rels = party_links.get(ca)

                        if not rels:
                            continue

                        overlap = cycle_accts.intersection(rels)

                        for cp in overlap:

                            if rels[cp] > max_other:
                                is_commercial = True
                                break

                        if is_commercial:
                            break

                if not is_commercial:

                    seen_cycles_add(cyc_key)

                    skip_hawala = False

                    if cfg.get("exclude_hawala_signature", False):

                        hw_cfg = DETECT_CONFIG.get("hawala", {})

                        hw_min, hw_max = hw_cfg.get(
                            "num_parties_range",
                            [3, 4]
                        )

                        hw_amt_lo, hw_amt_hi = hw_cfg.get(
                            "amount_range",
                            [80000, 1200000]
                        )

                        hw_channels = set(
                            c.lower()
                            for c in hw_cfg.get("channels", [])
                        )

                        path_amts = []
                        path_channels = set()

                        for pe in path:

                            amt = pe["amt"]

                            path_amts.append(amt)

                            ch = pe.get("channel")

                            if ch:
                                path_channels.add(str(ch).lower())

                        if (
                            hw_min <= path_len <= hw_max
                            and all(
                                hw_amt_lo <= a <= hw_amt_hi
                                for a in path_amts
                            )
                            and (
                                not hw_channels
                                or path_channels & hw_channels
                            )
                        ):
                            skip_hawala = True

                    if skip_hawala:
                        continue

                    gid = next_gid("CIRC")

                    amts = [pe["amt"] for pe in path]

                    max_amt = max(amts)
                    min_amt_path = min(amts)

                    drift = (
                        (max_amt - min_amt_path) / max_amt
                        if amts else 0
                    )

                    score = max(
                        0.85,
                        min(1.0, 0.98 - drift * 0.8)
                    )

                    flag(
                        [pe["idx"] for pe in path],
                        "Circular Transaction Loop",
                        gid,
                        score
                    )

                    t2_count += 1

            continue

        # ---------------------------------------------------
        # EXPAND DFS
        # ---------------------------------------------------

        prev_amt = cur["amt"]

        next_edges = edges_out_local.get(nxt_acct)

        if not next_edges:
            continue

        cur_day = cur["_day"]

        for e in next_edges:

            if e["dt"] > deadline:
                break

            if e["_day"] < cur_day:
                continue

            e_amt = e["amt"]

            if abs(e_amt - prev_amt) / max(prev_amt, 1) > amt_tol:
                continue

            recv = e["receiver"]

            if recv in visited and recv != origin:
                continue

            stack.append((
                e,
                path + (e,),
                visited | frozenset((recv,)),
                idx_set | frozenset((e["idx"],))
            ))

print(f"  Circular loop scenarios: {t2_count}")


# ## 8 -- T3: Funnel Account Networks
# **Pattern**: Many unrelated senders -> one account -> rapid large outflow.
# 

# In[23]:


print("T3: Detecting Funnel Account Networks...")
cfg = DETECT_CONFIG["funnel"]
win_days    = cfg["time_window_days"]
min_senders = cfg["min_unique_senders"]
out_days    = cfg["outflow_window_days"]
retention   = cfg["outflow_retention_pct"]
min_inflow  = cfg["min_total_inflow"]
t3_count = 0

for recv, inflows in edges_in.items():
    if len(inflows) < min_senders:
        continue

    for i in range(len(inflows)):
        w_end = inflows[i]["dt"] + timedelta(days=win_days)
        win_in = [inflows[i]]
        for j in range(i+1, len(inflows)):
            if inflows[j]["dt"] <= w_end:
                win_in.append(inflows[j])
            else:
                break

        if len(set(e["sender"] for e in win_in)) < min_senders:
            continue

        total_in = sum(e["amt"] for e in win_in)
        if total_in < min_inflow:
            continue

        last_in = max(e["dt"] for e in win_in)
        out_deadline = last_in + timedelta(days=out_days)
        outs = [e for e in edges_out.get(recv, []) if last_in <= e["dt"] <= out_deadline]
        total_out = sum(e["amt"] for e in outs)

        if total_out >= total_in * (1 - retention):
            gid = next_gid("FUNNEL")
            idxs = [e["idx"] for e in win_in] + [e["idx"] for e in outs]
            # Score: funnel is moderately specific. Strong score requires
            # both many senders AND high inflow.
            sender_n = len(set(e["sender"] for e in win_in))
            sender_factor = min(1.0, (sender_n - min_senders) / max(min_senders, 1))
            inflow_factor = min(1.0, total_in / max(min_inflow * 2, 1))
            score = 0.55 + 0.20 * sender_factor + 0.15 * inflow_factor
            flag(idxs, "Funnel Account Network", gid, score)
            t3_count += 1
            break

print(f"  Funnel network scenarios: {t3_count}")



# ## 9 -- T4: Pass-Through Transit Hubs
# **Pattern**: Receive large sum, forward 90%+ within minutes, retain almost nothing.
# 

# In[24]:


print("T4: Detecting Pass-Through Transit Hubs...")
cfg = DETECT_CONFIG["passthrough"]
gap_min   = cfg["time_gap_minutes"]
ret_pct   = cfg["retention_pct"]
min_amt   = cfg["min_amount"]
min_occ   = cfg["min_occurrences"]
require_diff_cp = cfg.get("require_different_counterparties", False)
max_net_ratio   = cfg.get("max_net_position_ratio", 1.0)
t4_count  = 0

# Pre-compute net position ratio per account (total_credits / total_debits)
acct_net_ratio = {}
if max_net_ratio < 1.0:
    for acct, txns in acct_txns.items():
        total_cr = sum(t["amt"] for t in txns if t["type"] in ("CR","CREDIT","C"))
        total_dr = sum(t["amt"] for t in txns if t["type"] in ("DR","DEBIT","D"))
        if total_cr + total_dr > 0:
            acct_net_ratio[acct] = abs(total_cr - total_dr) / (total_cr + total_dr)
        else:
            acct_net_ratio[acct] = 0

acct_events = defaultdict(list)

for acct in set(edges_in.keys()) & set(edges_out.keys()):
    # Net position filter: skip accounts that clearly retain funds
    if max_net_ratio < 1.0:
        if acct_net_ratio.get(acct, 1.0) > max_net_ratio:
            continue

    big_in = [e for e in edges_in[acct] if e["amt"] >= min_amt]
    outs   = edges_out[acct]

    for inf in big_in:
        deadline = inf["dt"] + timedelta(minutes=gap_min)
        matched = [o for o in outs
                   if inf["dt"] <= o["dt"] <= deadline
                   and o["receiver"] != inf["sender"]]

        # Counterparty diversity: inflow source must differ from outflow destinations
        if require_diff_cp and matched:
            matched = [o for o in matched if o["receiver"] != inf["sender"]]

        if not matched:
            continue
        total_out = sum(o["amt"] for o in matched)
        kept = inf["amt"] - total_out
        if 0 <= kept <= inf["amt"] * ret_pct:
            acct_events[acct].append({"inflow": inf, "outflows": matched})

for acct, evts in acct_events.items():
    if len(evts) >= min_occ:
        for evt in evts:
            gid = next_gid("PASS")
            idxs = [evt["inflow"]["idx"]] + [o["idx"] for o in evt["outflows"]]
            # Score: pass-through is specific (large inflow + near-immediate
            # forwarding of >94%). High score by default.
            inf = evt["inflow"]
            tot_out = sum(o["amt"] for o in evt["outflows"])
            fwd_pct = tot_out / max(inf["amt"], 1)
            score = 0.70 + 0.25 * min(1.0, fwd_pct)
            flag(idxs, "Pass-Through Transit Hub", gid, score)
            t4_count += 1

print(f"  Pass-through scenarios: {t4_count}")



# ## 10 -- T5: Rapid Multi-Hop Layering
# **Pattern**: Funds traverse 4+ accounts within hours. BFS forward chain tracing.
# 

# In[25]:


print("T5: Detecting Rapid Multi-Hop Layering...")
cfg = DETECT_CONFIG["layering"]
max_hrs   = cfg["max_chain_hours"]
min_hops  = cfg["min_hops"]
max_hops  = cfg["max_hops"]
decay_tol = cfg["amount_decay_tolerance"]
min_amt   = cfg["min_amount"]
search_limit_cfg = cfg.get("search_limit", 15000)
t5_count  = 0

starters = [e for elist in edges_out.values() for e in elist if e["amt"] >= min_amt]
starters.sort(key=lambda x: -x["amt"])
search_lim = min(len(starters), search_limit_cfg)
print(f"  Tracing from {search_lim:,} high-value edges...")

for si, start in enumerate(starters[:search_lim]):
    if si % 10000 == 0 and si > 0:
        print(f"    edge {si:,}/{search_lim:,}...")

    deadline = start["dt"] + timedelta(hours=max_hrs)
    best = [start]
    stack = [(start, [start])]
    explored = 0
    EXPLORE_BUDGET = 500

    while stack and explored < EXPLORE_BUDGET:
        cur, chain = stack.pop()
        explored += 1
        if len(chain) > max_hops:
            continue
        nxt = cur["receiver"]
        for e in edges_out.get(nxt, []):
            if e["dt"] > deadline:
                break
            if e["dt"].date() < cur["dt"].date():
                continue
            prev_amt = cur["amt"]
            if e["amt"] < prev_amt * (1 - decay_tol) or e["amt"] > prev_amt * (1 + 0.10):
                continue
            visited = {ed["sender"] for ed in chain} | {chain[-1]["receiver"]}
            if e["receiver"] in visited:
                continue
            new_chain = chain + [e]
            if len(new_chain) > len(best):
                best = new_chain
            stack.append((e, new_chain))

    if len(best) >= min_hops:
        gid = next_gid("LAYER")
        idxs = [e["idx"] for e in best]
        hop_factor = min(1.0, (len(best) - min_hops) / max(max_hops - min_hops, 1))
        score = 0.75 + 0.20 * hop_factor
        flag(idxs, "Rapid Multi-Hop Layering", gid, score)
        t5_count += 1

print(f"  Multi-hop layering scenarios: {t5_count}")


# ## 11 -- Apply Labels
# 

# ## 11 -- T6: Third-Party Payment Webs
# **Pattern**: Business receives payments from many unrelated individuals not matching its customer base.
# 

# In[26]:


print("T6: Detecting Third-Party Payment Webs...")
cfg = DETECT_CONFIG["third_party_web"]
win_days = cfg["time_window_days"]
min_payers = cfg["min_unique_payers"]
pay_lo, pay_hi = cfg["per_payment_amount_range"]
min_total = cfg.get("min_total_inflow", 0)
valid_ch = set(cfg["channels"])
hr_lo, hr_hi = cfg["hour_range"]
t6_count = 0

for recv, inflows in edges_in.items():
    valid_in = []
    for e in inflows:
        if not (pay_lo <= e["amt"] <= pay_hi):
            continue
        if "channel" in e and e["channel"] not in valid_ch:
            continue
        if "hour" in e and not (hr_lo <= e["hour"] <= hr_hi):
            continue
        valid_in.append(e)

    if len(valid_in) < min_payers:
        continue

    for i in range(len(valid_in)):
        w_end = valid_in[i]["dt"] + timedelta(days=win_days)
        cluster = [valid_in[i]]
        for j in range(i + 1, len(valid_in)):
            if valid_in[j]["dt"] <= w_end:
                cluster.append(valid_in[j])
            else:
                break
        senders = set(e["sender"] for e in cluster)
        if len(senders) < min_payers:
            continue
        total_in = sum(e["amt"] for e in cluster)
        if total_in < min_total:
            continue
        gid = next_gid("TPWEB")
        sender_factor = min(1.0, (len(senders) - min_payers) / max(min_payers, 1))
        score = 0.55 + 0.25 * sender_factor
        flag([e["idx"] for e in cluster], "Third-Party Payment Web", gid, score)
        t6_count += 1
        break

print(f"  Third-party payment web scenarios: {t6_count}")



# ## 12 -- T7: Money Mule Networks
# **Pattern**: Central controller sends to many mules, each mule forwards to a collector.
# 

# In[27]:


print("T7: Detecting Money Mule Networks...")
cfg = DETECT_CONFIG["money_mule"]
min_mules = cfg["min_mules"]
amt_lo, amt_hi = cfg["controller_amount_range"]
fwd_lo, fwd_hi = cfg["forward_pct_range"]
fwd_delay_hrs = cfg["forward_delay_hours"]
valid_ch = set(cfg["channels"])
hr_lo, hr_hi = cfg["hour_range"]
t7_count = 0

for controller, outflows in edges_out.items():
    valid_out = []
    for e in outflows:
        if not (amt_lo <= e["amt"] <= amt_hi):
            continue
        if "channel" in e and e["channel"] not in valid_ch:
            continue
        if "hour" in e and not (hr_lo <= e["hour"] <= hr_hi):
            continue
        valid_out.append(e)

    receivers = set(e["receiver"] for e in valid_out)
    if len(receivers) < min_mules:
        continue

    # Check if receivers (mules) forward funds
    mule_fwd_events = []
    for mule_acct in receivers:
        mule_ins = [vi for vi in valid_out if vi["receiver"] == mule_acct]
        mule_outs = edges_out.get(mule_acct, [])

        for mi in mule_ins:
            for mo in mule_outs:
                gap_hrs = (mo["dt"] - mi["dt"]).total_seconds() / 3600
                if 0 <= gap_hrs <= fwd_delay_hrs:
                    if fwd_lo * mi["amt"] <= mo["amt"] <= fwd_hi * mi["amt"]:
                        mule_fwd_events.append({"in": mi, "out": mo, "mule": mule_acct})
                        break
            if any(f["mule"] == mule_acct for f in mule_fwd_events):
                break

    forwarding_mules = set(f["mule"] for f in mule_fwd_events)
    if len(forwarding_mules) >= min_mules:
        gid = next_gid("MULE")
        idxs = []
        for fe in mule_fwd_events:
            idxs.extend([fe["in"]["idx"], fe["out"]["idx"]])
        # Score: mule network is moderately specific. Many mules is the
        # strongest signal. Don't go too high — Layering should usually win
        # if a chain exists in the same data.
        mule_factor = min(1.0, (len(forwarding_mules) - min_mules) / max(min_mules, 1))
        score = 0.55 + 0.20 * mule_factor
        flag(list(set(idxs)), "Money Mule Network", gid, score)
        t7_count += 1

print(f"  Money mule network scenarios: {t7_count}")



# ## 13 -- T8: High-Risk Corridor Transfers
# **Pattern**: Repeated transfers from one account to high-risk FATF countries.
# 

# In[28]:


print("T8: Detecting High-Risk Corridor Transfers...")
cfg = DETECT_CONFIG["high_risk_corridor"]
amt_lo, amt_hi = cfg["amount_range"]
target_countries = set(cfg["target_countries"])
min_xfers = cfg["min_transfers_per_account"]
win_days = cfg["time_window_days"]
valid_ch = set(cfg["channels"])
hr_lo, hr_hi = cfg["hour_range"]
t8_count = 0

# Need receiver country code column
rc_col = find_col(df, ["receiver_country_code", "receiver_country", "Receiver Country Code*"])
if rc_col:
    for acct, txns in acct_txns.items():
        corridor_txns = []
        for t in txns:
            idx = t["idx"]
            if idx not in df.index:
                continue
            recv_cc = str(df.at[idx, rc_col]).strip().upper() if rc_col in df.columns else ""
            if recv_cc not in target_countries:
                continue
            if not (amt_lo <= t["amt"] <= amt_hi):
                continue
            if "channel" in t and t["channel"] not in valid_ch:
                continue
            if "hour" in t and not (hr_lo <= t["hour"] <= hr_hi):
                continue
            if t["type"] not in ("DR", "DEBIT", "D"):
                continue
            corridor_txns.append(t)

        if len(corridor_txns) < min_xfers:
            continue

        corridor_txns.sort(key=lambda x: x["dt"])
        for i in range(len(corridor_txns)):
            w_end = corridor_txns[i]["dt"] + timedelta(days=win_days)
            cluster = [corridor_txns[i]]
            for j in range(i + 1, len(corridor_txns)):
                if corridor_txns[j]["dt"] <= w_end:
                    cluster.append(corridor_txns[j])
                else:
                    break
            if len(cluster) >= min_xfers:
                gid = next_gid("HRCORR")
                # Score: corridor is highly specific (target country + amount
                # band + channel). Very high score.
                score = min(1.0, 0.90 + (len(cluster) - min_xfers) * 0.02)
                flag([t["idx"] for t in cluster], "High-Risk Corridor Transfer", gid, score)
                t8_count += 1
                break
else:
    print("  WARNING: No receiver country column found, skipping")

print(f"  High-risk corridor scenarios: {t8_count}")


# ## 14 -- T9: Underground Banking (Hawala)
# **Pattern**: Triangular/quadrilateral settlements where amounts match across 3-4 parties.
# 

# In[29]:


print("T9: Detecting Underground Banking (Hawala)...")
cfg = DETECT_CONFIG["hawala"]
min_parties, max_parties = cfg["num_parties_range"]
amt_lo, amt_hi = cfg["amount_range"]
amt_tol = cfg["amount_tolerance_pct"]
win_days = cfg["time_window_days"]
valid_ch = set(cfg["channels"])
hr_lo, hr_hi = cfg["hour_range"]
t9_count = 0
seen_hawala = set()

# Filter starters with safe access
starters = []
for elist in edges_out.values():
    for e in elist:
        if not (amt_lo <= e["amt"] <= amt_hi):
            continue
        if "channel" in e and e["channel"] not in valid_ch:
            continue
        if "hour" in e and not (hr_lo <= e["hour"] <= hr_hi):
            continue
        starters.append(e)

starters.sort(key=lambda x: -x["amt"])
search_lim = min(len(starters), 15000)

for si, start in enumerate(starters[:search_lim]):
    origin = start["sender"]
    stack = [(start, [start])]
    while stack:
        cur, path = stack.pop()
        if len(path) > max_parties:
            continue
        nxt = cur["receiver"]
        deadline = start["dt"] + timedelta(days=win_days)
        if nxt == origin and min_parties <= len(path) <= max_parties:
            cyc_key = tuple(sorted(e["idx"] for e in path))
            if cyc_key not in seen_hawala:
                seen_hawala.add(cyc_key)
                gid = next_gid("HAWALA")
                # Score: Hawala has very specific signature (small parties,
                # balanced legs, hawala channels). Boost above Circular's
                # 0.85 floor so Hawala wins shared rings.
                amts = [e["amt"] for e in path]
                drift = (max(amts) - min(amts)) / max(amts) if amts else 0
                score = max(0.92, min(1.0, 0.99 - drift * 0.5))
                flag([e["idx"] for e in path], "Underground Banking (Hawala)", gid, score)

                t9_count += 1
            continue
        for e in edges_out.get(nxt, []):
            if e["dt"] > deadline:
                break
            # Allow same-day non-monotonic ordering (generator's unique_timestamp
            # can place legs out of order within a day)
            if e["dt"].date() < cur["dt"].date():
                continue
            if "channel" in e and e["channel"] and e["channel"] not in valid_ch:
                continue
            # Compare to previous leg, not start.
            prev_amt = cur["amt"]
            if abs(e["amt"] - prev_amt) / max(prev_amt, 1) > amt_tol:
                continue
            visited = {ed["sender"] for ed in path}
            if e["receiver"] in visited and e["receiver"] != origin:
                continue
            stack.append((e, path + [e]))

print(f"  Hawala scenarios: {t9_count}")


# ## 15 -- T10: Charity Abuse
# **Pattern**: NPO collects many donations then diverts funds to personal accounts.
# 

# In[30]:


print("T10: Detecting Charity Abuse...")
cfg = DETECT_CONFIG["charity_abuse"]
min_donors = cfg["min_donors"]
don_lo, don_hi = cfg["donation_amount_range"]
don_win = cfg["donation_window_days"]
div_win = cfg["diversion_window_days"]
div_ret = cfg["diversion_retention_pct"]
don_channels = set(cfg["donation_channels"])
div_hr_lo, div_hr_hi = cfg["diversion_hour_range"]
min_total = cfg.get("min_total_donation", 20000)
t10_count = 0

for recv, inflows in edges_in.items():
    valid_donations = []
    for e in inflows:
        if not (don_lo <= e["amt"] <= don_hi):
            continue
        if "channel" in e and e["channel"] not in don_channels:
            continue
        valid_donations.append(e)

    unique_donors = set(e["sender"] for e in valid_donations)
    if len(unique_donors) < min_donors:
        continue

    for i in range(len(valid_donations)):
        w_end = valid_donations[i]["dt"] + timedelta(days=don_win)
        cluster = [valid_donations[i]]
        for j in range(i + 1, len(valid_donations)):
            if valid_donations[j]["dt"] <= w_end:
                cluster.append(valid_donations[j])
            else:
                break

        donors = set(e["sender"] for e in cluster)
        if len(donors) < min_donors:
            continue

        total_donated = sum(e["amt"] for e in cluster)
        if total_donated < min_total:
            continue

        last_donation = max(e["dt"] for e in cluster)
        div_deadline = last_donation + timedelta(days=div_win)

        diversions = []
        donor_set = set(e["sender"] for e in cluster)
        for e in edges_out.get(recv, []):
            if not (last_donation <= e["dt"] <= div_deadline):
                continue
            if "hour" in e and not (div_hr_lo <= e["hour"] <= div_hr_hi):
                continue
            # Charity diversion goes to a SMALL set of beneficiaries that
            # are NOT among the donors. If the outflow goes back to donors,
            # it's not abuse — it's reimbursement / refund. Skip those.
            if e["receiver"] in donor_set:
                continue
            diversions.append(e)

        total_diverted = sum(e["amt"] for e in diversions)
        # Require diversions to go to a CONCENTRATED set of beneficiaries
        # (real charity abuse forwards to ~2-5 criminal accounts, not 50)
        diversion_recipients = set(e["receiver"] for e in diversions)
        if len(diversion_recipients) > 10:
            continue  # too dispersed — looks like legitimate distribution
        if total_diverted >= total_donated * (1 - div_ret):
            gid = next_gid("CHARITY")
            idxs = [e["idx"] for e in cluster] + [e["idx"] for e in diversions]
            # Score: charity is moderately specific (many small donors
            # then post-collection diversion). Reward strong diversion ratio.
            div_ratio = total_diverted / max(total_donated, 1)
            donor_factor = min(1.0, (len(donors) - min_donors) / max(min_donors, 1))
            score = 0.60 + 0.20 * donor_factor + 0.10 * min(1.0, div_ratio)
            flag(idxs, "Charity Abuse", gid, score)
            t10_count += 1
            break

print(f"  Charity abuse scenarios: {t10_count}")



# In[31]:


print("Applying labels to master dataframe...")
print("  Strategy: SCORE-BASED dedup with priority tiebreak")
print("  Each detector emits a confidence score; highest score wins")

# Specificity-based priority — lower number = more specific signature.
# Used ONLY as a tiebreaker when scores are within 5% of each other.
TYPOLOGY_PRIORITY = {
    "Structuring (Smurfing)":        1,
    "Underground Banking (Hawala)":  2,
    "Circular Transaction Loop":     3,
    "High-Risk Corridor Transfer":   4,
    "Pass-Through Transit Hub":      5,
    "Rapid Multi-Hop Layering":      6,
    "Charity Abuse":                 7,
    "Money Mule Network":            8,
    "Funnel Account Network":        9,
    "Third-Party Payment Web":       10,
}

# Step 1: gather all (typology, group_id, score) entries per index
label_map = defaultdict(list)
for entry in results:
    # results entries are now 4-tuples: (idx, typ, gid, score)
    if len(entry) == 4:
        idx, typ, gid, score = entry
    else:
        # back-compat with any detector that didn't update
        idx, typ, gid = entry[:3]
        score = 0.5
    label_map[idx].append({"typ": typ, "grp": gid, "score": score})

# Step 2: counts before dedup
typ_counts_before = defaultdict(int)
for idx, entries in label_map.items():
    seen_typs = set()
    for e in entries:
        if e["typ"] not in seen_typs:
            typ_counts_before[e["typ"]] += 1
            seen_typs.add(e["typ"])

print(f"\n  Detection counts (before dedup):")
for t, cnt in sorted(typ_counts_before.items(), key=lambda x: -x[1]):
    print(f"    {t:<45s} {cnt:>8,}")

# Step 3: reset labels — detector is sole source
df["is_aml"] = 0
df["aml_typology"] = ""
df["typology_group_id"] = ""

# Step 4: assign ONE typology per txn using score-then-priority
flagged_count = 0
multi_count = 0
single_count = 0
SCORE_TIE_TOLERANCE = 0.05  # within 5% = tie -> use priority

for idx, entries in label_map.items():
    if idx not in df.index:
        continue

    # group entries by typology, keep best score per typology
    by_typ = {}
    for e in entries:
        if e["typ"] not in by_typ or e["score"] > by_typ[e["typ"]]["score"]:
            by_typ[e["typ"]] = e

    candidates = list(by_typ.values())

    if len(candidates) > 1:
        # sort by score desc, then priority asc (lower priority # is more specific)
        max_score = max(c["score"] for c in candidates)
        # Group: top-tier = candidates within tolerance of max
        top = [c for c in candidates if c["score"] >= max_score - SCORE_TIE_TOLERANCE]
        # If tied, fall back to priority (most specific wins)
        chosen = min(top, key=lambda c: TYPOLOGY_PRIORITY.get(c["typ"], 99))
        multi_count += 1
    else:
        chosen = candidates[0]
        single_count += 1

    df.at[idx, "is_aml"] = 1
    df.at[idx, "aml_typology"] = chosen["typ"]
    df.at[idx, "typology_group_id"] = chosen["grp"]
    flagged_count += 1

# Step 5: summary
total = len(df)
print(f"\n{'='*70}")
print(f"DETECTION SUMMARY")
print(f"{'='*70}")
print(f"  Total transactions:       {total:>10,}")
print(f"  Flagged as AML:           {flagged_count:>10,}  ({flagged_count/total*100:.2f}%)")
print(f"  Clean:                    {total-flagged_count:>10,}  ({(total-flagged_count)/total*100:.2f}%)")
print(f"  Single typology:          {single_count:>10,}")
print(f"  Multi-typology (deduped): {multi_count:>10,}  (resolved by score then priority)")
print(f"{'='*70}")

# Verify zero multi-typology
aml_typs = df.loc[df["is_aml"]==1, "aml_typology"]
multi_check = aml_typs.str.contains(";", na=False).sum()
print(f"\n  Verification: multi-typology transactions = {multi_check} (should be 0)")

# Final distribution
print(f"\n  Final Typology Distribution:")
print(f"  {'Typology':<45s} {'Txns':>8s} {'%':>7s} {'Priority':>10s}")
print(f"  {'~'*75}")
for typ in sorted(df.loc[df["is_aml"]==1, "aml_typology"].unique()):
    if typ and typ != "nan":
        cnt = (df["aml_typology"] == typ).sum()
        pri = TYPOLOGY_PRIORITY.get(typ, 99)
        print(f"  {typ:<45s} {cnt:>8,} {cnt/flagged_count*100:>6.1f}% {pri:>10d}")

# Multi-typology diagnostics
print(f"\n  --- Multi-Typology Diagnostics ---")
print(f"  Transactions detected by multiple typologies: {multi_count:>8,}")
if multi_count > 0:
    multi_combos = defaultdict(int)
    for idx, entries in label_map.items():
        unique_typs = sorted(set(e["typ"] for e in entries))
        if len(unique_typs) > 1:
            combo = " + ".join(unique_typs)
            multi_combos[combo] += 1
    print(f"  Top 15 multi-typology combinations:")
    print(f"  {'Combination':<70s} {'Count':>8s}")
    print(f"  {'~'*80}")
    for combo, cnt in sorted(multi_combos.items(), key=lambda x: -x[1])[:15]:
        print(f"  {combo:<70s} {cnt:>8,}")
print(f"  All multi-typology transactions resolved to single typology via score+priority.")


# ## 12 -- Typology Distribution & Scenario Summary
# 

# In[32]:


flagged = df[df["is_aml"] == 1]

if len(flagged) > 0:
    typ_counts = flagged["aml_typology"].value_counts()
    grp_counts = flagged.groupby("aml_typology")["typology_group_id"].nunique()

    print(f"\n{'='*75}")
    print(f"{'Typology':<40s} {'Txns':>8s} {'Scenarios':>10s} {'% Flagged':>10s}")
    print(f"{'-'*75}")
    for typ in sorted(typ_counts.index):
        if not typ or typ == "nan":
            continue
        txns = typ_counts[typ]
        scen = grp_counts.get(typ, 0)
        pct = txns / len(flagged) * 100
        print(f"  {typ:<38s} {txns:>8,} {scen:>10,} {pct:>9.1f}%")

    print(f"\n  TOTAL FLAGGED        {len(flagged):>10,}")
    total_scen = grp_counts.sum()
    print(f"  TOTAL SCENARIOS      {total_scen:>10,}")

    # Single vs multi typology verification
    multi_check = flagged["aml_typology"].str.contains(";", na=False).sum()
    print(f"\n  Single-typology verification:")
    print(f"    Transactions with exactly 1 typology: {len(flagged) - multi_check:>10,}")
    print(f"    Transactions with multiple typologies: {multi_check:>10,} (should be 0)")
    if multi_check > 0:
        print(f"    WARNING: {multi_check} transactions still have multiple typologies!")
        samples = flagged[flagged["aml_typology"].str.contains(";", na=False)].head(5)
        for _, row in samples.iterrows():
            print(f"      {row.name}: {row['aml_typology']}")
    else:
        print(f"    PASSED: Every AML transaction has exactly one typology")
else:
    print("No AML transactions detected")



# ## 13 -- Export Labeled Datasets
# 

# In[33]:


for key in [
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD"
]:
    if key in os.environ:
        del os.environ[key]

print("Environment cache cleared")

import sys

modules_to_remove = [
    m for m in sys.modules
    if (
        m.startswith("db_utils")
    )
]

for m in modules_to_remove:
    del sys.modules[m]

print("Cleared cached modules:")
print(modules_to_remove)

import sys
import importlib

# Remove cached modules
for m in list(sys.modules):
    if m.startswith("db_config") or m.startswith("db_utils"):
        del sys.modules[m]

# Fresh import
import db_config
import db_utils

importlib.reload(db_config)
importlib.reload(db_utils)

print("Reloaded db_config and db_utils")
import db_config

print("HOST =", db_config.DB_HOST)
print("PORT =", db_config.DB_PORT)
print("DB   =", db_config.DB_NAME)
print("USER =", db_config.DB_USER)
print("PASS =", db_config.DB_PASSWORD)

# ── Database connection (PostgreSQL) ──
from db_utils import write_table, test_connection
test_connection()


# ## TO DELETE AFTER LOADING THE DATA INTO DATABASE

# In[ ]:


# import pandas as pd

# df = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_flagged.parquet")
# df.head()


# In[34]:


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

import io

import io
from psycopg2 import sql

import io
import pandas as pd
from psycopg2 import sql

def write_table_fast(df, table_name, mode="append"):

    conn = engine.raw_connection()
    cur = conn.cursor()

    try:

        # ==========================================
        # Replace mode
        # ==========================================
        if mode == "replace":

            cur.execute(
                sql.SQL("TRUNCATE TABLE {}").format(
                    sql.Identifier(table_name)
                )
            )

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

        # ==========================================
        # CSV buffer
        # ==========================================
        buffer = io.StringIO()

        df.to_csv(
            buffer,
            index=False,
            header=False,
            na_rep=""
        )

        buffer.seek(0)

        # ==========================================
        # COPY SQL
        # ==========================================
        copy_sql = sql.SQL("""
            COPY {} ({})
            FROM STDIN WITH CSV
        """).format(
            sql.Identifier(table_name),
            sql.SQL(",").join(
                map(sql.Identifier, df.columns)
            )
        )

        # ==========================================
        # Bulk insert
        # ==========================================
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

        print(f"Error loading table: {e}")

        raise

    finally:

        cur.close()
        conn.close()


# In[35]:


# Drop internal working columns
internal = [c for c in df.columns if c.startswith("_")]
creator_cols = [c for c in df.columns if c in ("is_aml_creator", "is_aml_detector",
                "aml_typology_creator", "aml_typology_detected", "aml_flag_source")]
df_out = df.drop(columns=[c for c in internal + creator_cols if c in df.columns])

for col in ["is_aml", "aml_typology", "typology_group_id"]:
    if col not in df_out.columns:
        df_out[col] = "" if col != "is_aml" else 0



# In[36]:


df_out = pd.DataFrame(df_out)
df_out['datestamp'] = pd.to_datetime(df_out['datestamp'],format="%d-%m-%Y",errors="coerce")
df_out['customer_cif_creation_date'] = pd.to_datetime(df_out['customer_cif_creation_date'],format="%d-%m-%Y",errors="coerce")
df_out['account_wallet_opening_date'] = pd.to_datetime(df_out['account_wallet_opening_date'],format="%d-%m-%Y",errors="coerce")
df_out['kyc_update_date'] = pd.to_datetime(df_out['kyc_update_date'],format="%d-%m-%Y",errors="coerce")
df_out['account_wallet_inoperative_date'] = pd.to_datetime(df_out['account_wallet_inoperative_date'])
df_out['date_of_incorporation'] = pd.to_datetime(df_out['date_of_incorporation'],format="%d-%m-%Y",errors="coerce")
df_out['date_of_birth'] = pd.to_datetime(df_out['date_of_birth'],format="%d-%m-%Y",errors="coerce")
df_out["professional_experience_years"] = pd.to_numeric(
    df_out["professional_experience_years"],
    errors="coerce"
).astype("Int64")

df_out.head()
df_out['cif_beneficial_owners'].unique()


# In[37]:


# Write to PostgreSQL (full refresh each run)
from datetime import datetime
df_out = pd.DataFrame(df_out)
df_out["loaded_at"] = datetime.now()

write_table_fast(df_out, "stg_transactions_flagged", mode="replace")
print(f"Detector output written: {len(df_out):,} rows")


# In[ ]:




