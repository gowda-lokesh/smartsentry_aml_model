#!/usr/bin/env python
# coding: utf-8

# # AML Synthetic Data Generator -- Complete Pipeline
# ---
# **Single notebook** that generates all master data tables and a fully-labeled transaction
# dataset with **all 92 schema columns** and embedded AML typologies.
# 
# ### Output Files
# | File | Description | Primary Key |
# |------|-------------|-------------|
# | `customers.csv` | Individual + Entity customer profiles | `customer_cif` |
# | `accounts.csv` | Bank accounts linked to customers | `account_number` |
# | `wallets.csv` | PPI wallets linked to customers | `wallet_id` |
# | `devices.csv` | Device fingerprints per customer | `device_id` |
# | `transactions_final.csv` | All transactions with 92 columns + AML labels | `transaction_id` |
# 
# ### AML Typologies (injected during generation)
# | # | Typology | Pattern |
# |---|----------|--------|
# | 1 | Structuring (Smurfing) | Sub-threshold cash deposits then consolidation |
# | 2 | Circular Transaction Loops | A -> B -> C -> A ring transfers |
# | 3 | Funnel Account Networks | Many-to-one then rapid outflow |
# | 4 | Pass-Through Transit Hubs | Receive large sum, forward within minutes |
# | 5 | Rapid Multi-Hop Layering | 8-10 hop chains within hours |
# 

# ## 1 -- Environment Setup
# 

# In[1]:


import os
print("Working directory:", os.getcwd())
print("Contents:", os.listdir())


# In[2]:


import os
import importlib
from project_config.loader import ensure_notebook_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ["ml_outputs", "phase2_outputs", "executed_notebooks"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
print(f"Output directory ready: {os.path.abspath(OUTPUT_DIR)}")


# In[3]:


import random, string, hashlib, uuid, json, os, math
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
import csv
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# ============================================================
# MASTER SEED - Controls ALL randomness for full reproducibility.
# Change this single value to get a different but equally
# reproducible dataset. Every run with the same seed produces
# identical output.
# ============================================================
MASTER_SEED = 42

random.seed(MASTER_SEED)

# Deterministic counter for device IDs (replaces uuid.uuid4 which uses OS entropy)
_deterministic_counter = 0
def deterministic_uuid():
    global _deterministic_counter
    _deterministic_counter += 1
    # Create a reproducible hash from seed + counter
    raw = f"SEED{MASTER_SEED}_DEV{_deterministic_counter}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]

# Deterministic session ID generator (replaces random.random() in hashlib)
_session_counter = 0
def deterministic_session(acct, date_str):
    global _session_counter
    _session_counter += 1
    raw = f"SEED{MASTER_SEED}_SESS{_session_counter}_{acct}_{date_str}".encode()
    return hashlib.md5(raw).hexdigest()[:24]


import os
from project_config.loader import load_generator_config, ensure_notebook_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = load_generator_config(_SETTINGS)
MASTER_SEED = CONFIG.get("master_seed", _SETTINGS.get("generator", {}).get("master_seed", 42))
random.seed(MASTER_SEED)
#print(f"CONFIG loaded from {_PATHS['generator_config_json']}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print(f"Master seed: {MASTER_SEED}")


# ## 2 -- Configuration (includes Fraud % and Typology Weights)
# 

# In[4]:


MASTER_SEED = 42
CONFIG = {
    # ── Reproducibility ──
    "master_seed": MASTER_SEED,

    # ── Volume ──
    "num_customers_individual": 3000,
    "num_customers_entity": 400,
    "num_accounts_per_customer_range": [1, 2],
    "num_wallets_fraction": 0.30,
    "num_transactions_per_account_range": [10, 120],
    "date_range_start": "2025-12-01",
    "date_range_end": "2026-03-01",

    # ── Fraud / Typology Configuration ──
    "target_fraud_pct": 0.20,   # 10% of final transactions will be AML-labeled
    # ── Weights kept evenly distributed; detector composition is matched
    #     by tuning detector thresholds, NOT by skewing creator weights.
    "typology_weights": {
        "Structuring (Smurfing)":       0.10,
        "Circular Transaction Loop":    0.10,
        "Funnel Account Network":       0.10,
        "Pass-Through Transit Hub":     0.10,
        "Rapid Multi-Hop Layering":     0.10,
        "Third-Party Payment Web":      0.10,
        "Money Mule Network":           0.10,
        "High-Risk Corridor Transfer":  0.10,
        "Underground Banking (Hawala)": 0.10,
        "Charity Abuse":                0.10,
    },
    # ── Calibrated to what each typology function actually emits per scenario.
    #     Old values were too low for Funnel/Money Mule/Charity/Third-Party,
    #     causing the planner to spawn too many scenarios -> overshoot.
    "avg_txns_per_scenario": {
        "Structuring (Smurfing)":       6,    # 3-6 deposits + 1-3 transfers ≈ 6
        "Circular Transaction Loop":    4,    # ring of 3-5, one txn per hop
        "Funnel Account Network":       35,   # 15-50 feeders + 2-3 outflows
        "Pass-Through Transit Hub":     2,    # 1 in + 1 out
        "Rapid Multi-Hop Layering":     9,    # 8-10 hops
        "Third-Party Payment Web":      10,   # 5-15 unrelated payers
        "Money Mule Network":           25,   # 5-20 mules x (in+out)
        "High-Risk Corridor Transfer":  5,    # 3-8 transfers per account
        "Underground Banking (Hawala)": 4,    # 3-4 parties closing the loop
        "Charity Abuse":                28,   # 10-40 donors + 2-5 diversions
    },

    # ── Typology Generation Parameters ──
    # These control exactly how each AML pattern is constructed.
    # The detector's DETECT_CONFIG should mirror these for alignment.
    "typology_generation": {
        "structuring": {
            "num_sources_range": [3, 6],           # Distinct accounts making sub-threshold deposits
            "deposit_amount_range": [8000, 9900],   # Cash deposit amount (just below 10K threshold)
            "transfer_amount_range": [7500, 9800],  # Outflow amount to consolidation target
            "deposit_hour_range": [9, 16],          # Business hours for deposits
            "transfer_hour_range": [10, 18],        # Hours for transfers out
            "transfer_delay_days_range": [1, 3],    # Days between deposit and transfer
            "deposit_channel": "Branch Cash",       # Channel for cash deposits
            "transfer_channels": ["NEFT", "IMPS", "UPI"],
        },
        "circular": {
            "ring_size_range": [3, 5],              # Number of accounts in the loop
            "base_amount_range": [50000, 500000],   # Starting amount for the ring
            "hop_amount_decay": [0.97, 1.0],        # Per-hop amount multiplier (0-3% fee)
            "hop_interval_days": 1,                 # Days between each hop
            "hop_hour_range": [10, 17],
            "channels": ["NEFT", "RTGS", "IMPS"],
        },
        "funnel": {
            "num_feeders_range": [15, 50],          # Distinct sender accounts
            "per_feeder_amount_range": [5000, 30000],
            "feeder_spread_days_range": [0, 5],     # Days over which feeders deposit
            "feeder_hour_range": [8, 20],
            "outflow_delay_days_range": [6, 10],    # Days after start for outflow
            "outflow_splits_range": [2, 3],         # Number of outflow transactions
            "outflow_split_pct_range": [0.3, 0.5],  # Each split as % of remaining
            "retention_pct": 0.05,                  # 5% retained, 95% forwarded
            "outflow_hour_range": [10, 16],
            "feeder_channels": ["UPI", "IMPS", "NEFT"],
        },
        "passthrough": {
            "inflow_amount_range": [200000, 2000000],
            "forward_pct_range": [0.96, 0.99],      # Forwards 96-99% of inflow
            "hour_range": [10, 17],
            "time_gap_hours": [0, 1],               # Outflow within 0-1 hours of inflow
            "inflow_channels": ["RTGS", "NEFT"],
            "outflow_channels": ["RTGS", "NEFT", "IMPS"],
        },
        "layering": {
            "num_hops_range": [8, 10],              # Chain length
            "base_amount_range": [100000, 1000000],
            "per_hop_decay": 0.99,                  # 1% decay per hop
            "per_hop_noise_range": [0.98, 1.0],     # Additional random noise per hop
            "hop_interval_minutes_range": [5, 30],  # Minutes between each hop
            "start_hour_range": [9, 14],
            "channels": ["IMPS", "NEFT", "UPI", "RTGS"],
        },
        "topup": {
            "passthrough_amount_range": [100000, 1500000],
            "passthrough_forward_pct": 0.97,
            "structuring_sources_range": [3, 5],
            "structuring_deposit_range": [8000, 9900],
            "structuring_transfer_range": [7500, 9800],
            "layering_hops_range": [6, 8],
            "layering_amount_range": [80000, 600000],
            "layering_decay": 0.99,
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
    },

        # ── Currency ──
    "primary_currency": "INR",
    "fx_currencies": ["USD", "GBP", "EUR", "AED", "SGD"],
    "fx_probability": 0.03,

    # ── Risk distribution ──
    "risk_weights": {"Low": 0.60, "Medium": 0.30, "High": 0.10},

    # ── PEP / HNI / Minor ──
    "pep_probability": 0.02,
    "hni_probability": 0.05,
    "minor_probability": 0.03,

    # ── Income bands (INR) ──
    "income_bands": {
        "Low":    [100000, 500000],
        "Medium": [500001, 2500000],
        "High":   [2500001, 10000000],
        "HNI":    [10000001, 100000000]
    },

    # ── Wallet limits (RBI PPI norms) ──
    "wallet_kyc_categories": {
        "Minimum KYC": {
            "per_txn": 10000, "daily": 10000, "monthly": 10000,
            "annual": 120000, "max_balance": 10000
        },
        "Full KYC": {
            "per_txn": 200000, "daily": 200000, "monthly": 200000,
            "annual": 2400000, "max_balance": 200000
        }
    },

    # ── IFSC pool ──
    "bank_ifsc_prefixes": [
        "SBIN0", "HDFC0", "ICIC0", "UTIB0", "PUNB0",
        "BARB0", "CNRB0", "KKBK0", "IOBA0", "BKID0"
    ],

    # ── MCC codes (ISO 18245) ──
    "mcc_codes": {
        "5411": "Grocery Stores", "5541": "Service Stations",
        "5812": "Restaurants", "5912": "Drug Stores",
        "4814": "Telecom Services", "6012": "Financial Institutions",
        "7011": "Hotels/Motels", "5691": "Clothing Stores",
        "5045": "Computers/Peripherals", "8062": "Hospitals",
        "4899": "Cable/Utilities", "5999": "Miscellaneous Retail",
        "6051": "Quasi Cash - Money Orders", "6211": "Security Brokers/Dealers",
        "7995": "Gambling/Betting", "7994": "Video Game Arcades",
        "5816": "Digital Goods - Games", "5817": "Digital Goods - Software",
        "5818": "Digital Goods - Large Digital Marketplace",
        "7801": "Government Licensed Online Casino", "7802": "Government Licensed Horse Racing",
        "6012": "Cryptocurrency Exchanges", "6050": "Quasi Cash - Crypto",
        "5967": "Direct Marketing - High Risk", "5966": "Outbound Telemarketing"
    },
    "high_risk_mccs": ["7995","7994","7801","7802","6051","6050","6012","5816","5817","5818","5967","5966"],
    "negative_list_names": [
        "HAWALA TRADERS LLC", "OFFSHORE SHELL CORP", "DARK WEB MERCHANTS",
        "SANCTIONED ENTITY FZE", "TERROR FINANCE TRUST", "BLOCKED ENTITY PVT LTD"
    ],
    "negative_list_devices": ["DEV_BLACKLISTED_001","DEV_BLACKLISTED_002","DEV_BLACKLISTED_003"],
    "negative_list_ips": ["192.168.99.1","10.0.0.99","172.16.99.1"],
    "negative_list_vpas": ["scam@upi","fraud@upi","blocked@upi"],

    # ── Transaction channels ──
    "bank_channels": [
        "NEFT", "RTGS", "IMPS", "UPI", "Branch Cash",
        "ATM", "Cheque", "Internet Banking", "Mobile Banking",
        "POS", "Demand Draft"
    ],
    "ppi_channels": ["UPI", "QR Scan", "In-App Transfer", "NFC Tap", "Online"],

    # ── Device / Auth ──
    "auth_methods": ["OTP", "PIN", "Biometric", "MPIN", "Password"],
    "browsers": [
        "Chrome/124.0", "Safari/17.4", "Firefox/125.0",
        "Edge/124.0", "SamsungBrowser/24.0"
    ],
    "app_versions": ["v8.1.2", "v8.2.0", "v9.0.1", "v9.1.0", "v7.5.3"],

    # ── Geography ──
    "indian_states": [
        "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana",
        "Gujarat", "Rajasthan", "Uttar Pradesh", "West Bengal", "Kerala",
        "Madhya Pradesh", "Punjab", "Haryana", "Bihar", "Odisha"
    ],
    "cities_by_state": {
        "Maharashtra":   ["Mumbai", "Pune", "Nagpur", "Nashik"],
        "Delhi":         ["New Delhi", "Dwarka", "Rohini", "Saket"],
        "Karnataka":     ["Bengaluru", "Mysuru", "Hubli", "Mangaluru"],
        "Tamil Nadu":    ["Chennai", "Coimbatore", "Madurai", "Salem"],
        "Telangana":     ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
        "Gujarat":       ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
        "Rajasthan":     ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
        "Uttar Pradesh": ["Lucknow", "Noida", "Agra", "Varanasi"],
        "West Bengal":   ["Kolkata", "Howrah", "Siliguri", "Durgapur"],
        "Kerala":        ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"],
        "Madhya Pradesh":["Bhopal", "Indore", "Gwalior", "Jabalpur"],
        "Punjab":        ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar"],
        "Haryana":       ["Gurugram", "Faridabad", "Karnal", "Panipat"],
        "Bihar":         ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
        "Odisha":        ["Bhubaneswar", "Cuttack", "Rourkela", "Puri"]
    },

    # ── Occupations ──
    "occupations_individual": [
        "Salaried - IT", "Salaried - Banking", "Salaried - Government",
        "Self-Employed - Retail", "Self-Employed - Professional",
        "Business Owner", "Freelancer", "Student", "Retired",
        "Agriculture", "Doctor", "Lawyer", "Consultant"
    ],
    "entity_types": [
        "Private Limited", "Public Limited", "LLP",
        "Partnership", "Trust", "Society", "Sole Proprietorship",
        "HUF", "Government Body"
    ],
    "entity_industries": [
        "IT Services", "Manufacturing", "Trading", "Real Estate",
        "Financial Services", "Healthcare", "Education", "Hospitality",
        "Logistics", "Textiles", "Pharmaceuticals", "Agriculture",
        "Construction", "Retail", "Import/Export"
    ]
}

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)
print("Configuration saved (including fraud % and typology weights)")


# ## 3 -- Full 92-Column Schema Reference
# Every transaction row will contain exactly these 92 fields plus 3 AML label fields (95 total).
# 

# In[5]:


# Definitive ordered list of ALL 92 schema columns + 3 label columns = 95 total
TRANSACTION_COLUMNS = [
    # -- Transaction Identifiers (SN 1-22) --
    "transaction_id",                    # SN1  - Unique transaction reference
    "timestamp",                         # SN2  - HH:MM:SS
    "datestamp",                          # SN3  - DD-MM-YYYY
    "transaction_amount",                # SN4  - Monetary value
    "currency",                          # SN5  - ISO currency code
    "transaction_type_dr_cr",            # SN6  - Debit (Dr) / Credit (Cr)
    "transaction_mode_channel_bank",     # SN7  - Bank channel (NEFT/RTGS/UPI etc)
    "cash_flag",                         # SN8  - Y/N physical cash
    "transaction_type_ppi",              # SN9  - PPI transaction type
    "transaction_mode_channel_ppi",      # SN10 - PPI digital channel
    "transaction_status",                # SN11 - Success/Failed/Pending/Reversed
    "wallet_balance_before",             # SN12 - PPI balance before
    "wallet_balance_after",              # SN13 - PPI balance after
    "source_of_funds_wallet",            # SN14 - Wallet load source
    "load_instrument_type",              # SN15 - Payment method for wallet load
    "load_source_account_card_details",  # SN16 - Masked source account/card
    "beneficiary_wallet_id_vpa",         # SN17 - Recipient wallet/UPI VPA
    "merchant_id",                       # SN18 - Merchant identifier
    "merchant_name",                     # SN19 - Business name
    "merchant_category_code",            # SN20 - MCC (ISO 18245)
    "merchant_location",                 # SN21 - Merchant location
    "refund_chargeback_flag",            # SN22 - Y/N refund/chargeback

    # -- Participant Information (SN 23-38) --
    "customer_account_number",           # SN23 - Customer bank account
    "account_wallet_status",             # SN24 - Active/Frozen/Dormant etc
    "non_face_to_face_flag",             # SN25 - Y/N remote onboarding
    "pep_flag",                          # SN26 - Politically Exposed Person
    "hni_flag",                          # SN27 - High Net Worth Individual
    "minor_flag",                        # SN28 - Under 18 flag
    "customer_branch_ifsc_code",         # SN29 - Home branch IFSC
    "customer_cif_id",                   # SN30 - CIF number
    "customer_cif_creation_date",        # SN31 - CIF creation date
    "annual_income",                     # SN32 - Annual income (INR)
    "counterparty_account_number",       # SN33 - Beneficiary/sender account
    "counterparty_branch_ifsc_swift",    # SN34 - Counterparty IFSC/SWIFT
    "customer_name",                     # SN35 - Full legal name
    "counterparty_name",                 # SN36 - Counterparty name
    "sender_country_code",               # SN37 - Sender ISO country
    "receiver_country_code",             # SN38 - Receiver ISO country

    # -- Customer Profile Data (SN 39-74) --
    "customer_current_risk_score",       # SN39 - Low/Medium/High
    "customer_type",                     # SN40 - Individual/Non-Individual
    "customer_entity_type",              # SN41 - Detailed entity type
    "account_category",                  # SN42 - Savings/Current/NRE etc
    "account_type",                      # SN43 - Regular/Premium/Basic etc
    "account_wallet_opening_date",       # SN44 - Account/wallet open date
    "customer_occupation_industry",      # SN45 - Occupation or industry
    "vkyc_flag",                         # SN46 - Video KYC flag
    "kyc_update_date",                   # SN47 - Last KYC update
    "account_wallet_inoperative_date",   # SN48 - Inoperative status date
    "source_of_funds",                   # SN49 - Wealth/income origin
    "tax_residency",                     # SN50 - India/Other
    "nationality",                       # SN51 - Country of citizenship
    "citizenship",                       # SN52 - Legal citizenship
    "residency",                         # SN53 - Resident/NRI
    "date_of_incorporation",             # SN54 - Entity incorporation date
    "place_of_incorporation",            # SN55 - Entity registration location
    "beneficial_owner_types",            # SN56 - UBO classification
    "passive_nfe",                       # SN57 - Passive Non-Financial Entity
    "address_registered_office",         # SN58 - Entity registered office
    "address_place_of_business",         # SN59 - Entity business address
    "address_beneficial_owners",         # SN60 - UBO addresses
    "address_individual_customer",       # SN61 - Individual residential address
    "date_of_birth",                     # SN62 - Individual DOB
    "father_spouse_name",                # SN63 - Father/spouse name
    "identification_proof_doc_no",       # SN64 - OVD document number
    "entity_identification_proof_doc_no",# SN65 - Entity registration number
    "credit_summation_period",           # SN66 - Total credits in period
    "debit_summation_period",            # SN67 - Total debits in period
    "professional_experience_years",     # SN68 - Work experience
    "cif_beneficial_owners",             # SN69 - UBO customer IDs
    "name_beneficial_owners",            # SN70 - UBO names
    "mobile_number",                     # SN71 - Registered mobile
    "pan",                               # SN72 - PAN number
    "aadhaar_number",                    # SN73 - Aadhaar (masked)
    "email_id",                          # SN74 - Email address

    # -- Wallet Account Data (SN 75-82) --
    "wallet_kyc_category",               # SN75 - Min KYC / Full KYC
    "wallet_account_id",                 # SN76 - Wallet ID
    "escrow_account_linked",             # SN77 - Escrow account
    "transaction_limit_per_txn",         # SN78 - Per-txn limit
    "daily_transaction_limit",           # SN79 - Daily limit
    "monthly_transaction_limit",         # SN80 - Monthly limit
    "annual_transaction_limit",          # SN81 - Annual limit
    "maximum_wallet_balance_limit",      # SN82 - Max wallet balance

    # -- Device & Channel Data (SN 83-92) --
    "device_id_fingerprint",             # SN83 - Device ID/fingerprint
    "ip_address",                        # SN84 - Originating IP
    "geo_location_city_country",         # SN85 - Geo from IP/GPS
    "gps_coordinates_lat",               # SN86 - GPS latitude
    "gps_coordinates_lon",               # SN86 - GPS longitude (split)
    "browser_app_information",           # SN87 - Browser/app version
    "session_id",                        # SN88 - Session identifier
    "authentication_method",             # SN89 - OTP/PIN/Biometric
    "vpn_flag",                          # SN90 - VPN/proxy/Tor flag
    "emulator_flag",                     # SN91 - Emulator flag
    "customer_address_lat",              # SN92 - Registered address lat
    "customer_address_lon",              # SN92 - Registered address lon (split)

    # -- AML Labels (3 additional) --
    "is_aml",                            # 0/1 label
    "aml_typology",                      # Typology name or empty
    "typology_group_id",                 # Scenario group ID or empty
]

print(f"Total columns defined: {len(TRANSACTION_COLUMNS)}")
print(f"  Schema columns (SN 1-92): {len(TRANSACTION_COLUMNS) - 3}")
print(f"  AML label columns: 3")



# ## 4 -- Helper Functions
# 

# In[6]:


# -- Realistic ID generators --

def generate_cif():
    return str(random.randint(100000000000, 999999999999))

def generate_account_number(ifsc_prefix):
    bank = ifsc_prefix[:4]
    if bank == "SBIN":
        return str(random.randint(10000000000, 99999999999))
    elif bank == "HDFC":
        return str(random.randint(10000000000000, 99999999999999))
    elif bank == "ICIC":
        return str(random.randint(100000000000, 999999999999))
    elif bank == "UTIB":
        return str(random.randint(9100000000000, 9299999999999))
    elif bank == "PUNB":
        return str(random.randint(1000000000000000, 9999999999999999))
    else:
        length = random.choice([11, 12, 13, 14])
        return str(random.randint(10**(length-1), 10**length - 1))

def generate_ifsc():
    prefix = random.choice(CONFIG["bank_ifsc_prefixes"])
    return prefix + str(random.randint(10000, 99999))

def generate_pan(is_entity=False):
    first3 = ''.join(random.choices(string.ascii_uppercase, k=3))
    fourth = random.choice(["C","H","F","A","T","B","L","J","G"]) if is_entity else "P"
    fifth = random.choice(string.ascii_uppercase)
    digits = ''.join(random.choices(string.digits, k=4))
    last = random.choice(string.ascii_uppercase)
    return first3 + fourth + fifth + digits + last

def generate_aadhaar():
    return str(random.randint(2, 9)) + ''.join(random.choices(string.digits, k=11))

def generate_aadhaar_masked():
    return "XXXX-XXXX-" + ''.join(random.choices(string.digits, k=4))

def generate_mobile():
    return str(random.choice([6,7,8,9])) + ''.join(random.choices(string.digits, k=9))

def generate_email(name):
    domains = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com"]
    clean = name.lower().replace(" ", ".").replace("'", "")
    return f"{clean}{random.randint(1,999)}@{random.choice(domains)}"

def generate_vpa(name):
    handles = ["oksbi", "okaxis", "okicici", "okhdfcbank", "ybl", "paytm", "ibl", "upi"]
    clean = name.lower().replace(" ", "")[:10]
    return f"{clean}{random.randint(1,99)}@{random.choice(handles)}"

def generate_wallet_id():
    return "WLT" + ''.join(random.choices(string.digits, k=12))

def generate_merchant_id():
    return "MID" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))

def generate_device_id():
    return deterministic_uuid()

def generate_ip():
    first_octet = random.choice([49, 59, 103, 106, 117, 122, 157, 182, 203])
    return f"{first_octet}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def generate_gps(city, state):
    coords = {
        "Mumbai": (19.076, 72.878), "Pune": (18.520, 73.856),
        "New Delhi": (28.614, 77.209), "Bengaluru": (12.972, 77.594),
        "Chennai": (13.083, 80.271), "Hyderabad": (17.385, 78.487),
        "Ahmedabad": (23.023, 72.571), "Kolkata": (22.573, 88.364),
        "Jaipur": (26.912, 75.787), "Lucknow": (26.847, 80.947),
        "Kochi": (9.932, 76.267), "Bhopal": (23.260, 77.413),
        "Chandigarh": (30.734, 76.779), "Patna": (25.611, 85.144),
        "Bhubaneswar": (20.297, 85.825),
    }
    base = coords.get(city, (20.5 + random.uniform(-5, 5), 78.0 + random.uniform(-5, 5)))
    return (round(base[0] + random.uniform(-0.05, 0.05), 6),
            round(base[1] + random.uniform(-0.05, 0.05), 6))

def random_date(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))

def weighted_choice(weight_dict):
    items = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(items, weights=weights, k=1)[0]

def generate_txn_id():
    return "TXN" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

# -- Name generators --
FIRST_NAMES_M = ["Rahul","Amit","Suresh","Vikram","Arjun","Rohan","Kiran","Deepak",
    "Manoj","Rajesh","Sanjay","Anil","Pradeep","Nikhil","Varun",
    "Aditya","Akash","Harsh","Yash","Gaurav","Sachin","Pranav",
    "Ankit","Ravi","Sandeep","Mohit","Tushar","Vivek","Ashish","Naveen"]
FIRST_NAMES_F = ["Priya","Anita","Sunita","Kavita","Neha","Pooja","Swati","Meera",
    "Divya","Sneha","Ritu","Pallavi","Shruti","Aditi","Anjali",
    "Nisha","Rekha","Sonal","Tanvi","Bhavna","Komal","Jyoti",
    "Asha","Geeta","Lata","Mamta","Rashmi","Sapna","Shweta","Vandana"]
LAST_NAMES = ["Sharma","Verma","Patel","Gupta","Singh","Kumar","Reddy","Nair",
    "Iyer","Joshi","Deshmukh","Rao","Pillai","Mehta","Shah",
    "Agarwal","Mishra","Bhat","Yadav","Chauhan","Tiwari","Pandey",
    "Malhotra","Kapoor","Bose","Chatterjee","Mukherjee","Das","Banerjee","Sen"]
ENTITY_PREFIXES = ["Shri","Om","Bharat","National","Premier","Golden","Silver",
    "Diamond","Royal","Star","Global","United","Pioneer","Excel","Apex"]
ENTITY_SUFFIXES = ["Enterprises","Trading Co","Industries","Solutions","Services",
    "Exports","Imports","Associates","Group","Holdings",
    "Infra","Ventures","Technologies","Logistics","Capital"]
MERCHANT_NAMES = [
    "Reliance Fresh","Big Bazaar","DMart","Spencer's","Amazon","Flipkart",
    "Zomato","Swiggy","Uber","Ola","IRCTC","MakeMyTrip","Cleartrip",
    "Apollo Pharmacy","PharmEasy","1mg","Paytm Mall","Myntra","Ajio",
    "Croma","Vijay Sales","BookMyShow","PVR Cinemas","Starbucks India",
    "McDonald's","Domino's","KFC","Jio Mart","Tata Neu","Samsung Store",
    # High-risk merchants (for PPI rules 12,39,48,50)
    "BetWin Online Gaming","Lucky Casino India","CryptoEx Exchange",
    "ForexTrader Pro","GameZone Premium","PokerStar India",
    "Fantasy Sports Hub","Digital Betting Corp","CoinSwap Crypto",
    "Adult Content Platform","Dark Market Deals"
]

def random_individual_name():
    if random.random() < 0.5:
        return random.choice(FIRST_NAMES_M) + " " + random.choice(LAST_NAMES)
    return random.choice(FIRST_NAMES_F) + " " + random.choice(LAST_NAMES)

def random_entity_name():
    return random.choice(ENTITY_PREFIXES) + " " + random.choice(ENTITY_SUFFIXES)

def random_father_name():
    return random.choice(FIRST_NAMES_M) + " " + random.choice(LAST_NAMES)

print("Helpers loaded")





# ## 5 -- Generate Customer Master Data
# 

# In[7]:


customers = []
cif_set = set()

def make_individual_customer():
    cif = generate_cif()
    while cif in cif_set:
        cif = generate_cif()
    cif_set.add(cif)

    state = random.choice(CONFIG["indian_states"])
    city = random.choice(CONFIG["cities_by_state"][state])
    gps = generate_gps(city, state)
    name = random_individual_name()

    is_pep = random.random() < CONFIG["pep_probability"]
    is_hni = random.random() < CONFIG["hni_probability"]
    is_minor = random.random() < CONFIG["minor_probability"]
    risk = weighted_choice(CONFIG["risk_weights"])
    if is_pep:
        risk = "High"

    if is_hni:
        income_band = CONFIG["income_bands"]["HNI"]
    elif risk == "High":
        income_band = CONFIG["income_bands"][random.choice(["Medium", "High"])]
    else:
        income_band = CONFIG["income_bands"][risk]
    annual_income = random.randint(*income_band)

    dob = random_date("1960-01-01", "2006-12-31") if not is_minor else random_date("2008-01-01", "2015-12-31")
    cif_creation = random_date("2015-01-01", "2024-06-30")
    kyc_date = random_date(cif_creation.strftime("%Y-%m-%d"), "2025-03-15")

    # NRI status drives residency, tax residency, and later account category
    _residency = random.choices(["Resident", "NRI"], weights=[0.95, 0.05])[0]

    # NRIs have foreign country associations
    _nri_countries = ["US", "GB", "AE", "SG", "CA", "AU", "DE", "NL", "JP", "QA", "KW", "OM", "BH"]
    _nri_country = random.choice(_nri_countries) if _residency == "NRI" else ""

    return {
        "customer_cif": cif, "customer_name": name,
        "customer_type": "Individual", "customer_entity_type": "Individual",
        "date_of_birth": dob.strftime("%d-%m-%Y"),
        "father_spouse_name": random_father_name(),
        "nationality": "Indian",
        "citizenship": "Indian" if _residency == "Resident" else random.choice(["Indian", "Dual - Indian/" + _nri_country]),
        "residency": _residency,
        "tax_residency": "Other" if _residency == "NRI" else random.choices(["India", "Other"], weights=[0.99, 0.01])[0],
        "pan": generate_pan(False), "aadhaar": generate_aadhaar(),
        "aadhaar_masked": generate_aadhaar_masked(),
        "identification_doc_no": "AADHAAR-" + generate_aadhaar()[:4] + "XXXX" + generate_aadhaar()[-4:],
        "mobile_number": generate_mobile(), "email_id": generate_email(name),
        "address_individual": f"{random.randint(1,500)}, {random.choice(['MG Road','Station Road','NH Highway','Park Street','Ring Road','Gandhi Nagar','Nehru Place'])}, {city}, {state} - {random.randint(100000,999999)}",
        "state": state, "city": city,
        "address_lat": gps[0], "address_lon": gps[1],
        "occupation_industry": random.choice(CONFIG["occupations_individual"]),
        "annual_income": annual_income,
        "professional_experience_years": 0 if is_minor else random.randint(0, 40),
        "source_of_funds": random.choice(["Foreign Employment","Overseas Business","Investment Returns","Rental Income"]) if _residency == "NRI" else random.choice(["Salary","Business Income","Investment Returns","Rental Income","Family Support","Pension","Agriculture Income"]),
        "_nri_country": _nri_country,
        "customer_risk_score": risk,
        "pep_flag": "Y" if is_pep else "N",
        "hni_flag": "Y" if is_hni else "N",
        "minor_flag": "Y" if is_minor else "N",
        "non_face_to_face_flag": random.choices(["Y","N"], weights=[0.35,0.65])[0],
        "vkyc_flag": random.choices(["Y","N"], weights=[0.25,0.75])[0],
        "cif_creation_date": cif_creation.strftime("%d-%m-%Y"),
        "kyc_update_date": kyc_date.strftime("%d-%m-%Y"),
        "date_of_incorporation": "", "place_of_incorporation": "",
        "beneficial_owner_types": "", "passive_nfe": "",
        "address_registered_office": "", "address_place_of_business": "",
        "address_beneficial_owners": "", "entity_identification_doc_no": "",
        "cif_beneficial_owners": "", "name_beneficial_owners": "",
    }

def make_entity_customer():
    cif = generate_cif()
    while cif in cif_set:
        cif = generate_cif()
    cif_set.add(cif)

    state = random.choice(CONFIG["indian_states"])
    city = random.choice(CONFIG["cities_by_state"][state])
    gps = generate_gps(city, state)
    entity_name = random_entity_name()
    entity_type = random.choice(CONFIG["entity_types"])
    risk = weighted_choice(CONFIG["risk_weights"])
    inc_date = random_date("1990-01-01", "2024-01-01")
    cif_creation = random_date(inc_date.strftime("%Y-%m-%d"), "2024-06-30")
    if cif_creation < inc_date:
        cif_creation = inc_date + timedelta(days=random.randint(1, 365))
    kyc_date = random_date(cif_creation.strftime("%Y-%m-%d"), "2025-03-15")
    ubo_name = random_individual_name()

    return {
        "customer_cif": cif, "customer_name": entity_name,
        "customer_type": "Non-Individual", "customer_entity_type": entity_type,
        "date_of_birth": "", "father_spouse_name": "",
        "nationality": "Indian", "citizenship": "",
        "residency": "Resident", "tax_residency": "India",
        "pan": generate_pan(True), "aadhaar": "", "aadhaar_masked": "",
        "identification_doc_no": "", "mobile_number": generate_mobile(),
        "email_id": generate_email(entity_name.split()[0]),
        "address_individual": "", "state": state, "city": city,
        "address_lat": gps[0], "address_lon": gps[1],
        "occupation_industry": random.choice(CONFIG["entity_industries"]),
        "annual_income": random.randint(1000000, 500000000),
        "professional_experience_years": "",
        "source_of_funds": random.choice(["Business Revenue","Investment","Government Grant","Donations","Trading Profits"]),
        "customer_risk_score": risk,
        "pep_flag": "N", "hni_flag": "N", "minor_flag": "N",
        "non_face_to_face_flag": random.choices(["Y","N"], weights=[0.20,0.80])[0],
        "vkyc_flag": "N",
        "cif_creation_date": cif_creation.strftime("%d-%m-%Y"),
        "kyc_update_date": kyc_date.strftime("%d-%m-%Y"),
        "date_of_incorporation": inc_date.strftime("%d-%m-%Y"),
        "place_of_incorporation": f"{city}, {state}",
        "beneficial_owner_types": random.choice(["Shareholding > 25%","Control through other means","Senior Managing Official"]),
        "passive_nfe": random.choices(["Y","N"], weights=[0.15,0.85])[0],
        "address_registered_office": f"{random.randint(1,200)} Corporate Park, {city}, {state} - {random.randint(100000,999999)}",
        "address_place_of_business": f"Plot {random.randint(1,500)}, Industrial Area, {city}, {state}",
        "address_beneficial_owners": f"{random.randint(1,999)}, Residential Colony, {city}, {state}",
        "entity_identification_doc_no": f"U{random.randint(10000,99999)}{random.choice(string.ascii_uppercase)}{random.choice(string.ascii_uppercase)}{random.randint(1990,2024)}PTC{random.randint(100000,999999)}",
        "cif_beneficial_owners": generate_cif(),
        "name_beneficial_owners": ubo_name,
    }

print("Generating customers...")
for _ in range(CONFIG["num_customers_individual"]):
    customers.append(make_individual_customer())
for _ in range(CONFIG["num_customers_entity"]):
    customers.append(make_entity_customer())
random.shuffle(customers)
cust_lookup = {c["customer_cif"]: c for c in customers}
print(f"Total customers: {len(customers):,}")




# ## 6 -- Generate Accounts
# 

# In[8]:


accounts = []
acct_number_set = set()
cif_to_accounts = defaultdict(list)

ACCOUNT_CATEGORIES = ["Savings","Current","Salary","NRE","NRO","Fixed Deposit","Recurring Deposit"]
ACCOUNT_TYPES = ["Regular","Premium","Basic","Corporate","BSBD"]

for cust in customers:
    cif = cust["customer_cif"]
    n_accts = random.randint(*CONFIG["num_accounts_per_customer_range"])
    ifsc = generate_ifsc()

    for j in range(n_accts):
        acct_num = generate_account_number(ifsc[:5])
        while acct_num in acct_number_set:
            acct_num = generate_account_number(ifsc[:5])
        acct_number_set.add(acct_num)

        cif_dt = datetime.strptime(cust["cif_creation_date"], "%d-%m-%Y")
        start_bound = max(datetime.strptime("2015-01-01", "%Y-%m-%d"), cif_dt)
        open_date = random_date(start_bound.strftime("%Y-%m-%d"), "2024-12-31")

        is_dormant = random.random() < 0.05
        status = "Dormant" if is_dormant else random.choices(["Active","Frozen","Closed"], weights=[0.90,0.03,0.02])[0]
        inop_date = random_date(open_date.strftime("%Y-%m-%d"), "2025-03-15").strftime("%d-%m-%Y") if is_dormant else ""

        # ── Account Category Rules (real-world constraints) ──
        is_nri = cust.get("residency") == "NRI"
        is_minor = cust.get("minor_flag") == "Y"
        is_entity = cust["customer_type"] == "Non-Individual"

        if is_entity:
            # Entities: Current, FD, Cash Credit, OD only
            cat = random.choice(["Current", "Fixed Deposit"])
            acct_type = random.choice(["Corporate", "Regular"])
        elif is_nri:
            # NRIs: NRE, NRO, FCNR (mapped to FD), Savings (NRO sub-type)
            cat = random.choices(["NRE", "NRO", "Fixed Deposit", "Savings"],
                                 weights=[0.35, 0.35, 0.15, 0.15])[0]
            acct_type = random.choice(["Regular", "Premium"])
        elif is_minor:
            # Minors: Savings, RD only (no Current, no Salary, no NRE/NRO)
            cat = random.choices(["Savings", "Recurring Deposit"], weights=[0.85, 0.15])[0]
            acct_type = random.choice(["Basic", "Regular"])
        else:
            # Regular individuals: Savings, Current, Salary, FD, RD (no NRE/NRO)
            cat = random.choices(["Savings", "Current", "Salary", "Fixed Deposit", "Recurring Deposit"],
                                  weights=[0.40, 0.15, 0.25, 0.10, 0.10])[0]
            acct_type = random.choice(["Regular", "Premium", "Basic", "BSBD"])

        acct = {
            "account_number": acct_num, "customer_cif": cif,
            "customer_branch_ifsc": ifsc, "account_category": cat,
            "account_type": acct_type, "account_status": status,
            "account_opening_date": open_date.strftime("%d-%m-%Y"),
            "inoperative_status_date": inop_date,
            "credit_summation_period": 0, "debit_summation_period": 0,
        }
        accounts.append(acct)
        cif_to_accounts[cif].append(acct_num)

acct_lookup = {a["account_number"]: a for a in accounts}
print(f"Total accounts: {len(accounts):,}")




# ## 7 -- Generate Wallets
# 

# In[9]:


wallets = []
wallet_id_set = set()
cif_to_wallet = {}
wallet_lookup = {}

individual_cifs = [c["customer_cif"] for c in customers if c["customer_type"] == "Individual"]
wallet_cifs = random.sample(individual_cifs, int(len(individual_cifs) * CONFIG["num_wallets_fraction"]))

for cif in wallet_cifs:
    wid = generate_wallet_id()
    while wid in wallet_id_set:
        wid = generate_wallet_id()
    wallet_id_set.add(wid)

    kyc_cat = random.choices(["Minimum KYC","Full KYC"], weights=[0.40,0.60])[0]
    limits = CONFIG["wallet_kyc_categories"][kyc_cat]
    linked_accts = cif_to_accounts.get(cif, [])
    linked_acct = linked_accts[0] if linked_accts else ""
    cust = cust_lookup[cif]
    cif_year = cust["cif_creation_date"].split("-")[-1]
    open_date = random_date(f"{cif_year}-01-01", "2025-01-31")

    w = {
        "wallet_id": wid, "customer_cif": cif, "wallet_kyc_category": kyc_cat,
        "wallet_status": random.choices(["Active","Suspended","Closed"], weights=[0.90,0.05,0.05])[0],
        "wallet_opening_date": open_date.strftime("%d-%m-%Y"),
        "escrow_account_linked": f"ESC{random.randint(10**12, 10**13-1)}",
        "per_txn_limit": limits["per_txn"], "daily_txn_limit": limits["daily"],
        "monthly_txn_limit": limits["monthly"], "annual_txn_limit": limits["annual"],
        "max_balance_limit": limits["max_balance"], "linked_bank_account": linked_acct,
    }
    wallets.append(w)
    cif_to_wallet[cif] = wid
    wallet_lookup[wid] = w

print(f"Total wallets: {len(wallets):,}")



# ## 8 -- Generate Device Profiles
# 

# In[10]:


devices = []
cif_to_devices = defaultdict(list)

for cust in customers:
    cif = cust["customer_cif"]
    n_devices = random.choices([1,2,3], weights=[0.60,0.30,0.10])[0]
    for _ in range(n_devices):
        dev = {
            "customer_cif": cif,
            "device_id": generate_device_id(), "ip_address": generate_ip(),
            "geo_city": cust["city"], "geo_country": "IN",
            "gps_lat": round(cust["address_lat"] + random.uniform(-0.01, 0.01), 6),
            "gps_lon": round(cust["address_lon"] + random.uniform(-0.01, 0.01), 6),
            "browser_app_info": random.choice(CONFIG["browsers"] + CONFIG["app_versions"]),
            "vpn_flag": random.choices(["Y","N"], weights=[0.03,0.97])[0],
            "emulator_flag": random.choices(["Y","N"], weights=[0.01,0.99])[0],
        }
        devices.append(dev)
        cif_to_devices[cif].append(dev)

print(f"Total device profiles: {len(devices):,}")



# ## 9 -- Transaction Builder (Full 95 Columns)
# 

# In[11]:


used_timestamps = defaultdict(set)
txn_id_set = set()

def unique_timestamp(account_number, txn_date, preferred_hour=None):
    for _ in range(86400):
        h = preferred_hour if preferred_hour is not None else random.randint(6, 23)
        m = random.randint(0, 59)
        s = random.randint(0, 59)
        ts = f"{h:02d}:{m:02d}:{s:02d}"
        key = (txn_date, ts)
        if key not in used_timestamps[account_number]:
            used_timestamps[account_number].add(key)
            return ts
    return f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

active_accounts_list = [a for a in accounts if a["account_status"] in ("Active","Dormant")]
all_acct_numbers = [a["account_number"] for a in active_accounts_list]
active_only_numbers = [a["account_number"] for a in accounts if a["account_status"] == "Active"]

def build_full_txn(acct_num, cp_acct, amount, txn_date_str, txn_time,
                   direction, channel, is_aml=0, aml_typology="", group_id="",
                   is_ppi=False, ppi_txn_type="", ppi_channel=""):
    # Resolve all entities
    acct_info = acct_lookup.get(acct_num, {})
    cif = acct_info.get("customer_cif", "")
    cust = cust_lookup.get(cif, {})
    cp_info = acct_lookup.get(cp_acct, {})
    cp_cif = cp_info.get("customer_cif", "")
    cp_cust = cust_lookup.get(cp_cif, {})

    # Device
    dev_list = cif_to_devices.get(cif, [])
    dev = random.choice(dev_list) if dev_list else None

    # Wallet
    wallet_id = cif_to_wallet.get(cif, "")
    w_info = wallet_lookup.get(wallet_id, {}) if is_ppi and wallet_id else {}

    # Cash flag
    is_cash = "Y" if channel in ("Branch Cash", "ATM") else "N"

    # Merchant
    has_merchant = channel in ("POS","UPI") or (is_ppi and ppi_txn_type == "P2M") or random.random() < 0.2
    # MCC selection with higher probability of high-risk for some customers
    if has_merchant:
        if is_ppi and ppi_txn_type == "P2M" and random.random() < 0.15:
            mcc = random.choice(CONFIG.get("high_risk_mccs", list(CONFIG["mcc_codes"].keys())))
        else:
            mcc = random.choice(list(CONFIG["mcc_codes"].keys()))
    else:
        mcc = ""
    m_name = random.choice(MERCHANT_NAMES) if has_merchant else ""
    m_id = generate_merchant_id() if has_merchant else ""
    m_loc = f"{cust.get('city','')}, {cust.get('state','')}" if has_merchant else ""

    # Wallet balances
    w_before = round(random.uniform(0, float(w_info.get("max_balance_limit", 200000))), 2) if is_ppi else ""
    if is_ppi and isinstance(w_before, float):
        w_after = round(max(0, w_before - amount), 2) if direction == "Dr" else round(w_before + amount, 2)
    else:
        w_after = ""

    # FX
    is_fx = random.random() < CONFIG["fx_probability"] if not is_aml else False
    currency = random.choice(CONFIG["fx_currencies"]) if is_fx else CONFIG["primary_currency"]
    # Country codes: NRI accounts use NRI country for relevant direction
    sender_cc, receiver_cc = "IN", "IN"
    _nri_cc = cust.get("_nri_country", "") if cust else ""
    _acct_info = acct_lookup.get(acct_num, {})
    _acct_cat = _acct_info.get("account_category", "")

    if _acct_cat == "NRE" and _nri_cc:
        # NRE: Inward remittance from abroad (Cr) or transfer from NRE (Dr stays IN)
        if direction == "Cr":
            sender_cc = _nri_cc
        # NRE debits stay IN→IN (transferred within India or repatriated)
    elif _acct_cat == "NRO" and _nri_cc:
        # NRO: Indian income credited (Cr from IN), withdrawals/remittances outward (Dr)
        if direction == "Dr" and random.random() < 0.30:
            receiver_cc = _nri_cc  # 30% NRO debits go to NRI country
    elif is_fx:
        if direction == "Cr":
            sender_cc = random.choice(["US","GB","AE","SG","DE"])
        else:
            receiver_cc = random.choice(["US","GB","AE","SG","DE"])

    status = random.choices(["Success","Failed","Pending","Reversed"], weights=[0.92,0.04,0.02,0.02])[0] if not is_aml else "Success"
    session_id = deterministic_session(acct_num, txn_date_str)

    cp_name = cp_cust.get("customer_name", "") if cp_cust else random_individual_name()

    txn = {
        # -- Transaction Identifiers (SN 1-22) --
        "transaction_id": "",  # set externally
        "timestamp": txn_time,
        "datestamp": txn_date_str,
        "transaction_amount": round(amount, 2),
        "currency": currency,
        "transaction_type_dr_cr": direction,
        "transaction_mode_channel_bank": channel if not is_ppi else "",
        "cash_flag": is_cash,
        "transaction_type_ppi": ppi_txn_type,
        "transaction_mode_channel_ppi": ppi_channel,
        # PPI: inject failed transactions before successful ones for rules 8,30,38
        "transaction_status": "Failed" if (is_ppi and random.random() < 0.03) else status,
        "wallet_balance_before": w_before,
        "wallet_balance_after": w_after,
        "source_of_funds_wallet": random.choice(["Bank Transfer","UPI","Debit Card","Credit Card","Net Banking"]) if is_ppi else "",
        "load_instrument_type": random.choice(["Debit Card","Net Banking","UPI","Credit Card"]) if is_ppi else "",
        "load_source_account_card_details": f"XXXX-XXXX-{random.randint(1000,9999)}" if is_ppi else "",
        # Occasionally use negative-list VPA for rule PPI-66
        "beneficiary_wallet_id_vpa": random.choice(CONFIG.get("negative_list_vpas",[""])) if (is_ppi and random.random() < 0.002) else (generate_vpa(cp_name) if (is_ppi or channel == "UPI") else ""),
        "merchant_id": m_id,
        "merchant_name": m_name,
        "merchant_category_code": mcc,
        "merchant_location": m_loc,
        # Refund flag: higher for PPI P2M, creates data for PPI rule 21
        "refund_chargeback_flag": random.choices(["Y","N"], weights=[0.08,0.92])[0] if (is_ppi and ppi_txn_type == "P2M") else (random.choices(["Y","N"], weights=[0.02,0.98])[0] if not is_aml else "N"),

        # -- Participant Information (SN 23-38) --
        "customer_account_number": acct_num,
        "account_wallet_status": acct_info.get("account_status", ""),
        "non_face_to_face_flag": cust.get("non_face_to_face_flag", ""),
        "pep_flag": cust.get("pep_flag", "N"),
        "hni_flag": cust.get("hni_flag", "N"),
        "minor_flag": cust.get("minor_flag", "N"),
        "customer_branch_ifsc_code": acct_info.get("customer_branch_ifsc", ""),
        "customer_cif_id": cif,
        "customer_cif_creation_date": cust.get("cif_creation_date", ""),
        "annual_income": cust.get("annual_income", ""),
        "counterparty_account_number": cp_acct,
        "counterparty_branch_ifsc_swift": cp_info.get("customer_branch_ifsc", generate_ifsc()),
        "customer_name": cust.get("customer_name", ""),
        "counterparty_name": cp_name,
        "sender_country_code": sender_cc,
        "receiver_country_code": receiver_cc,

        # -- Customer Profile Data (SN 39-74) --
        "customer_current_risk_score": cust.get("customer_risk_score", ""),
        "customer_type": cust.get("customer_type", ""),
        "customer_entity_type": cust.get("customer_entity_type", ""),
        "account_category": acct_info.get("account_category", ""),
        "account_type": acct_info.get("account_type", ""),
        "account_wallet_opening_date": acct_info.get("account_opening_date", ""),
        "customer_occupation_industry": cust.get("occupation_industry", ""),
        "vkyc_flag": cust.get("vkyc_flag", ""),
        "kyc_update_date": cust.get("kyc_update_date", ""),
        "account_wallet_inoperative_date": acct_info.get("inoperative_status_date", ""),
        "source_of_funds": cust.get("source_of_funds", ""),
        "tax_residency": cust.get("tax_residency", ""),
        "nationality": cust.get("nationality", ""),
        "citizenship": cust.get("citizenship", ""),
        "residency": cust.get("residency", ""),
        "date_of_incorporation": cust.get("date_of_incorporation", ""),
        "place_of_incorporation": cust.get("place_of_incorporation", ""),
        "beneficial_owner_types": cust.get("beneficial_owner_types", ""),
        "passive_nfe": cust.get("passive_nfe", ""),
        "address_registered_office": cust.get("address_registered_office", ""),
        "address_place_of_business": cust.get("address_place_of_business", ""),
        "address_beneficial_owners": cust.get("address_beneficial_owners", ""),
        "address_individual_customer": cust.get("address_individual", ""),
        "date_of_birth": cust.get("date_of_birth", ""),
        "father_spouse_name": cust.get("father_spouse_name", ""),
        "identification_proof_doc_no": cust.get("identification_doc_no", ""),
        "entity_identification_proof_doc_no": cust.get("entity_identification_doc_no", ""),
        "credit_summation_period": acct_info.get("credit_summation_period", 0),
        "debit_summation_period": acct_info.get("debit_summation_period", 0),
        "professional_experience_years": cust.get("professional_experience_years", ""),
        "cif_beneficial_owners": cust.get("cif_beneficial_owners", ""),
        "name_beneficial_owners": cust.get("name_beneficial_owners", ""),
        "mobile_number": cust.get("mobile_number", ""),
        "pan": cust.get("pan", ""),
        "aadhaar_number": cust.get("aadhaar_masked", ""),
        "email_id": cust.get("email_id", ""),

        # -- Wallet Account Data (SN 75-82) --
        "wallet_kyc_category": w_info.get("wallet_kyc_category", "") if is_ppi else "",
        "wallet_account_id": wallet_id if is_ppi else "",
        "escrow_account_linked": w_info.get("escrow_account_linked", "") if is_ppi else "",
        "transaction_limit_per_txn": w_info.get("per_txn_limit", "") if is_ppi else "",
        "daily_transaction_limit": w_info.get("daily_txn_limit", "") if is_ppi else "",
        "monthly_transaction_limit": w_info.get("monthly_txn_limit", "") if is_ppi else "",
        "annual_transaction_limit": w_info.get("annual_txn_limit", "") if is_ppi else "",
        "maximum_wallet_balance_limit": w_info.get("max_balance_limit", "") if is_ppi else "",

        # -- Device & Channel Data (SN 83-92) --
        # Inject negative-list device for ~0.1% of PPI txns (PPI rule 65)
        "device_id_fingerprint": random.choice(CONFIG.get("negative_list_devices",[""])) if (is_ppi and random.random()<0.001) else (dev["device_id"] if dev else ""),
        "ip_address": dev["ip_address"] if dev else "",
        "geo_location_city_country": f"{dev['geo_city']}, {dev['geo_country']}" if dev else "",
        "gps_coordinates_lat": dev["gps_lat"] if dev else "",
        "gps_coordinates_lon": dev["gps_lon"] if dev else "",
        "browser_app_information": dev["browser_app_info"] if dev else "",
        "session_id": session_id,
        "authentication_method": random.choice(CONFIG["auth_methods"]),
        "vpn_flag": dev["vpn_flag"] if dev else "N",
        "emulator_flag": dev["emulator_flag"] if dev else "N",
        "customer_address_lat": cust.get("address_lat", ""),
        "customer_address_lon": cust.get("address_lon", ""),

        # -- AML Labels --
        "is_aml": is_aml,
        "aml_typology": aml_typology,
        "typology_group_id": group_id,
    }

    # Assign unique txn ID
    tid = generate_txn_id()
    while tid in txn_id_set:
        tid = generate_txn_id()
    txn_id_set.add(tid)
    txn["transaction_id"] = tid

    return txn

print(f"Transaction builder ready. Output columns: {len(TRANSACTION_COLUMNS)}")







# ## 10 -- Compute Fraud Injection Plan
# 

# In[12]:


# Estimate base transaction count
est_base_txns = 0
for acct in active_accounts_list:
    if acct["account_status"] == "Dormant":
        est_base_txns += 2
    else:
        lo, hi = CONFIG["num_transactions_per_account_range"]
        est_base_txns += (lo + hi) // 2

target_pct = CONFIG["target_fraud_pct"]
target_aml = int(est_base_txns * target_pct / (1.0 - target_pct))

weights = CONFIG["typology_weights"]
total_w = sum(weights.values())
norm_w = {k: v / total_w for k, v in weights.items()}
avg_per = CONFIG["avg_txns_per_scenario"]

SCENARIO_COUNTS = {}
print("=" * 75)
print("FRAUD INJECTION PLAN")
print("=" * 75)
print(f"  Estimated base txns:      {est_base_txns:>10,}")
print(f"  Target fraud %:           {target_pct*100:>9.1f}%")
print(f"  Target AML txns:          {target_aml:>10,}")
print()
#print(f"  {'Typology':<35s} {'Weight':>7s} {'Target':>10s} {'Avg/Scen':>9s} {'Scenarios':>10s}")
print(f"  {'-'*73}")

for typ in weights:
    txn_target = int(target_aml * norm_w[typ])
    n_sc = max(1, math.ceil(txn_target / avg_per[typ]))
    SCENARIO_COUNTS[typ] = n_sc
    print(f"  {typ:<35s} {norm_w[typ]*100:>6.1f}% {txn_target:>10,} {avg_per[typ]:>9} {n_sc:>10,}")

print(f"  {'-'*73}")
print(f"  {'TOTAL':<35s} {'100.0%':>7s} {target_aml:>10,}")
print("=" * 75)



# ## 11 -- Generate All Transactions (Base + Embedded Typologies)
# Base (clean) transactions and all 5 AML typologies are generated in a single pass.
# 

# In[13]:


# Should be defined before Phase 2 starts
target_pct = CONFIG.get("target_fraud_rate", 0.20)
# SCENARIO_COUNTS already built in Fraud Injection Plan cell — do NOT override
# # SCENARIO_COUNTS already built in Fraud Injection Plan cell
# SCENARIO_COUNTS = CONFIG.get("scenario_counts", CONFIG.get("typology_scenario_counts", {}))


# In[14]:


def random_date_str():
    """Generate a random date string in dd-mm-yyyy format within CONFIG date range."""
    start = datetime.strptime(CONFIG["date_range_start"], "%Y-%m-%d")
    end = datetime.strptime(CONFIG["date_range_end"], "%Y-%m-%d")
    delta = (end - start).days
    rand_date = start + timedelta(days=random.randint(0, delta))
    return rand_date.strftime("%d-%m-%Y")


# In[15]:


print(type(SCENARIO_COUNTS))
print(SCENARIO_COUNTS)


# In[16]:


transactions = []
acct_credit_sums = defaultdict(float)
acct_debit_sums  = defaultdict(float)

# ====================================================================
# PRE-ASSIGNMENT: Assign each active account to ONE typology only
# ====================================================================
# WHY THIS IS NEEDED
# ------------------
# The original code calls  random.choice(active_only_numbers)  independently
# inside every typology loop.  The same account can therefore appear in
# Structuring AND Money Mule AND Hawala, giving it multiple aml_typology labels.
# When the ML pipeline aggregates to account level, the "primary" typology
# is ambiguous and the transaction counts per typology don't add up.
#
# FIX:
#   1. Calculate how many accounts each typology needs (accounts_needed).
#   2. Shuffle active_only_numbers once and slice non-overlapping groups.
#   3. Store {account_number: typology_name} in TYPOLOGY_ACCOUNT_MAP.
#   4. Every pick_n() call uses only that typology's account pool.
#   5. Base (clean) transactions use only accounts with no typology assigned.

print("=" * 75)
print("PRE-ASSIGNMENT: One typology per account")
print("=" * 75)

# Accounts needed per typology:
#   Each scenario touches  accounts_per_scenario  accounts on average.
#   accounts_needed = SCENARIO_COUNTS[typ] * accounts_per_scenario_estimate
ACCOUNTS_PER_SCENARIO_ESTIMATE = {
    "Structuring (Smurfing)":      7,   # 1 target + 3-6 sources
    "Circular Transaction Loop":   4,   # 3-5 ring members
    "Funnel Account Network":      20,  # 1 funnel + 15-50 feeders + 1 dest (use median)
    "Pass-Through Transit Hub":    3,   # 1 transit + 1 src + 1 dst
    "Rapid Multi-Hop Layering":    6,   # num_hops+1 chain (median ~5-7)
    "Third-Party Payment Web":     8,   # 1 biz + 3-12 payers
    "Money Mule Network":          8,   # 1 controller + 3-6 mules + 1 collector
    "High-Risk Corridor Transfer": 2,   # 1 sender + 1 receiver
    "Underground Banking (Hawala)":4,   # 3-6 parties (median 4)
    "Charity Abuse":               10,  # 1 npo + 5-15 donors + 2-4 diversion targets
}

accounts_needed = {}
for typ, n_sc in SCENARIO_COUNTS.items():
    est = ACCOUNTS_PER_SCENARIO_ESTIMATE.get(typ, 5)
    accounts_needed[typ] = n_sc * est

total_aml_accounts_needed = sum(accounts_needed.values())
print(f"  Total active accounts available : {len(active_only_numbers):,}")
print(f"  Total accounts needed for AML   : {total_aml_accounts_needed:,}")

# Safety check — if demand exceeds supply, scale down proportionally
if total_aml_accounts_needed > len(active_only_numbers):
    scale = len(active_only_numbers) / total_aml_accounts_needed * 0.95
    accounts_needed = {k: max(1, int(v * scale)) for k, v in accounts_needed.items()}
    total_aml_accounts_needed = sum(accounts_needed.values())
    print(f"  WARNING: Scaled down to {total_aml_accounts_needed:,} (pool too small)")

print()
header = f"  {'Typology':<40s} {'Count':>8s}"
print(header)

print(f"  {'-'*51}")
for typ, n in accounts_needed.items():
    print(f"  {typ:<40s} {n:>9,}")
print(f"  {'-'*51}")
print(f"  {'TOTAL':<40s} {total_aml_accounts_needed:>9,}")
print()

# Shuffle once and slice non-overlapping pools
shuffled = active_only_numbers[:]
random.shuffle(shuffled)

TYPOLOGY_POOLS   = {}   # typology_name -> list of account_numbers (exclusive pool)
TYPOLOGY_ACCT_MAP = {}  # account_number -> typology_name (for labelling clean txns)

cursor = 0
for typ in SCENARIO_COUNTS:
    n = accounts_needed[typ]
    pool = shuffled[cursor: cursor + n]
    TYPOLOGY_POOLS[typ] = pool
    for acc in pool:
        TYPOLOGY_ACCT_MAP[acc] = typ
    cursor += n

# Remaining accounts are CLEAN (no AML typology)
clean_accounts = set(shuffled[cursor:])
print(f"  Clean accounts (no typology): {len(clean_accounts):,}")
print(f"  AML-assigned accounts       : {cursor:,}")
print()

# Validation: zero overlap between pools
all_aml_accounts = [a for pool in TYPOLOGY_POOLS.values() for a in pool]
assert len(all_aml_accounts) == len(set(all_aml_accounts)), "OVERLAP DETECTED in typology pools!"
print("✅ Pre-assignment complete — zero overlap between typology account pools")
print("=" * 75)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: pick from a SPECIFIC POOL (not the global active_only_numbers)
# ─────────────────────────────────────────────────────────────────────────────
def pick_from(pool, n, exclude=None):
    """Sample n accounts from pool, excluding any in exclude set."""
    exclude = set(exclude or [])
    candidates = [a for a in pool if a not in exclude]
    return random.sample(candidates, min(n, len(candidates)))

# ====================================================================
# PHASE 1: Generate base (clean) transactions
# ====================================================================
print("PHASE 1: Generating base transactions (clean accounts only)...")
for idx, acct in enumerate(active_accounts_list):
    if idx % 2000 == 0:
        print(f"  Account {idx+1:,}/{len(active_accounts_list):,}...")

    acct_num = acct["account_number"]
    cif = acct["customer_cif"]
    cust = cust_lookup.get(cif)
    if not cust:
        continue

    # FIX: AML-assigned accounts still get their base (normal) transactions,
    # but we also generate the AML-typology transactions in Phase 2.
    # This mirrors the original design — an AML account has normal txns AND fraud txns.
    n_txns = random.randint(*CONFIG["num_transactions_per_account_range"])
    if acct["account_status"] == "Dormant":
        n_txns = random.randint(0, 5)

    income = cust["annual_income"] if isinstance(cust["annual_income"], int) else 500000
    monthly_budget = income / 12
    wallet_id = cif_to_wallet.get(cif, "")

    for _ in range(n_txns):
        txn_date_obj = random_date(CONFIG["date_range_start"], CONFIG["date_range_end"])
        txn_date = txn_date_obj.strftime("%d-%m-%Y")
        txn_time = unique_timestamp(acct_num, txn_date)

        direction = random.choices(["Dr","Cr"], weights=[0.55,0.45])[0]

        r = random.random()
        if r < 0.50:
            amount = round(random.uniform(50, 5000), 2)
        elif r < 0.80:
            amount = round(random.uniform(5000, 50000), 2)
        elif r < 0.95:
            amount = round(random.uniform(50000, 200000), 2)
        else:
            amount = round(random.uniform(200000, min(monthly_budget * 2, 5000000)), 2)

        channel = random.choice(CONFIG["bank_channels"])
        is_ppi  = random.random() < 0.15 and wallet_id != ""
        ppi_channel  = random.choice(CONFIG["ppi_channels"]) if is_ppi else ""
        ppi_txn_type = random.choice(["P2P","P2M","Bill Pay","Recharge","Cash Out"]) if is_ppi else ""

        cp_acct = random.choice(all_acct_numbers)
        while cp_acct == acct_num:
            cp_acct = random.choice(all_acct_numbers)

        # is_aml=0, aml_typology="" — this is a clean transaction
        txn = build_full_txn(acct_num, cp_acct, amount, txn_date, txn_time,
                             direction, channel, is_aml=0, aml_typology="",
                             is_ppi=is_ppi, ppi_txn_type=ppi_txn_type, ppi_channel=ppi_channel)
        transactions.append(txn)

        if direction == "Cr":
            acct_credit_sums[acct_num] += amount
        else:
            acct_debit_sums[acct_num] += amount

n_base = len(transactions)
print(f"  Base transactions generated: {n_base:,}")

# ====================================================================
# PHASE 2: Generate AML typology transactions (ONE typology per account)
# ====================================================================
print("\nPHASE 2: Injecting AML typologies (pre-assigned account pools)...")
aml_txns_generated = 0
TG = CONFIG["typology_generation"]


# ─────────────────────────────────────────────────────────────────────
# T1: Structuring (Smurfing)
# Pool: TYPOLOGY_POOLS["Structuring (Smurfing)"]
# ─────────────────────────────────────────────────────────────────────
cfg_s = TG["structuring"]
POOL_S = TYPOLOGY_POOLS.get("Structuring (Smurfing)", [])
#print(f"  T1: Structuring ({SCENARIO_COUNTS['Structuring (Smurfing)']} scenarios, pool={len(POOL_S)} accounts)...")
for si in range(SCENARIO_COUNTS["Structuring (Smurfing)"]):
    gid = f"STRUCT_{si+1:05d}"
    if len(POOL_S) < 2:
        break
    target = random.choice(POOL_S)
    n_src  = random.randint(*cfg_s["num_sources_range"])
    sources = pick_from(POOL_S, n_src, exclude=[target])
    if len(sources) < cfg_s["num_sources_range"][0]:
        continue
    ds = random_date_str()
    bh = random.randint(*cfg_s["deposit_hour_range"])
    for i, src in enumerate(sources):
        dep = random.uniform(*cfg_s["deposit_amount_range"])
        ts  = unique_timestamp(src, ds, bh + (i % 6))
        txn = build_full_txn(src, src, dep, ds, ts, "Cr", cfg_s["deposit_channel"],
                             is_aml=1, aml_typology="Structuring (Smurfing)", group_id=gid)
        txn["cash_flag"] = "Y"
        transactions.append(txn)
        aml_txns_generated += 1
    nd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=random.randint(*cfg_s["transfer_delay_days_range"]))
    nds = nd.strftime("%d-%m-%Y")
    for src in sources:
        amt = random.uniform(*cfg_s["transfer_amount_range"])
        ts  = unique_timestamp(src, nds, random.randint(*cfg_s["transfer_hour_range"]))
        txn = build_full_txn(src, target, amt, nds, ts, "Dr",
                             random.choice(cfg_s["transfer_channels"]),
                             is_aml=1, aml_typology="Structuring (Smurfing)", group_id=gid)
        transactions.append(txn)
        aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T2: Circular Transaction Loops
# ─────────────────────────────────────────────────────────────────────
cfg_c  = TG["circular"]
POOL_C = TYPOLOGY_POOLS.get("Circular Transaction Loop", [])
print(f"  T2: Circular loops ({SCENARIO_COUNTS['Circular Transaction Loop']} scenarios, pool={len(POOL_C)} accounts)...")
for si in range(SCENARIO_COUNTS["Circular Transaction Loop"]):
    gid     = f"CIRC_{si+1:05d}"
    ring_sz = random.randint(*cfg_c["ring_size_range"])
    ring    = pick_from(POOL_C, ring_sz)
    if len(ring) < cfg_c["ring_size_range"][0]:
        continue
    base_amt = random.uniform(*cfg_c["base_amount_range"])
    ds = random_date_str()
    for hop in range(len(ring)):
        sender   = ring[hop]
        receiver = ring[(hop + 1) % len(ring)]
        amt = base_amt * random.uniform(*cfg_c["hop_amount_decay"])
        hd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=hop * cfg_c["hop_interval_days"])
        hds = hd.strftime("%d-%m-%Y")
        ch  = random.choice(cfg_c["channels"])
        ts  = unique_timestamp(sender, hds, random.randint(*cfg_c["hop_hour_range"]))
        txn = build_full_txn(sender, receiver, amt, hds, ts, "Dr", ch,
                             is_aml=1, aml_typology="Circular Transaction Loop", group_id=gid)
        transactions.append(txn); aml_txns_generated += 1
        ts2  = unique_timestamp(receiver, hds, random.randint(*cfg_c["hop_hour_range"]))
        txn2 = build_full_txn(receiver, sender, amt, hds, ts2, "Cr", ch,
                              is_aml=1, aml_typology="Circular Transaction Loop", group_id=gid)
        transactions.append(txn2); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T3: Funnel Account Networks
# ─────────────────────────────────────────────────────────────────────
cfg_f  = TG["funnel"]
POOL_F = TYPOLOGY_POOLS.get("Funnel Account Network", [])
print(f"  T3: Funnel networks ({SCENARIO_COUNTS['Funnel Account Network']} scenarios, pool={len(POOL_F)} accounts)...")
for si in range(SCENARIO_COUNTS["Funnel Account Network"]):
    gid    = f"FUNNEL_{si+1:05d}"
    if len(POOL_F) < 3:
        break
    funnel = random.choice(POOL_F)
    n_feed = random.randint(*cfg_f["num_feeders_range"])
    feeders = pick_from(POOL_F, n_feed, exclude=[funnel])
    dest_list = pick_from(POOL_F, 1, exclude=[funnel] + feeders)
    if not dest_list or len(feeders) < cfg_f["num_feeders_range"][0]:
        continue
    dest = dest_list[0]
    ds = random_date_str()
    total_in = 0
    for feeder in feeders:
        amt = random.uniform(*cfg_f["per_feeder_amount_range"])
        total_in += amt
        day_off = random.randint(*cfg_f["feeder_spread_days_range"])
        fd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=day_off)
        fds = fd.strftime("%d-%m-%Y")
        ts  = unique_timestamp(feeder, fds, random.randint(*cfg_f["feeder_hour_range"]))
        txn = build_full_txn(feeder, funnel, amt, fds, ts, "Dr",
                             random.choice(cfg_f["feeder_channels"]),
                             is_aml=1, aml_typology="Funnel Account Network", group_id=gid)
        transactions.append(txn); aml_txns_generated += 1
    out_d = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=random.randint(*cfg_f["outflow_delay_days_range"]))
    n_sp  = random.randint(*cfg_f["outflow_splits_range"])
    rem   = total_in * (1.0 - cfg_f["retention_pct"])
    for s in range(n_sp):
        amt = rem * random.uniform(*cfg_f["outflow_split_pct_range"]) if s < n_sp-1 else rem
        rem -= amt
        od  = out_d + timedelta(days=s)
        ods = od.strftime("%d-%m-%Y")
        ts  = unique_timestamp(funnel, ods, random.randint(*cfg_f["outflow_hour_range"]))
        txn = build_full_txn(funnel, dest, amt, ods, ts, "Dr",
                             "RTGS" if amt > 200000 else "NEFT",
                             is_aml=1, aml_typology="Funnel Account Network", group_id=gid)
        transactions.append(txn); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T4: Pass-Through Transit Hubs
# ─────────────────────────────────────────────────────────────────────
cfg_p  = TG["passthrough"]
POOL_P = TYPOLOGY_POOLS.get("Pass-Through Transit Hub", [])
print(f"  T4: Pass-through ({SCENARIO_COUNTS['Pass-Through Transit Hub']} scenarios, pool={len(POOL_P)} accounts)...")
for si in range(SCENARIO_COUNTS["Pass-Through Transit Hub"]):
    gid     = f"PASS_{si+1:05d}"
    if len(POOL_P) < 3:
        break
    transit  = random.choice(POOL_P)
    src_list = pick_from(POOL_P, 1, exclude=[transit])
    dst_list = pick_from(POOL_P, 1, exclude=[transit] + src_list)
    if not src_list or not dst_list:
        continue
    src, dst = src_list[0], dst_list[0]
    amt_in  = random.uniform(*cfg_p["inflow_amount_range"])
    amt_out = amt_in * random.uniform(*cfg_p["forward_pct_range"])
    ds   = random_date_str()
    hour = random.randint(*cfg_p["hour_range"])
    ts1  = unique_timestamp(transit, ds, hour)
    txn1 = build_full_txn(transit, src, amt_in, ds, ts1, "Cr",
                          random.choice(cfg_p["inflow_channels"]),
                          is_aml=1, aml_typology="Pass-Through Transit Hub", group_id=gid)
    txn1["counterparty_account_number"] = src
    transactions.append(txn1); aml_txns_generated += 1
    ts2  = unique_timestamp(transit, ds, min(hour + random.choice(cfg_p["time_gap_hours"]), 23))
    txn2 = build_full_txn(transit, dst, amt_out, ds, ts2, "Dr",
                          random.choice(cfg_p["outflow_channels"]),
                          is_aml=1, aml_typology="Pass-Through Transit Hub", group_id=gid)
    transactions.append(txn2); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T5: Rapid Multi-Hop Layering
# ─────────────────────────────────────────────────────────────────────
cfg_l  = TG["layering"]
POOL_L = TYPOLOGY_POOLS.get("Rapid Multi-Hop Layering", [])
print(f"  T5: Multi-hop layering ({SCENARIO_COUNTS['Rapid Multi-Hop Layering']} scenarios, pool={len(POOL_L)} accounts)...")
for si in range(SCENARIO_COUNTS["Rapid Multi-Hop Layering"]):
    gid    = f"LAYER_{si+1:05d}"
    n_hops = random.randint(*cfg_l["num_hops_range"])
    chain  = pick_from(POOL_L, n_hops + 1)
    if len(chain) < cfg_l["num_hops_range"][0] + 1:
        continue
    base_amt = random.uniform(*cfg_l["base_amount_range"])
    ds = random_date_str()
    sh = random.randint(*cfg_l["start_hour_range"])
    for hop in range(len(chain) - 1):
        amt     = base_amt * (cfg_l["per_hop_decay"] ** hop) * random.uniform(*cfg_l["per_hop_noise_range"])
        min_off = hop * random.randint(*cfg_l["hop_interval_minutes_range"])
        hh      = min(sh + min_off // 60, 23)
        day_off = min_off // (24 * 60)
        hd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=day_off)
        hds = hd.strftime("%d-%m-%Y")
        ts  = unique_timestamp(chain[hop], hds, hh)
        txn = build_full_txn(chain[hop], chain[hop+1], amt, hds, ts, "Dr",
                             random.choice(cfg_l["channels"]),
                             is_aml=1, aml_typology="Rapid Multi-Hop Layering", group_id=gid)
        transactions.append(txn); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T6: Third-Party Payment Webs
# ─────────────────────────────────────────────────────────────────────
if "Third-Party Payment Web" in SCENARIO_COUNTS:
    cfg_tp  = TG["third_party_web"]
    POOL_TP = TYPOLOGY_POOLS.get("Third-Party Payment Web", [])
    print(f"  T6: Third-party payment webs ({SCENARIO_COUNTS['Third-Party Payment Web']} scenarios, pool={len(POOL_TP)} accounts)...")
    for si in range(SCENARIO_COUNTS["Third-Party Payment Web"]):
        gid      = f"TPWEB_{si+1:05d}"
        if len(POOL_TP) < 2:
            break
        biz_acct = random.choice(POOL_TP)
        n_payers = random.randint(*cfg_tp["num_unrelated_payers_range"])
        payers   = pick_from(POOL_TP, n_payers, exclude=[biz_acct])
        if len(payers) < cfg_tp["num_unrelated_payers_range"][0]:
            continue
        ds = random_date_str()
        for payer in payers:
            amt    = random.uniform(*cfg_tp["per_payment_amount_range"])
            day_off = random.randint(*cfg_tp["payment_spread_days_range"])
            pd_dt  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=day_off)
            pds    = pd_dt.strftime("%d-%m-%Y")
            ts     = unique_timestamp(payer, pds, random.randint(*cfg_tp["payment_hour_range"]))
            txn    = build_full_txn(payer, biz_acct, amt, pds, ts, "Dr",
                                    random.choice(cfg_tp["payment_channels"]),
                                    is_aml=1, aml_typology="Third-Party Payment Web", group_id=gid)
            transactions.append(txn); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T7: Money Mule Networks
# ─────────────────────────────────────────────────────────────────────
if "Money Mule Network" in SCENARIO_COUNTS:
    cfg_mm  = TG["money_mule"]
    POOL_MM = TYPOLOGY_POOLS.get("Money Mule Network", [])
    print(f"  T7: Money mule networks ({SCENARIO_COUNTS['Money Mule Network']} scenarios, pool={len(POOL_MM)} accounts)...")
    for si in range(SCENARIO_COUNTS["Money Mule Network"]):
        gid        = f"MULE_{si+1:05d}"
        if len(POOL_MM) < 3:
            break
        controller = random.choice(POOL_MM)
        n_mules    = random.randint(*cfg_mm["num_mules_range"])
        mules      = pick_from(POOL_MM, n_mules, exclude=[controller])
        collector  = pick_from(POOL_MM, 1, exclude=[controller] + mules)
        if len(mules) < cfg_mm["num_mules_range"][0] or not collector:
            continue
        collector = collector[0]
        ds = random_date_str()
        for mule in mules:
            amt = random.uniform(*cfg_mm["controller_to_mule_amount_range"])
            ts  = unique_timestamp(controller, ds, random.randint(*cfg_mm["hour_range"]))
            txn = build_full_txn(controller, mule, amt, ds, ts, "Dr",
                                 random.choice(cfg_mm["channels"]),
                                 is_aml=1, aml_typology="Money Mule Network", group_id=gid)
            transactions.append(txn); aml_txns_generated += 1
            fwd_amt  = amt * random.uniform(*cfg_mm["mule_forward_pct_range"])
            delay_h  = random.randint(*cfg_mm["mule_forward_delay_hours_range"])
            fwd_date = datetime.strptime(ds, "%d-%m-%Y") + timedelta(hours=delay_h)
            fds      = fwd_date.strftime("%d-%m-%Y")
            ts2      = unique_timestamp(mule, fds, min(fwd_date.hour + random.randint(0,2), 23))
            txn2     = build_full_txn(mule, collector, fwd_amt, fds, ts2, "Dr",
                                      random.choice(cfg_mm["channels"]),
                                      is_aml=1, aml_typology="Money Mule Network", group_id=gid)
            transactions.append(txn2); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T8: High-Risk Corridor Transfers
# ─────────────────────────────────────────────────────────────────────
if "High-Risk Corridor Transfer" in SCENARIO_COUNTS:
    cfg_hrc  = TG["high_risk_corridor"]
    POOL_HRC = TYPOLOGY_POOLS.get("High-Risk Corridor Transfer", [])
    print(f"  T8: High-risk corridor transfers ({SCENARIO_COUNTS['High-Risk Corridor Transfer']} scenarios, pool={len(POOL_HRC)} accounts)...")
    for si in range(SCENARIO_COUNTS["High-Risk Corridor Transfer"]):
        gid          = f"HRCORR_{si+1:05d}"
        if len(POOL_HRC) < 2:
            break
        sender_acct  = random.choice(POOL_HRC)
        recv_list    = pick_from(POOL_HRC, 1, exclude=[sender_acct])
        if not recv_list:
            continue
        receiver_acct  = recv_list[0]
        target_country = random.choice(cfg_hrc["target_countries"])
        n_transfers    = random.randint(*cfg_hrc["frequency_per_account_range"])
        ds = random_date_str()
        for ti in range(n_transfers):
            amt     = random.uniform(*cfg_hrc["amount_range"])
            day_off = random.randint(*cfg_hrc["spread_days_range"])
            td  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=day_off * ti)
            tds = td.strftime("%d-%m-%Y")
            ts  = unique_timestamp(sender_acct, tds, random.randint(*cfg_hrc["hour_range"]))
            txn = build_full_txn(sender_acct, receiver_acct, amt, tds, ts, "Dr",
                                 random.choice(cfg_hrc["channels"]),
                                 is_aml=1, aml_typology="High-Risk Corridor Transfer", group_id=gid)
            txn["receiver_country_code"] = target_country
            transactions.append(txn); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T9: Underground Banking (Hawala)
# ─────────────────────────────────────────────────────────────────────
if "Underground Banking (Hawala)" in SCENARIO_COUNTS:
    cfg_hw  = TG["hawala"]
    POOL_HW = TYPOLOGY_POOLS.get("Underground Banking (Hawala)", [])
    print(f"  T9: Underground banking / hawala ({SCENARIO_COUNTS['Underground Banking (Hawala)']} scenarios, pool={len(POOL_HW)} accounts)...")
    for si in range(SCENARIO_COUNTS["Underground Banking (Hawala)"]):
        gid      = f"HAWALA_{si+1:05d}"
        n_parties = random.randint(*cfg_hw["num_parties_range"])
        parties  = pick_from(POOL_HW, n_parties)
        if len(parties) < cfg_hw["num_parties_range"][0]:
            continue
        base_amt = random.uniform(*cfg_hw["settlement_amount_range"])
        ds = random_date_str()
        for hop in range(len(parties)):
            sender   = parties[hop]
            receiver = parties[(hop + 1) % len(parties)]
            amt      = base_amt * random.uniform(*cfg_hw["leg_amount_variation_pct"])
            day_off  = random.randint(*cfg_hw["settlement_spread_days_range"])
            hd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=day_off + hop)
            hds = hd.strftime("%d-%m-%Y")
            ts  = unique_timestamp(sender, hds, random.randint(*cfg_hw["hour_range"]))
            txn = build_full_txn(sender, receiver, amt, hds, ts, "Dr",
                                 random.choice(cfg_hw["channels"]),
                                 is_aml=1, aml_typology="Underground Banking (Hawala)", group_id=gid)
            transactions.append(txn); aml_txns_generated += 1

# ─────────────────────────────────────────────────────────────────────
# T10: Charity Abuse
# ─────────────────────────────────────────────────────────────────────
if "Charity Abuse" in SCENARIO_COUNTS:
    cfg_ca  = TG["charity_abuse"]
    POOL_CA = TYPOLOGY_POOLS.get("Charity Abuse", [])
    print(f"  T10: Charity abuse ({SCENARIO_COUNTS['Charity Abuse']} scenarios, pool={len(POOL_CA)} accounts)...")
    for si in range(SCENARIO_COUNTS["Charity Abuse"]):
        gid      = f"CHARITY_{si+1:05d}"
        if len(POOL_CA) < 3:
            break
        npo_acct = random.choice(POOL_CA)
        n_donors = random.randint(*cfg_ca["num_donors_range"])
        donors   = pick_from(POOL_CA, n_donors, exclude=[npo_acct])
        diversion_targets = pick_from(POOL_CA, random.randint(*cfg_ca["diversion_splits_range"]),
                                      exclude=[npo_acct] + donors)
        if len(donors) < cfg_ca["num_donors_range"][0] or not diversion_targets:
            continue
        ds = random_date_str()
        total_donated = 0
        for donor in donors:
            amt = random.uniform(*cfg_ca["donation_amount_range"])
            total_donated += amt
            day_off = random.randint(*cfg_ca["donation_spread_days_range"])
            dd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=day_off)
            dds = dd.strftime("%d-%m-%Y")
            ts  = unique_timestamp(donor, dds, random.randint(8, 20))
            txn = build_full_txn(donor, npo_acct, amt, dds, ts, "Dr",
                                 random.choice(cfg_ca["donation_channels"]),
                                 is_aml=1, aml_typology="Charity Abuse", group_id=gid)
            transactions.append(txn); aml_txns_generated += 1
        divert_total = total_donated * cfg_ca["diversion_pct"]
        divert_delay = random.randint(*cfg_ca["diversion_delay_days_range"])
        rem = divert_total
        for di, tgt in enumerate(diversion_targets):
            amt = rem * random.uniform(0.2, 0.5) if di < len(diversion_targets)-1 else rem
            rem -= amt
            od  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=divert_delay + di)
            ods = od.strftime("%d-%m-%Y")
            ts  = unique_timestamp(npo_acct, ods, random.randint(*cfg_ca["diversion_hour_range"]))
            txn = build_full_txn(npo_acct, tgt, amt, ods, ts, "Dr",
                                 random.choice(["NEFT","IMPS"]),
                                 is_aml=1, aml_typology="Charity Abuse", group_id=gid)
            transactions.append(txn); aml_txns_generated += 1

print(f"\n  AML transactions injected: {aml_txns_generated:,}")

# ====================================================================
# PHASE 3: Top-up if needed to hit target % (also CONFIG-driven)
# ====================================================================
actual_total = len(transactions)
actual_aml   = sum(1 for t in transactions if t["is_aml"] == 1)
target_aml_exact = int(n_base * target_pct / (1.0 - target_pct))
deficit = target_aml_exact - actual_aml
cfg_tu = TG["topup"]

if deficit > 0:
    print(f"\nPHASE 3: Top-up needed ({deficit:,} more AML txns)...")
    topup_round = 0
    while deficit > 0:
        topup_round += 1
        gid    = f"TOPUP_{topup_round:05d}"
        choice = topup_round % 3
        if choice == 0:
            tr = random.choice(POOL_P) if POOL_P else random.choice(active_only_numbers)
            s  = pick_from(POOL_P or active_only_numbers, 1, exclude=[tr])
            d  = pick_from(POOL_P or active_only_numbers, 1, exclude=[tr] + s)
            if not s or not d:
                continue
            ds  = random_date_str(); h = random.randint(*cfg_p["hour_range"])
            ai  = random.uniform(*cfg_tu["passthrough_amount_range"])
            ao  = ai * cfg_tu["passthrough_forward_pct"]
            ts1 = unique_timestamp(tr, ds, h)
            t1  = build_full_txn(tr, s[0], ai, ds, ts1, "Cr", "RTGS",
                                 is_aml=1, aml_typology="Pass-Through Transit Hub", group_id=gid)
            transactions.append(t1); deficit -= 1
            ts2 = unique_timestamp(tr, ds, min(h+1, 23))
            t2  = build_full_txn(tr, d[0], ao, ds, ts2, "Dr", "NEFT",
                                 is_aml=1, aml_typology="Pass-Through Transit Hub", group_id=gid)
            transactions.append(t2); deficit -= 1
        elif choice == 1:
            tgt  = random.choice(POOL_S) if POOL_S else random.choice(active_only_numbers)
            srcs = pick_from(POOL_S or active_only_numbers,
                             random.randint(*cfg_tu["structuring_sources_range"]), exclude=[tgt])
            if len(srcs) < cfg_tu["structuring_sources_range"][0]:
                continue
            ds = random_date_str()
            for src in srcs:
                ts = unique_timestamp(src, ds, random.randint(*cfg_s["deposit_hour_range"]))
                t  = build_full_txn(src, src, random.uniform(*cfg_tu["structuring_deposit_range"]),
                                    ds, ts, "Cr", "Branch Cash",
                                    is_aml=1, aml_typology="Structuring (Smurfing)", group_id=gid)
                t["cash_flag"] = "Y"; transactions.append(t); deficit -= 1
            nd  = datetime.strptime(ds, "%d-%m-%Y") + timedelta(days=1)
            nds = nd.strftime("%d-%m-%Y")
            for src in srcs:
                ts = unique_timestamp(src, nds, random.randint(*cfg_s["transfer_hour_range"]))
                t  = build_full_txn(src, tgt, random.uniform(*cfg_tu["structuring_transfer_range"]),
                                    nds, ts, "Dr", "IMPS",
                                    is_aml=1, aml_typology="Structuring (Smurfing)", group_id=gid)
                transactions.append(t); deficit -= 1
        else:
            ch = pick_from(POOL_L or active_only_numbers, random.randint(*cfg_tu["layering_hops_range"]))
            if len(ch) < cfg_tu["layering_hops_range"][0]:
                continue
            ba = random.uniform(*cfg_tu["layering_amount_range"])
            ds = random_date_str(); sh = random.randint(*cfg_l["start_hour_range"])
            for hp in range(len(ch)-1):
                ts = unique_timestamp(ch[hp], ds, min(sh+hp, 23))
                t  = build_full_txn(ch[hp], ch[hp+1], ba*(cfg_tu["layering_decay"]**hp),
                                    ds, ts, "Dr", "IMPS",
                                    is_aml=1, aml_typology="Rapid Multi-Hop Layering", group_id=gid)
                transactions.append(t); deficit -= 1
    print("  Top-up complete")
else:
    print("\nPHASE 3: No top-up needed")

# ─────────────────────────────────────────────────────────────────────
# POST-GENERATION VALIDATION: confirm one typology per account
# ─────────────────────────────────────────────────────────────────────
print("\n── Post-generation typology uniqueness check ──")
from collections import defaultdict as _dd
acct_typs = _dd(set)
for t in transactions:
    if t["is_aml"] == 1 and t["aml_typology"]:
        acct_typs[t["customer_account_number"]].add(t["aml_typology"])

multi_typ = {a: typs for a, typs in acct_typs.items() if len(typs) > 1}
if multi_typ:
    print(f"  ⚠️  {len(multi_typ)} accounts have multiple typologies:")
    for a, typs in list(multi_typ.items())[:5]:
        print(f"    {a}: {typs}")
else:
    print(f"  ✅ All {len(acct_typs):,} AML accounts have exactly ONE typology assigned")

random.shuffle(transactions)
total       = len(transactions)
aml_count   = sum(1 for t in transactions if t["is_aml"] == 1)
clean_count = total - aml_count

#print(f"\n{\'=\' * 75}")
print(f"GENERATION COMPLETE")
#print(f"{\'=\' * 75}")
print(f"  Clean transactions:  {clean_count:>10,}  ({clean_count/total*100:.2f}%)")
print(f"  AML transactions:    {aml_count:>10,}  ({aml_count/total*100:.2f}%)")
print(f"  Total transactions:  {total:>10,}")
print(f"  Columns per row:     {len(TRANSACTION_COLUMNS):>10}")
#print(f"{\'=\' * 75}")


# ## 11b -- Inject PPI Rule Trigger Scenarios
# Creates specific transaction groups to ensure every PPI rule has at least one trigger.
# Covers: MCC concentration, refund abuse, multi-wallet clusters, promo cycling, negative list injection.
# 

# In[17]:


print("Injecting PPI-specific rule trigger scenarios...")
ppi_scenario_count = 0

ppi_cifs = [c["customer_cif"] for c in customers if c["customer_type"] == "Individual" and c["customer_cif"] in cif_to_wallet]
if not ppi_cifs:
    print("  No PPI wallets found, skipping PPI scenarios")
else:
    random.shuffle(ppi_cifs)
    scenario_cifs = ppi_cifs[:min(50, len(ppi_cifs))]

    for sc_idx, cif in enumerate(scenario_cifs):
        cust = cust_lookup[cif]
        wallet_id = cif_to_wallet.get(cif, "")
        accts = cif_to_accounts.get(cif, [])
        if not accts or not wallet_id: continue
        acct_num = accts[0]
        base_date = random_date(CONFIG["date_range_start"], CONFIG["date_range_end"])
        cp = random.choice(all_acct_numbers)
        while cp == acct_num:
            cp = random.choice(all_acct_numbers)

        scenario_type = sc_idx % 12

        if scenario_type == 0:
            for i in range(8):
                txn_date = base_date + timedelta(days=i)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(3000, 8000), 2),
                                     ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2M", ppi_channel="UPI")
                txn["merchant_category_code"] = random.choice(CONFIG.get("high_risk_mccs", ["7995"]))
                txn["merchant_name"] = random.choice(["BetWin Online","CryptoEx Exchange","GameZone Premium"])
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 1:
            merchant_id = generate_merchant_id()
            for i in range(6):
                txn_date = base_date + timedelta(days=i*3)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(3500, 8000), 2),
                                     ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2M", ppi_channel="UPI")
                txn["refund_chargeback_flag"] = "Y"
                txn["merchant_id"] = merchant_id
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 2:
            for i in range(5):
                txn_date = base_date + timedelta(days=i)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(40000, 80000), 2),
                                     ds, ts, "Dr", "In-App Transfer", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2P", ppi_channel="In-App Transfer")
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 3:
            shared_ip = f"103.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
            for i in range(5):
                txn_date = base_date + timedelta(hours=i)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(5000, 30000), 2),
                                     ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2P", ppi_channel="UPI")
                txn["ip_address"] = shared_ip
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 4:
            shared_vpa = f"carousel{random.randint(1000,9999)}@upi"
            for i in range(6):
                txn_date = base_date + timedelta(days=i)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(5000, 40000), 2),
                                     ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2P", ppi_channel="UPI")
                txn["beneficiary_wallet_id_vpa"] = shared_vpa
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 5:
            for i in range(8):
                txn_date = base_date + timedelta(hours=i*3)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(100, 500), 2),
                                     ds, ts, "Dr", "In-App Transfer", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2P", ppi_channel="In-App Transfer")
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 6:
            for i in range(22):
                txn_date = base_date + timedelta(minutes=i*30)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(50, 99), 2),
                                     ds, ts, "Dr", "QR Scan", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2M", ppi_channel="QR Scan")
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 7:
            ds = base_date.strftime("%d-%m-%Y")
            ts = unique_timestamp(acct_num, ds)
            txn = build_full_txn(acct_num, cp, round(random.uniform(10000, 50000), 2),
                                 ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                 ppi_txn_type="P2P", ppi_channel="UPI")
            txn["device_id_fingerprint"] = random.choice(CONFIG.get("negative_list_devices", ["DEV_BL"]))
            txn["beneficiary_wallet_id_vpa"] = random.choice(CONFIG.get("negative_list_vpas", ["scam@upi"]))
            transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 8:
            m_id = generate_merchant_id()
            mccs_used = random.sample(list(CONFIG["mcc_codes"].keys()), min(3, len(CONFIG["mcc_codes"])))
            for i, mcc in enumerate(mccs_used):
                txn_date = base_date + timedelta(days=i*30)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(5000, 20000), 2),
                                     ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2M", ppi_channel="UPI")
                txn["merchant_id"] = m_id
                txn["merchant_category_code"] = mcc
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 9:
            ds = base_date.strftime("%d-%m-%Y")
            ts1 = unique_timestamp(acct_num, ds)
            amt_val = round(random.uniform(10000, 25000), 2)
            txn_fail = build_full_txn(acct_num, cp, amt_val,
                                      ds, ts1, "Cr", "UPI", is_aml=0, is_ppi=True,
                                      ppi_txn_type="Load", ppi_channel="UPI")
            txn_fail["transaction_status"] = "Failed"
            txn_fail["load_instrument_type"] = "Debit Card"
            transactions.append(txn_fail)
            txn_date2 = base_date + timedelta(minutes=3)
            ds2 = txn_date2.strftime("%d-%m-%Y")
            ts2 = unique_timestamp(acct_num, ds2)
            txn_success = build_full_txn(acct_num, cp, amt_val,
                                          ds2, ts2, "Cr", "UPI", is_aml=0, is_ppi=True,
                                          ppi_txn_type="Load", ppi_channel="UPI")
            txn_success["transaction_status"] = "Success"
            txn_success["load_instrument_type"] = "Credit Card"
            transactions.append(txn_success)
            ppi_scenario_count += 2

        elif scenario_type == 10:
            foreign_names = ["Zhang Wei","Kim Soo-jin","Ahmed Al-Rashid","Ivan Petrov","Maria Garcia"]
            for i, name in enumerate(foreign_names):
                txn_date = base_date + timedelta(days=i)
                ds = txn_date.strftime("%d-%m-%Y")
                ts = unique_timestamp(acct_num, ds)
                txn = build_full_txn(acct_num, cp, round(random.uniform(5000, 15000), 2),
                                     ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                     ppi_txn_type="P2P", ppi_channel="UPI")
                txn["counterparty_name"] = name
                transactions.append(txn); ppi_scenario_count += 1

        elif scenario_type == 11:
            ds = base_date.strftime("%d-%m-%Y")
            ts = unique_timestamp(acct_num, ds)
            txn = build_full_txn(acct_num, cp, round(random.uniform(10000, 50000), 2),
                                 ds, ts, "Dr", "UPI", is_aml=0, is_ppi=True,
                                 ppi_txn_type="P2M", ppi_channel="UPI")
            txn["merchant_category_code"] = random.choice(CONFIG.get("high_risk_mccs", ["7995"]))
            transactions.append(txn); ppi_scenario_count += 1

    print(f"  PPI scenarios injected: {ppi_scenario_count:,} transactions across {len(scenario_cifs)} wallets")
    print(f"  Total transactions now: {len(transactions):,}")


# ## 12 -- Timestamp Collision Check
# 

# In[18]:


ts_check = defaultdict(list)
for t in transactions:
    ts_check[t["customer_account_number"]].append((t["datestamp"], t["timestamp"]))

collision_count = 0
for acct, ts_list in ts_check.items():
    c = Counter(ts_list)
    for k, v in c.items():
        if v > 1:
            collision_count += 1

if collision_count == 0:
    print("OK: No timestamp collisions detected")
else:
    print(f"Resolving {collision_count} timestamp collisions...")
    seen = defaultdict(set)
    for t in transactions:
        acct = t["customer_account_number"]
        key = (t["datestamp"], t["timestamp"])
        while key in seen[acct]:
            parts = t["timestamp"].split(":")
            s = int(parts[2]) + 1
            if s >= 60:
                s = 0; m = int(parts[1]) + 1
                if m >= 60:
                    m = 0; h = int(parts[0]) + 1
                    if h >= 24: h = 0
                    parts[0] = f"{h:02d}"
                parts[1] = f"{m:02d}"
            parts[2] = f"{s:02d}"
            t["timestamp"] = ":".join(parts)
            key = (t["datestamp"], t["timestamp"])
        seen[acct].add(key)
    print("  OK: All collisions resolved")



# ## 13 -- Update Account Credit/Debit Summations
# 

# In[19]:


# Recompute from final transactions
final_credit = defaultdict(float)
final_debit = defaultdict(float)
for t in transactions:
    acct = t["customer_account_number"]
    amt = float(t["transaction_amount"])
    if t["transaction_type_dr_cr"] == "Cr":
        final_credit[acct] += amt
    else:
        final_debit[acct] += amt

# Update in transactions and accounts
for t in transactions:
    acct = t["customer_account_number"]
    t["credit_summation_period"] = round(final_credit[acct], 2)
    t["debit_summation_period"] = round(final_debit[acct], 2)

for a in accounts:
    a["credit_summation_period"] = round(final_credit[a["account_number"]], 2)
    a["debit_summation_period"] = round(final_debit[a["account_number"]], 2)

print("Account summations updated")



# In[20]:


# TYPOLOGY SUMMARY


# In[21]:


# Column count check
sample = transactions[0]
ordered_keys = TRANSACTION_COLUMNS
missing = [k for k in ordered_keys if k not in sample]
extra = [k for k in sample if k not in ordered_keys]
print(f"Column validation:")
print(f"  Expected:  {len(ordered_keys)}")
print(f"  In data:   {len(sample)}")
print(f"  Missing:   {missing if missing else 'None'}")
print(f"  Extra:     {extra if extra else 'None'}")

# Referential integrity
cif_cust = {c["customer_cif"] for c in customers}
cif_acct = {a["customer_cif"] for a in accounts}
cif_wall = {w["customer_cif"] for w in wallets}
print(f"\nReferential integrity:")
print(f"  Account CIFs in customers?  {cif_acct.issubset(cif_cust)}")
print(f"  Wallet CIFs in customers?   {cif_wall.issubset(cif_cust)}")

# Uniqueness
print(f"  Unique account numbers?     {len(set(a['account_number'] for a in accounts)) == len(accounts)}")
print(f"  Unique wallet IDs?          {len(set(w['wallet_id'] for w in wallets)) == len(wallets)}")

# Timestamp uniqueness
ts_per = defaultdict(list)
for t in transactions:
    ts_per[t["customer_account_number"]].append((t["datestamp"], t["timestamp"]))
dups = sum(sum(v-1 for v in Counter(tsl).values() if v > 1) for tsl in ts_per.values())
print(f"  Timestamp duplicates:       {dups} (should be 0)")

# Typology distribution
print(f"\n-- Typology Distribution --")
typ_counts = defaultdict(int)
typ_groups = defaultdict(set)
for t in transactions:
    typ = t.get("aml_typology", "")
    if typ:
        typ_counts[typ] += 1
        typ_groups[typ].add(t.get("typology_group_id", ""))

total_aml = sum(typ_counts.values())
for typ in sorted(typ_counts):
    cnt = typ_counts[typ]
    grps = len(typ_groups[typ])
    print(f"  {typ:<35s} {cnt:>7,} txns ({cnt/total_aml*100:>5.1f}% of AML) [{grps:>5} scenarios]")

print(f"\n  {'TOTAL AML':<35s} {total_aml:>7,} ({total_aml/len(transactions)*100:.2f}% of total)")
print(f"  {'TOTAL CLEAN':<35s} {len(transactions)-total_aml:>7,} ({(len(transactions)-total_aml)/len(transactions)*100:.2f}% of total)")
print(f"  {'GRAND TOTAL':<35s} {len(transactions):>7,}")

print(f"\n{'=' * 60}")
print(f"PIPELINE COMPLETE - All data ready for model training!")
print(f"{'=' * 60}")



# In[22]:


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



# In[23]:


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


# In[24]:


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


# In[25]:


import db_config

print("HOST =", db_config.DB_HOST)
print("PORT =", db_config.DB_PORT)
print("DB   =", db_config.DB_NAME)
print("USER =", db_config.DB_USER)
print("PASS =", db_config.DB_PASSWORD)


# In[26]:


# ── Database connection (PostgreSQL) ──
from db_utils import write_table, test_connection
test_connection()


# In[27]:


import db_utils, inspect
print("_coerce_dates present:", hasattr(db_utils, "_coerce_dates"))
print()
#print(inspect.getsource(db_utils.write_table))


# In[28]:


import pandas as pd
from db_utils import _coerce_dates

# the actual customers data
test = pd.DataFrame(customers)
print("Before:", test['date_of_birth'].head(3).tolist())
print("dtype before:", test['date_of_birth'].dtype)

fixed = _coerce_dates(test)
print("After: ", fixed['date_of_birth'].head(3).tolist())
print("dtype after:", fixed['date_of_birth'].dtype)


# In[29]:


# import pandas as pd
# from db_utils import get_engine

# db_types = pd.read_sql("""
#     SELECT column_name, data_type
#     FROM information_schema.columns
#     WHERE table_schema='public' AND table_name='wallets'
# """, get_engine()).set_index('column_name')['data_type'].to_dict()

# cust_df = pd.DataFrame(wallets)
# print("Potential type mismatches in 'wallets':\n")
# for col in cust_df.columns:
#     if col not in db_types:
#         continue
#     dbt = db_types[col]
#     vals = cust_df[col].dropna()
#     if dbt in ('numeric','integer','bigint','smallint') and len(vals):
#         # is the actual data actually numeric?
#         non_numeric = vals[pd.to_numeric(vals, errors='coerce').isna()]
#         if len(non_numeric):
#             print(f"  {col:<32s} DB={dbt:<10s} but data has text e.g. {non_numeric.head(3).tolist()}")


# In[30]:


customers = pd.DataFrame(customers)
customers["professional_experience_years"] = (
    pd.to_numeric(
        customers["professional_experience_years"],
        errors="coerce"
    )
)


# In[31]:


wallets = pd.DataFrame(wallets)
print(wallets["wallet_opening_date"].dtype)
print(wallets["wallet_opening_date"].head())
wallets['wallet_opening_date'] = pd.to_datetime(wallets['wallet_opening_date'])


# In[32]:


import pandas as pd

print("Saving all datasets to PostgreSQL...")

write_table(pd.DataFrame(customers), "customers", mode="replace")
print("Customers data uploaded")

write_table(pd.DataFrame(accounts),  "accounts",  mode="replace")
print("accounts data uploaded")

write_table(pd.DataFrame(wallets),   "wallets",   mode="replace")
print("wallets data uploaded")

write_table(pd.DataFrame(devices),   "devices",   mode="replace")
print("devices data uploaded")



# In[33]:


transactions = pd.DataFrame(transactions)


# In[34]:


transactions['datestamp'] = pd.to_datetime(transactions['datestamp'],format="%d-%m-%Y",errors="coerce")
transactions['customer_cif_creation_date'] = pd.to_datetime(transactions['customer_cif_creation_date'],format="%d-%m-%Y",errors="coerce")
transactions['account_wallet_opening_date'] = pd.to_datetime(transactions['account_wallet_opening_date'],format="%d-%m-%Y",errors="coerce")
transactions['kyc_update_date'] = pd.to_datetime(transactions['kyc_update_date'],format="%d-%m-%Y",errors="coerce")
transactions['account_wallet_inoperative_date'] = pd.to_datetime(transactions['account_wallet_inoperative_date'])
transactions['date_of_incorporation'] = pd.to_datetime(transactions['date_of_incorporation'],format="%d-%m-%Y",errors="coerce")
transactions['date_of_birth'] = pd.to_datetime(transactions['date_of_birth'],format="%d-%m-%Y",errors="coerce")
transactions["professional_experience_years"] = pd.to_numeric(
    transactions["professional_experience_years"],
    errors="coerce"
).astype("Int64")

transactions.head()
transactions['cif_beneficial_owners'].unique()


# In[35]:


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


# In[39]:


import io
from psycopg2 import sql

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


# In[37]:


from datetime import datetime
txn_df = pd.DataFrame(transactions)
txn_df["loaded_at"] = datetime.now()


# In[ ]:


txn_df = pd.DataFrame(transactions).reindex(columns=TRANSACTION_COLUMNS)
write_table_fast(txn_df, "stg_transactions_generated_typology", mode="replace")
print("Generated data table has been uploaded to the database")

