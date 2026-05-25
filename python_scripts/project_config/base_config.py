#!/usr/bin/env python
# coding: utf-8
"""
config/base_config.py
═══════════════════════════════════════════════════════════════════════════════
SmartSentry AML — Master Configuration File

All parameters that control synthetic data generation are defined here.
No hardcoded values should appear in any generator module — everything
references this file.

SECTIONS
─────────
  1.  Random Seed & Reproducibility
  2.  Population Sizes
  3.  Simulation Time Window
  4.  Fraud Configuration
  5.  Customer Schema & Distributions
  6.  Account Schema & Distributions
  7.  Device Schema & Distributions
  8.  Beneficiary Schema & Distributions
  9.  Transaction Schema & Distributions
  10. Noise Injection Configuration
  11. Output Configuration
  11b. Graph Feature Configuration (graph_feature_generator)
  12. PK / FK Schema Reference  (documentation block)
  13. Feature Catalogue          (documentation block — all column names + meanings)
"""
# In[2]:


from datetime import datetime


# In[3]:


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RANDOM SEED & REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42          # Set to None to disable deterministic mode

# ═══════════════════════════════════════════════════════════════════════════════
# 2. POPULATION SIZES
# ═══════════════════════════════════════════════════════════════════════════════
NUM_CUSTOMERS      = 20_000    # Unique customer entities
NUM_ACCOUNTS       = 25_000    # Bank accounts (some customers have multiple)
NUM_DEVICES        = 18_000    # Unique device fingerprints
NUM_BENEFICIARIES  = 30_000    # External payment destinations

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION TIME WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 6, 30)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FRAUD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_FRAUD_RATE = 0.04      # Target proportion of fraudulent transactions (4%)

# Per-typology injection volumes
FRAUD_VOLUMES = {
    # --- Mule Ring ---
    # Multiple accounts (mules) share one device and funnel funds to a
    # single high-risk beneficiary exit node. Star-topology network pattern.
    "mule_ring": {
        "num_rings"          : 60,    # Number of distinct mule rings
        "accounts_per_ring"  : 10,    # Mule accounts per ring
        "txns_per_account"   : 8,     # Transactions each mule sends
        "day_min"            : 60,    # Earliest day offset from START_DATE
        "day_max"            : 120,   # Latest day offset from START_DATE
        "amount_mean"        : 9.5,   # Lognormal mean for transaction amount
        "amount_sigma"       : 0.5,   # Lognormal sigma for transaction amount
        "channels"           : ["mobile", "web"],
    },

    # --- Layering ---
    # Funds hop through 3–4 internal accounts in rapid succession to
    # obscure the audit trail. Linear chain A→B→C→D graph pattern.
    "layering": {
        "num_chains"         : 900,   # Number of layering chains
        "chain_length"       : 3,     # Number of hops per chain
        "day_min"            : 100,
        "day_max"            : 150,
        "hop_delay_min_mins" : 5,     # Min minutes between hops
        "hop_delay_max_mins" : 30,    # Max minutes between hops
        "amount_mean_base"   : 9.0,   # Lognormal mean (decreases per hop)
        "amount_mean_decay"  : 0.2,   # Amount reduction per hop (layering fees)
        "amount_sigma"       : 0.8,
        "channels"           : ["mobile", "web"],
    },

    # --- Account Takeover (ATO) ---
    # Compromised accounts transact from an unfamiliar device, immediately
    # transferring large sums to high-risk beneficiaries.
    "ATO": {
        "count"              : 2000,  # Number of ATO transactions
        "day_min"            : 120,
        "day_max"            : 180,
        "amount_mean"        : 10.5,
        "amount_sigma"       : 0.8,
        "channels"           : ["mobile", "web"],
    },

    # --- Smurfing / Structuring ---
    # Multiple cash transactions just below the reporting threshold, spread
    # over 2–3 days to avoid detection. Always cash-based (branch/ATM).
    "smurfing": {
        "num_groups"         : 400,   # Number of structuring groups
        "min_txns_per_group" : 4,     # Min transactions per group
        "max_txns_per_group" : 8,     # Max transactions per group
        "amount_min"         : 8_500, # INR — just below ₹10,000 threshold
        "amount_max"         : 9_999,
        "day_window"         : 2,     # Spread over N days
        "channels"           : ["branch", "atm"],
        "force_cash_flag"    : True,  # Override cash_flag to 1
    },

    # --- Identity Fraud ---
    # Newly opened accounts (< 60 days old) immediately make large transfers
    # to high-risk beneficiaries — synthetic identity fraud pattern.
    "identity_fraud": {
        "count"                  : 800,
        "max_account_open_days"  : 60,  # Only target new accounts
        "day_min"                : 0,
        "day_max"                : 30,
        "amount_mean"            : 11.0,
        "amount_sigma"           : 0.6,
        "channels"               : ["mobile", "web"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CUSTOMER SCHEMA & DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Occupations and their prevalence in the customer base
OCCUPATIONS = [
    "salaried",           # Regular employment income
    "self_employed",      # Own business, variable income
    "business_owner",     # Company director / proprietor
    "student",            # Low income, high-risk for synthetic fraud
    "retired",            # Fixed income, dormancy risk
    "government_employee",# Stable income, PEP-adjacent risk
    "freelancer",         # Variable income, offshore exposure
    "unemployed",         # High-risk: no income justification
]
OCCUPATION_PROBS = [0.30, 0.15, 0.12, 0.08, 0.10, 0.10, 0.08, 0.07]

# Industry sectors
INDUSTRIES = [
    "finance",            # Banks, NBFCs — inherent ML/TF exposure
    "retail",             # Cash-heavy, structuring risk
    "healthcare",         # Moderate risk
    "technology",         # Cross-border, crypto exposure
    "real_estate",        # High layering / placement risk
    "manufacturing",      # Trade-based ML risk
    "education",          # Low risk baseline
    "hospitality",        # Cash-heavy, smurfing risk
    "construction",       # Cash-heavy, high layering risk
    "unknown",            # Missing / not disclosed — elevated risk
]
INDUSTRY_PROBS = [0.12, 0.14, 0.10, 0.13, 0.09, 0.10, 0.08, 0.08, 0.07, 0.09]

# KYC level assigned during onboarding
KYC_LEVELS = ["low", "medium", "high"]
KYC_PROBS  = [0.20,  0.50,     0.30]

# Customer risk ratings (CDD / EDD driven)
RISK_RATINGS = ["low",  "medium", "high", "very_high"]
RISK_PROBS   = [0.50,    0.30,    0.15,    0.05]

# Account types (from Section 4.2 of design doc)
ACCOUNT_TYPES = ["retail", "corporate", "savings", "current", "business"]
ACCOUNT_PROBS = [0.35,     0.15,       0.25,      0.15,     0.10]

# Declared income bracket
INCOME_BRACKETS = ["low",  "medium", "high"]
INCOME_PROBS    = [0.30,    0.50,    0.20]

# Country risk of customer's home jurisdiction
COUNTRY_RISKS = ["low",  "medium", "high"]
COUNTRY_PROBS = [0.60,    0.30,    0.10]

# Customer age range
CUSTOMER_AGE_MIN = 21
CUSTOMER_AGE_MAX = 70

# Days since account opening (used as customer tenure proxy)
CUSTOMER_SINCE_DAYS_MIN = 30
CUSTOMER_SINCE_DAYS_MAX = 3650

# PEP (Politically Exposed Person) prevalence
PEP_PREVALENCE = 0.03    # 3% of customers are PEPs

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ACCOUNT SCHEMA & DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Account balance (lognormal — realistic wealth distribution)
AVG_BALANCE_LOGNORMAL_MEAN  = 10    # ln(₹) → median balance ≈ ₹22,026
AVG_BALANCE_LOGNORMAL_SIGMA = 1

# Account age in days
ACCOUNT_OPEN_DAYS_MIN = 30
ACCOUNT_OPEN_DAYS_MAX = 2000

# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEVICE SCHEMA & DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

OS_TYPES  = ["android", "ios", "windows", "unknown"]
OS_PROBS  = [0.45,      0.40,  0.10,      0.05]

DEVICE_AGE_DAYS_MIN = 30
DEVICE_AGE_DAYS_MAX = 1500

ROOTED_DEVICE_RATE  = 0.05    # 5% of devices are rooted/jailbroken
VPN_USAGE_RATE      = 0.08    # 8% of devices route through VPN
EMULATOR_RATE       = 0.03    # 3% are emulated devices (fraud signal)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. BENEFICIARY SCHEMA & DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

BENE_TYPES  = ["individual", "merchant", "crypto", "offshore"]
BENE_PROBS  = [0.60,         0.30,       0.05,     0.05]

# These types are considered high-risk exit nodes for fraud
HIGH_RISK_BENE_TYPES = {"crypto", "offshore"}

BENE_COUNTRY_RISKS  = ["low",  "medium", "high"]
BENE_COUNTRY_PROBS  = [0.65,    0.25,    0.10]

# Pre-assigned beneficiaries per account for legit transactions
BENE_PER_ACCOUNT_MIN = 2
BENE_PER_ACCOUNT_MAX = 5

# ═══════════════════════════════════════════════════════════════════════════════
# 9. TRANSACTION SCHEMA & DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Channels and their usage split
CHANNELS      = ["mobile", "web",  "branch", "atm"]
CHANNEL_PROBS = [0.50,     0.30,   0.10,     0.10]

# Transaction types available per channel
TXN_TYPES_BY_CHANNEL = {
    "mobile" : ["UPI", "IMPS", "wallet_transfer"],
    "web"    : ["NEFT", "RTGS", "online_transfer"],
    "branch" : ["cash_deposit", "cash_withdrawal", "DD"],
    "atm"    : ["cash_withdrawal", "balance_enquiry"],
}

# Transaction types that involve physical cash → cash_flag = 1
CASH_TXN_TYPES = {"cash_deposit", "cash_withdrawal", "DD"}

# Lognormal amount parameters keyed by KYC level
# (KYC level is a proxy for customer wealth)
AMOUNT_BY_KYC = {
    "low"    : {"mean": 7.0, "sigma": 1.0},   # median ≈ ₹1,097
    "medium" : {"mean": 8.0, "sigma": 1.0},   # median ≈ ₹2,981
    "high"   : {"mean": 9.5, "sigma": 1.0},   # median ≈ ₹13,360
}
AMOUNT_DEFAULT = {"mean": 8.0, "sigma": 1.0}

# Average number of transactions per account (Poisson lambda)
AVG_TXNS_PER_ACCOUNT = 50

# Probability a receiver is another internal account vs external beneficiary
INTERNAL_TRANSFER_PROB = 0.40

# Hour weights for transaction timing
# Fraud transactions concentrate on late night / early morning
HOUR_WEIGHTS_NORMAL = [1,1,1,1,1,1,2,3,5,6,6,6,5,6,6,6,5,5,4,4,3,2,2,1]
HOUR_WEIGHTS_FRAUD  = [4,4,4,4,3,2,1,1,1,1,1,1,1,1,1,1,2,2,3,3,4,4,4,4]

# ═══════════════════════════════════════════════════════════════════════════════
# 10. NOISE INJECTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_CONFIG = {
    # Missing values — simulate real-world data quality issues
    "missing_values": {
        "enabled"     : True,
        "columns"     : {
            # column_name : fraction of rows to set NULL
            "beneficiary_id"          : 0.02,   # 2% external txns missing bene
            "device_id"               : 0.01,   # 1% missing device (branch txns)
            "occupation"              : 0.03,
            "beneficiary_country_risk": 0.02,
        },
    },

    # Duplicate transactions — simulate processing errors / replay attacks
    "duplicates": {
        "enabled"          : True,
        "duplicate_rate"   : 0.005,   # 0.5% of rows duplicated
        "id_suffix"        : "_DUP",  # Appended to transaction_id of duplicate
    },

    # Amount rounding noise — simulate currency conversion / fee artefacts
    "amount_noise": {
        "enabled"     : True,
        "noise_std"   : 0.5,          # Gaussian std dev added to amount (INR)
        "clip_min"    : 1.0,          # Minimum amount after noise
    },

    # Timestamp jitter — simulate clock skew between systems
    "timestamp_jitter": {
        "enabled"     : True,
        "max_jitter_seconds": 30,     # ± seconds of random jitter
    },

    # Label noise — simulate analyst misclassification
    "label_noise": {
        "enabled"        : True,
        "flip_rate_fraud" : 0.02,    # 2% of fraud rows mislabelled as normal
        "flip_rate_legit" : 0.001,   # 0.1% of legit rows mislabelled as fraud
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 11. OUTPUT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "outputs"

OUTPUT_FILES = {
    "customers"        : "customers.csv",
    "accounts"         : "accounts.csv",
    "devices"          : "devices.csv",
    "beneficiaries"    : "beneficiaries.csv",
    "transactions"     : "transactions.csv",
    "graph_edges"      : "graph_edges.csv",
    "feature_catalogue": "feature_catalogue.csv",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 11b. GRAPH FEATURE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Used by graph_feature_generator notebook. Input = transaction table with additional
# features; output = same table + graph-based AML features.

GRAPH_FEATURE_CONFIG = {
    "input_path"       : "transaction_additional_feature.parquet",  # or .csv fallback
    "output_path"      : "transaction_with_graph_features.parquet",
    "input_dir"        : "outputs",   # directory for input_path / output_path
    "rolling_days"     : 30,           # window for degree, inflow, outflow, counterparty stats
    "rolling_days_7d"  : 7,            # for 7d pass-through / outflow_to_inflow
    "rolling_hours_24h": 24,           # for 24h pass-through
    "features_enabled" : [
        "sender_in_degree_30d", "sender_out_degree_30d", "sender_total_inflow_30d",
        "sender_total_outflow_30d", "sender_unique_counterparties_30d", "sender_repeat_counterparty_ratio",
        "receiver_in_degree_30d", "receiver_out_degree_30d", "receiver_unique_senders_30d",
        "pass_through_ratio_24h", "pass_through_ratio_7d", "outflow_to_inflow_ratio_7d",
        "avg_time_gap_in_out", "accounts_per_device", "devices_per_account",
        "device_shared_high_risk_ratio", "shared_device_fraud_count",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 12. PK / FK SCHEMA REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
"""
TABLE             PRIMARY KEY       FOREIGN KEYS
──────────────────────────────────────────────────────────────────────────────
customers         customer_id       —
accounts          account_id        customer_id       → customers.customer_id
devices           device_id         —
beneficiaries     beneficiary_id    —
transactions      transaction_id    customer_id       → customers.customer_id
                                    sender_account_id → accounts.account_id
                                    receiver_account_id → accounts.account_id (nullable)
                                    beneficiary_id    → beneficiaries.beneficiary_id (nullable)
                                    device_id         → devices.device_id
graph_edges       —                 transaction_id    → transactions.transaction_id
                                    source_account_id → accounts.account_id
                                    customer_id       → customers.customer_id

NULL SEMANTICS IN TRANSACTIONS
──────────────────────────────
  receiver_account_id is populated → INTERNAL transfer between two known accounts
  beneficiary_id      is populated → EXTERNAL transfer to an outside entity
  Exactly one of the two is non-NULL per row (XOR constraint).

QUICK JOIN RECIPES
──────────────────
  # Transactions + customer profile (single key):
  transactions.merge(customers, on="customer_id")

  # Transactions + account details:
  transactions.merge(accounts, left_on="sender_account_id", right_on="account_id")

  # Transactions + device info:
  transactions.merge(devices, on="device_id")

  # Transactions + beneficiary info (external only):
  transactions[transactions.beneficiary_id.notna()].merge(beneficiaries, on="beneficiary_id")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 13. FEATURE CATALOGUE
# Column name → (table, data_type, description)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_CATALOGUE = {

    # ── CUSTOMERS TABLE ───────────────────────────────────────────────────────
    "customer_id": (
        "customers", "string",
        "Primary key. Unique customer identifier (format: C0, C1, …)."
    ),
    "age": (
        "customers", "integer",
        "Customer age in years. Range: 21–70."
    ),
    "customer_risk_rating": (
        "customers", "categorical",
        "Overall CDD risk rating assigned during onboarding. "
        "Values: low | medium | high | very_high."
    ),
    "pep_flag": (
        "customers", "binary",
        "Politically Exposed Person flag. 1 = PEP (elected officials, senior "
        "government roles, their close associates). Triggers Enhanced Due Diligence."
    ),
    "occupation": (
        "customers", "categorical",
        "Customer's declared occupation. "
        "Values: salaried | self_employed | business_owner | student | retired | "
        "government_employee | freelancer | unemployed."
    ),
    "industry": (
        "customers", "categorical",
        "Industry sector of customer's business or employer. Used for risk segmentation. "
        "Values: finance | retail | healthcare | technology | real_estate | "
        "manufacturing | education | hospitality | construction | unknown."
    ),
    "account_type": (
        "customers", "categorical",
        "Type of account held. Values: retail | corporate | savings | current | business."
    ),
    "kyc_level": (
        "customers", "categorical",
        "Know Your Customer verification depth. "
        "low = basic ID only; medium = standard verification; high = full EDD completed."
    ),
    "income_bracket": (
        "customers", "categorical",
        "Declared income range. Values: low | medium | high. "
        "Used to contextualise transaction amounts."
    ),
    "country_risk": (
        "customers", "categorical",
        "Risk level of customer's home country/jurisdiction. "
        "Based on FATF grey/black list status and corruption indices. "
        "Values: low | medium | high."
    ),
    "customer_since_days": (
        "customers", "integer",
        "Number of days since the customer relationship was established. "
        "Range: 30–3650 days (~10 years max)."
    ),

    # ── ACCOUNTS TABLE ────────────────────────────────────────────────────────
    "account_id": (
        "accounts", "string",
        "Primary key. Unique account identifier (format: A0, A1, …)."
    ),
    # customer_id repeated in accounts (FK)
    "avg_balance": (
        "accounts", "float",
        "Average account balance in INR (lognormally distributed). "
        "Used as denominator in amount_to_balance_ratio feature."
    ),
    "account_open_days": (
        "accounts", "integer",
        "Number of days since account was opened. Range: 30–2000. "
        "New accounts (<60 days) are higher risk for identity fraud."
    ),

    # ── DEVICES TABLE ─────────────────────────────────────────────────────────
    "device_id": (
        "devices", "string",
        "Primary key. Unique device fingerprint identifier (format: D0, D1, …)."
    ),
    "device_age_days": (
        "devices", "integer",
        "Age of device in days since first seen. Range: 30–1500."
    ),
    "rooted_flag": (
        "devices", "binary",
        "1 = device is rooted (Android) or jailbroken (iOS). "
        "Indicates potential tampering and elevated fraud risk."
    ),
    "os_type": (
        "devices", "categorical",
        "Operating system of the device. "
        "Values: android | ios | windows | unknown."
    ),
    "vpn_flag": (
        "devices", "binary",
        "1 = transaction was routed through a VPN or proxy. "
        "Geo-masking indicator; elevated risk when combined with high-value transfers."
    ),
    "emulator_flag": (
        "devices", "binary",
        "1 = device is an emulated environment (not a real physical device). "
        "Strong fraud signal — used in automated fraud attacks."
    ),

    # ── BENEFICIARIES TABLE ───────────────────────────────────────────────────
    "beneficiary_id": (
        "beneficiaries", "string",
        "Primary key. Unique external payment destination (format: B0, B1, …)."
    ),
    "beneficiary_type": (
        "beneficiaries", "categorical",
        "Type of external beneficiary. "
        "individual = person-to-person; merchant = business payment; "
        "crypto = cryptocurrency exchange; offshore = foreign/shell entity. "
        "crypto and offshore are high-risk types."
    ),
    "beneficiary_country_risk": (
        "beneficiaries", "categorical",
        "Country risk of the beneficiary's jurisdiction. "
        "Values: low | medium | high."
    ),

    # ── TRANSACTIONS TABLE (core fields) ──────────────────────────────────────
    "transaction_id": (
        "transactions", "string",
        "Primary key. Unique transaction identifier (format: T0, T1, …). "
        "Oversampled fraud rows get suffix to maintain uniqueness."
    ),
    "customer_id": (
        "transactions", "string",
        "Foreign key → customers.customer_id. "
        "Denormalised from accounts so any transaction can be linked directly "
        "to the customer profile with a single join."
    ),
    "sender_account_id": (
        "transactions", "string",
        "Foreign key → accounts.account_id. "
        "The account that initiated (debited) the transaction."
    ),
    "receiver_account_id": (
        "transactions", "string",
        "Foreign key → accounts.account_id. NULLABLE. "
        "Populated only for INTERNAL transfers between two known accounts. "
        "NULL when beneficiary_id is set (external transfer)."
    ),
    "beneficiary_id": (
        "transactions", "string",
        "Foreign key → beneficiaries.beneficiary_id. NULLABLE. "
        "Populated only for EXTERNAL transfers to outside entities. "
        "NULL when receiver_account_id is set (internal transfer)."
    ),
    "device_id": (
        "transactions", "string",
        "Foreign key → devices.device_id. "
        "Device fingerprint used to initiate the transaction."
    ),
    "timestamp": (
        "transactions", "datetime",
        "Full datetime of the transaction (UTC). Format: YYYY-MM-DD HH:MM:SS."
    ),
    "amount": (
        "transactions", "float",
        "Transaction amount in INR (Indian Rupees). Lognormally distributed. "
        "Fraud transactions are generally higher-value than legit."
    ),
    "channel": (
        "transactions", "categorical",
        "Originating channel of the transaction. "
        "Values: mobile | web | branch | atm."
    ),
    "debit_credit": (
        "transactions", "categorical",
        "Direction from sender's perspective. Always 'debit' in this dataset "
        "(outgoing from the sender account)."
    ),
    "transaction_type": (
        "transactions", "categorical",
        "Specific payment instrument used. "
        "UPI | IMPS | wallet_transfer | NEFT | RTGS | online_transfer | "
        "cash_deposit | cash_withdrawal | DD | balance_enquiry."
    ),
    "cash_flag": (
        "transactions", "binary",
        "1 = transaction involves physical cash (cash_deposit, cash_withdrawal, DD). "
        "Key feature for structuring / smurfing detection."
    ),
    "fraud_type": (
        "transactions", "categorical",
        "Ground truth label for the fraud typology. "
        "normal | mule_ring | layering | ATO | smurfing | identity_fraud."
    ),
    "label": (
        "transactions", "binary",
        "Binary fraud label. 0 = legitimate transaction, 1 = fraudulent. "
        "This is the model target variable."
    ),

    # ── TRANSACTIONS TABLE (enriched / derived features) ──────────────────────
    "hour": (
        "transactions", "integer",
        "Hour of day (0–23) extracted from timestamp. "
        "Fraud transactions concentrate in hours 0–4 and 20–23."
    ),
    "day_of_week": (
        "transactions", "integer",
        "Day of week (0=Monday … 6=Sunday). Used for weekend anomaly detection."
    ),
    "is_night": (
        "transactions", "binary",
        "1 = transaction occurred between 22:00 and 05:59. "
        "Derived from hour. Strong fraud signal."
    ),
    "is_weekend": (
        "transactions", "binary",
        "1 = transaction occurred on Saturday or Sunday. "
        "Weekend high-value transactions are flagged by rule engine."
    ),
    "month": (
        "transactions", "integer",
        "Calendar month (1–12). Useful for seasonality analysis."
    ),
    "amount_to_balance_ratio": (
        "transactions", "float",
        "Transaction amount divided by sender's average account balance. "
        "Ratio > 1 means transaction exceeds typical balance. "
        "Clipped at balance >= 1 to avoid division by zero."
    ),
    "high_risk_beneficiary": (
        "transactions", "binary",
        "1 = beneficiary is in the high-risk pool "
        "(type: crypto or offshore, OR country_risk: high)."
    ),
    "txn_velocity_cumulative": (
        "transactions", "integer",
        "Cumulative count of transactions sent by this account up to this point. "
        "Proxy for overall account activity level."
    ),
    "rolling_7d_txn_count": (
        "transactions", "integer",
        "Number of transactions sent by this account in the 7 days prior to this transaction. "
        "High values indicate burst behaviour."
    ),
    "rolling_7d_txn_sum": (
        "transactions", "float",
        "Total amount sent by this account in the 7 days prior (INR). "
        "Used for velocity-based AML rules."
    ),
    "rolling_30d_txn_count": (
        "transactions", "integer",
        "Transaction count in the 30-day rolling window. "
        "Basis for monthly velocity rules."
    ),
    "rolling_30d_txn_sum": (
        "transactions", "float",
        "Total amount sent in the 30-day rolling window (INR)."
    ),
    "amount_zscore": (
        "transactions", "float",
        "Z-score of this transaction's amount relative to the sender's 30-day rolling "
        "mean and std. Values > 3 indicate anomalous spikes."
    ),
    "dormancy_flag": (
        "transactions", "binary",
        "1 = more than 30 days have elapsed since this account's previous transaction. "
        "Dormant account reactivation is a key AML signal."
    ),

    # ── RULE ENGINE FEATURES ──────────────────────────────────────────────────
    "rule_trigger_count": (
        "transactions", "integer",
        "Total number of AML rules triggered by this transaction. "
        "Higher values indicate greater suspicion. Used as ML meta-feature."
    ),
    "max_rule_severity": (
        "transactions", "integer",
        "Severity (1–3) of the most severe rule triggered. "
        "0 = no rules fired; 3 = at least one high-severity rule fired."
    ),
    "weighted_rule_score": (
        "transactions", "integer",
        "Sum of severities of all triggered rules. "
        "Reflects cumulative suspicion weight across all rules."
    ),

    # ── GRAPH EDGES TABLE ─────────────────────────────────────────────────────
    "source_account_id": (
        "graph_edges", "string",
        "FK → accounts.account_id. The sending account (graph source node)."
    ),
    "target_id": (
        "graph_edges", "string",
        "The receiving node. FK → accounts.account_id if target_type='account', "
        "FK → beneficiaries.beneficiary_id if target_type='beneficiary'."
    ),
    "target_type": (
        "graph_edges", "categorical",
        "Tells the consumer which table target_id belongs to. "
        "Values: account | beneficiary."
    ),
    "relation": (
        "graph_edges", "categorical",
        "The fraud typology label of this edge. "
        "normal | mule_ring | layering | ATO | smurfing | identity_fraud."
    ),
}

