# SmartSentry AML — Synthetic Transaction Pipeline Documentation

> **Version:** V7 | **Seed:** 42 | **Date Range:** 2025-09-01 → 2025-12-31
>
> Comprehensive technical reference for the end-to-end synthetic AML transaction dataset generation pipeline.

---

## Table of Contents

1. [Overall Pipeline Overview](#1-overall-pipeline-overview)
2. [Configuration Reference](#2-configuration-reference)
3. [Data Generation Process](#3-data-generation-process)
4. [Typology Injection](#4-typology-injection)
5. [Feature Engineering](#5-feature-engineering)
6. [Graph / Network Features](#6-graph--network-features)
7. [Rule-Based Features & Fraud Intensity Score](#7-rule-based-features--fraud-intensity-score)
8. [Data Quality & QC Checks](#8-data-quality--qc-checks)
9. [Code Flow — Step-by-Step](#9-code-flow--step-by-step)
10. [Final Dataset Structure](#10-final-dataset-structure)
11. [Mathematical Formulas Reference](#11-mathematical-formulas-reference)
12. [Pipeline Flow Diagram](#12-pipeline-flow-diagram)

---

## 1. Overall Pipeline Overview

The SmartSentry AML pipeline is a **five-notebook, multi-stage data factory** that produces a fully-labelled synthetic financial transactions dataset enriched with fraud/AML typologies, velocity features, network graph metrics, and a rule-based Fraud Intensity Score (FIS). The dataset is designed to train and benchmark AML detection models in a controlled, reproducible environment.

### What the Pipeline Produces

| Artifact | Description |
|---|---|
| `customers.csv` | 3,000 customer entities with KYC, risk, demographic attributes |
| `accounts.csv` | 4,500 bank accounts linked to customers (denormalised) |
| `devices.csv` | 4,000 unique device fingerprints |
| `beneficiaries.csv` | 4,000 external payment destinations including shell companies |
| `txns_with_typologies.parquet` | Raw transactions with 8 fraud typologies injected |
| `transactions_enriched.parquet` | Transactions enriched with rolling velocity and balance features |
| `stg_transactions.parquet` | Final transactions with 50 AML rules and FIS score |
| `stg_transactions_graph.parquet` | Final output with graph/network features appended |

### Notebook Execution Order

```
01_customer_generator.ipynb
        ↓
02a_merging_typologies.ipynb
        ↓
02b_rolling_features.ipynb
        ↓
02c_rule_based_features.ipynb
        ↓
03_graph_feature_generator.ipynb
```

### Design Principles

- **No magic numbers** — all constants are defined in a central configuration cell per notebook.
- **Reproducibility** — `RANDOM_SEED = 42` is applied globally across `numpy`, `random`, and per-RNG instances.
- **Denormalised schema** — customer risk attributes are embedded into the accounts table so transactions need only one join to get full entity context.
- **Ground truth labels** — `label = 1` for all injected fraud rows, `label = 0` for legitimate rows. Oversampling brings the dataset to a 10% fraud rate.

---

## 2. Configuration Reference

### 2.1 Entity Population Sizes (Notebook 01)

| Parameter | Default | Purpose |
|---|---|---|
| `NUM_CUSTOMERS` | 3,000 | Total unique customer entities |
| `NUM_ACCOUNTS` | 4,500 | Bank accounts — allows 1-2 accounts per customer on average |
| `NUM_DEVICES` | 4,000 | Unique device fingerprints available for assignment |
| `NUM_BENEFICIARIES` | 4,000 | External payment destinations |
| `RANDOM_SEED` | 42 | Global RNG seed for full reproducibility |

### 2.2 Customer Distributions (Notebook 01)

| Parameter | Values / Range | Purpose |
|---|---|---|
| `CUSTOMER_AGE_MIN/MAX` | 21–70 | Age range for customer demographics |
| `KYC_LEVELS` / `KYC_PROBS` | low 20%, medium 50%, high 30% | KYC depth distribution |
| `RISK_RATINGS` / `RISK_PROBS` | low 50%, medium 30%, high 15%, very_high 5% | CDD risk tier |
| `PEP_PREVALENCE` | 0.03 (3%) | Politically Exposed Persons fraction |
| `OCCUPATION_PROBS` | salaried 30%, self_employed 15%, business_owner 12%, student 8%, retired 10%, government 10%, freelancer 8%, unemployed 7% | Occupation mix |
| `INCOME_BRACKETS` / `INCOME_PROBS` | low 30%, medium 50%, high 20% | Declared income distribution |
| `COUNTRY_RISKS` / `COUNTRY_PROBS` | low 60%, medium 30%, high 10% | Home jurisdiction FATF risk |
| `CUSTOMER_SINCE_DAYS_MIN/MAX` | 30–3,650 | Customer tenure range in days |

### 2.3 Account Distributions (Notebook 01)

| Parameter | Default | Purpose |
|---|---|---|
| `AVG_BALANCE_LOG_MEAN` | 10 | Log-mean for lognormal balance (median ≈ ₹22,026) |
| `AVG_BALANCE_LOG_SIGMA` | 1.0 | Log-sigma for balance spread |
| `ACCOUNT_OPEN_DAYS_MIN/MAX` | 30–2,000 | Account age range |
| `ACCOUNT_TYPES` / `ACCOUNT_PROBS` | retail 35%, corporate 15%, savings 25%, current 15%, business 10% | Account type mix |

### 2.4 Device Distributions (Notebook 01)

| Parameter | Default | Purpose |
|---|---|---|
| `OS_TYPES` / `OS_PROBS` | android 45%, ios 40%, windows 10%, unknown 5% | OS distribution |
| `DEVICE_AGE_MIN/MAX` | 30–1,500 days | Device age range |
| `ROOTED_DEVICE_RATE` | 0.05 (5%) | Jailbroken/rooted devices — elevated risk |
| `VPN_USAGE_RATE` | 0.08 (8%) | VPN/proxy usage rate — geo-masking signal |
| `EMULATOR_RATE` | 0.03 (3%) | Emulated environment rate — strong fraud signal |

### 2.5 Beneficiary Distributions (Notebook 01)

| Parameter | Default | Purpose |
|---|---|---|
| `BENE_TYPES` / `BENE_PROBS` | individual 55%, business 25%, shell_company 8%, crypto 7%, offshore 5% | Beneficiary type mix |
| `BENE_COUNTRY_PROBS` | low 60%, medium 25%, high 15% | Jurisdiction risk of beneficiaries |
| `BENE_PER_ACCOUNT_MIN/MAX` | 2–5 | Pre-assigned known beneficiaries per account |
| `SHELL_COMPANY_JURISDICTIONS` | BVI 25%, Cayman 20%, Panama 15%, Delaware 10%, Luxembourg 10%, Seychelles 8%, Mauritius 7%, Singapore 5% | Secrecy jurisdiction mix for shell companies |
| `SHELL_AGE_MIN/MAX` | 30–1,825 days | Shell company incorporation age |

### 2.6 Shared Identity (Synthetic Identity / Mule Detection)

| Parameter | Default | Purpose |
|---|---|---|
| `SHARED_KYC_FRAC` | 0.065 (6.5%) | Customers sharing a KYC ID (synthetic identity rings) |
| `SHARED_KYC_GROUP_MIN/MAX` | 2–4 | Size of each shared KYC group |
| `SHARED_PHONE_FRAC` | 0.015 (1.5%) | Customers sharing a phone hash (mule recruitment signal) |
| `SHARED_EMAIL_FRAC` | 0.015 (1.5%) | Customers sharing an email hash (identity farm signal) |

### 2.7 Transaction Simulation (Notebook 02a)

| Parameter | Default | Purpose |
|---|---|---|
| `START_DATE` / `END_DATE` | 2025-09-01 → 2025-12-31 | 121-day simulation window |
| `TARGET_FRAUD_RATE` | 0.10 (10%) | Post-rebalancing fraud fraction |
| `AVG_TXNS_PER_ACCOUNT` | 50 / 121 days × 121 ≈ 50 | Average transactions per account across the window |
| `DORMANT_ACCOUNT_FRAC` | 0.05 (5%) | Accounts with dormant-then-reactivated patterns |
| `INTERNAL_TRANSFER_PROB` | 0.40 (40%) | Fraction of transactions routed to another account vs. beneficiary |

### 2.8 Transaction Amount Parameters

| KYC Level | Log-Mean | Log-Sigma | Approximate Median |
|---|---|---|---|
| `low` | 7.0 | 1.0 | ₹1,097 |
| `medium` | 8.0 | 1.0 | ₹2,981 |
| `high` | 9.5 | 1.0 | ₹13,360 |

### 2.9 Hour-of-Day Transaction Weights

```
Normal transactions: heavier during business hours (9–17)
Fraud transactions:  heavier at night (0–5) and evenings (20–24)
```

| Period | Normal Weight | Fraud Weight |
|---|---|---|
| 00:00–05:59 (night) | 1 | 4 |
| 06:00–08:59 (early morning) | 2–3 | 1 |
| 09:00–17:59 (business hours) | 5–6 | 1 |
| 18:00–21:59 (evening) | 3–5 | 2–3 |
| 22:00–23:59 (late night) | 1–2 | 4 |

### 2.10 IP Risk Score Constants (Notebook 02a)

| Parameter | Default | Purpose |
|---|---|---|
| `IP_HOME_FRAC` | 0.70 | 70% of IPs are home (private) IPs |
| `IP_ROAM_FRAC` | 0.20 | 20% are roaming (public) IPs |
| `IP_HOME_BASE_SCORE` | 0.05 | Base risk score for home IPs |
| `IP_ROAM_BASE_SCORE` | 0.30 | Base risk score for roaming IPs |
| `IP_RISK_BASE_SCORE` | 0.80 | Base risk score for flagged/risk IPs |
| `HIGH_RISK_IP_PREFIXES` | 10 known TOR/darknet prefixes | Prefixes triggering risk classification |

### 2.11 Graph Feature Config (Notebook 03)

| Parameter | Default | Purpose |
|---|---|---|
| `rolling_days` | 30 | Primary rolling window for graph metrics |
| `rolling_days_7d` | 7 | Secondary rolling window |
| `rolling_hours_24h` | 24 | Short-term rolling window |

---

## 3. Data Generation Process

### 3.1 Customer Generation

Each of the 3,000 customers is assigned:
- A unique `customer_id` (`C0`–`C2999`)
- Demographics: age (21–70), occupation, industry
- Risk profile: KYC level, CDD risk rating, PEP flag, income bracket, country risk, tenure
- Home city: one of 20 major Indian cities, sampled by population weight
- Home latitude/longitude (seeds per-transaction geo coordinates)
- Shared identity tokens: `shared_kyc_id`, `shared_phone_hash`, `shared_email_hash` (NULL for unique customers)

**Shared identity assignment** uses a grouping function:

```python
def assign_shared_ids(df, frac, group_min, group_max, prefix, seed_offset):
    # Randomly permute customer indices
    # Assign frac × N customers into groups of size [group_min, group_max]
    # Each group gets a label like "KYC_00042"
```

This models real AML patterns: synthetic identity rings (shared KYC), mule recruitment (shared phone), and identity farms (shared email).

### 3.2 Account Generation

4,500 accounts are generated with:
- A unique `account_id` (`A0`–`A4499`)
- A random `customer_id` (customers may have 1+ accounts)
- `avg_balance` drawn from `LogNormal(μ=10, σ=1)` → median ≈ ₹22,026
- `account_open_days` uniform in [30, 2000]
- All customer risk attributes **denormalised** into the accounts table for fast lookup during transaction join

**Denormalisation** means: account rows carry `kyc_level`, `customer_risk_rating`, `pep_flag`, `country_risk`, `income_bracket`, `occupation`, `industry`, `home_city`, and all shared identity columns. This avoids multi-table joins at transaction time.

### 3.3 Device Generation

4,000 devices are generated, each with:
- A unique `device_id` (`D0`–`D3999`)
- OS type (`android`, `ios`, `windows`, `unknown`)
- Age in days: uniform [30, 1500]
- Risk flags: `rooted_flag`, `vpn_flag`, `emulator_flag`

Each account is mapped to one **home device** (`account_device_map`). Fraud injectors deliberately assign a **different** device to simulate Account Takeover (ATO).

### 3.4 Beneficiary Generation

4,000 beneficiaries with:
- `beneficiary_type`: individual, business, shell_company, crypto, offshore
- `beneficiary_country_risk`: low / medium / high (shells skew high)

Shell company beneficiaries additionally have:
- `shell_jurisdiction`: one of 8 secrecy jurisdictions (BVI, Cayman, Panama, etc.)
- `shell_age_days`: days since incorporation
- `ubo_disclosed`: 1 = UBO known, 0 = hidden (75% of shells hide UBO)
- `shell_newly_incorporated`: 1 if age < 180 days (highest layering risk)

A **high-risk beneficiary pool** is pre-built:

```
high_risk_mask = (beneficiary_type IN {crypto, offshore}) OR (country_risk == 'high')
```

Fraud injectors always route to beneficiaries from this pool.

### 3.5 Transaction Simulation

**Legitimate transactions** are generated per account:

- **Normal accounts**: Poisson(λ = AVG_TXNS_PER_ACCOUNT) transactions, spread uniformly across the 121-day window
- **Dormant accounts** (5% of accounts): Two activity clusters:
  - Cluster 1 (early): 3–8 transactions in days [0, ~25]
  - Cluster 2 (late reactivation): 5–15 transactions in days [~66, 121]
- For each transaction: timestamp drawn with business-hours weighting, amount from KYC-stratified lognormal, channel sampled from `[mobile, web, branch, atm]` at probabilities `[0.50, 0.30, 0.10, 0.10]`
- 40% of transactions are internal transfers (to another account); 60% go to a pre-assigned beneficiary

**Transaction row structure** (18 core columns):

```
transaction_id, customer_id, sender_account_id, receiver_account_id,
beneficiary_id, device_id, timestamp, amount, channel, debit_credit,
transaction_type, cash_flag, synthetic_flow_id, flow_depth, hop_number,
time_since_origin_ts, fraud_type, label
```

---

## 4. Typology Injection

Eight distinct fraud/AML typologies are injected. All injected rows carry `label = 1`.

### 4.1 Mule Ring

**Pattern:** A group of accounts share one device and funnel funds to a common high-risk exit beneficiary.

| Parameter | Value |
|---|---|
| Number of rings | ~60 × (date_range / 121) |
| Accounts per ring | 10 |
| Transactions per account | 8 |
| Amount | LogNormal(μ=9.5, σ=0.5) |
| Channels | mobile, web |
| Timing | Second 50–99% of date window |

**Signals injected:**
- All ring members share one `device_id` (device collision)
- All route to the same `beneficiary_id` (exit node concentration)
- Tagged with a common `synthetic_flow_id`, `flow_depth`, and `hop_number`
- Shared identity: same `shared_kyc_id` per flow (all ring members use same KYC document)

**Detection logic:**
```
Multiple accounts → single device → single high-risk beneficiary
```

### 4.2 Layering Chains

**Pattern:** Funds cycle through a chain of accounts A→B→C→A, each hop slightly later in time, with amounts decaying at each hop.

| Parameter | Value |
|---|---|
| Number of chains | ~900 × (date_range / 121) |
| Chain length | 3 hops |
| Hop delay | 5–30 minutes per hop |
| Amount decay | μ decreases by 0.2 per hop |
| Timing | 60–99% of date window |

**Amount formula at each hop:**

```
amount_hop_i ~ LogNormal(μ_base - i × 0.2, σ=0.8)
```

**Signals injected:**
- Shared `synthetic_flow_id` across the chain
- `hop_number` = 1, 2, 3 with `time_since_origin_ts` (seconds since hop 1)
- Chain closes back to originating account
- Shared `shared_phone_hash` per flow

### 4.3 Account Takeover (ATO)

**Pattern:** An adversary uses a new, foreign device (not the account's home device) to make a large, rapid transfer to a high-risk beneficiary.

| Parameter | Value |
|---|---|
| Count | ~2,000 × (date_range / 121) |
| Amount | LogNormal(μ=10.5, σ=0.8) |
| Device | Randomly chosen device ≠ home device |
| Timing | 70–99% of date window |

**Key signal:** `device_id` ≠ `account_device_map[account_id]`

Shared identity: same `shared_email_hash` per account (same takeover infrastructure)

### 4.4 Smurfing (Structuring)

**Pattern:** Multiple cash transactions just below the reporting threshold (₹8,500–₹9,999) on the same account within a 2-day window.

| Parameter | Value |
|---|---|
| Number of groups | ~400 × (date_range / 121) |
| Transactions per group | 4–8 |
| Amount range | ₹8,500–₹9,999 |
| Window | 2 days |
| Channels | branch, atm |
| `cash_flag` | Always 1 |

**Structuring formula:**

```
amount ~ Uniform(8500, 9999)
Multiple transactions within [base_day, base_day + 2]
All routed to same high-risk beneficiary
```

Shared identity: same `shared_email_hash` per account

### 4.5 Identity Fraud

**Pattern:** Large transfers on very new accounts (opened < 60 days ago) in the early part of the simulation window.

| Parameter | Value |
|---|---|
| Count | up to 800 (limited by new account pool) |
| Max account age | 60 days |
| Amount | LogNormal(μ=11.0, σ=0.6) |
| Timing | Days 0–25% of window |

**Trigger condition:**
```
account_open_days < 60 AND amount > E(LogNormal(11, 0.6)) ≈ ₹60,496
```

Shared identity: same `shared_email_hash` + `shared_phone_hash` per customer (identity farm)

### 4.6 Dormant ATO

**Pattern:** After a dormancy period, an adversary takes over a reactivated dormant account using a foreign device.

| Parameter | Value |
|---|---|
| Targeted dormant accounts | 60% of dormant set |
| Transactions per account | 2–5 |
| Amount | LogNormal(μ=11.0, σ=0.7) |
| Timing | Second dormant window (55–100% of date range) |
| Device | Different from account's home device |

### 4.7 Dormant Smurfing

**Pattern:** Cash structuring on dormant accounts after reactivation.

| Parameter | Value |
|---|---|
| Targeted dormant accounts | 25% of dormant set |
| Transactions per group | 3–6 |
| Amount | ₹8,500–₹9,999 |
| Window | 3 days |
| `cash_flag` | Always 1 |

Shared identity: `shared_kyc_id` + `shared_phone_hash` per account (organised cell)

### 4.8 Dormant → Offshore

**Pattern:** Reactivated dormant accounts immediately wire to crypto/offshore beneficiaries.

| Parameter | Value |
|---|---|
| Targeted dormant accounts | 30% of dormant set |
| Transactions per account | 1–3 |
| Amount | LogNormal(μ=11.5, σ=0.5) |
| Exit pool | crypto + offshore beneficiaries only |

Shared identity: `shared_kyc_id` per account

### 4.9 Typology Summary Table

| Typology | Label | Device Signal | Amount Signal | Flow Signal | Identity Signal |
|---|---|---|---|---|---|
| Mule Ring | 1 | Shared device | High (μ=9.5) | flow_id, hop_number | shared_kyc_id |
| Layering | 1 | Normal | Decaying per hop | flow_id, hop_number, time_since_origin | shared_phone_hash |
| ATO | 1 | Foreign device | Very high (μ=10.5) | None | shared_email_hash |
| Smurfing | 1 | Normal | ₹8.5k–₹9.9k + cash | None | shared_email_hash |
| Identity Fraud | 1 | Normal | Very high (μ=11.0) | None | shared_email + phone |
| Dormant ATO | 1 | Foreign device | Very high (μ=11.0) | None | shared_email_hash |
| Dormant Smurfing | 1 | Normal | ₹8.5k–₹9.9k + cash | None | shared_kyc + phone |
| Dormant→Offshore | 1 | Normal | Very high (μ=11.5) | None | shared_kyc_id |

---

## 5. Feature Engineering

### 5.1 Entity Enrichment (FK Joins)

After merging legitimate and fraud transactions, three joins add entity attributes:

1. **Account join** — adds 16 columns: `avg_balance`, `account_open_days`, `kyc_level`, `country_risk`, `income_bracket`, `customer_risk_rating`, `pep_flag`, `occupation`, `industry`, `account_type`, `home_lat`, `home_lon`, `home_city`, `shared_kyc_id`, `shared_phone_hash`, `shared_email_hash`
2. **Device join** — adds: `device_age_days`, `rooted_flag`, `os_type`, `vpn_flag`, `emulator_flag`
3. **Beneficiary join** — adds: `beneficiary_type`, `beneficiary_country_risk`

### 5.2 IP Risk Score

Each transaction is assigned an IP address and a risk score based on four IP types and six additive signals.

**IP Type Assignment:**

```
If vpn_flag == 1:
    ip_type = 'risk'
Else:
    r ~ Uniform(0, 1)
    if r < 0.70:  ip_type = 'home'
    elif r < 0.90: ip_type = 'roam'
    else:          ip_type = 'risk'
```

**Home IP Format:** `10.{hash(customer+account) >> 8 & 0xFF}.{hash & 0xFF}.{hash(account) & 0xFF}`

**Risk IPs:** Pre-generated from 10 high-risk prefixes (Tor exit nodes, darknet relays).

**IP Risk Score Formula:**

```
ip_risk_score = clip(
    base_score
    + 0.15 × vpn_flag
    + 0.10 × rooted_flag
    + 0.10 × emulator_flag
    + 0.10 × is_night
    + 0.05 × (country_risk == 'high')
    − 0.05 × (kyc_level == 'high'),
    0.0, 1.0
)
```

| IP Type | Base Score | Typical Range |
|---|---|---|
| home | 0.05 | 0.00–0.20 |
| roam | 0.30 | 0.20–0.69 |
| risk | 0.80 | 0.70–1.00 |

### 5.3 Dormancy Flag

```python
dormancy_flag = 1 if sender_account_id in dormant_account_set else 0
```

5% of accounts are designated dormant at the start. This flag is set on all their transactions.

### 5.4 Temporal Features (Notebook 02b)

Extracted directly from `timestamp`:

| Feature | Formula | Description |
|---|---|---|
| `txn_hour` | `timestamp.hour` | Hour of transaction (0–23) |
| `txn_day_of_week` | `timestamp.dayofweek` | 0=Monday, 6=Sunday |
| `txn_day_of_month` | `timestamp.day` | Day of month (1–31) |
| `txn_month` | `timestamp.month` | Month (9–12) |
| `txn_year` | `timestamp.year` | Year |
| `txn_quarter` | `timestamp.quarter` | Quarter (3 or 4) |
| `is_weekend` | `dayofweek ∈ {5, 6}` | 1 if Saturday or Sunday |
| `is_night` | `hour ∈ {22,23,0,1,2,3,4,5}` | 1 if 22:00–05:59 |
| `is_business_hours` | `hour ∈ [9,17] AND is_weekend == 0` | 1 if 09:00–17:59 on weekday |
| `is_early_morning` | `hour ∈ [0, 8]` | 1 if 00:00–08:59 |
| `txn_date` | `timestamp.date` | Calendar date (used for daily balance resets) |

**Mutual exclusivity guarantee:** `is_night` and `is_business_hours` never overlap (validated via assertion).

### 5.5 Account-Level Event Log (Melted Format)

Each transaction is exploded into **two rows** — one from the sender perspective (debit) and one from the receiver perspective (credit):

```
sender_df:   account_id = sender_account_id,  debit_credit = 'debit',  signed_amount = -amount
receiver_df: account_id = receiver_account_id, debit_credit = 'credit', signed_amount = +amount
```

This produces `acct_events` with `2 × N_transactions` rows, enabling symmetric rolling computations for both sides of every transaction.

### 5.6 Account-Level Rolling Velocity Features

For each transaction row in `acct_events`, four windows are computed: **1h, 24h, 7d, 30d**.

**Window definition:** Inclusive on both ends — `[t - W, t]`

**Computed per window W ∈ {1h, 24h, 7d, 30d}:**

| Feature | Formula |
|---|---|
| `acct_txn_count_{W}` | `COUNT(events where timestamp ∈ [t−W, t])` |
| `acct_inflow_amt_{W}` | `SUM(amount where debit_credit='credit' AND timestamp ∈ [t−W, t])` |
| `acct_outflow_amt_{W}` | `SUM(amount where debit_credit='debit' AND timestamp ∈ [t−W, t])` |
| `acct_inflow_count_{W}` | `COUNT(events where debit_credit='credit' AND timestamp ∈ [t−W, t])` |
| `acct_outflow_count_{W}` | `COUNT(events where debit_credit='debit' AND timestamp ∈ [t−W, t])` |

Total account-level velocity columns per transaction: **5 metrics × 4 windows = 20 columns**, prefixed `sender_acct_*` and `receiver_acct_*` after the final merge.

**Implementation detail:** The algorithm iterates over each account group, uses numpy array comparisons for the time window mask, and pre-allocates output arrays for performance.

### 5.7 Customer-Level Rolling Velocity Features

A customer may own multiple accounts. Customer-level velocity aggregates activity **across all owned accounts**.

**Key concept:** `cust_id_for_rollup = account_owner_customer_id ?? customer_id` — this resolves both the sender's and receiver's customer identity.

**Computed per window W ∈ {1h, 24h, 7d, 30d}:**

| Feature | Formula |
|---|---|
| `cust_txn_count_{W}` | `COUNT(all events for this customer across all accounts, timestamp ∈ [t−W, t])` |
| `cust_inflow_amt_{W}` | `SUM(inflow amounts for this customer, timestamp ∈ [t−W, t])` |
| `cust_outflow_amt_{W}` | `SUM(outflow amounts for this customer, timestamp ∈ [t−W, t])` |
| `cust_inflow_count_{W}` | `COUNT(inflow events for this customer, timestamp ∈ [t−W, t])` |
| `cust_outflow_count_{W}` | `COUNT(outflow events for this customer, timestamp ∈ [t−W, t])` |

Total customer-level velocity columns: **5 metrics × 4 windows = 20 columns**, prefixed `sender_cust_*`.

### 5.8 Running Balance Features

Per account, a chronological running balance chain is maintained. The seed balance is computed as:

```
seed = max(
    avg_balance_from_accounts_table,
    max_single_outflow_for_account × 10,
    p95(all_transaction_amounts) × 10       ← GLOBAL_MIN_SEED
)
```

This ensures no account starts with a negative balance immediately.

**Per-transaction balance calculation:**

| Feature | Formula |
|---|---|
| `balance_before_txn` | Balance immediately before this transaction |
| `running_balance_txn_amount` | `+amount` (credit) or `−amount` (debit) |
| `balance_after_txn` | `balance_before_txn + running_balance_txn_amount` |
| `cumulative_daily_balance_change` | `balance_after_txn − day_start_balance` (resets at midnight) |
| `current_balance` | Last known `balance_after_txn` for this account |
| `bal_ratio_after_to_current` | `balance_after_txn / current_balance` (NaN if current_balance = 0) |

**Balance chain integrity check:** `balance_before[i+1] = balance_after[i]` for all consecutive rows (validated with tolerance 0.01).

After computation, all balance columns are split into `sender_*` and `receiver_*` prefixed versions and merged back to the transaction-level dataframe.

### 5.9 Flow Tracking Features

These columns are injected at typology time and verified/computed in Notebook 02b:

| Feature | Description |
|---|---|
| `synthetic_flow_id` | Flow identifier (e.g., `FLOW_00042`) — links all hops of a layering chain or mule ring |
| `flow_depth` | Total number of hops in the flow |
| `hop_number` | This transaction's position in the chain (1, 2, 3, ...) |
| `time_since_origin_ts` | Seconds elapsed since `hop_number = 1` transaction |

For rows where `time_since_origin_ts` is missing:
```
time_since_origin_ts = (timestamp - origin_timestamp).total_seconds()
```
where `origin_timestamp` = timestamp of the `hop_number == 1` transaction in the same flow.

---

## 6. Graph / Network Features

### 6.1 Graph Construction Logic

The graph is constructed implicitly through **rolling time-window aggregations** over two edge types:

- **Outflow edges:** `sender_account_id → {receiver_account_id | beneficiary_id}` — all transactions where this account sends money
- **Inflow edges:** `sender_account_id → receiver_account_id` — only for internal account-to-account transfers (no beneficiary)

A **counterparty** column unifies receiver account and beneficiary:

```python
_counterparty = receiver_account_id ?? str(beneficiary_id)
```

### 6.2 Sender-Side Graph Features (30-Day Window)

| Feature | Formula | Fraud Relevance |
|---|---|---|
| `sender_out_degree_30d` | `COUNT(outflow events in past 30d)` | High out-degree = fan-out pattern |
| `sender_total_outflow_30d` | `SUM(amount, outflow, 30d)` | Large total outflow vs. inflow |
| `sender_in_degree_30d` | `COUNT(inflow events in past 30d)` | Asymmetry: high out + low in = mule |
| `sender_total_inflow_30d` | `SUM(amount, inflow, 30d)` | Pass-through comparison |
| `sender_unique_counterparties_30d` | `NUNIQUE(_counterparty in past 30d)` | High unique = new payee explosion |
| `sender_repeat_counterparty_ratio` | `1 − (unique_counterparties / out_degree)` | Low ratio = always new payees (suspicious) |

### 6.3 Receiver-Side Graph Features (30-Day Window)

| Feature | Formula | Fraud Relevance |
|---|---|---|
| `receiver_in_degree_30d` | `COUNT(inflow events in past 30d)` | Many senders = money mule collector |
| `receiver_total_inflow_30d` | `SUM(amount, inflow, 30d)` | Total funds received |
| `receiver_unique_senders_30d` | `NUNIQUE(sender_account_id, past 30d)` | High = many depositors → aggregator |
| `receiver_account_outflow_30d` | `sender_out_degree_30d` for the receiver account | How quickly receiver disperses funds |

### 6.4 Pass-Through and Volume Ratio Features

These features detect accounts where funds flow in and are immediately forwarded — a core money laundering signal.

| Feature | Formula | Fraud Relevance |
|---|---|---|
| `inflow_outflow_volume_balance_ratio_24h` | `min(inflow_24h, outflow_24h) / inflow_24h` | Approaches 1.0 for pure pass-through accounts |
| `inflow_outflow_volume_balance_ratio_7d` | `min(inflow_7d, outflow_7d) / inflow_7d` | 7-day version of above |
| `outflow_to_inflow_ratio_7d` | `outflow_7d / inflow_7d` | >1.0 = spending more than received (fraud or structuring) |

**Interpretation:**
- `volume_balance_ratio ≈ 1.0` means all received funds are forwarded — characteristic of layering and mule accounts
- `outflow_to_inflow_ratio > 1.0` indicates the account is spending beyond its known inflows

### 6.5 Temporal Flow Feature

| Feature | Description | Formula |
|---|---|---|
| `avg_time_gap_in_out` | Rolling 24h average of seconds between receiving funds and sending them onwards | Rolling mean of `(next_outflow_ts − inflow_ts)` in seconds |

**Short `avg_time_gap_in_out`** (e.g., < 60 seconds) is a strong indicator of automated pass-through — the hallmark of mule and layering accounts in real-time payment networks.

**Implementation:** For each inflow event, the gap to the immediate next outflow for the same account is computed. These gaps are then rolled over a 24-hour window using pandas rolling mean. Merge to the main dataframe uses an as-of join (most recent rolling value ≤ transaction timestamp).

### 6.6 Device-Level Graph Features

| Feature | Formula | Fraud Relevance |
|---|---|---|
| `accounts_per_device` | `NUNIQUE(sender_account_id per device_id, 30d)` | >1 = device shared across accounts (mule ring signal) |
| `devices_per_account` | `NUNIQUE(device_id per account_id, 30d)` | >1 = account used from multiple devices (ATO signal) |
| `device_shared_high_risk_ratio` | `COUNT(fraud OR high_risk_bene, 30d) / COUNT(all txns, 30d)` | High ratio = device implicated in known fraud |
| `shared_device_fraud_count` | `SUM(label, 30d per device)` | Total confirmed fraud transactions on this device |

**Double-counting prevention:** `device_shared_high_risk_ratio` uses a union flag:

```python
_fraud_or_high_risk = (label == 1) OR (high_risk_beneficiary == 1)
```

This ensures a transaction that is both fraudulent and high-risk only counts once in the numerator.

### 6.7 Graph Feature Fraud Detection Relationships

```
Mule Ring Detection:
  accounts_per_device > 1     → multiple accounts share a device
  sender_repeat_counterparty_ratio ≈ 0  → always routing to same exit

Layering Detection:
  volume_balance_ratio ≈ 1.0  → all inflows forwarded
  avg_time_gap_in_out < 300s  → near-instant forwarding
  hop_number > 1              → mid-chain position

ATO Detection:
  devices_per_account > 1     → new device for this account
  shared_device_fraud_count > 0 → device previously flagged

Smurfing Detection:
  sender_out_degree_30d high  → many outflows
  sender_unique_counterparties_30d low → same beneficiary repeatedly
```

---

## 7. Rule-Based Features & Fraud Intensity Score

### 7.1 AML Rule Engine (50 Rules)

Rules are defined as `(name, severity, condition_lambda)` triples. Severity ∈ {1, 2, 3}. Each rule fires as a binary column `rule_{name}` = 0 or 1.

**Rule Categories and Counts:**

| Category | Count | Severity Range | Example Rules |
|---|---|---|---|
| [A] Amount & Cash | 5 | 1–3 | large_cash_deposit, cash_just_below_threshold, high_value_transfer |
| [B] Temporal | 4 | 2–3 | night_transaction, weekend_high_value, dormant_account_activation |
| [C] Account Velocity | 7 | 2–3 | high_acct_velocity_1hr, high_acct_volume_24hr, amount_spike_30d |
| [D] Customer Velocity | 4 | 2–3 | high_cust_velocity_1hr, high_cust_volume_30d |
| [E] Balance & KYC | 3 | 2–3 | low_kyc_high_amount, new_account_large_txn, high_amount_to_balance |
| [F] KYC / Risk Profile | 5 | 2–3 | very_high_risk_customer, pep_high_value, high_risk_country_sender |
| [G] Beneficiary Risk | 4 | 2–3 | high_risk_beneficiary, crypto_transfer, offshore_transfer |
| [H] Device / IP | 4 | 1–3 | rooted_device, vpn_proxy_detected, emulator_detected |
| [I] Channel | 2 | 2 | atm_high_withdrawal, branch_night_txn |
| [J] Cash Structuring | 2 | 3 | structuring_pattern, multiple_small_cash |
| [K] Occupation / Industry | 4 | 1–2 | student_high_value, unemployed_large_transfer, freelancer_offshore |
| [L] Combined Multi-Signal | 6 | 2–3 | pep_crypto_transfer, vpn_offshore, emulator_crypto, new_account_offshore |

**Key Rule Thresholds:**

| Rule | Condition |
|---|---|
| `large_cash_deposit` | `cash_flag=1 AND amount > 50,000` |
| `cash_just_below_threshold` | `cash_flag=1 AND 8,000 ≤ amount ≤ 9,999` |
| `high_value_transfer` | `amount > 100,000` |
| `dormant_account_activation` | `dormancy_flag=1 AND amount > 5,000` |
| `high_acct_velocity_1hr` | `txn_count_last_1hr > 5` |
| `structuring_pattern` | `cash_flag=1 AND 8,500 ≤ amount ≤ 9,999` |
| `very_high_risk_customer` | `customer_risk_rating = 'very_high'` |
| `emulator_crypto` | `emulator_flag=1 AND beneficiary_type='crypto'` |
| `new_account_offshore` | `account_open_days < 90 AND beneficiary_type='offshore'` |

**Aggregate rule columns:**

```python
rule_trigger_count  = SUM(all rule columns)              # total rules fired
max_rule_severity   = MAX(rule_col × severity)           # highest severity fired
weighted_rule_score = SUM(rule_col × severity)           # sum of all severities fired
```

### 7.2 Fraud Intensity Score (FIS)

The FIS is a composite [0–100] risk score that aggregates rule, behavioural, customer, IP, and device signals.

**Component Definitions:**

| Component | Weight | Formula |
|---|---|---|
| Rule Risk | ×30 | `weighted_rule_score / max(weighted_rule_score in dataset)` |
| Behaviour Risk | ×25 | `(velocity_ratio + amount_spike_ratio) / 2` |
| Customer Risk | ×15 | `(rating_num/max_rating)×0.5 + pep_flag×0.3 + ((max_kyc−kyc_num)/max_kyc)×0.2` |
| IP Risk | ×15 | `ip_risk_score.clip(0, 1)` |
| Device Risk | ×15 | `(vpn_flag + rooted_flag + emulator_flag) / 3` |

**Sub-formula: Velocity Ratio**

```
velocity_ratio = clip(txn_count_1h / (txn_count_30d + 1), 0, 5) / 5
```

**Sub-formula: Amount Spike Ratio**

```
daily_avg_outflow = (outflow_amt_30d / 30) + 1
amount_spike_ratio = clip(amount / daily_avg_outflow, 0, 5) / 5
```

**Sub-formula: Customer Risk**

```
rating_map:  low=1, medium=2, high=3, very_high=4
kyc_map:     low=1, medium=2, high=3

customer_risk = clip(
    (rating_num / max_rating) × 0.5
    + pep_flag × 0.3
    + ((max_kyc − kyc_num) / max_kyc) × 0.2,
    0, 1
)
```

Note: High KYC level **reduces** customer risk (negative contribution from KYC gap term).

**FIS Raw Score:**

```
fis_raw = rule_risk×30 + behaviour_risk×25 + customer_risk×15 + ip_risk×15 + device_risk×15
```

**FIS Final Score (P99 Scaling):**

```
p99 = fis_raw.quantile(0.99)
fis_score = clip(fis_raw / p99, 0, 1) × 100
```

**FIS Band Assignment:**

| Band | Score Range |
|---|---|
| `very_low` | (0, 20] |
| `low` | (20, 40] |
| `medium` | (40, 60] |
| `high` | (60, 80] |
| `critical` | (80, 100] |

---

## 8. Data Quality & QC Checks

### 8.1 Entity Table Validations (Notebook 01)

| Check | Assertion |
|---|---|
| Customer PK uniqueness | `customers['customer_id'].is_unique` |
| Account PK uniqueness | `accounts['account_id'].is_unique` |
| Device PK uniqueness | `devices['device_id'].is_unique` |
| Beneficiary PK uniqueness | `beneficiaries['beneficiary_id'].is_unique` |
| Account FK integrity | `accounts['customer_id'].isin(customers['customer_id']).all()` |
| No null PKs | `customers['customer_id'].notna().all()` |
| Correct row counts | `len(customers) == NUM_CUSTOMERS` |
| Shared identity non-empty | `customers['shared_kyc_id'].notna().sum() > 0` |
| Beneficiary type validity | `beneficiaries['beneficiary_type'].isin(BENE_TYPES).all()` |

### 8.2 Transaction Generation Validations (Notebook 02a)

| Check | Assertion |
|---|---|
| 8 typologies present | `fraud_df['fraud_type'].nunique() == 8` |
| Smurfing is always cash | `fraud_df[smurf_mask]['cash_flag'].eq(1).all()` |
| Post-rebalancing fraud rate | `txns['label'].mean() ≈ TARGET_FRAUD_RATE (0.10)` |

### 8.3 Temporal Feature Validations (Notebook 02b — Test Cases)

| Test | Check |
|---|---|
| TEST 1: Hours in range | `txn_hour.between(0, 23).all()` |
| TEST 1: Night flag consistency | Night rows have `hour ∈ {22,23,0,1,2,3,4,5}` |
| TEST 1: Business hours on weekdays only | `is_business_hours=1` implies `is_weekend=0` |
| TEST 1: No overlap | `(is_night=1 AND is_business_hours=1).sum() == 0` |
| TEST 2: Velocity manual check | 24h account count manually recomputed for a sample row |
| TEST 2b: Inflow/Outflow amounts | Manual sum matches feature value (tolerance 0.01) |
| TEST 4: Cross-account customer velocity | Customer 24h count validated for multi-account customers |
| TEST 5: Window inclusivity | Every account's first transaction counts itself (`acct_txn_count_1h ≥ 1`) |

### 8.4 Balance Chain Integrity (Notebook 02b)

```python
breaks = sum(
    abs(balance_after[i] - balance_before[i+1]) > 0.01
    for i in range(len(account_slice) - 1)
)
# Expected: 0 breaks
```

### 8.5 Final Column Preservation (Notebook 02b)

After feature merges, all 44 original input columns are verified to be present:

```python
missing_cols = [c for c in ORIGINAL_INPUT_COLS if c not in final_df.columns]
# Expected: missing_cols = []
```

---

## 9. Code Flow — Step-by-Step

### Step 1 — `01_customer_generator.ipynb`

| Cell | Action | Dataset State |
|---|---|---|
| C1 | Import libraries, create `./outputs/` directory | — |
| C2 | Define all configuration constants | — |
| C3 | Build `FEATURE_CATALOGUE` and save as CSV | — |
| C4 | Generate `customers` DataFrame (3,000 rows) including shared identity assignment | `customers`: 3,000 × 14 |
| C5 | Print distribution summary | — |
| C6 | Generate `accounts` DataFrame (4,500 rows) and denormalise customer attributes | `accounts`: 4,500 × 18 |
| C7 | Generate `devices` DataFrame (4,000 rows), build `account_device_map` | `devices`: 4,000 × 6 |
| C9 | Generate `beneficiaries` DataFrame (4,000 rows) with shell company attributes | `beneficiaries`: 4,000 × 7 |
| C10 | PK/FK integrity validation report | — |
| C15 | Save all four tables to CSV | 4 CSV files written |

### Step 2 — `02a_merging_typologies.ipynb`

| Cell | Action | Dataset State |
|---|---|---|
| A-1 | Import libraries | — |
| A-2 | Define simulation configuration (dates, channels, fraud volumes, IP config) | — |
| A-3 | Load reference tables, build `account_device_map`, `account_beneficiaries_map`, `high_risk_bene_pool`, `dormant_account_set` | Reference structures ready |
| A-4 | Define transaction builder utilities (`build_row`, `random_ts`, `weighted_hour`) | — |
| A-5 | Generate legitimate transactions (Poisson + dormant pattern) | `legit_df`: ~225,000 rows |
| A-6 | Inject 8 fraud typologies | `fraud_df`: ~50,000 rows |
| A-7a | Concatenate + oversample to 10% fraud rate | `combined`: ~275,000 rows |
| A-7b | FK joins: accounts, devices, beneficiaries | +22 columns |
| A-8 | Compute IP addresses, geo-coordinates, and `ip_risk_score` | +4 columns |
| A-9 | Overwrite shared identity columns for fraud rows by typology | Shared identity enriched |
| C15 | Add `dormancy_flag` | +1 column |
| A-10 | Save `txns_with_typologies.parquet` | **Stage 1 output** |

### Step 3 — `02b_rolling_features.ipynb`

| Cell | Action | Dataset State |
|---|---|---|
| B-3 | Load parquet, sort by timestamp globally | `df`: Stage 1 |
| Sec 1 | Compute temporal decomposition features | +10 columns |
| Sec 2 | Build `acct_events` long-format event log (2× rows) | `acct_events`: 2N rows |
| Sec 3 | Compute account-level rolling velocity (all 4 windows) | +20 columns on `acct_events` |
| Sec 4 | Compute customer-level rolling velocity (all 4 windows) | +20 columns on `acct_events` |
| Sec 5 | Compute running balance per account | +6 columns on `acct_events`, split to sender/receiver |
| Sec 6 | Verify/derive `time_since_origin_ts` for flow transactions | `time_since_origin_ts` finalised |
| Sec 7 | Merge all sender_ and receiver_ feature columns back to transaction-level `final_df` | `final_df`: Stage 2 |
| Tests | Run 5 validation test cases | Assertions pass |
| C27 | Save `transactions_enriched.parquet` | **Stage 2 output** |

### Step 4 — `02c_rule_based_features.ipynb`

| Cell | Action | Dataset State |
|---|---|---|
| C-3 | Load `transactions_enriched.parquet` | `result` loaded |
| C-4 | Apply 50 AML rules row-by-row; compute `rule_trigger_count`, `max_rule_severity`, `weighted_rule_score` | +53 columns |
| C-6 | Compute FIS components, raw score, P99-scaled score, and band | +6 columns (`fis_raw`, `fis_score`, `fis_band`, numeric maps, risk components) |
| C-9 | Print final table summary | — |
| C-11 | Save `stg_transactions.parquet` | **Stage 3 output** |

### Step 5 — `03_graph_feature_generator.ipynb`

| Cell | Action | Dataset State |
|---|---|---|
| Cell 1 | Load config, define input/output paths | — |
| Cell 2 | Load `stg_transactions.parquet`, sort by timestamp | `df` loaded |
| Cell 3 | Build `inflow` and `outflow` edge tables | Two edge DataFrames |
| Cell 4 | Compute sender-side rolling 30d features (out-degree, outflow, unique counterparties, devices_per_account, repeat ratio) | Intermediate tables |
| Cell 5 | Compute receiver-side rolling 30d features (in-degree, inflow, unique senders) | Intermediate table |
| Cell 6 | Compute 24h and 7d inflow/outflow for ratio features | Intermediate tables |
| Cell 7 | Compute device-level rolling features (accounts_per_device, shared_high_risk_ratio, fraud_count) | Intermediate table |
| Cell 8 | Compute rolling `avg_time_gap_in_out` (24h window) | Intermediate table |
| Cell 10 | Merge all features to `df` (Steps 1–8 with memory cleanup) | +18 graph feature columns |
| Cell 11 | Drop helper columns, apply feature filter | Clean columns |
| Cell 12 | Save `stg_transactions_graph.parquet` | **Final output** |

---

## 10. Final Dataset Structure

### 10.1 Approximate Scale

| Metric | Value |
|---|---|
| Total rows | ~275,000 transactions |
| Fraud rows (label=1) | ~27,500 (10%) |
| Legitimate rows (label=0) | ~247,500 (90%) |
| Total columns (final) | ~160+ |

### 10.2 Feature Groups

| Group | Prefix / Pattern | Count | Description |
|---|---|---|---|
| Transaction identifiers | `transaction_id`, `customer_id`, etc. | 6 | PKs and FKs |
| Transaction attributes | `amount`, `channel`, `transaction_type`, etc. | 7 | Core transaction fields |
| Flow / typology | `synthetic_flow_id`, `flow_depth`, `hop_number`, `time_since_origin_ts`, `fraud_type`, `label` | 6 | Injected labels and chain metadata |
| Customer/Account profile | `avg_balance`, `account_open_days`, `kyc_level`, `customer_risk_rating`, etc. | 16 | Denormalised entity attributes |
| Device attributes | `device_age_days`, `rooted_flag`, `vpn_flag`, `emulator_flag`, `os_type` | 5 | Device risk signals |
| Beneficiary attributes | `beneficiary_type`, `beneficiary_country_risk` | 2 | Exit node risk |
| IP / Geo | `ip_address`, `ip_risk_score`, `geo_lat`, `geo_lon` | 4 | Network location signals |
| Shared identity | `shared_kyc_id`, `shared_phone_hash`, `shared_email_hash` | 3 | Identity collision signals |
| Temporal | `txn_hour`, `txn_day_of_week`, `is_weekend`, `is_night`, etc. | 10 | Time-based features |
| Sender account velocity | `sender_acct_{metric}_{window}` | 20 | 5 metrics × 4 windows |
| Sender customer velocity | `sender_cust_{metric}_{window}` | 20 | 5 metrics × 4 windows |
| Receiver account velocity | `receiver_acct_{metric}_{window}` | 20 | 5 metrics × 4 windows |
| Sender balance | `sender_balance_*`, `sender_running_balance_*` | 6 | Running balance signals |
| Receiver balance | `receiver_balance_*`, `receiver_running_balance_*` | 6 | Receiver-side balance |
| AML Rules | `rule_{name}` | 50 | Binary rule triggers |
| Rule aggregates | `rule_trigger_count`, `max_rule_severity`, `weighted_rule_score` | 3 | Rule summary metrics |
| FIS | `fraud_intensity_score_raw`, `fraud_intensity_score`, `fis_band`, `customer_risk_rating_num`, `kyc_level_num` | 5 | Fraud Intensity Score |
| Graph features | `sender_in_degree_30d`, `receiver_in_degree_30d`, `accounts_per_device`, etc. | 18 | Network topology metrics |

---

## 11. Mathematical Formulas Reference

### 11.1 Amount Simulation

```
amount ~ LogNormal(μ_kyc, σ_kyc)

μ_low    = 7.0,  σ = 1.0  →  median ≈ ₹1,097
μ_medium = 8.0,  σ = 1.0  →  median ≈ ₹2,981
μ_high   = 9.5,  σ = 1.0  →  median ≈ ₹13,360
```

### 11.2 Rolling Window Formula (Inclusive)

```
For transaction at time t, window W:
    t_start = t − W
    result  = SUM/COUNT( events where t_start ≤ event.timestamp ≤ t )
```

### 11.3 Account Velocity Metrics

```
acct_txn_count_{W}     = |{ e ∈ A : t−W ≤ e.ts ≤ t }|
acct_inflow_amt_{W}    = Σ e.amount  for e ∈ A, e.type='credit', t−W ≤ e.ts ≤ t
acct_outflow_amt_{W}   = Σ e.amount  for e ∈ A, e.type='debit',  t−W ≤ e.ts ≤ t
acct_inflow_count_{W}  = |{ e ∈ A : e.type='credit', t−W ≤ e.ts ≤ t }|
acct_outflow_count_{W} = |{ e ∈ A : e.type='debit',  t−W ≤ e.ts ≤ t }|
```

### 11.4 Balance Update Chain

```
seed_balance    = max(avg_balance, max_outflow × 10, p95_all_amounts × 10)
signed_amount   = +amount  if credit
                  −amount  if debit
balance_after   = balance_before + signed_amount
cum_daily_delta = balance_after − balance_at_day_start
bal_ratio       = balance_after / current_balance   (0 if current_balance = 0)
```

### 11.5 IP Risk Score

```
ip_risk_score = clip(
    base_score(ip_type)
    + 0.15·vpn + 0.10·rooted + 0.10·emulator
    + 0.10·is_night + 0.05·[country_risk='high']
    − 0.05·[kyc='high'],
    0.0, 1.0
)

base_score: home=0.05, roam=0.30, risk=0.80
```

### 11.6 Graph Pass-Through Ratios

```
volume_balance_ratio_{W} = min(inflow_{W}, outflow_{W}) / inflow_{W}
outflow_to_inflow_{W}    = outflow_{W} / inflow_{W}

Interpretation:
  volume_balance_ratio → 1.0  : pure pass-through (all inflows forwarded)
  outflow_to_inflow   >  1.0  : spending exceeds known inflows
```

### 11.7 Graph Repeat Counterparty Ratio

```
repeat_counterparty_ratio = 1 − (unique_counterparties_30d / out_degree_30d)

Range: [0, 1]
  → 0.0 : every transaction goes to a new counterparty (new payee explosion)
  → 1.0 : all transactions go to the same counterparty (concentrated exit)
```

### 11.8 Device High-Risk Ratio

```
device_shared_high_risk_ratio =
    COUNT(transactions where (label=1 OR high_risk_beneficiary=1), 30d per device)
    ─────────────────────────────────────────────────────────────────────────────
    COUNT(all transactions, 30d per device)
```

### 11.9 Fraud Intensity Score

```
velocity_ratio      = clip(txn_count_1h / (txn_count_30d + 1), 0, 5) / 5
amount_spike_ratio  = clip(amount / (outflow_30d/30 + 1),       0, 5) / 5
behaviour_risk      = (velocity_ratio + amount_spike_ratio) / 2

customer_risk       = clip(
    (rating_num / max_rating) × 0.50
    + pep_flag × 0.30
    + ((max_kyc − kyc_num) / max_kyc) × 0.20,
    0, 1)

fis_raw   = rule_risk×30 + behaviour_risk×25 + customer_risk×15
           + ip_risk×15  + device_risk×15

fis_score = clip(fis_raw / P99(fis_raw), 0, 1) × 100
```

### 11.10 Weighted Rule Score

```
weighted_rule_score = Σ (rule_i_fired × severity_i)   for i = 1..50

severity ∈ {1, 2, 3}:
  1 = Low priority (informational)
  2 = Medium priority (elevated risk)
  3 = High priority (strong fraud indicator)
```

### 11.11 Layering Amount Decay

```
amount_hop_i ~ LogNormal(μ_base − i × 0.2,  σ = 0.8)

hop 1: μ = 9.0  → median ≈ ₹8,103
hop 2: μ = 8.8  → median ≈ ₹6,634
hop 3: μ = 8.6  → median ≈ ₹5,431
```

---

## 12. Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SmartSentry AML Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│  NOTEBOOK 01: Entity Generation           │
│                                           │
│  customers.csv    (3,000 rows)            │
│  accounts.csv     (4,500 rows)            │  ← Denormalised customer attrs
│  devices.csv      (4,000 rows)            │
│  beneficiaries.csv(4,000 rows)            │  ← Shell company attrs
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  NOTEBOOK 02a: Transactions + Typologies  │
│                                           │
│  ① Legitimate txns (Poisson + dormant)    │
│     ~225k rows, label=0                  │
│                                           │
│  ② Fraud injection (8 typologies)         │
│     Mule Ring, Layering, ATO,            │
│     Smurfing, Identity Fraud,            │
│     Dormant ATO/Smurfing/Offshore        │
│     label=1                              │
│                                           │
│  ③ Oversample → 10% fraud rate           │
│  ④ FK joins (accounts, devices, benes)   │
│  ⑤ IP address + ip_risk_score            │
│  ⑥ Shared identity enrichment           │
│  ⑦ Dormancy flag                        │
│                                           │
│  → txns_with_typologies.parquet          │
│    (~275k rows × ~45 cols)               │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  NOTEBOOK 02b: Feature Engineering       │
│                                           │
│  ① Temporal features (10 cols)           │
│  ② Melt → account event log (2N rows)    │
│  ③ Account rolling velocity              │
│     (1h/24h/7d/30d → 20 cols)           │
│  ④ Customer rolling velocity             │
│     (1h/24h/7d/30d → 20 cols)           │
│  ⑤ Running balance chain (6 cols)        │
│  ⑥ Flow tracking (time_since_origin_ts)  │
│  ⑦ Merge sender_ + receiver_ features   │
│                                           │
│  → transactions_enriched.parquet         │
│    (~275k rows × ~120 cols)              │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  NOTEBOOK 02c: Rules + FIS               │
│                                           │
│  ① 50 AML rule engine                    │
│     rule_{name} (50 binary cols)         │
│     rule_trigger_count                   │
│     max_rule_severity                    │
│     weighted_rule_score                  │
│                                           │
│  ② Fraud Intensity Score                 │
│     5 components × weights (30/25/15/15/15)│
│     P99-scaled [0–100]                   │
│     fis_band (5 tiers)                   │
│                                           │
│  → stg_transactions.parquet              │
│    (~275k rows × ~130 cols)              │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│  NOTEBOOK 03: Graph Feature Generator    │
│                                           │
│  ① Build inflow + outflow edge tables    │
│  ② Sender-side 30d rolling graph:        │
│     in/out degree, inflow/outflow sums,  │
│     unique counterparties, repeat ratio  │
│  ③ Receiver-side 30d rolling graph:      │
│     in-degree, unique senders,           │
│     receiver outflow                     │
│  ④ Volume balance ratios (24h / 7d)     │
│  ⑤ Outflow-to-inflow ratio (7d)         │
│  ⑥ avg_time_gap_in_out (rolling 24h)   │
│  ⑦ Device-level: accounts_per_device,   │
│     devices_per_account,                 │
│     shared_high_risk_ratio,              │
│     shared_device_fraud_count            │
│                                           │
│  → stg_transactions_graph.parquet       │
│    (~275k rows × ~160+ cols)            │
└───────────────────────────────────────────┘

Data Flow Summary:
  customers/accounts/devices/beneficiaries
           │
           ▼
  raw transactions (label 0/1)
           │
           ▼
  + IP layer + shared identity + dormancy
           │
           ▼
  + temporal + velocity + balance + flow
           │
           ▼
  + 50 AML rules + FIS score [0–100]
           │
           ▼
  + graph / network topology features
           │
           ▼
  FINAL: stg_transactions_graph.parquet
         (~275k rows × 160+ features)
         Ready for AML model training
```

---

*Documentation generated from source notebooks: `01_customer_generator.ipynb`, `02a_merging_typologies.ipynb`, `02b_rolling_features.ipynb`, `02c_rule_based_features.ipynb`, `03_graph_feature_generator.ipynb`*

*SmartSentry AML — Synthetic Data Pipeline V7*
