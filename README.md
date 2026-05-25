# SmartSentry AML — Anti-Money Laundering Detection Framework

An end-to-end machine learning pipeline for detecting and classifying money laundering transactions across 10 AML typologies, built for Indian banking compliance (RBI/FIU-IND/PMLA).

## Overview

SmartSentry AML is a two-phase detection system designed to identify suspicious financial transactions and classify them into specific money laundering patterns. The framework addresses one of the most critical challenges in banking compliance: distinguishing genuine criminal financial activity from normal business operations across millions of daily transactions.

The system operates in two sequential phases. **Phase 1** performs binary classification — answering the fundamental question "Is this transaction suspicious?" — using a LightGBM gradient boosting model that achieves an AUC of approximately 0.94. Transactions flagged by Phase 1 are then passed to **Phase 2**, which performs **multi-label typology classification** — answering "What TYPE(S) of money laundering is this?" — across 10 distinct AML patterns. Phase 2 reports both a primary typology (the highest-probability class, ~76% accuracy) and the full set of typologies whose probability exceeds a configurable threshold (default 0.30, ~82% recall). The multi-label output reflects real-world AML patterns where a single transaction may legitimately fit multiple money-laundering signatures.

The complete pipeline spans synthetic data generation for model development, graph-based pattern detection for labeling, a 126-rule regulatory compliance engine aligned with RBI and FIU-IND requirements, comprehensive feature engineering covering transaction velocity, account balance behavior, and device/IP risk signals, and finally the two-phase ML modeling framework with hyperparameter tuning, class imbalance handling, and production-ready output generation.

### AML Typologies Covered

| # | Typology | Detection Method |
|---|---|---|
| 1 | Structuring (Smurfing) | Sub-threshold cash deposits (₹8K–₹10K) |
| 2 | Circular Transaction Loop | Closed-loop A→B→C→A transfers |
| 3 | Funnel Account Network | Many-to-one fund aggregation |
| 4 | Pass-Through Transit Hub | Immediate receive-and-forward |
| 5 | Rapid Multi-Hop Layering | Sequential 8–10 hop chains |
| 6 | Third-Party Payment Web | Unrelated payer convergence |
| 7 | Money Mule Network | Controller→mule→collector star pattern |
| 8 | High-Risk Corridor Transfer | Cross-border to FATF jurisdictions |
| 9 | Underground Banking (Hawala) | Matched bilateral settlements |
| 10 | Charity Abuse | Donation diversion from NPO accounts |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SmartSentry AML Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [00] Data Generator ──→ [01] Typology Detector ──→ [02] Rules Engine  │
│       (Synthetic)              (Graph-based)            (126 rules)     │
│                                                                         │
│                         ──→ [03] Feature Engineering                    │
│                              (Velocity / Balance / IP)                  │
│                                                                         │
│                         ──→ [04] Phase 1: Binary AML Detection          │
│                              (LightGBM, SMOTE, threshold tuning)       │
│                                                                         │
│                         ──→ [05] Phase 2: Typology Classification       │
│                              (LightGBM multiclass, 10 typologies)      │
│                                                                         │
│  Output: fraud_risk_score (0-100) + typology probabilities + priority  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Notebook | Input | Output |
|---|---|---|---|
| **Data Generation** | `aml_generator_complete_pipeline.ipynb` | CONFIG parameters | `transactions_generated_typology_V2.parquet` |
| **Typology Detection** | `01__aml_typology_detector.ipynb` | Raw transactions | `stg_transactions_flagged.parquet` |
| **Rules Engine** | `02__aml_rules_engine.ipynb` | Flagged transactions | `stg_transactions_rules_V3.parquet` |
| **Feature Engineering** | `03__aml_feature_engineering.ipynb` | Rules output | `stg_transactions_features_V2.parquet` |
| **Phase 1 Model** | `04__aml_ml_preparation.ipynb` | Features output | Model + predictions + artifacts |
| **Phase 2 Model** | `05__aml_phase2_typology_classifier.ipynb` | Phase 1 artifacts | Typology probabilities + production tables |

---

## Module Descriptions

### Module 0: Synthetic Data Generator (`aml_generator_complete_pipeline.ipynb`)

The data generator is the foundation of the entire framework. It creates a realistic synthetic banking dataset that mirrors the statistical properties of actual Indian bank transaction data, enabling model development and testing without requiring access to sensitive customer information.

The generator operates in two phases. In **Phase 1 (Clean Transaction Generation)**, it creates approximately 300,000 normal transactions across 3,400 customers (3,000 individuals and 400 entities) with realistic income-proportional spending patterns. Each customer is assigned a profile including occupation, income band, risk score, PEP status, and device fingerprints. Transactions are generated with proper channel distributions (UPI, NEFT, RTGS, IMPS, Branch Cash, ATM), directional balance tracking (Dr/Cr), and temporal patterns that reflect genuine banking behavior. Clean transactions are specifically designed so that only 50-60% trigger any compliance rules, ensuring the ML model has sufficient negative examples to learn from.

In **Phase 2 (AML Typology Injection)**, the generator injects money laundering patterns into the dataset. Each AML typology has its own dedicated account pool — accounts are pre-assigned to exactly one typology to prevent multi-label contamination. The generator creates complete scenarios for each of the 10 typologies with configurable parameters: ring sizes for Circular loops, feeder counts for Funnel networks, hop counts for Layering chains, donor counts for Charity abuse, and so on. The target fraud rate is approximately 20-22%, resulting in roughly 88,000 AML transactions with known ground truth labels.

The generator also includes a PPI (Prepaid Payment Instrument) scenario injection module that creates wallet-specific rule-triggering patterns such as high-risk MCC concentration, merchant refund abuse, multi-wallet same-PAN scenarios, and UPI carousel patterns. These PPI scenarios are marked as `is_aml=0` (not money laundering) but are designed to trigger specific PPI compliance rules in the rules engine.

### Module 1: Typology Detector (`01__aml_typology_detector.ipynb`)

The typology detector is a graph-based pattern recognition engine that analyzes the transaction dataset to independently identify AML patterns. This module is critical because in a production deployment, there are no creator labels — the detector must find suspicious patterns purely from transaction data.

The detector begins by constructing a directed transaction graph where each account is a node and each transaction is a weighted, timestamped edge. It builds two edge indices: `edges_out` (outflows from each account) and `edges_in` (inflows to each account), both sorted chronologically. This graph representation enables efficient traversal for pattern matching.

Ten independent detection algorithms then scan the graph, each looking for a specific typology pattern. The **Structuring detector** searches for clusters of cash deposits just below the ₹10,000 reporting threshold followed by consolidation transfers. The **Circular Loop detector** uses depth-first search to find closed paths (A→B→C→A) with amount tolerance for per-hop decay. The **Funnel detector** identifies accounts receiving from 15+ unique senders within a time window followed by concentrated outflows. The **Layering detector** traces sequential multi-hop chains where each hop's amount is within a decay tolerance of the previous hop — a critical fix from earlier versions that compared against the start amount. The **Money Mule detector** finds star patterns where a controller sends to multiple mules who then forward to concentrated collectors. The **Hawala detector** looks for matched bilateral settlement patterns among 3-4 parties.

Each detector is parameterized via `DETECT_CONFIG`, which contains thresholds mathematically derived from the creator's generation parameters with small buffers. After all detectors run, a priority-based dedup step ensures each transaction is assigned to exactly one typology — the most specific pattern wins (Structuring > Hawala > Corridor > Pass-Through > Circular > Funnel > Mule > Third-Party > Charity > Layering).

### Module 2: Rules Engine (`02__aml_rules_engine.ipynb`)

The rules engine implements 126 compliance rules derived from RBI Master Directions, PMLA 2002 requirements, FIU-IND reporting guidelines, and RBI PPI norms. Unlike the ML model which learns patterns from data, the rules engine encodes explicit regulatory knowledge — specific thresholds, combinations, and scenarios that regulators require banks to monitor.

The engine operates in four stages. **Column Resolution** maps the raw transaction columns to internal working names (e.g., `customer_account_number` → `_acct`, `transaction_amount` → `_amt`). **Derived Feature Computation** creates 22 aggregated features that rules need but don't exist in the raw data — cumulative cash amounts over 30 days, distinct channels per day, unique depositors per week, device novelty flags, high-risk MCC percentages, name similarity scores, and PPI wallet aggregations. **Rule Execution** applies all 126 rules as vectorized pandas operations, each producing a binary column (`rule_<name>` = 0 or 1). **Composite Scoring** computes a weighted `rule_score` using severity-based weights (High=3, Medium=2, Low=1) and assigns alert levels (Critical, High, Medium, Low).

The 126 rules are organized into 13 functional groups covering frequency anomalies, cumulative threshold proximity, credit-debit sequences, account lifecycle events, direct regulatory flags, large transaction detection, new account monitoring, dormant reactivation, entity-specific checks, intrabank patterns, device/digital signals, occupation-based profiling, and PPI wallet-specific rules. Each rule has a severity rating calibrated from a bank operations perspective — for example, `rule_structuring_pattern` is severity 3 (High) because it directly indicates CTR avoidance, while `rule_attempted_failed` is severity 1 (Low) because failed transactions are overwhelmingly caused by network issues.

### Module 3: Feature Engineering (`03__aml_feature_engineering.ipynb`)

The feature engineering module transforms the rule-enriched transaction data into a comprehensive feature matrix suitable for machine learning. While the rules engine captures explicit regulatory knowledge, the feature engineering module captures implicit behavioral patterns that distinguish money launderers from legitimate customers.

The module computes four categories of features across multiple time windows. **Sender Account Velocity** features capture transaction frequency and volume from the sender's account across 1-hour, 24-hour, 7-day, and 30-day windows — including transaction counts, total amounts, unique counterparties, and inflow/outflow amounts. **Receiver Account Velocity** features mirror the same calculations from the receiver's perspective, capturing inflow patterns, unique sender counts, and balance changes. **Balance Tracking** features compute the sender's running balance, percentage of balance moved per transaction, and cumulative daily balance change — critical for detecting pass-through and drain patterns. **IP Risk** features derive a composite risk score from VPN detection, emulator detection, cross-border flags, night-time activity, and shared IP indicators.

Additionally, the module computes **volume balance ratios** at 24-hour and 7-day windows (inflow-to-outflow ratios that reveal pass-through behavior) and a **Fraud Insight Score (FIS)** using a 4-component formula: `rule_score × 20 + behaviour_score × 55 + ip_risk × 15 + device_risk × 10`. The FIS was originally intended as the prediction target but analysis proved it cannot separate AML from clean transactions (1.0x ratio between AML and clean FIS distributions), confirming that the ML model using `is_aml` as the target is the correct approach.

### Module 4: Phase 1 — Binary AML Detection (`04__aml_ml_preparation.ipynb`)

Phase 1 is the core detection model that answers the binary question: "Is this transaction suspicious?" It takes the feature-engineered dataset and trains a LightGBM gradient boosting classifier to distinguish AML from clean transactions.

The module begins with **feature classification** into three tiers. Protected features (126 rule flags + encoded categoricals) are never removed by feature selection — they encode irreplaceable domain knowledge. Selectable features (velocity, balance, IP signals) are subject to correlation analysis and importance-based filtering. Excluded features (labels, IDs, post-hoc scores, typology signals) are dropped to prevent data leakage. Label encoding is used for categorical features because LightGBM handles it natively.

**Feature selection** proceeds through multicollinearity detection (VIF computation, greedy removal of features with |r| > 0.90) and zero-importance removal (features that LightGBM never uses for splitting). Feature interactions are then added — velocity × account age, amount × counterparty spread, passthrough ratio, rules per lakh, drain speed, cross-border × amount, risky device × burst, and funnel signal.

The **training pipeline** uses a typology-aware stratified split that ensures all 10 typologies appear in both training and test sets. For multi-label transactions (where the detector assigned multiple typologies), a rarest-first assignment ensures rare typologies like Hawala aren't absorbed by common ones like Funnel. SMOTE oversampling is applied to balance the classes. Twelve hyperparameter configurations are tested — varying tree depth, regularization, learning rate, and imbalance strategy (class weights vs SMOTE vs both). The best configuration is selected using a 99% F1 floor with maximum recall: find the peak F1 across all configs, keep everything within 1% of that peak, and among those pick the highest recall.

**Threshold calibration** sweeps from 0.15 to 0.70 in 0.025 steps, applying the same 99% F1 floor logic to select the operating point that maximizes detection while maintaining investigation quality. The output includes per-typology detection rates, feature importance rankings, ROC/PR curves, and saved model artifacts for Phase 2.

### Module 5: Phase 2 — Typology Classification (`05__aml_phase2_typology_classifier.ipynb`)

Phase 2 is a multi-label classifier that operates exclusively on transactions already flagged as AML by Phase 1. For each flagged transaction, it produces a probability distribution across all 10 typologies and emits both a primary prediction (the highest-probability class) and the full set of typologies whose probability crosses a configurable threshold. This dual output reflects how investigators actually reason about money laundering — many real transactions fit more than one signature (a structured cash deposit that is also part of a funnel, a layering hop that is also part of a hawala settlement, etc.) and forcing a single label loses information.

The module loads Phase 1 artifacts (trained model, feature list, train/test splits, metadata) and scores the full dataset with Phase 1 to generate `_phase1_score` as an additional feature for Phase 2. The AML-only subset is then prepared with typology labels, and a stratified split ensures all 10 typologies appear in both training and test sets.

**Typology-discriminating interaction features** are engineered specifically for Phase 2 — these capture signals that distinguish typologies from each other rather than distinguishing AML from clean. Eight interactions are computed: cash × amount (separates Structuring from everything else), cross-border × amount (separates Corridor and Hawala from domestic patterns), per-row rule count (different typologies trigger different rule clusters), counterparties × log(amount) (Funnel/Mule have many, Corridor has few), velocity × balance drain (Pass-Through signal), inflow/outflow ratio (separates Funnel from Circular), proximity to ₹10,000 (Structuring signal), and country-risk × amount (Corridor signal).

**Feature selection** trains a quick LightGBM model and removes features with zero gain for typology classification — many Phase 1 features that excel at separating AML from clean are useless for distinguishing between AML typologies. **Class weights** use inverse-frequency with a rare-class boost (up to 2.5×) for typologies below the median sample count, ensuring the model doesn't ignore rare patterns like Hawala in favor of common ones like Funnel or Layering.

**Hyperparameter tuning** runs three LightGBM configurations in parallel via `ThreadPoolExecutor` (LightGBM releases the GIL during training): Baseline, Deep+Reg (deeper trees with stronger L1/L2), and Wide+Shallow. Each is trained with 500-round boosting and early stopping at 20 patience. The best configuration is chosen by `0.5·accuracy + 0.5·macro_f1` so a strong overall result cannot mask poor recall on rare typologies.

**Multi-label evaluation** is a distinct step. After model selection, the same `y_proba_2` is used to compute both metrics:

- **Primary accuracy** — the argmax prediction matches the true label (~76%).
- **Multi-label accuracy** — the true label appears in the set of typologies above `TYPOLOGY_THRESHOLD = 0.30` (~82%).

The evaluation cell additionally produces a per-typology recall comparison (primary vs multi-label) and a threshold sensitivity table sweeping from 0.10 to 0.50, so investigators can see the recall/specificity trade-off and pick an operating point that matches their case-load capacity.

**Production output** is built in a single vectorized pass over the full dataset. For each predicted-AML row, the model emits a primary typology, the full set of typologies that crossed the threshold (with confidence percentages inline, e.g. `"Funnel Account Network (87%); Money Mule Network (42%)"`), an investigation priority (Critical / High / Medium / Low), a plain-English business explanation that combines the typology pattern with the specific compliance rules that triggered, individual probability columns for each typology, and audit columns listing both the rule names that fired and a human-readable summary of the top three rules. The full pipeline including rule explanation generation is fully numpy-vectorized and runs in 2–4 minutes on a 4M-row dataset (down from 30+ minutes in the prior per-row implementation).

---

## Project Structure

```
smartsentry_aml_model/
├── python_scripts/
│   ├── aml_generator_complete_pipeline.ipynb   # Module 0: Synthetic data generator
│   ├── 01__aml_typology_detector.ipynb         # Module 1: Graph-based typology detection
│   ├── 02__aml_rules_engine.ipynb              # Module 2: 126-rule compliance engine
│   ├── 03__aml_feature_engineering.ipynb        # Module 3: Velocity/balance/IP features
│   ├── 04__aml_ml_preparation.ipynb            # Module 4: Phase 1 binary AML model
│   └── 05__aml_phase2_typology_classifier.ipynb # Module 5: Phase 2 typology classifier
├── outputs_updated/
│   ├── stg_transactions_flagged.parquet         # Detector output
│   ├── stg_transactions_rules_V2.parquet        # Rules engine output
│   ├── stg_transactions_features_V2.parquet     # Feature-engineered dataset
│   ├── ml_outputs/                              # Phase 1 model artifacts
│   │   ├── final_lgb_model.txt                  # Trained LightGBM model
│   │   ├── model_metadata.json                  # Features, threshold, strategy
│   │   ├── X_train.parquet / X_test.parquet     # Train/test splits
│   │   └── y_train.parquet / y_test.parquet     # Labels
│   └── phase2_outputs/                          # Phase 2 model artifacts
│       ├── phase2_typology_model.txt            # Trained LightGBM multiclass model
│       ├── phase1_aml_detection.parquet         # Phase 1 binary output (re-emitted by Phase 2)
│       ├── phase2_typology_classification.parquet  # Multi-label typology output
│       ├── combined_aml_output.parquet          # Phase 1 + Phase 2 merged
│       ├── combined_aml_output.csv              # Same as above, CSV for Excel users
│       ├── phase2_accuracy_summary.csv          # Headline metrics (primary, multi-label, F1, threshold)
│       ├── phase2_evaluation.png                # Confusion matrix + recall + threshold sensitivity
│       ├── model_parameters_full.json           # Full model config + recomputed metrics
│       └── typology_mapping.json                # Index ↔ typology name

```

---

## Setup

### Prerequisites

- Python 3.10+
- 8GB+ RAM (dataset is ~390K rows × 200+ columns)

### Installation

```bash
git clone <repository-url>
cd smartsentry_aml_model

pip install -r requirements.txt
```

Pipeline paths and parameters are centralized in `config/settings.yaml` and `config/generator_config.json`. Notebooks load them via `config/loader.py` — see `config/README.md`.

### Required Packages

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.23 | Numerical operations |
| `lightgbm` | ≥3.3 | Gradient boosting models |
| `scikit-learn` | ≥1.2 | Train/test split, metrics |
| `imbalanced-learn` | ≥0.10 | SMOTE oversampling |
| `pyarrow` | ≥10.0 | Parquet I/O |
| `matplotlib` / `seaborn` | ≥3.6 / ≥0.12 | Visualization |

---

## Usage

Run notebooks sequentially — each reads the previous notebook's output:

```bash
# Step 1: Generate synthetic transactions (~390K rows)
jupyter notebook aml_generator_complete_pipeline.ipynb

# Step 2: Detect typologies via graph traversal
jupyter notebook 01__aml_typology_detector.ipynb

# Step 3: Apply 126 regulatory rules
jupyter notebook 02__aml_rules_engine.ipynb

# Step 4: Engineer velocity/balance/IP features
jupyter notebook 03__aml_feature_engineering.ipynb

# Step 5: Train Phase 1 binary AML detector
jupyter notebook 04__aml_ml_preparation.ipynb

# Step 6: Train Phase 2 typology classifier
jupyter notebook 05__aml_phase2_typology_classifier.ipynb
```

> **Important:** Skipping a step causes downstream notebooks to use stale data. After any change to a notebook, re-run all subsequent notebooks.

---

## Configuration

### Global Parameters

Key parameters in `aml_generator_complete_pipeline.ipynb` (Cell 4 CONFIG):

```python
CONFIG = {
    "num_customers_individual": 3000,       # Individual customer profiles
    "num_customers_entity": 400,            # Business/entity profiles
    "num_transactions_per_account_range": [10, 120],  # Txns per account over 3 months
    "target_fraud_rate": 0.20,              # 20% AML rate in generated data
    "minor_probability": 0.01,              # 1% minors (reduced from 3% to control rule triggers)
    "risk_weights": {"Low": 0.40, "Medium": 0.45, "High": 0.15},  # Customer risk distribution
    "fx_probability": 0.03,                 # 3% foreign exchange transactions
    "income_bands": {
        "Low":    [200000, 500000],          # ₹2L–5L annual
        "Medium": [500001, 2500000],         # ₹5L–25L annual
        "High":   [2500001, 10000000],       # ₹25L–1Cr annual
        "HNI":    [10000001, 100000000]      # ₹1Cr–10Cr annual
    },
}
```

### Typology Weights

Controls relative proportion of each AML typology in the generated dataset:

```python
"typology_weights": {
    "Structuring (Smurfing)":       0.10,   # 12% of AML scenarios
    "Circular Transaction Loop":    0.10,
    "Funnel Account Network":       0.10,
    "Pass-Through Transit Hub":     0.10,
    "Rapid Multi-Hop Layering":     0.10,
    "Third-Party Payment Web":      0.10,
    "Money Mule Network":           0.10,
    "High-Risk Corridor Transfer":  0.10,
    "Underground Banking (Hawala)":  0.10,
    "Charity Abuse":                0.10,
}
```

### Typology Generation Parameters (Creator Config)

These parameters define exactly how each AML pattern is constructed in the synthetic data. The detector's `DETECT_CONFIG` thresholds are derived from these values.

#### T1: Structuring (Smurfing)

Multiple accounts make cash deposits just below the ₹10,000 CTR reporting threshold, then consolidate via electronic transfer to a target account.

```python
"structuring": {
    "num_sources_range": [3, 6],           # 3–6 distinct depositor accounts
    "deposit_amount_range": [8000, 9900],  # Each deposit: ₹8,000–₹9,900 (below ₹10K threshold)
    "transfer_amount_range": [7500, 9800], # Consolidation transfer amount
    "deposit_hour_range": [9, 16],         # Deposits during business hours
    "transfer_hour_range": [10, 18],       # Transfers during extended hours
    "transfer_delay_days_range": [1, 3],   # 1–3 days gap between deposits and transfer
    "deposit_channel": "Branch Cash",      # Cash deposits only
    "transfer_channels": ["NEFT", "IMPS", "UPI"],  # Electronic transfer out
}
```

#### T2: Circular Transaction Loop

Funds flow in a closed ring (A→B→C→A) with small per-hop decay simulating fees, returning to the originating account to obscure the money trail.

```python
"circular": {
    "ring_size_range": [3, 5],             # 3–5 accounts in the loop
    "base_amount_range": [50000, 500000],  # Starting amount: ₹50K–₹5L
    "hop_amount_decay": [0.97, 1.0],       # 0–3% decay per hop (simulates fees)
    "hop_interval_days": 1,                # 1 day between each hop
    "hop_hour_range": [10, 17],            # Business hours
    "channels": ["NEFT", "RTGS", "IMPS"],
}
```

#### T3: Funnel Account Network

Multiple feeder accounts send funds to a single aggregation account, which then forwards 95% of the total to a small number of destination accounts.

```python
"funnel": {
    "num_feeders_range": [15, 50],         # 15–50 distinct feeder accounts
    "per_feeder_amount_range": [5000, 30000],  # Each feeder sends ₹5K–₹30K
    "feeder_spread_days_range": [0, 5],    # Feeders deposit over 0–5 days
    "feeder_hour_range": [8, 20],
    "outflow_delay_days_range": [6, 10],   # 6–10 days after first feeder deposit
    "outflow_splits_range": [2, 3],        # Outflow split into 2–3 transactions
    "outflow_split_pct_range": [0.3, 0.5], # Each split: 30–50% of remaining
    "retention_pct": 0.05,                 # 5% retained in funnel account
    "outflow_hour_range": [10, 16],
    "feeder_channels": ["UPI", "IMPS", "NEFT"],
}
```

#### T4: Pass-Through Transit Hub

An account receives a large inflow and immediately forwards 96–99% to a different account, acting as a transit point to obscure the fund origin.

```python
"passthrough": {
    "inflow_amount_range": [200000, 2000000],  # ₹2L–₹20L inflow
    "forward_pct_range": [0.96, 0.99],         # Forwards 96–99% immediately
    "hour_range": [10, 17],
    "time_gap_hours": [0, 1],                  # Outflow within 0–1 hours of inflow
    "inflow_channels": ["RTGS", "NEFT"],
    "outflow_channels": ["RTGS", "NEFT", "IMPS"],
}
```

#### T5: Rapid Multi-Hop Layering

Funds are rapidly transferred through a chain of 8–10 intermediary accounts, with each hop reducing the amount by ~1% and adding random noise to obscure the trail.

```python
"layering": {
    "num_hops_range": [8, 10],                 # 8–10 sequential hops
    "base_amount_range": [100000, 1000000],    # Starting amount: ₹1L–₹10L
    "per_hop_decay": 0.99,                     # 1% decay per hop
    "per_hop_noise_range": [0.98, 1.0],        # Additional 0–2% random noise per hop
    "hop_interval_minutes_range": [5, 30],     # 5–30 minutes between hops
    "start_hour_range": [9, 14],               # Chains start during morning hours
    "channels": ["IMPS", "NEFT", "UPI", "RTGS"],
}
```

#### T6: Third-Party Payment Web

Multiple unrelated payers send funds to a central account, creating a web of seemingly unrelated payments that actually converge for consolidation.

```python
"third_party_web": {
    "num_unrelated_payers_range": [5, 15],     # 5–15 distinct payers
    "per_payment_amount_range": [10000, 100000],  # Each payment: ₹10K–₹1L
    "payment_spread_days_range": [0, 10],      # Payments spread over 0–10 days
    "payment_channels": ["NEFT", "IMPS", "UPI"],
    "payment_hour_range": [9, 18],
}
```

#### T7: Money Mule Network

A controller distributes funds to 5–20 mule accounts, each of which forwards 85–95% to collector accounts within 1–24 hours.

```python
"money_mule": {
    "num_mules_range": [5, 20],                # 5–20 mule accounts
    "controller_to_mule_amount_range": [20000, 200000],  # ₹20K–₹2L per mule
    "mule_forward_pct_range": [0.85, 0.95],    # Each mule forwards 85–95%
    "mule_forward_delay_hours_range": [1, 24],  # Forward within 1–24 hours
    "channels": ["IMPS", "UPI", "NEFT"],
    "hour_range": [8, 22],                     # Extended hours (mules operate flexibly)
}
```

#### T8: High-Risk Corridor Transfer

Repeated transfers to FATF high-risk or grey-listed jurisdictions, indicating potential terror financing or sanctions evasion.

```python
"high_risk_corridor": {
    "amount_range": [50000, 500000],           # ₹50K–₹5L per transfer
    "target_countries": ["AE", "PK", "BD", "NP", "LK", "MM", "AF"],  # FATF jurisdictions
    "channels": ["RTGS", "NEFT", "SWIFT"],     # Cross-border channels
    "hour_range": [10, 17],
    "frequency_per_account_range": [3, 8],     # 3–8 transfers per account
    "spread_days_range": [1, 15],              # Spread over 1–15 days
}
```

#### T9: Underground Banking (Hawala)

Matched bilateral settlements among 3–4 parties where amounts in both directions are within 5% of each other, indicating informal value transfer without physical money movement.

```python
"hawala": {
    "num_parties_range": [3, 4],               # 3–4 settlement parties
    "settlement_amount_range": [100000, 1000000],  # ₹1L–₹10L per settlement
    "leg_amount_variation_pct": [0.95, 1.05],  # ±5% variation between legs
    "settlement_spread_days_range": [0, 3],    # Settlements within 0–3 days
    "channels": ["NEFT", "RTGS", "Branch Cash"],
    "hour_range": [10, 16],
}
```

#### T10: Charity Abuse

Donations from 10–40 donors flow into an NPO/Trust account, after which 80% of the total is diverted to personal accounts of trustees within 3–10 days.

```python
"charity_abuse": {
    "num_donors_range": [10, 40],              # 10–40 unique donors
    "donation_amount_range": [1000, 50000],    # ₹1K–₹50K per donation
    "donation_spread_days_range": [0, 14],     # Donations over 0–14 days
    "donation_channels": ["UPI", "NEFT", "IMPS"],
    "diversion_delay_days_range": [3, 10],     # Diversion starts 3–10 days after donations
    "diversion_pct": 0.80,                     # 80% of donations diverted
    "diversion_splits_range": [2, 5],          # Split across 2–5 personal accounts
    "diversion_hour_range": [10, 16],
}
```

### Detector Config (DETECT_CONFIG)

The detector's `DETECT_CONFIG` in `01__aml_typology_detector.ipynb` (Cell 5) is derived programmatically from the creator parameters above. Each detection threshold is calculated from the creator's exact generation values with small buffers to handle edge cases. The key derivation principles are:

| Principle | Example |
|---|---|
| Time windows = creator spread + 1–3 day buffer | Funnel feeder spread [0,5] → detector window = 7 days |
| Amount tolerances = creator decay range + 1–2% buffer | Circular decay [0.97,1.0] → detector tolerance = 4% |
| Count thresholds = creator minimum or slightly below | Funnel feeders [15,50] → detector min_senders = 10 |
| Percentage thresholds = creator exact value + 3–5% buffer | Passthrough forward [0.96,0.99] → detector retention = 0.07 |

The detector also includes a **priority-based dedup** system that assigns each transaction to exactly one typology when multiple detectors flag the same transaction. Priority order (most specific first): Structuring (1) > Hawala (2) > Corridor (3) > Pass-Through (4) > Circular (5) > Funnel (6) > Mule (7) > Third-Party (8) > Charity (9) > Layering (10).

---

## Model Details

### Phase 1: Binary AML Detection

| Aspect | Detail |
|---|---|
| **Algorithm** | LightGBM (gradient boosted decision trees) |
| **Objective** | Binary classification (`is_aml` = 0 or 1) |
| **Class Imbalance** | SMOTE + `scale_pos_weight` (tested independently and combined) |
| **Tuning** | 12 hyperparameter configurations compared |
| **Threshold** | Fine-grained sweep (0.15–0.70, step 0.025), selected via 99% F1 floor + max recall |
| **Features** | ~200 (126 rule flags + encoded categoricals + velocity/balance/IP) |
| **Target Metrics** | AUC ≥ 0.94, Recall ≥ 80%, F1 ≥ 0.77 |

### Phase 2: Typology Classification

| Aspect | Detail |
|---|---|
| **Algorithm** | LightGBM multiclass (`objective="multiclass"`, `is_unbalance=True`) |
| **Training Data** | AML transactions only (`is_aml` = 1) |
| **Classes** | 10 typologies |
| **Output Mode** | Multi-label — primary typology + all typologies above `TYPOLOGY_THRESHOLD` (default 0.30) |
| **Class Weights** | Inverse frequency with rare-class boost (up to 2.5×) |
| **Feature Selection** | Gain-based importance; zero-importance features removed via quick 300-round model |
| **Interaction Features** | 8 typology-discriminating interactions (cash×amt, xborder×amt, rule count, cp×log(amt), drain speed, I/O ratio, ₹10K proximity, country×amt) |
| **Hyperparameter Tuning** | 3 configs (Baseline, Deep+Reg, Wide+Shallow) in parallel via `ThreadPoolExecutor`, 500 rounds with 20-patience early stopping |
| **Selection Criterion** | `0.5·accuracy + 0.5·macro_f1` — guards against rare-class neglect |
| **Reported Metrics** | Primary accuracy (argmax) ≈ 76%, multi-label accuracy (≥0.30) ≈ 82%, macro F1 ≥ 0.75 |
| **Threshold Calibration** | OOF-guided per-class threshold lowering for biased classes; sensitivity table sweeps 0.10 → 0.50 |

### Feature Tiers

| Tier | Count | Examples | Treatment |
|---|---|---|---|
| **Protected** | ~160 | 126 rule flags, encoded categoricals | Never removed by feature selection |
| **Selectable** | ~50 | Velocity, balance, IP features | Subject to correlation/importance filtering |
| **Excluded** | ~50 | IDs, labels, FIS, typology signals | Dropped to prevent leakage |

---

## Data Description

### Transaction Schema (Key Columns)

| Column Group | Examples | Count |
|---|---|---|
| **Transaction IDs** | `transaction_id`, `timestamp`, `datestamp`, `transaction_amount` | 22 |
| **Participant Info** | `customer_account_number`, `counterparty_account_number`, `customer_cif_id` | 16 |
| **Customer Profile** | `customer_type`, `occupation_industry`, `annual_income`, `pep_flag` | 36 |
| **Wallet Data** | `wallet_kyc_category`, `wallet_balance_before/after`, `monthly_transaction_limit` | 8 |
| **Device/Channel** | `device_id_fingerprint`, `ip_address`, `vpn_flag`, `emulator_flag` | 10 |
| **AML Labels** | `is_aml`, `aml_typology`, `typology_group_id` | 3 |

### Engineered Features

| Category | Examples | Window |
|---|---|---|
| **Sender Velocity** | `sender_acct_txn_count_1h/24h/7d/30d`, `sender_acct_outflow_amt_30d` | 1h–30d |
| **Receiver Velocity** | `receiver_acct_inflow_amt_30d`, `receiver_acct_unique_senders_7d` | 7d–30d |
| **Balance Tracking** | `sender_balance_before_txn`, `sender_pct_balance_moved` | Per-txn |
| **IP Risk** | `ip_risk_score`, `ip_flag_cross_border`, `ip_flag_country_high_risk` | Per-txn |
| **Rule Flags** | `rule_structuring_pattern`, `rule_negative_list_country`, ... (126 total) | Per-txn |

---

## Outputs

### Production Tables

| File | Description | Key Columns |
|---|---|---|
| `phase1_aml_detection.parquet` | Binary AML scores for all transactions | `fraud_risk_score` (0–100), `alert_source`, `rule_trigger_count` |
| `phase2_typology_classification.parquet` | Typology probabilities + multi-label flags for flagged transactions | `predicted_typology`, `all_matched_typologies` (e.g. `"Funnel (87%); Mule (42%)"`), `num_typologies_matched`, `typology_confidence`, `investigation_priority`, `rules_triggered_count`, `rules_triggered_list`, `rule_explanation`, `business_explanation`, `prob_<typology>` × 10 |
| `combined_aml_output.parquet` | Merged Phase 1 + Phase 2 | All columns from both phases |

### Phase 2 Output Fields (Multi-Label)

For every transaction flagged by Phase 1, Phase 2 emits the following columns:

| Column | Description |
|---|---|
| `predicted_typology` | Primary typology — the class with the highest probability |
| `typology_confidence` | Probability of the primary typology (0.0 — 1.0) |
| `all_matched_typologies` | Semicolon-separated list of every typology whose probability exceeds `TYPOLOGY_THRESHOLD`, with confidence percentages inline. Example: `"Funnel Account Network (87%); Money Mule Network (42%)"`. Always sorted by descending confidence. |
| `num_typologies_matched` | Count of typologies above threshold (≥ 1) |
| `prob_<typology>` × 10 | One column per typology with the raw probability — supports custom downstream thresholding |
| `investigation_priority` | Critical / High / Medium / Low — see below |
| `rules_triggered_count` | Number of compliance rules that fired for this transaction |
| `rules_triggered_list` | Semicolon-separated list of `rule_*` column names that fired (full audit trail) |
| `rule_explanation` | Human-readable summary of the top 3 fired rules (e.g. "Sub-threshold cash structuring detected | FATF/sanctioned country involved | +2 more rules") |
| `business_explanation` | Combined typology + rule narrative shown to investigators (e.g. `"Multiple inflows from different sources converging into single account | Also flagged: Money Mule Network || Rules: Sub-threshold cash structuring detected | High frequency intrabank transfers"`) |

The multi-label threshold is configurable. The default of 0.30 was chosen from the threshold sensitivity table in evaluation: it adds ~5pp recall over primary-only while keeping the average matched-typologies-per-transaction at ~1.15 (i.e. only ~15% of flagged transactions get a second label). Lowering to 0.10 lifts recall to ~92% but average matches climb to ~1.7 — useful only when investigation cost per case is low.

### Investigation Priority Logic

| Priority | Criteria |
|---|---|
| **Critical** | Rule + ML confirmed AND typology confidence ≥ 50% |
| **High** | Fraud risk score ≥ 70 OR typology confidence ≥ 60% |
| **Medium** | Fraud risk score ≥ 40 OR rule triggered |
| **Low** | ML-only detection with low confidence |

### Alert Source Categories

| Source | Meaning |
|---|---|
| `Rule + ML Confirmed` | Both rules and ML model flag the transaction (highest confidence) |
| `Rule Triggered` | Only regulatory rules fire |
| `ML Behavioural Alert` | Only ML detects anomaly (no rules triggered) |
| `Normal` | Neither rules nor ML flag the transaction |

---

## Rules Engine

126 rules organized into 13 groups:

| Group | Rules | Severity Distribution | Examples |
|---|---|---|---|
| Frequency Anomaly | 7 | 1H, 5M, 1L | Baseline frequency deviation, off-hours activity |
| Cumulative Band | 6 | 3H, 2M, 1L | Cash 8.5–10L threshold, structuring pattern |
| Credit-Debit Sequence | 6 | 2H, 3M, 1L | Remittance then cash withdrawal |
| Lifecycle | 9 | 4H, 3M, 2L | Dormant activation, zero-balance cycling |
| Direct Flags | 12 | 8H, 2M, 2L | FATF country, negative list device |
| Large Txn Dynamic | 10 | 4H, 4M, 2L | PEP large transaction, FX cash |
| New Account/Income | 8 | 1H, 3M, 4L | Age-amount mismatch, new account cash |
| Dormant Reactivation | 4 | 2H, 1M, 1L | 75% drain within 7 days |
| Entity Specific | 5 | 4H, 0M, 1L | Trust foreign remittance, shell company |
| Intrabank/Frequency | 5 | 1H, 2M, 2L | Rapid burst, sole prop burst |
| Device/Digital | 15 | 8H, 4M, 3L | Device hopping, VPN + new city |
| Occupation/Profile | 9 | 1H, 2M, 6L | PEP high velocity, minor anomalous |
| PPI Wallet | 30 | 10H, 10M, 10L | Multi-wallet KYC, refund abuse |

**Severity:** High (40) = immediate investigation | Medium (50) = 48hr review | Low (36) = routine monitoring

---

## Logging & Monitoring

Each notebook produces structured console output with progress indicators, diagnostic tables, and summary statistics. Key monitoring points include:

- **Data Generator:** Typology distribution, fraud rate, account pool utilization, PPI scenario counts
- **Detector:** Per-typology detection counts (before/after dedup), detection summary with flagged percentage, multi-typology verification (should be 0)
- **Rules Engine:** Rule trigger diagnostics (% of transactions with ≥1 rule), top 20 most-triggered rules, per-typology rule coverage, rules that never triggered, severity distribution
- **Feature Engineering:** Feature counts per category, missing value summary, FIS score distribution
- **Phase 1:** Hyperparameter comparison table, threshold tuning table with eligible zone, per-typology detection rate, feature importance rankings, ROC/PR curves
- **Phase 2:** Hyperparameter comparison table (3 configs), best-config selection criterion (`0.5·acc + 0.5·macro_f1`), primary vs multi-label accuracy box, per-typology recall comparison (primary vs multi-label, with status flags ✓ / ⚡ / ⚠), threshold sensitivity table (0.10 → 0.50), confusion matrix heatmap, recall bar chart, threshold-vs-recall curve, multi-label sample examples, alert source distribution, investigation priority distribution, rule-explanation coverage

All plots are saved as PNG files in the output directory for documentation and audit purposes.

---


## Regulatory Alignment

| Regulation | Coverage |
|---|---|
| **RBI Master Direction on KYC** | Risk-based monitoring, PEP enhanced due diligence |
| **PMLA 2002** | STR/CTR threshold detection, structuring identification |
| **FIU-IND Reporting** | FATF country flags, ₹10L cash threshold proximity |
| **FEMA** | Foreign remittance monitoring, cross-border corridor detection |
| **RBI PPI Guidelines** | Wallet limit monitoring, KYC category-based rules |


---

## License

Proprietary — For internal use only. Not for redistribution.
