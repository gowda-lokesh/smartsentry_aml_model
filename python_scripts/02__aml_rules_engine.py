#!/usr/bin/env python
# coding: utf-8

# # AML Rules Engine — 151 Rules (76 Bank + 75 PPI)
# ---
# **76 Bank transaction rules** (13 groups, dynamic thresholds)
# **75 PPI/Wallet rules** (merged where overlapping, placeholder where external data needed)
# **17 merged rules** — same column with conditional PPI/Bank thresholds
# 
# Total unique rule columns: ~134 (17 overlaps merged into single columns)
# 

# ## 1 — Environment Setup
# 

# In[1]:



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

from project_config.loader import ensure_notebook_path, get_run_mode, get_artifact_path

_SETTINGS, _PATHS = ensure_notebook_path()
OUTPUT_DIR = str(_PATHS["outputs_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
RUN_MODE = get_run_mode(_SETTINGS)




# In[2]:


# ── Database connection (PostgreSQL) ──
from db_utils import read_table, write_table, save_model, load_model, test_connection
test_connection()      # prints a one-line OK on connect


# In[3]:


#RUN_MODE = "train"
#RUN_MODE = "predict"
print(RUN_MODE)


# In[4]:


from project_config.loader import get_artifact_path
if RUN_MODE == "predict":
    df = read_table("transaction_temp")
else:
    df = read_table("stg_transactions_flagged")
#INPUT_FILE = os.environ.get("AML_INPUT_FILE", _default_input)
print("Environment ready")


# In[ ]:


# df = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_flagged.parquet")


# In[5]:


df.head()


# ## 2 — Load Data
# 

# In[6]:


# Optional summary if labels exist (training mode only)
if "is_aml" in df.columns:
    flagged = df[df["is_aml"] == 1]
    print(f"  AML-flagged subset: {len(flagged):,} rows")
else:
    print(f"  Skipping AML summary (inference mode)")


# ## 3 — Column Resolution, Dynamic Thresholds & PPI Detection
# 

# In[7]:


#print(df['transaction_type_ppi'].unique())
df = df.fillna("")
#print(df['transaction_type_ppi'].unique())


# In[8]:


def find_col(candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

COL = {
    "amount":find_col(["transaction_amount","amount"]),
    "cash_flag":find_col(["cash_flag","Cash Flag"]),
    "timestamp":find_col(["timestamp","Timestamp"]),
    "datestamp":find_col(["datestamp","Datestamp"]),
    "txn_type":find_col(["transaction_type_dr_cr","txn_type"]),
    "channel":find_col(["transaction_mode_channel_bank","channel_bank"]),
    "status":find_col(["transaction_status","txn_status"]),
    "acct":find_col(["customer_account_number","account_number"]),
    "cp_acct":find_col(["counterparty_account_number","cp_account_number"]),
    "cif":find_col(["customer_cif_id","cif_id"]),
    "acct_status":find_col(["account_wallet_status","account_status"]),
    "acct_open":find_col(["account_wallet_opening_date","account_open_date"]),
    "risk_score":find_col(["customer_current_risk_score","risk_score"]),
    "pep_flag":find_col(["pep_flag","PEP Flag"]),
    "vpn_flag":find_col(["vpn_flag","VPN Flag"]),
    "emulator_flag":find_col(["emulator_flag","Emulator Flag"]),
    "cust_type":find_col(["customer_type","Customer Type"]),
    "entity_type":find_col(["customer_entity_type","entity_type"]),
    "acct_category":find_col(["account_category","Account Category"]),
    "acct_type":find_col(["account_type","Account Type"]),
    "occupation":find_col(["customer_occupation_industry","occupation"]),
    "annual_income":find_col(["annual_income","Annual Income"]),
    "sender_cc":find_col(["sender_country_code","sender_country"]),
    "receiver_cc":find_col(["receiver_country_code","receiver_country"]),
    "wallet_kyc":find_col(["wallet_kyc_category","wallet_kyc"]),
    "device_id":find_col(["device_id_fingerprint","device_id"]),
    "ip_address":find_col(["ip_address"]),
    "credit_sum":find_col(["credit_summation_period","credit_sum_period"]),
    "debit_sum":find_col(["debit_summation_period","debit_sum_period"]),
    "tax_residency":find_col(["tax_residency","Tax Residency"]),
    "geo_location":find_col(["geo_location_city_country","geo_location"]),
    "inop_date":find_col(["inoperative_status_date","inop_date"]),
    "bo_address":find_col(["address_beneficial_owners","bo_address"]),
    "reg_office":find_col(["address_registered_office","reg_office"]),
    # PPI-specific columns
    "ppi_txn_type":find_col(["transaction_type_ppi"]),
    "ppi_channel":find_col(["transaction_mode_channel_ppi"]),
    "wallet_bal_before":find_col(["wallet_balance_before"]),
    "wallet_bal_after":find_col(["wallet_balance_after"]),
    "wallet_id":find_col(["wallet_account_id"]),
    "load_instrument":find_col(["load_instrument_type"]),
    "load_source":find_col(["load_source_account_card_details"]),
    "bene_wallet_vpa":find_col(["beneficiary_wallet_id_vpa"]),
    "wallet_max_bal":find_col(["maximum_wallet_balance_limit"]),
    "per_txn_limit":find_col(["transaction_limit_per_txn"]),
    "monthly_txn_limit":find_col(["monthly_transaction_limit"]),
    "minor_flag":find_col(["minor_flag"]),
    "dob":find_col(["date_of_birth"]),
    "mobile":find_col(["mobile_number"]),
    "email":find_col(["email_id"]),
    "pan":find_col(["pan"]),
    "aadhaar":find_col(["aadhaar"]),
    "residency":find_col(["residency"]),
    "address_lat":find_col(["address_lat"]),
    "address_lon":find_col(["address_lon"]),
    "merchant_id":  find_col(["merchant_id","Merchant ID"]),
    "merchant_name":find_col(["merchant_name","Merchant Name"]),
    "mcc":         find_col(["merchant_category_code","mcc","MCC"]),
    "refund_flag": find_col(["refund_chargeback_flag","refund_flag"]),
    "customer_name":find_col(["customer_name","Customer Name"]),
    "cp_name":     find_col(["counterparty_name","Counterparty Name"]),
    "acct_wallet_status":find_col(["account_wallet_status","account_status"]),
}

# HIGH-RISK MCC CODES (gambling, crypto, gaming, adult, telemarketing)
HIGH_RISK_MCCS = {"7995","7994","7801","7802","6051","6050","6012","5816","5817","5818","5967","5966"}

# NEGATIVE LISTS (embedded — in production these come from external DB)
NEGATIVE_DEVICES = {"DEV_BLACKLISTED_001","DEV_BLACKLISTED_002","DEV_BLACKLISTED_003"}
NEGATIVE_IPS = {"192.168.99.1","10.0.0.99","172.16.99.1"}
NEGATIVE_VPAS = {"scam@upi","fraud@upi","blocked@upi"}
NEGATIVE_NAMES = {"HAWALA TRADERS LLC","OFFSHORE SHELL CORP","DARK WEB MERCHANTS",
                  "SANCTIONED ENTITY FZE","TERROR FINANCE TRUST","BLOCKED ENTITY PVT LTD"}

# HIGH-RISK DOMESTIC REGIONS (Indian trafficking/smuggling corridors)
HIGH_RISK_REGIONS = {"MANIPUR","MIZORAM","NAGALAND","JAMMU","KASHMIR","NORTH EAST",
                     "INDO-MYANMAR","INDO-NEPAL","GOLDEN TRIANGLE","KUTCH","JAISALMER"}

print(f"Columns resolved: {sum(1 for v in COL.values() if v)} of {len(COL)}")
print(f"  Reference lists: {len(HIGH_RISK_MCCS)} high-risk MCCs, {len(NEGATIVE_NAMES)} negative names")

# ═══ Dynamic Threshold Framework ═══
print("Building dynamic thresholds...")
_amt_col = COL["amount"]
amt_s = pd.to_numeric(df[_amt_col], errors="coerce").fillna(0) if _amt_col else pd.Series(0, index=df.index)

def txn_category(ch):
    ch = str(ch).upper()
    if any(k in ch for k in ["CASH","ATM","BRANCH CASH"]): return "CASH"
    if any(k in ch for k in ["FOREIGN","FOREX","SWIFT","WIRE","REMIT"]): return "FOREIGN"
    return "INTRABANK"

if COL["channel"] and COL["channel"] in df.columns:
    df["_txn_cat"] = df[COL["channel"]].apply(txn_category)
else:
    df["_txn_cat"] = "INTRABANK"

if COL["risk_score"] and COL["risk_score"] in df.columns:
    _rr = df[COL["risk_score"]].astype(str).str.strip().str.upper()
    df["_risk_cat"] = _rr.map({"LOW":"LOW","MEDIUM":"MEDIUM","MED":"MEDIUM","HIGH":"HIGH","VERY_HIGH":"HIGH","VERY HIGH":"HIGH","1":"LOW","2":"MEDIUM","3":"HIGH"}).fillna("MEDIUM")
else:
    df["_risk_cat"] = "MEDIUM"

PCTLS = {"LOW":0.99,"MEDIUM":0.97,"HIGH":0.95}
DYN_T = {}
for risk in ["LOW","MEDIUM","HIGH"]:
    for cat in ["CASH","INTRABANK","FOREIGN"]:
        mask = (df["_risk_cat"]==risk) & (df["_txn_cat"]==cat)
        sub = amt_s[mask]
        DYN_T[(risk,cat)] = sub.quantile(PCTLS[risk]) if len(sub)>10 else 50000
        if risk=="HIGH" and COL["credit_sum"]:
            bal = pd.to_numeric(df.loc[mask, COL["credit_sum"]], errors="coerce").fillna(0)
            aqb5 = bal.mean()*5 if len(bal)>0 else 0
            if aqb5>0: DYN_T[(risk,cat)] = min(DYN_T[(risk,cat)], aqb5)
        DYN_T[(risk,cat)] = max(DYN_T[(risk,cat)], 1000)

df["_dyn_thresh"] = df.apply(lambda r: DYN_T.get((r["_risk_cat"],r["_txn_cat"]),50000), axis=1)

DYN_T_R = {}
for risk in ["LOW","MEDIUM","HIGH"]:
    sub = amt_s[df["_risk_cat"]==risk]
    DYN_T_R[risk] = sub.quantile(PCTLS[risk]) if len(sub)>10 else 50000
df["_dyn_thresh_risk"] = df["_risk_cat"].map(DYN_T_R).fillna(50000)

print(f"  Thresholds computed for {len(DYN_T)} risk×category combinations")

# ═══ PPI Detection ═══
df["_is_ppi"] = False
if COL["ppi_txn_type"] and COL["ppi_txn_type"] in df.columns:
    df["_is_ppi"] = df[COL["ppi_txn_type"]].astype(str).str.strip() != ""
    df["_is_ppi"] = df["_is_ppi"] & (df[COL["ppi_txn_type"]].astype(str).str.strip().str.upper() != "NAN")

ppi_count = df["_is_ppi"].sum()
bank_count = (~df["_is_ppi"]).sum()
print(f"\n  PPI transactions:  {ppi_count:>10,} ({ppi_count/len(df)*100:.1f}%)")
print(f"  Bank transactions: {bank_count:>10,} ({bank_count/len(df)*100:.1f}%)")




# ## 4 — Build Derived Features (Bank + PPI)
# 

# In[9]:


print("Building derived features...")

df["_amt"] = pd.to_numeric(df[COL["amount"]], errors="coerce").fillna(0) if COL["amount"] else 0
df["_cash"] = df[COL["cash_flag"]].astype(str).str.strip().str.upper().isin(["Y","1","YES","TRUE"]) if COL["cash_flag"] else False
df["_dr_cr"] = df[COL["txn_type"]].astype(str).str.strip().str.upper() if COL["txn_type"] else ""
df["_is_credit"] = df["_dr_cr"].str.contains("CR|CREDIT", case=False, na=False)
df["_is_debit"] = df["_dr_cr"].str.contains("DR|DEBIT", case=False, na=False)

def parse_dt(row):
    ds = str(row.get(COL["datestamp"],"") if COL["datestamp"] else "")
    ts = str(row.get(COL["timestamp"],"00:00:00") if COL["timestamp"] else "00:00:00")
    for fmt in ["%d-%m-%Y %H:%M:%S","%d/%m/%Y %H:%M:%S","%Y-%m-%d %H:%M:%S"]:
        try: return datetime.strptime(f"{ds} {ts}".strip(), fmt)
        except: continue
    return pd.NaT

df["_dt"] = df.apply(parse_dt, axis=1)
df["_hour"] = df["_dt"].apply(lambda x: x.hour if pd.notna(x) else 12)
df["_dow"] = df["_dt"].apply(lambda x: x.weekday() if pd.notna(x) else 0)
df["_is_night"] = df["_hour"].apply(lambda h: h>=22 or h<6)
df["_is_weekend"] = df["_dow"].isin([5,6])
df["_is_round"] = df["_amt"].apply(lambda x: x>0 and x==int(x) and int(x)%1000==0)

def parse_date_col(cn):
    if not cn or cn not in df.columns: return pd.NaT
    return pd.to_datetime(df[cn], format="%d-%m-%Y", errors="coerce")

ao = parse_date_col(COL["acct_open"])
df["_acct_open_days"] = (df["_dt"]-ao).dt.days.fillna(9999) if ao is not None and not isinstance(ao,type(pd.NaT)) else 9999

io = parse_date_col(COL["inop_date"])
if io is not None and not isinstance(io,type(pd.NaT)):
    df["_was_inoperative"] = io.notna(); df["_days_since_reactivation"] = (df["_dt"]-io).dt.days.fillna(9999)
else:
    df["_was_inoperative"] = False; df["_days_since_reactivation"] = 9999

def fc(k):
    if COL[k] and COL[k] in df.columns: return df[COL[k]].astype(str).str.strip().str.upper().isin(["Y","1","YES","TRUE"])
    return pd.Series(False, index=df.index)
def sc(k, default=""):
    if COL[k] and COL[k] in df.columns: return df[COL[k]].astype(str).str.strip()
    return pd.Series(default, index=df.index)

df["_pep"]=fc("pep_flag"); df["_vpn"]=fc("vpn_flag"); df["_emulator"]=fc("emulator_flag")
df["_minor"]=fc("minor_flag")
df["_channel"]=sc("channel","").str.upper()
df["_risk"]=sc("risk_score","").str.upper()
df["_occupation"]=sc("occupation","").str.lower()
df["_income"]=pd.to_numeric(df[COL["annual_income"]],errors="coerce").fillna(0) if COL["annual_income"] else 0
df["_cust_type"]=sc("cust_type","").str.lower()
df["_entity_type"]=sc("entity_type","").str.lower() if COL["entity_type"] else df["_cust_type"]
df["_acct_cat"]=sc("acct_category","").str.lower()
df["_acct_type"]=sc("acct_type","").str.lower()
df["_is_entity"]=df["_cust_type"].str.contains("non-individual|corporate|business|company|trust|society|association|partnership",case=False,na=False)
df["_is_trust_society"]=df["_entity_type"].str.contains("trust|society|association|npo|ngo",case=False,na=False)
df["_is_partnership"]=df["_entity_type"].str.contains("partnership",case=False,na=False)
df["_acct"]=sc("acct",""); df["_cif"]=sc("cif","") if COL["cif"] else df["_acct"]
df["_cp_acct"]=sc("cp_acct","")
df["_sender_cc"]=sc("sender_cc","IN").str.upper(); df["_receiver_cc"]=sc("receiver_cc","IN").str.upper()
df["_tax_res"]=sc("tax_residency","INDIA").str.upper()
df["_device_id"]=sc("device_id",""); df["_ip"]=sc("ip_address","")
df["_geo"]=sc("geo_location","").str.lower()
df["_wallet_kyc"]=sc("wallet_kyc","").str.lower()
df["_bo_city"]=sc("bo_address","").str.lower() if COL["bo_address"] else ""
df["_reg_city"]=sc("reg_office","").str.lower() if COL["reg_office"] else ""
df["_balance_proxy"]=pd.to_numeric(df[COL["credit_sum"]],errors="coerce").fillna(1).replace(0,1) if COL["credit_sum"] else 1

# PPI-specific derived features
df["_ppi_type"]=sc("ppi_txn_type","").str.upper()
df["_ppi_channel"]=sc("ppi_channel","").str.upper()
df["_is_p2p"]=df["_ppi_type"].str.contains("P2P",case=False,na=False)
df["_is_p2m"]=df["_ppi_type"].str.contains("P2M",case=False,na=False)
df["_is_load"]=df["_ppi_type"].str.contains("LOAD|TOP.?UP",case=False,na=False)
df["_wallet_id"]=sc("wallet_id","")
df["_wallet_bal_before"]=pd.to_numeric(df[COL["wallet_bal_before"]],errors="coerce").fillna(0) if COL["wallet_bal_before"] else 0
df["_wallet_bal_after"]=pd.to_numeric(df[COL["wallet_bal_after"]],errors="coerce").fillna(0) if COL["wallet_bal_after"] else 0
df["_wallet_max_bal"]=pd.to_numeric(df[COL["wallet_max_bal"]],errors="coerce").fillna(200000) if COL["wallet_max_bal"] else 200000
df["_monthly_limit"]=pd.to_numeric(df[COL["monthly_txn_limit"]],errors="coerce").fillna(100000) if COL["monthly_txn_limit"] else 100000
df["_load_source"]=sc("load_source","")
df["_load_instrument"]=sc("load_instrument","")
df["_bene_vpa"]=sc("bene_wallet_vpa","")
df["_is_small_kyc"]=df["_wallet_kyc"].str.contains("min",case=False,na=False)
df["_residency"]=sc("residency","RESIDENT").str.upper()
df["_addr_lat"]=pd.to_numeric(df[COL["address_lat"]],errors="coerce").fillna(0) if COL["address_lat"] else 0
df["_addr_lon"]=pd.to_numeric(df[COL["address_lon"]],errors="coerce").fillna(0) if COL["address_lon"] else 0

# Country / risk lists
FATF={"KP","IR","MM","AF","SY","YE","HT","SD","ML","BF","MZ","TZ","CD","SS","LY","PK","NI","JM","TR","PH","AE"}
TAX_H={"CH","MU","KY","VG","PA","BM","GG","JE","IM","BZ","LI","MC","AD","SM","GI","LU","SG","HK","BH"}
df["_sender_high_risk_cc"]=df["_sender_cc"].isin(FATF)
df["_receiver_high_risk_cc"]=df["_receiver_cc"].isin(FATF)
df["_is_cross_border"]=(df["_sender_cc"]!=df["_receiver_cc"])&(df["_sender_cc"]!="")&(df["_receiver_cc"]!="")
df["_involves_tax_haven"]=df["_sender_cc"].isin(TAX_H)|df["_receiver_cc"].isin(TAX_H)
df["_is_offshore"]=df["_is_cross_border"]|df["_receiver_high_risk_cc"]|df["_involves_tax_haven"]
df["_is_foreign_remittance"]=df["_channel"].str.contains("FOREIGN|FOREX|SWIFT|WIRE|REMIT",case=False,na=False)|df["_is_cross_border"]
df["_is_intrabank"]=df["_channel"].str.contains("INTRA|INTERNAL|NEFT|IMPS|UPI",case=False,na=False)

# Age computation
df["_age"]=0
if COL["dob"] and COL["dob"] in df.columns:
    dob = pd.to_datetime(df[COL["dob"]], format="%d-%m-%Y", errors="coerce")
    df["_age"] = ((df["_dt"] - dob).dt.days / 365.25).fillna(30).astype(int)


# ═══ MCC / Merchant derived features (for PPI rules 12,32,39,50,72) ═══
df["_mcc"] = df[COL["mcc"]].astype(str).str.strip() if COL["mcc"] and COL["mcc"] in df.columns else ""
df["_merchant_id"] = df[COL["merchant_id"]].astype(str).str.strip() if COL["merchant_id"] and COL["merchant_id"] in df.columns else ""
df["_is_high_risk_mcc"] = df["_mcc"].isin(HIGH_RISK_MCCS)
df["_refund_flag"] = df[COL["refund_flag"]].astype(str).str.strip().str.upper().isin(["Y","1","YES","TRUE"]) if COL["refund_flag"] and COL["refund_flag"] in df.columns else False

# ═══ Name matching (for PPI rules 40,61-64,67) ═══
df["_cust_name"] = df[COL["customer_name"]].astype(str).str.strip().str.upper() if COL["customer_name"] and COL["customer_name"] in df.columns else ""
df["_cp_name"] = df[COL["cp_name"]].astype(str).str.strip().str.upper() if COL["cp_name"] and COL["cp_name"] in df.columns else ""
df["_cp_name_on_negative"] = df["_cp_name"].isin(NEGATIVE_NAMES) | df["_cust_name"].isin(NEGATIVE_NAMES)
df["_merchant_name"] = df[COL["merchant_name"]].astype(str).str.strip().str.upper() if COL["merchant_name"] and COL["merchant_name"] in df.columns else ""
df["_merchant_on_negative"] = df["_merchant_name"].isin(NEGATIVE_NAMES)

# Simple name similarity (token overlap ratio as proxy for fuzzy match)
def name_similarity(a, b):
    if not a or not b or a == "" or b == "": return 1.0
    a_tokens = set(str(a).upper().split())
    b_tokens = set(str(b).upper().split())
    if not a_tokens or not b_tokens: return 1.0
    overlap = len(a_tokens & b_tokens)
    total = max(len(a_tokens | b_tokens), 1)
    return overlap / total

df["_name_similarity"] = df.apply(lambda r: name_similarity(r["_cust_name"], r["_cp_name"]), axis=1)

# ═══ Device/IP/VPA on negative list (for PPI rules 65,66) ═══
df["_device_on_negative"] = df["_device_id"].isin(NEGATIVE_DEVICES)
df["_ip_on_negative"] = df["_ip"].isin(NEGATIVE_IPS)
df["_vpa_on_negative"] = df["_bene_vpa"].isin(NEGATIVE_VPAS)

# ═══ Transaction status (for bank rule 58, PPI rules 8,30,38) ═══
df["_status"] = df[COL["acct_wallet_status"]].astype(str).str.strip().str.upper() if COL["acct_wallet_status"] and COL["acct_wallet_status"] in df.columns else ""
df["_txn_status"] = df[COL["status"]].astype(str).str.strip().str.upper() if COL["status"] and COL["status"] in df.columns else ""
df["_is_failed"] = df["_txn_status"].isin(["FAILED","REVERSED","ATTEMPTED","DECLINED"])

# ═══ Wallet status (for PPI rule 74) ═══
df["_wallet_status"] = df["_status"]
df["_wallet_reported"] = df["_wallet_status"].str.contains("SUSPEND|FROZEN|BLOCK|REVIEW|REPORT", case=False, na=False)

# ═══ High-risk domestic region (for PPI rule 69) ═══
df["_in_high_risk_region"] = df["_geo"].str.upper().apply(lambda g: any(r in str(g) for r in HIGH_RISK_REGIONS)) if isinstance(df["_geo"], pd.Series) else False

# ═══ Load instrument type (for PPI rule 38) ═══
df["_load_instrument"] = df[COL["load_instrument"]].astype(str).str.strip() if COL["load_instrument"] and COL["load_instrument"] in df.columns else ""

print(f"  Derived features: {sum(1 for c in df.columns if c.startswith('_'))} columns")
print(f"  PPI txns: {df['_is_ppi'].sum():,} | P2P: {df['_is_p2p'].sum():,} | P2M: {df['_is_p2m'].sum():,} | Load: {df['_is_load'].sum():,}")




# ## 5 — Velocity, Aggregation & Window Features (Bank + PPI)
# 

# In[10]:


import numpy as np
print("Computing velocity features (this may take a minute)...")

# Sort by account + datetime for rolling calculations
df = df.sort_values(["_acct", "_dt"]).reset_index(drop=True)

# -- Account-level velocity --
# For each transaction, count/sum prior txns from same account within window
def compute_velocity(group_col, prefix):
    # group_col: column to group by (_acct or _cif)
    # prefix: column name prefix
    counts_1hr = []
    counts_24hr = []
    counts_7d = []
    sums_24hr = []
    sums_7d = []

    grouped = df.groupby(group_col)
    total_groups = len(grouped)
    processed = 0

    for gname, gdf in grouped:
        processed += 1
        if processed % 3000 == 0:
            print(f"    {prefix}: {processed:,}/{total_groups:,} groups...")

        dts = gdf["_dt"].values
        amts = gdf["_amt"].values
        n = len(gdf)

        c1 = np.zeros(n, dtype=int)
        c24 = np.zeros(n, dtype=int)
        c7 = np.zeros(n, dtype=int)
        s24 = np.zeros(n, dtype=float)
        s7 = np.zeros(n, dtype=float)

        for i in range(n):
            t = dts[i]
            if pd.isna(t):
                continue
            for j in range(i - 1, -1, -1):
                if pd.isna(dts[j]):
                    continue
                diff = (t - dts[j]) / np.timedelta64(1, 'h')
                if diff > 168:  # > 7 days in hours
                    break
                if diff <= 1:
                    c1[i] += 1
                if diff <= 24:
                    c24[i] += 1
                    s24[i] += amts[j]
                if diff <= 168:
                    c7[i] += 1
                    s7[i] += amts[j]

        counts_1hr.extend(c1)
        counts_24hr.extend(c24)
        counts_7d.extend(c7)
        sums_24hr.extend(s24)
        sums_7d.extend(s7)

    df[f"{prefix}_count_1hr"] = counts_1hr
    df[f"{prefix}_count_24hr"] = counts_24hr
    df[f"{prefix}_count_7d"] = counts_7d
    df[f"{prefix}_sum_24hr"] = sums_24hr
    df[f"{prefix}_sum_7d"] = sums_7d

compute_velocity("_acct", "_acct_vel")
compute_velocity("_cif", "_cust_vel")

# -- Dormancy flag: gap > 30 days since last txn from same account --
print("  Computing dormancy flags...")
df["_prev_txn_gap_days"] = df.groupby("_acct")["_dt"].diff().dt.days.fillna(0)
df["_dormancy"] = df["_prev_txn_gap_days"] > 30

# -- Amount z-score (30-day rolling mean/std per account) --
print("  Computing 30-day amount z-scores...")
zscore_vals = np.zeros(len(df))
for acct, gdf in df.groupby("_acct"):
    idxs = gdf.index.values
    dts = gdf["_dt"].values
    amts = gdf["_amt"].values
    for i in range(len(idxs)):
        t = dts[i]
        if pd.isna(t):
            continue
        window_amts = []
        for j in range(i - 1, -1, -1):
            if pd.isna(dts[j]):
                continue
            diff_days = (t - dts[j]) / np.timedelta64(1, 'D')
            if diff_days > 30:
                break
            window_amts.append(amts[j])
        if len(window_amts) >= 3:
            mu = np.mean(window_amts)
            sigma = np.std(window_amts)
            if sigma > 0:
                zscore_vals[idxs[i]] = (amts[i] - mu) / sigma

df["_amt_zscore_30d"] = zscore_vals

# -- Amount to balance ratio --
df["_amt_to_balance"] = df["_amt"] / df["_balance_proxy"]

print("  Velocity features complete")

# ═══ PPI-SPECIFIC VELOCITY FEATURES ═══
print("  Computing PPI-specific velocity features...")

# Wallet-level velocity (for PPI transactions, group by wallet_id)
if df["_is_ppi"].any() and df["_wallet_id"].str.strip().ne("").any():
    ppi_df = df[df["_is_ppi"]].copy()

    # Load count per wallet in 7d and 30d
    df["_wallet_load_count_7d"] = 0
    df["_wallet_load_sum_7d"] = 0.0
    df["_wallet_load_sum_30d"] = 0.0
    df["_wallet_p2p_count_1hr"] = 0
    df["_wallet_p2p_count_24hr"] = 0
    df["_wallet_unique_bene_24hr"] = 0
    df["_wallet_unique_bene_7d"] = 0
    df["_wallet_load_sources_30d"] = 0

    for wid, gdf in ppi_df.groupby("_wallet_id"):
        if not wid or wid == "nan": continue
        idxs = gdf.index.values; dts = gdf["_dt"].values; amts = gdf["_amt"].values
        is_load = gdf["_is_load"].values; is_p2p = gdf["_is_p2p"].values
        benes = gdf["_bene_vpa"].values; sources = gdf["_load_source"].values

        for i in range(len(idxs)):
            t = dts[i]
            if pd.isna(t): continue
            lc7=0; ls7=0.0; ls30=0.0; pc1=0; pc24=0
            bene_set_24=set(); bene_set_7=set(); src_set_30=set()

            if is_load[i] and sources[i]: src_set_30.add(sources[i])
            if is_p2p[i] and benes[i]: bene_set_24.add(benes[i]); bene_set_7.add(benes[i])

            for j in range(i-1,-1,-1):
                if pd.isna(dts[j]): continue
                dh = (t-dts[j])/np.timedelta64(1,'h')
                dd = dh/24
                if dd > 30: break
                if is_load[j]:
                    if dd <= 7: lc7+=1; ls7+=amts[j]
                    ls30+=amts[j]
                    if sources[j]: src_set_30.add(sources[j])
                if is_p2p[j]:
                    if dh <= 1: pc1+=1
                    if dh <= 24: pc24+=1; 
                    if benes[j] and dh<=24: bene_set_24.add(benes[j])
                    if benes[j] and dd<=7: bene_set_7.add(benes[j])

            df.at[idxs[i],"_wallet_load_count_7d"] = lc7
            df.at[idxs[i],"_wallet_load_sum_7d"] = ls7
            df.at[idxs[i],"_wallet_load_sum_30d"] = ls30
            df.at[idxs[i],"_wallet_p2p_count_1hr"] = pc1
            df.at[idxs[i],"_wallet_p2p_count_24hr"] = pc24
            df.at[idxs[i],"_wallet_unique_bene_24hr"] = len(bene_set_24)
            df.at[idxs[i],"_wallet_unique_bene_7d"] = len(bene_set_7)
            df.at[idxs[i],"_wallet_load_sources_30d"] = len(src_set_30)
else:
    for c in ["_wallet_load_count_7d","_wallet_load_sum_7d","_wallet_load_sum_30d",
              "_wallet_p2p_count_1hr","_wallet_p2p_count_24hr","_wallet_unique_bene_24hr",
              "_wallet_unique_bene_7d","_wallet_load_sources_30d"]:
        df[c] = 0


    # ═══ MCC AGGREGATION per wallet (for PPI rules 12,39,50) ═══
    print("  Computing MCC aggregation per wallet...")
    df["_high_risk_mcc_sum_30d"] = 0.0
    df["_total_p2m_sum_30d"] = 0.0
    df["_high_risk_mcc_pct"] = 0.0

    ppi_p2m = df[df["_is_ppi"] & df["_is_p2m"]]
    if len(ppi_p2m) > 0:
        for wid, gdf in ppi_p2m.groupby("_wallet_id"):
            if not wid or wid == "nan": continue
            idxs = gdf.index.values; dts = gdf["_dt"].values
            amts = gdf["_amt"].values; mccs = gdf["_is_high_risk_mcc"].values
            for i in range(len(idxs)):
                t = dts[i]
                if pd.isna(t): continue
                hr_sum = amts[i] if mccs[i] else 0; total = amts[i]
                for j in range(i-1,-1,-1):
                    if pd.isna(dts[j]): continue
                    if (t-dts[j])/np.timedelta64(1,'D') > 30: break
                    total += amts[j]
                    if mccs[j]: hr_sum += amts[j]
                df.at[idxs[i],"_high_risk_mcc_sum_30d"] = hr_sum
                df.at[idxs[i],"_total_p2m_sum_30d"] = total
                df.at[idxs[i],"_high_risk_mcc_pct"] = hr_sum/max(total,1)

    # ═══ REFUND COUNT per merchant per wallet (for PPI rule 21) ═══
    print("  Computing refund counts per merchant...")
    df["_refund_count_merchant_30d"] = 0
    ppi_refunds = df[df["_is_ppi"] & df["_refund_flag"]]
    if len(ppi_refunds) > 0:
        for (wid, mid), gdf in ppi_refunds.groupby(["_wallet_id","_merchant_id"]):
            if not wid or wid == "nan" or not mid or mid == "nan": continue
            for idx in gdf.index:
                df.at[idx, "_refund_count_merchant_30d"] = len(gdf)

    # ═══ CROSS-WALLET AGGREGATION (for PPI rules 44-47,52) ═══
    print("  Computing cross-wallet aggregations...")

    # Count wallets per PAN
    if COL["pan"] and COL["pan"] in df.columns:
        pan_wallet_count = df[df["_is_ppi"]].groupby(df[COL["pan"]].astype(str).str.strip())["_wallet_id"].nunique()
        df["_pan_wallet_count"] = df[COL["pan"]].astype(str).str.strip().map(pan_wallet_count).fillna(0).astype(int)
    else:
        df["_pan_wallet_count"] = 0

    # Count wallets per mobile
    if COL["mobile"] and COL["mobile"] in df.columns:
        mob_wallet_count = df[df["_is_ppi"]].groupby(df[COL["mobile"]].astype(str).str.strip())["_wallet_id"].nunique()
        df["_mobile_wallet_count"] = df[COL["mobile"]].astype(str).str.strip().map(mob_wallet_count).fillna(0).astype(int)
    else:
        df["_mobile_wallet_count"] = 0

    # Count wallets per load source
    if COL["load_source"] and COL["load_source"] in df.columns:
        src_wallet_count = df[df["_is_ppi"] & df["_is_load"]].groupby(df[COL["load_source"]].astype(str).str.strip())["_wallet_id"].nunique()
        df["_load_src_wallet_count"] = df[COL["load_source"]].astype(str).str.strip().map(src_wallet_count).fillna(0).astype(int)
    else:
        df["_load_src_wallet_count"] = 0

    # Count wallets per IP (24h window approximation using full dataset)
    ip_wallet_count = df[df["_is_ppi"]].groupby("_ip")["_wallet_id"].nunique()
    df["_ip_wallet_count"] = df["_ip"].map(ip_wallet_count).fillna(0).astype(int)

    # Count wallets per VPA
    vpa_wallet_count = df[df["_is_ppi"]].groupby("_bene_vpa")["_wallet_id"].nunique()
    df["_vpa_wallet_count"] = df["_bene_vpa"].map(vpa_wallet_count).fillna(0).astype(int)

    # ═══ MCC CHANGE COUNT per merchant (for PPI rule 32) ═══
    print("  Computing MCC change counts per merchant...")
    df["_merchant_mcc_change_count"] = 0
    if COL["mcc"] and COL["mcc"] in df.columns and COL["merchant_id"] and COL["merchant_id"] in df.columns:
        for mid, gdf in df[df["_merchant_id"] != ""].groupby("_merchant_id"):
            if len(gdf) < 2: continue
            unique_mccs = gdf["_mcc"].nunique()
            if unique_mccs >= 2:
                for idx in gdf.index:
                    df.at[idx, "_merchant_mcc_change_count"] = unique_mccs

print(f"  PPI velocity features complete")
print(f"  All velocity and aggregation features complete")




# In[11]:


print("Computing missing derived features...")

# ═══ Cleanup from previous runs to avoid merge suffix conflicts ═══
cleanup_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
if cleanup_cols:
    df.drop(columns=cleanup_cols, inplace=True, errors="ignore")
    print(f"  Cleaned {len(cleanup_cols)} leftover merge columns")

target_cols = [
    "_cust_baseline_freq_30d", "_cum_cash_30d", "_cum_foreign_30d",
    "_cum_credits_30d", "_cum_debits_30d", "_distinct_channels_24h",
    "_distinct_depositors_7d", "_is_new_device", "_is_new_city",
    "_distinct_devices_2h", "_high_risk_mcc_pct", "_high_risk_mcc_sum_30d",
    "_is_high_risk_mcc", "_refund_flag", "_refund_count_merchant_30d",
    "_merchant_mcc_change_count", "_name_similarity",
    "_pan_wallet_count", "_mobile_wallet_count", "_ip_wallet_count",
    "_vpa_wallet_count", "_load_src_wallet_count", "_in_high_risk_region"
]
existing = [c for c in target_cols if c in df.columns]
if existing:
    df.drop(columns=existing, inplace=True, errors="ignore")
    print(f"  Dropped {len(existing)} existing target columns for fresh computation")

# ── 1. Customer baseline frequency (30-day) ──
if "_acct" in df.columns and "_dt" in df.columns:
    cust_col = next((c for c in ["_cif", "_customer_cif"] if c in df.columns), None)
    if cust_col:
        cust_freq = df.groupby(cust_col).size().reset_index(name="_cbf_temp")
        df = df.merge(cust_freq, on=cust_col, how="left")
        df["_cust_baseline_freq_30d"] = df["_cbf_temp"].fillna(0)
        df.drop(columns=["_cbf_temp"], inplace=True)
        print(f"  _cust_baseline_freq_30d computed from {cust_col}")
    else:
        df["_cust_baseline_freq_30d"] = 0
        print(f"  _cust_baseline_freq_30d defaulted (no CIF column)")
else:
    df["_cust_baseline_freq_30d"] = 0

# ── 2. Cumulative cash 30d ──
df["_cum_cash_30d"] = 0.0
if "_cash" in df.columns and "_amt" in df.columns and "_acct" in df.columns:
    cash_mask = df["_cash"] == True
    if cash_mask.any():
        cash_sums = df[cash_mask].groupby("_acct")["_amt"].sum().reset_index(name="_tmp")
        df = df.merge(cash_sums, on="_acct", how="left")
        df["_cum_cash_30d"] = df["_tmp"].fillna(0)
        df.drop(columns=["_tmp"], inplace=True)
        print(f"  _cum_cash_30d computed")

# ── 3. Cumulative foreign 30d ──
df["_cum_foreign_30d"] = 0.0
if "_is_foreign_remittance" in df.columns and "_acct" in df.columns:
    foreign_mask = df["_is_foreign_remittance"] == True
    if foreign_mask.any():
        foreign_sums = df[foreign_mask].groupby("_acct")["_amt"].sum().reset_index(name="_tmp")
        df = df.merge(foreign_sums, on="_acct", how="left")
        df["_cum_foreign_30d"] = df["_tmp"].fillna(0)
        df.drop(columns=["_tmp"], inplace=True)
        print(f"  _cum_foreign_30d computed")

# ── 4. Cumulative credits/debits 30d ──
df["_cum_credits_30d"] = 0.0
df["_cum_debits_30d"] = 0.0
if "_is_credit" in df.columns and "_acct" in df.columns:
    cr_sums = df[df["_is_credit"]].groupby("_acct")["_amt"].sum().reset_index(name="_tmp_cr")
    df = df.merge(cr_sums, on="_acct", how="left")
    df["_cum_credits_30d"] = df["_tmp_cr"].fillna(0)
    df.drop(columns=["_tmp_cr"], inplace=True)

    dr_sums = df[df["_is_debit"]].groupby("_acct")["_amt"].sum().reset_index(name="_tmp_dr")
    df = df.merge(dr_sums, on="_acct", how="left")
    df["_cum_debits_30d"] = df["_tmp_dr"].fillna(0)
    df.drop(columns=["_tmp_dr"], inplace=True)
    print(f"  _cum_credits_30d / _cum_debits_30d computed")

# ── 5. Distinct channels 24h — FIXED ──
# OLD: counted unique channels across ENTIRE period (always high)
# NEW: count unique channels on the SAME DAY as each transaction
df["_distinct_channels_24h"] = 1  # default: just this transaction's channel
if "_acct" in df.columns and "_channel" in df.columns and "_dt" in df.columns:
    # Group by account + date, count unique channels per day
    df["_txn_date"] = df["_dt"].dt.date
    daily_channels = df.groupby(["_acct", "_txn_date"])["_channel"].transform("nunique")
    df["_distinct_channels_24h"] = daily_channels.fillna(1).astype(int)
    df.drop(columns=["_txn_date"], inplace=True, errors="ignore")
    print(f"  _distinct_channels_24h computed (per-day per-account)")

# ── 6. Distinct depositors 7d ──
df["_distinct_depositors_7d"] = 0
if "_acct" in df.columns and "_cp_acct" in df.columns and "_is_credit" in df.columns:
    credit_mask = df["_is_credit"] == True
    if credit_mask.any():
        dep_counts = df[credit_mask].groupby("_acct")["_cp_acct"].nunique().reset_index(name="_tmp")
        df = df.merge(dep_counts, on="_acct", how="left")
        df["_distinct_depositors_7d"] = df["_tmp"].fillna(0).astype(int)
        df.drop(columns=["_tmp"], inplace=True)
        print(f"  _distinct_depositors_7d computed")

# ── 7. Device novelty flags ──
df["_is_new_device"] = False
df["_is_new_city"] = False
device_col = next((c for c in ["device_id_fingerprint", "device_id", "_device"] if c in df.columns), None)
city_col = next((c for c in ["ip_city", "customer_city", "_city"] if c in df.columns), None)

if device_col and "_acct" in df.columns:
    try:
        df_sorted = df.sort_values("_dt").reset_index(drop=False)
        orig_idx = df_sorted["index"]
        first_dev = df_sorted.groupby("_acct")[device_col].transform("first")
        new_dev = (df_sorted[device_col] != first_dev)
        result = pd.Series(False, index=df.index)
        result.iloc[orig_idx.values] = new_dev.values
        df["_is_new_device"] = result
        print(f"  _is_new_device computed from {device_col}")
    except Exception as e:
        print(f"  _is_new_device failed: {e}")

if city_col and "_acct" in df.columns:
    try:
        df_sorted = df.sort_values("_dt").reset_index(drop=False)
        orig_idx = df_sorted["index"]
        first_cit = df_sorted.groupby("_acct")[city_col].transform("first")
        new_cit = (df_sorted[city_col] != first_cit)
        result = pd.Series(False, index=df.index)
        result.iloc[orig_idx.values] = new_cit.values
        df["_is_new_city"] = result
        print(f"  _is_new_city computed from {city_col}")
    except Exception as e:
        print(f"  _is_new_city failed: {e}")

# ── 8. Distinct devices 2h — FIXED ──
df["_distinct_devices_2h"] = 1
if device_col and "_acct" in df.columns and "_dt" in df.columns:
    df["_txn_date"] = df["_dt"].dt.date
    daily_devices = df.groupby(["_acct", "_txn_date"])[device_col].transform("nunique")
    df["_distinct_devices_2h"] = daily_devices.fillna(1).astype(int)
    df.drop(columns=["_txn_date"], inplace=True, errors="ignore")
    print(f"  _distinct_devices_2h computed (per-day per-account)")


# ── 9. High-risk MCC features ──
HIGH_RISK_MCCS = {"7995","7994","7801","7802","6051","6050","6012","5816","5817","5818","5967","5966"}
mcc_col = next((c for c in ["merchant_category_code", "_mcc"] if c in df.columns), None)

df["_high_risk_mcc_pct"] = 0.0
df["_high_risk_mcc_sum_30d"] = 0.0
df["_is_high_risk_mcc"] = False

if mcc_col and "_acct" in df.columns:
    df["_is_high_risk_mcc"] = df[mcc_col].astype(str).isin(HIGH_RISK_MCCS)

    acct_total = df.groupby("_acct").size().reset_index(name="_at")
    hr_mask = df["_is_high_risk_mcc"] == True
    if hr_mask.any():
        acct_hr = df[hr_mask].groupby("_acct").size().reset_index(name="_ah")
        mcc_stats = acct_total.merge(acct_hr, on="_acct", how="left")
        mcc_stats["_ah"] = mcc_stats["_ah"].fillna(0)
        mcc_stats["_pct"] = mcc_stats["_ah"] / mcc_stats["_at"]
        df = df.merge(mcc_stats[["_acct", "_pct"]], on="_acct", how="left")
        df["_high_risk_mcc_pct"] = df["_pct"].fillna(0)
        df.drop(columns=["_pct"], inplace=True)

        hr_sums = df[hr_mask].groupby("_acct")["_amt"].sum().reset_index(name="_tmp")
        df = df.merge(hr_sums, on="_acct", how="left")
        df["_high_risk_mcc_sum_30d"] = df["_tmp"].fillna(0)
        df.drop(columns=["_tmp"], inplace=True)
        print(f"  _high_risk_mcc_pct / _high_risk_mcc_sum_30d computed from {mcc_col}")
    else:
        print(f"  No high-risk MCC transactions found")

# ── 10. Refund flag and count ──
df["_refund_flag"] = False
df["_refund_count_merchant_30d"] = 0
refund_col = next((c for c in ["refund_chargeback_flag", "refund_flag"] if c in df.columns), None)
if refund_col:
    df["_refund_flag"] = df[refund_col].astype(str).str.upper().isin(["Y", "YES", "1", "TRUE"])
    merchant_col = next((c for c in ["merchant_id", "_merchant_id"] if c in df.columns), None)
    if merchant_col and "_acct" in df.columns:
        ref_mask = df["_refund_flag"] == True
        if ref_mask.any():
            refund_counts = df[ref_mask].groupby(["_acct", merchant_col]).size().reset_index(name="_rc")
            refund_max = refund_counts.groupby("_acct")["_rc"].max().reset_index(name="_tmp")
            df = df.merge(refund_max, on="_acct", how="left")
            df["_refund_count_merchant_30d"] = df["_tmp"].fillna(0).astype(int)
            df.drop(columns=["_tmp"], inplace=True)
    print(f"  _refund_flag / _refund_count_merchant_30d computed")

# ── 11. Merchant MCC change count ──
df["_merchant_mcc_change_count"] = 0
merchant_col = next((c for c in ["merchant_id", "_merchant_id"] if c in df.columns), None)
if merchant_col and mcc_col:
    valid_merchant = df[merchant_col].astype(str).str.strip()
    valid_mask = (valid_merchant != "") & (valid_merchant != "nan")
    if valid_mask.any():
        mcc_changes = df[valid_mask].groupby(merchant_col)[mcc_col].nunique().reset_index(name="_tmp")
        df = df.merge(mcc_changes, on=merchant_col, how="left")
        df["_merchant_mcc_change_count"] = (df["_tmp"].fillna(1) - 1).clip(lower=0).astype(int)
        df.drop(columns=["_tmp"], inplace=True)
        print(f"  _merchant_mcc_change_count computed")

# ── 12. Name similarity ──
df["_name_similarity"] = 1.0
cust_name_col = next((c for c in ["customer_name", "_cust_name"] if c in df.columns), None)
cp_name_col = next((c for c in ["counterparty_name", "_cp_name"] if c in df.columns), None)
if cust_name_col and cp_name_col:
    from difflib import SequenceMatcher
    def quick_sim(row):
        a = str(row[cust_name_col]).upper().strip()
        b = str(row[cp_name_col]).upper().strip()
        if not a or not b or a == "NAN" or b == "NAN":
            return 1.0
        return SequenceMatcher(None, a, b).ratio()
    df["_name_similarity"] = df.apply(quick_sim, axis=1)
    print(f"  _name_similarity computed")

# ── 13. PPI wallet aggregations ──
pan_col = next((c for c in ["pan", "customer_pan"] if c in df.columns), None)
mobile_col = next((c for c in ["mobile_number", "customer_mobile"] if c in df.columns), None)
ip_col = next((c for c in ["ip_address"] if c in df.columns), None)
wallet_col = next((c for c in ["wallet_id", "ppi_wallet_id"] if c in df.columns), None)
vpa_col = next((c for c in ["beneficiary_wallet_id_vpa", "_bene_vpa"] if c in df.columns), None)
load_src_col = next((c for c in ["load_instrument_type", "load_source"] if c in df.columns), None)

df["_pan_wallet_count"] = 0
df["_mobile_wallet_count"] = 0
df["_ip_wallet_count"] = 0
df["_vpa_wallet_count"] = 0
df["_load_src_wallet_count"] = 0

wallet_aggs = [
    (pan_col, "_pan_wallet_count"),
    (mobile_col, "_mobile_wallet_count"),
    (ip_col, "_ip_wallet_count"),
    (vpa_col, "_vpa_wallet_count"),
]

for key_col, target_name in wallet_aggs:
    if key_col and wallet_col:
        valid = df[(df[wallet_col].astype(str).str.strip() != "") & (df[key_col].astype(str).str.strip() != "")]
        if len(valid) > 0:
            counts = valid.groupby(key_col)[wallet_col].nunique().reset_index(name="_tmp")
            df = df.merge(counts, on=key_col, how="left")
            df[target_name] = df["_tmp"].fillna(0).astype(int)
            df.drop(columns=["_tmp"], inplace=True)
            print(f"  {target_name} computed from {key_col}")

if load_src_col and wallet_col:
    valid = df[(df[wallet_col].astype(str).str.strip() != "") & (df[load_src_col].astype(str).str.strip() != "")]
    if len(valid) > 0:
        counts = valid.groupby(wallet_col)[load_src_col].nunique().reset_index(name="_tmp")
        df = df.merge(counts, on=wallet_col, how="left")
        df["_load_src_wallet_count"] = df["_tmp"].fillna(0).astype(int)
        df.drop(columns=["_tmp"], inplace=True)
        print(f"  _load_src_wallet_count computed from {load_src_col}")

# ── 14. High-risk region flag ──
HIGH_RISK_REGIONS = {"J&K", "JAMMU", "KASHMIR", "NORTHEAST", "MANIPUR", "NAGALAND", "MIZORAM"}
df["_in_high_risk_region"] = False
region_col = next((c for c in ["customer_state", "ip_state", "customer_city"] if c in df.columns), None)
if region_col:
    df["_in_high_risk_region"] = df[region_col].astype(str).str.upper().apply(
        lambda x: any(r in x for r in HIGH_RISK_REGIONS)
    )
    print(f"  _in_high_risk_region computed from {region_col}")

# ═══ Final summary ═══
computed = [c for c in target_cols if c in df.columns]
missing = [c for c in target_cols if c not in df.columns]
print(f"\n  Derived features computed: {len(computed)}/{len(target_cols)}")
if missing:
    print(f"  Still missing: {missing}")
else:
    print(f"  All derived features ready. Rule warnings should be resolved.")
print(f"  DataFrame shape: {df.shape}")


# ## 6 — Rule Definitions (151 Rules = 76 Bank + 75 PPI)
# 17 overlapping rules merged into single columns with conditional PPI/Bank thresholds.
# Each rule: `(col_name, group, readable, severity, condition_fn, regulatory_ref, threshold_type)`
# 

# In[12]:


RULES = [
    # ═══ G1: FREQUENCY ANOMALY (7 rules) ═══
    ("rule_freq_2x_business", "Frequency", "Freq >2x Baseline (Business)", 2,
     lambda d: d["_is_entity"] & (d["_cust_baseline_freq_30d"]>0) & (d["_cust_vel_count_7d"]>d["_cust_baseline_freq_30d"]*2*7/30) & (d["_amt"]>=d["_dyn_thresh"]),
     "Bank-1", "DYN"),
    # Rationale: 2x baseline is unusual but not alarming — could be seasonal business spike

    ("rule_freq_2x_individual", "Frequency", "Freq >2x Baseline (Individual)", 2,
     lambda d: (~d["_is_entity"]) & (d["_cust_baseline_freq_30d"]>0) & (d["_cust_vel_count_7d"]>d["_cust_baseline_freq_30d"]*2*7/30) & (d["_amt"]>=d["_dyn_thresh"]),
     "Bank-2", "DYN"),
    # Rationale: Same as above — individuals may have wedding/medical expenses causing spikes

    ("rule_series_credits_7d", "Frequency", "Series of Credits 7d", 2,
     lambda d: d["_is_credit"] & (d["_acct_vel_count_7d"]>10) & (d["_amt"]>=d["_dyn_thresh"]),
     "Bank-3,5", "DYN"),
    # Rationale: Series of credits could be salary + freelance + refunds — common for gig workers

    ("rule_series_debits_7d", "Frequency", "Series of Debits 7d", 1,
     lambda d: d["_is_debit"] & (d["_acct_vel_count_7d"]>10) & (d["_amt"]>=d["_dyn_thresh"]),
     "Bank-4,6", "DYN"),
    # Rationale: Series of debits is normal spending behavior — lowest concern

    ("rule_repeated_counterparty_7d", "Frequency", "Repeated Counterparty 7d", 1,
     lambda d: (d["_acct_vel_count_7d"]>3) & (d["_cp_acct"]!="") & (d["_amt"]>=d["_dyn_thresh"]),
     "Bank-7,8", "DYN"),
    # Rationale: Repeated counterparty is often landlord rent, EMI, regular supplier — very common

    ("rule_off_hours_activity", "Frequency", "Off-Hours High Activity", 2,
     lambda d: ((d["_is_ppi"]) & (d["_hour"]>=0) & (d["_hour"]<5) & (d["_acct_vel_count_24hr"]>5)) | ((~d["_is_ppi"]) & d["_is_night"] & (d["_amt"]>d["_dyn_thresh"])),
     "Bank-42|PPI-9", "DYN"),
    # Rationale: 12AM-5AM high-value activity is unusual — warrants review but could be NRI timezone

    ("rule_cross_entity_common", "Frequency", "Cross-Entity Common Person", 2,
     lambda d: d["_is_entity"] & (d["_cust_vel_count_24hr"]>5) & (d["_amt"]>=d["_dyn_thresh"]),
     "Bank-10", "DYN"),
    # Rationale: Director controlling multiple entities with rapid transfers — potential layering

    # ═══ G2: CUMULATIVE BAND (6 rules) ═══
    ("rule_integrated_cash_8_5_10L", "Cumulative", "Integrated Cash 8.5-10L 30d", 3,
     lambda d: (d["_cum_cash_30d"]>=850000)&(d["_cum_cash_30d"]<1000000),
     "Bank-11", "CUM"),
    # Rationale: Just below 10L CTR threshold — classic structuring signal, mandatory RBI reporting

    ("rule_foreign_remit_4_5L", "Cumulative", "Foreign Remit 4-5L 30d", 3,
     lambda d: (d["_cum_foreign_30d"]>=400000)&(d["_cum_foreign_30d"]<500000),
     "Bank-12", "CUM"),
    # Rationale: Just below 5L foreign remittance threshold — FEMA/RBI reporting trigger

    ("rule_npo_receipts_8_5_10L", "Cumulative", "NPO Receipts 8.5-10L", 3,
     lambda d: d["_is_trust_society"]&(d["_cum_credits_30d"]>=850000)&(d["_cum_credits_30d"]<1000000),
     "Bank-13", "CUM"),
    # Rationale: FCRA-regulated entity near threshold — high regulatory risk

    ("rule_structuring_pattern", "Cumulative", "Structuring Sub-Threshold", 3,
     lambda d: ((d["_is_ppi"]) & d["_is_load"] & (d["_amt"]<9500) & (d["_wallet_load_sum_7d"]>50000)) | ((~d["_is_ppi"]) & d["_cash"] & (d["_amt"]>=8000) & (d["_amt"]<=9999) & (d["_acct_vel_count_7d"]>3)),
     "Bank-11,16|PPI-3", "CUM"),
    # Rationale: Core structuring/smurfing indicator — RBI/FIU-IND top priority

    ("rule_multiple_parties_cash", "Cumulative", "Multiple Parties Cash Deposit 7d", 2,
     lambda d: d["_is_credit"]&d["_cash"]&(d["_distinct_depositors_7d"]>=3)&(~d["_is_entity"]),
     "Bank-16", "STATIC"),
    # Rationale: Multiple people depositing cash into one account — suspicious but could be joint family

    ("rule_round_amount_struct_7d", "Cumulative", "Round Amount Structuring >50K/7d", 2,
     lambda d: d["_is_round"]&(d["_acct_vel_sum_7d"]>50000)&(d["_acct_vel_count_7d"]>3),
     "PPI-26", "CUM"),
    # Rationale: Round amounts are common in India (rent, salary) — medium not high

    # ═══ G3: CREDIT-DEBIT SEQUENCE (6 rules) ═══
    ("rule_large_cr_dr_business", "CrDr Sequence", "Large Cr then Dr (Business) 30d", 1,
     lambda d: d["_is_entity"]&(d["_cum_credits_30d"]>d["_dyn_thresh"])&(d["_cum_debits_30d"]>d["_dyn_thresh"])&(d["_cum_debits_30d"]>=d["_cum_credits_30d"]*0.60),
     "Bank-14", "DYN"),
    # Rationale: Businesses naturally receive and spend — 60% outflow is normal operations

    ("rule_large_cr_dr_individual", "CrDr Sequence", "Large Cr then Dr (Individual) 30d", 1,
     lambda d: (~d["_is_entity"])&(d["_cum_credits_30d"]>d["_dyn_thresh"])&(d["_cum_debits_30d"]>d["_dyn_thresh"])&(d["_cum_debits_30d"]>=d["_cum_credits_30d"]*0.60),
     "Bank-15", "DYN"),
    # Rationale: Individuals spend most of income — 60% is normal household budget

    ("rule_credit_then_cash", "CrDr Sequence", "Credit then Cash Withdrawal", 2,
     lambda d: d["_cash"]&d["_is_debit"]&(d["_cum_credits_30d"]>100000)&(d["_amt"]>=d["_cum_credits_30d"]*0.50),
     "Bank-18", "CUM"),
    # Rationale: Receiving electronic then withdrawing 50%+ as cash — potential conversion

    ("rule_remit_then_cash_75pct", "CrDr Sequence", "Remittance then Cash >=75%", 3,
     lambda d: d["_cash"]&d["_is_debit"]&(d["_cum_foreign_30d"]>0)&(d["_amt"]>=d["_cum_foreign_30d"]*0.75),
     "Bank-21", "CUM"),
    # Rationale: Foreign remittance immediately converted to cash — strong hawala/conversion signal

    ("rule_remit_then_transfer", "CrDr Sequence", "Remittance then Transfer Out", 2,
     lambda d: d["_is_debit"]&d["_is_intrabank"]&(d["_cum_foreign_30d"]>100000)&(d["_amt"]>=d["_cum_foreign_30d"]*0.50),
     "Bank-23", "CUM"),
    # Rationale: Foreign funds re-routed domestically — could be legitimate investment or layering

    ("rule_rapid_load_transfer", "CrDr Sequence", "Rapid Load then Transfer (PPI)", 3,
     lambda d: d["_is_ppi"]&(d["_is_p2p"]|d["_is_p2m"])&(d["_wallet_load_sum_7d"]>0)&(d["_amt"]>=d["_wallet_load_sum_7d"]*0.90),
     "PPI-4,51", "CUM"),
    # Rationale: Load wallet then immediately transfer 90%+ — classic pass-through/mule pattern

    # ═══ G4: LIFECYCLE (9 rules) ═══
    ("rule_dormant_activation", "Lifecycle", "Dormant Account/Wallet Activation", 3,
     lambda d: ((d["_is_ppi"]) & (d["_prev_txn_gap_days"]>90) & (d["_amt"]>5000)) | ((~d["_is_ppi"]) & d["_dormancy"] & (d["_amt"]>5000)),
     "Bank-24,25|PPI-7", "STATIC"),
    # Rationale: Dormant account suddenly used for high value — strong mule/takeover signal

    ("rule_acct_closed_90d", "Lifecycle", "Account Closed Within 90d", 2,
     lambda d: (d["_acct_open_days"]>0)&(d["_acct_open_days"]<90)&(d["_acct_vel_count_7d"]<=5)&(d["_amt"]>10000),
     "Bank-17", "STATIC"),
    # Rationale: Very new account with high value — needs monitoring but many new accounts are legitimate

    ("rule_unusual_type_spike", "Lifecycle", "Unusual Txn Type Spike", 1,
     lambda d: d["_amt_zscore_30d"]>4,
     "Bank-22", "STATIC"),
    # Rationale: Statistical outlier — could be one-time purchase (car, jewelry) — low concern alone

    ("rule_dormant_business_15d", "Lifecycle", "Dormant Business 15d", 1,
     lambda d: d["_is_entity"]&(d["_prev_txn_gap_days"]>15)&(d["_amt"]>5000),
     "Bank-24", "STATIC"),
    # Rationale: Businesses have seasonal gaps — 15 days is very short dormancy period

    ("rule_new_wallet_high_value", "Lifecycle", "New Wallet <7d High Value (PPI)", 3,
     lambda d: d["_is_ppi"]&(d["_acct_open_days"]>=0)&(d["_acct_open_days"]<7)&(d["_acct_vel_sum_7d"]>25000),
     "PPI-13", "STATIC"),
    # Rationale: Brand new wallet immediately transacting high value — strong mule/fraud signal

    ("rule_balance_parking", "Lifecycle", "Balance Parking High Bal Low Activity (PPI)", 1,
     lambda d: d["_is_ppi"]&(d["_wallet_bal_before"]>25000)&(d["_acct_vel_count_7d"]<1),
     "PPI-15", "STATIC"),
    # Rationale: Holding balance without activity — could be saving, not necessarily suspicious

    ("rule_zero_balance_cycling", "Lifecycle", "Zero Balance Drain Cycling (PPI)", 3,
     lambda d: d["_is_ppi"]&(d["_wallet_bal_after"]<10)&(d["_wallet_bal_before"]>1000)&(d["_acct_vel_count_7d"]>3),
     "PPI-31", "STATIC"),
    # Rationale: Repeatedly draining wallet to zero — classic mule behavior

    ("rule_short_lived_wallet", "Lifecycle", "Short-Lived Wallet High Throughput (PPI)", 3,
     lambda d: d["_is_ppi"]&(d["_acct_open_days"]<30)&(d["_acct_vel_sum_7d"]>50000),
     "PPI-37", "STATIC"),
    # Rationale: New wallet with high throughput — strong mule indicator

    ("rule_monthly_limit_exhaustion", "Lifecycle", "Monthly Limit Exhaustion (PPI)", 2,
     lambda d: d["_is_ppi"]&(d["_acct_vel_sum_7d"]>d["_monthly_limit"]*0.95),
     "PPI-35", "STATIC"),
    # Rationale: Pushing wallet limits — suspicious but could be legitimate heavy user

    # ═══ G5: DIRECT FLAGS (12 rules) ═══
    ("rule_negative_list_country", "Direct Flag", "FATF/Negative List Country", 3,
     lambda d: d["_sender_high_risk_cc"]|d["_receiver_high_risk_cc"],
     "Bank-26|PPI-61,62,63,64", "FLAG"),
    # Rationale: FATF blacklist/greylist — mandatory enhanced due diligence per RBI

    ("rule_sole_prop_personal", "Direct Flag", "Sole Prop Personal Account", 1,
     lambda d: d["_occupation"].str.contains("sole.*proprietor|self.*employed|business.*owner",case=False,na=False)&d["_acct_cat"].str.contains("saving|salary",case=False,na=False)&(d["_amt"]>50000),
     "Bank-27,28,29", "FLAG"),
    # Rationale: Very common in India — self-employed using savings account is normal practice

    ("rule_tax_haven_remit", "Direct Flag", "Tax Haven Remittance", 3,
     lambda d: d["_involves_tax_haven"]&(d["_amt"]>10000),
     "Bank-30", "FLAG"),
    # Rationale: Tax haven transfers — mandatory regulatory scrutiny

    ("rule_loan_repay_cash", "Direct Flag", "Loan Repayment Cash", 1,
     lambda d: d["_cash"]&d["_channel"].str.contains("LOAN|EMI|REPAY",case=False,na=False),
     "Bank-31", "FLAG"),
    # Rationale: Cash loan repayment is common in rural/semi-urban India

    ("rule_insurance_lump", "Direct Flag", "Insurance Premium Lump Sum", 1,
     lambda d: d["_channel"].str.contains("INSURANCE|PREMIUM|LIC|POLICY",case=False,na=False)&(d["_amt"]>100000),
     "Bank-57", "FLAG"),
    # Rationale: Annual premium payments are normal — not suspicious by itself

    ("rule_attempted_failed", "Direct Flag", "Attempted/Failed Transaction", 1,
     lambda d: d["_is_failed"],
     "Bank-58|PPI-8", "FLAG"),
    # Rationale: System failures, network issues, wrong PIN — overwhelmingly innocent

    ("rule_cc_cash_1L", "Direct Flag", "Credit Card Cash >=1L", 3,
     lambda d: d["_cash"]&d["_channel"].str.contains("CREDIT.*CARD|CC.*PAY",case=False,na=False)&(d["_amt"]>=100000),
     "Bank-60", "STATIC"),
    # Rationale: Large credit card cash advance — potential money conversion scheme

    ("rule_ppi_small_kyc_load_breach", "Direct Flag", "Small KYC Load Breach >10K (PPI)", 3,
     lambda d: d["_is_ppi"]&d["_is_small_kyc"]&d["_is_load"]&(d["_amt"]>10000),
     "PPI-1", "FLAG"),
    # Rationale: RBI limit violation — regulatory non-compliance

    ("rule_ppi_small_kyc_bal_breach", "Direct Flag", "Small KYC Balance Breach >10K (PPI)", 3,
     lambda d: d["_is_ppi"]&d["_is_small_kyc"]&(d["_wallet_bal_after"]>10000),
     "PPI-2", "FLAG"),
    # Rationale: RBI limit violation — regulatory non-compliance

    ("rule_ppi_client_reported", "Direct Flag", "Client-Reported Suspicious (PPI)", 3,
     lambda d: d["_wallet_reported"],
     "PPI-74", "FLAG"),
    # Rationale: Client/merchant reported — direct escalation required

    ("rule_ppi_negative_list_device", "Direct Flag", "Device/IP/Mobile Negative List (PPI)", 3,
     lambda d: d["_device_on_negative"] | d["_ip_on_negative"] | d["_vpa_on_negative"] | d["_merchant_on_negative"],
     "PPI-65,66", "FLAG"),
    # Rationale: Known bad device/IP — strong fraud signal

    ("rule_ppi_adverse_media", "Direct Flag", "Adverse Media Match (PPI)", 3,
     lambda d: d["_cp_name_on_negative"] | d["_merchant_on_negative"],
     "PPI-67", "FLAG"),
    # Rationale: Counterparty in adverse media — regulatory requirement

    # ═══ G6: LARGE TXN DYNAMIC (10 rules) ═══
    ("rule_large_intra_business", "Large Txn DYN", "Large Intrabank (Business)", 1,
     lambda d: d["_is_entity"]&d["_is_intrabank"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-32", "DYN"),
    # Rationale: Intrabank business transfers are normal treasury operations

    ("rule_large_intra_individual", "Large Txn DYN", "Large Intrabank (Individual)", 1,
     lambda d: (~d["_is_entity"])&d["_is_intrabank"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-33", "DYN"),
    # Rationale: Moving money between own accounts — common for FD creation, loan prep

    ("rule_large_cash_business", "Large Txn DYN", "Large Cash (Business)", 2,
     lambda d: d["_is_entity"]&d["_cash"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-34", "DYN"),
    # Rationale: Large business cash — CTR threshold awareness, needs monitoring

    ("rule_large_cash_individual", "Large Txn DYN", "Large Cash (Individual)", 2,
     lambda d: (~d["_is_entity"])&d["_cash"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-35", "DYN"),
    # Rationale: Individual large cash — potential structuring avoidance

    ("rule_pep_large_any", "Large Txn DYN", "PEP Large Any Type", 3,
     lambda d: d["_pep"]&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-43", "DYN"),
    # Rationale: PEP enhanced due diligence — mandatory per PMLA

    ("rule_partnership_sleeping", "Large Txn DYN", "Partnership Sleeping Partner Large", 2,
     lambda d: d["_is_partnership"]&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-44", "DYN"),
    # Rationale: Sleeping partner with large transactions — potential shell entity

    ("rule_large_foreign_indiv", "Large Txn DYN", "Large Foreign Remit (Individual)", 2,
     lambda d: (~d["_is_entity"])&d["_is_foreign_remittance"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-47", "DYN"),
    # Rationale: Individual large foreign remittance — FEMA monitoring required

    ("rule_large_foreign_biz", "Large Txn DYN", "Large Foreign Remit (Business)", 2,
     lambda d: d["_is_entity"]&d["_is_foreign_remittance"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-48", "DYN"),
    # Rationale: Business foreign remittance — trade-based laundering risk

    ("rule_fx_cash_large", "Large Txn DYN", "FX Cash Large", 3,
     lambda d: d["_cash"]&d["_is_foreign_remittance"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-49", "DYN"),
    # Rationale: Foreign remittance + cash = strong hawala/conversion indicator

    ("rule_trust_large_cash", "Large Txn DYN", "Trust Large Cash", 3,
     lambda d: d["_is_trust_society"]&d["_cash"]&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-51,54", "DYN"),
    # Rationale: Trust/Society handling large cash — FCRA/NPO misuse risk

    # ═══ G7: NEW ACCOUNT / INCOME (8 rules) ═══
    ("rule_new_indiv_cash_30pct", "New Acct Income", "New Individual Cash >30% Income", 2,
     lambda d: (~d["_is_entity"])&(d["_acct_open_days"]<180)&d["_cash"]&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.30),
     "Bank-36", "PCT"),
    # Rationale: New account + large cash relative to income — needs monitoring

    ("rule_new_indiv_noncash_50pct", "New Acct Income", "New Individual Non-Cash >50% Income", 1,
     lambda d: (~d["_is_entity"])&(d["_acct_open_days"]<180)&(~d["_cash"])&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.50),
     "Bank-37", "PCT"),
    # Rationale: Non-cash 50% of income could be property/vehicle purchase — common

    ("rule_new_biz_noncash_50pct", "New Acct Income", "New Business Non-Cash >50%", 1,
     lambda d: d["_is_entity"]&(d["_acct_open_days"]<180)&(~d["_cash"])&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.50),
     "Bank-38", "PCT"),
    # Rationale: New business with large non-cash — startup capital injection is normal

    ("rule_new_biz_cash_30pct", "New Acct Income", "New Business Cash >30%", 2,
     lambda d: d["_is_entity"]&(d["_acct_open_days"]<180)&d["_cash"]&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.30),
     "Bank-39", "PCT"),
    # Rationale: New business + large cash — potential front company

    ("rule_new_biz_25pct", "New Acct Income", "New Business <6mo >25%", 1,
     lambda d: d["_is_entity"]&(d["_acct_open_days"]<180)&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.25),
     "Bank-40", "PCT"),
    # Rationale: 25% of declared turnover is routine business — low concern

    ("rule_new_employed_25pct", "New Acct Income", "Newly Employed >25% Income", 1,
     lambda d: (~d["_is_entity"])&(d["_acct_open_days"]<180)&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.25),
     "Bank-41", "PCT"),
    # Rationale: New employee spending 25% income — furniture, deposits, relocation expenses

    ("rule_large_intra_same_cust", "New Acct Income", "Large Intrabank Same Customer", 1,
     lambda d: d["_is_intrabank"]&(d["_acct_vel_count_7d"]>5)&(d["_amt"]>DYN_T_R.get("MEDIUM",50000)),
     "Bank-52", "DYN"),
    # Rationale: Moving between own accounts — FD, RD, sweep — entirely normal

    ("rule_age_amount_mismatch", "New Acct Income", "Age <25 High Value", 1,
     lambda d: (d["_age"]>0)&(d["_age"]<25)&(d["_acct_vel_sum_7d"]>75000),
     "PPI-28", "STATIC"),
    # Rationale: Young professionals in IT earn well — 75K/week is normal for many under-25s

    # ═══ G8: DORMANT REACTIVATION (4 rules) ═══
    ("rule_dormant_75pct_drain", "Dormant React", "Dormant 75% Drain 7d", 3,
     lambda d: d["_was_inoperative"]&(d["_days_since_reactivation"]<7)&d["_is_debit"]&(d["_amt"]>d["_balance_proxy"]*0.75),
     "Bank-45", "STATIC"),
    # Rationale: Reactivated then 75% drained within a week — very strong mule signal

    ("rule_dormant_credit_50pct", "Dormant React", "Dormant Credit >50% Income", 2,
     lambda d: d["_was_inoperative"]&(d["_days_since_reactivation"]<30)&d["_is_credit"]&(~d["_cash"])&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.50),
     "Bank-46", "PCT"),
    # Rationale: Dormant account receives large credit — could be inheritance or layering

    ("rule_dormant_cash_30pct", "Dormant React", "Dormant Cash >30% Income", 3,
     lambda d: d["_was_inoperative"]&(d["_days_since_reactivation"]<30)&d["_is_credit"]&d["_cash"]&(d["_income"]>0)&(d["_amt"]>d["_income"]*0.30),
     "Bank-50", "PCT"),
    # Rationale: Dormant + cash deposit — strong structuring/smurfing indicator

    ("rule_dormant_activation_gen", "Dormant React", "Dormant Activation General", 1,
     lambda d: d["_dormancy"]&(d["_amt"]>5000),
     "Bank-24,25", "STATIC"),
    # Rationale: Generic dormant activation — 5K is low threshold, many legitimate reactivations

    # ═══ G9: ENTITY SPECIFIC (5 rules) ═══
    ("rule_trust_foreign_remit", "Entity", "Trust Foreign Remittance", 3,
     lambda d: d["_is_trust_society"]&d["_is_foreign_remittance"]&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-53", "DYN"),
    # Rationale: Trust receiving foreign funds — FCRA compliance mandatory

    ("rule_offshore_entity", "Entity", "Offshore Entity Transfer", 3,
     lambda d: d["_is_offshore"]&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-56", "DYN"),
    # Rationale: Offshore entity — high shell company / tax evasion risk

    ("rule_crypto_nft_large", "Entity", "Crypto/NFT Large", 3,
     lambda d: d["_channel"].str.contains("CRYPTO|NFT|BLOCKCHAIN",case=False,na=False)&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-55", "DYN"),
    # Rationale: Crypto large value — RBI has strict guidelines, mandatory reporting

    ("rule_bo_address_mismatch", "Entity", "BO Address != Reg Office", 1,
     lambda d: d["_is_entity"]&(d["_bo_city"]!="")&(d["_reg_city"]!="")&(d["_bo_city"]!=d["_reg_city"])&(d["_amt"]>DYN_T_R.get("MEDIUM",50000)),
     "Bank-64", "DYN"),
    # Rationale: Many businesses have BO in different city — branch offices, remote directors

    ("rule_ppi_merchant_passthrough", "Entity", "Merchant Pass-Through (PPI)", 3,
     lambda d: d["_is_ppi"] & d["_is_p2m"] & (d["_merchant_id"] != "") & (d["_acct_vel_count_7d"] > 20) & (d["_refund_flag"]),
     "PPI-48,49", "FLAG"),
    # Rationale: Merchant with high refunds + high velocity — fraud ring indicator

    # ═══ G10: INTRABANK / FREQUENCY (5 rules) ═══
    ("rule_intra_high_freq", "Intrabank", "Intrabank High Freq Same Customer", 1,
     lambda d: d["_is_intrabank"]&(d["_acct_vel_count_7d"]>10),
     "Bank-19", "STATIC"),
    # Rationale: Moving between own accounts is normal banking — FD, sweep, salary allocation

    ("rule_high_freq_foreign", "Intrabank", "High Freq Foreign Remittance 7d", 2,
     lambda d: d["_is_foreign_remittance"]&(d["_acct_vel_count_7d"]>5),
     "Bank-20", "STATIC"),
    # Rationale: Multiple foreign remittances in a week — needs FEMA monitoring

    ("rule_rapid_burst", "Intrabank", "Rapid Burst / High Velocity P2P", 3,
     lambda d: ((d["_is_ppi"]) & d["_is_p2p"] & (d["_wallet_p2p_count_1hr"]>10)) | ((~d["_is_ppi"]) & (d["_acct_vel_count_1hr"]>3)),
     "Bank-9|PPI-5", "STATIC"),
    # Rationale: >3 transactions per hour or >10 P2P per hour — bot/automated behavior

    ("rule_sole_prop_burst", "Intrabank", "Sole Prop Burst", 2,
     lambda d: d["_occupation"].str.contains("sole.*proprietor|self.*employed",case=False,na=False)&(d["_acct_vel_count_1hr"]>3)&(d["_amt"]>=d["_dyn_thresh"]),
     "Bank-9", "DYN"),
    # Rationale: Sole proprietor rapid high-value — potential cash business structuring

    ("rule_early_morning_cluster", "Intrabank", "Early Morning 4-6AM Cluster (PPI)", 2,
     lambda d: d["_is_ppi"]&(d["_hour"]>=4)&(d["_hour"]<6)&(~d["_is_weekend"])&(d["_acct_vel_count_24hr"]>5),
     "PPI-34", "STATIC"),
    # Rationale: 4-6AM weekday PPI cluster — unusual timing for legitimate activity

    # ═══ G11: DEVICE / DIGITAL (15 rules) ═══
    ("rule_multi_channel_24h", "Device", "Multi-Channel >=4 in 24h", 1,
     lambda d: d["_distinct_channels_24h"]>=4,
     "Bank-61|PPI-61", "STATIC"),
    # Rationale: Using 4+ channels is increasingly common — UPI, NEFT, ATM, POS in one day is normal

    ("rule_vpn_emulator_detected", "Device", "VPN/Proxy/Emulator/TOR Detected", 2,
     lambda d: (d["_vpn"]|d["_emulator"]),
     "Bank-62|PPI-36", "FLAG"),
    # Rationale: VPN use is growing for privacy — suspicious but not conclusive alone

    ("rule_auth_degrade_high_risk", "Device", "Auth Degradation + High Risk", 3,
     lambda d: d["_risk"].isin(["HIGH","VERY_HIGH","VERY HIGH","3"])&(d["_vpn"]|d["_emulator"])&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-62", "DYN"),
    # Rationale: High risk customer + VPN + large amount — strong fraud convergence

    ("rule_tax_res_mismatch_xborder", "Device", "Tax Residency Mismatch Cross-Border", 3,
     lambda d: d["_tax_res"].isin(["INDIA","IN"])&d["_is_cross_border"]&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-63", "DYN"),
    # Rationale: Indian tax resident sending cross-border — FEMA/DTAA scrutiny

    ("rule_crypto_then_cash_48h", "Device", "Crypto then Cash 48h", 3,
     lambda d: d["_cash"]&d["_is_debit"]&(d["_cum_credits_30d"]>50000)&(d["_amt"]>=d["_cum_credits_30d"]*0.75),
     "Bank-65|PPI-65", "CUM"),
    # Rationale: Credits followed by large cash withdrawal — conversion pattern

    ("rule_new_device_new_loc", "Device", "New Device + New City + Large", 2,
     lambda d: ((d["_is_ppi"]) & d["_is_new_device"] & (d["_amt"]>10000)) | ((~d["_is_ppi"]) & d["_is_new_device"] & d["_is_new_city"] & (d["_amt"]>d["_dyn_thresh"])),
     "Bank-66|PPI-19", "DYN"),
    # Rationale: New device from new city — could be travel or account takeover

    ("rule_device_hopping", "Device", "Device/IP Hopping", 3,
     lambda d: ((d["_is_ppi"]) & (d["_distinct_devices_2h"]>=3)) | ((~d["_is_ppi"]) & (d["_distinct_devices_2h"]>=3) & (d["_amt"]>d["_dyn_thresh"])),
     "Bank-67|PPI-57", "DYN"),
    # Rationale: 3+ devices in 2 hours — strong bot/fraud ring indicator

    ("rule_vpn_emu_new_city", "Device", "VPN/Emu + New City + Large", 3,
     lambda d: (d["_vpn"]|d["_emulator"])&d["_is_new_city"]&(d["_amt"]>d["_dyn_thresh"]),
     "Bank-68", "DYN"),
    # Rationale: VPN + new city + large amount — strong takeover/fraud convergence

    ("rule_session_integrity", "Device", "Multi-Auth Multi-Device 30min", 3,
     lambda d: (d["_distinct_devices_2h"]>=2)&(d["_amt"]>d["_dyn_thresh_risk"]),
     "Bank-69|PPI-58", "DYN"),
    # Rationale: Multiple devices authenticating simultaneously — session hijack

    ("rule_shell_company", "Device", "Shell Company Anomaly", 3,
     lambda d: d["_is_entity"]&(d["_acct_vel_count_7d"]>20)&(d["_acct_vel_sum_7d"]<d["_balance_proxy"]*0.20),
     "Bank-70", "STATIC"),
    # Rationale: High frequency but tiny amounts relative to balance — potential shell

    ("rule_impossible_travel", "Device", "Impossible Travel / Location Anomaly", 2,
     lambda d: (d["_is_new_city"])&(d["_amt"]>10000)&(d["_acct_vel_count_1hr"]>0),
     "PPI-10,56", "STATIC"),
    # Rationale: Different city within an hour — suspicious but VPN can cause false positives

    ("rule_ip_geo_change", "Device", "IP Geo Change from 30d Pattern (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_new_city"]&(d["_amt"]>5000),
     "PPI-20", "STATIC"),
    # Rationale: IP location change is common — travel, mobile networks, VPN

    ("rule_ppi_cross_state_p2p", "Device", "Cross-State P2P >5 Cities/24h (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_p2p"]&(d["_distinct_channels_24h"]>=4),
     "PPI-27", "STATIC"),
    # Rationale: Channel diversity doesn't necessarily mean cross-state — low signal

    ("rule_auth_failure_spike", "Device", "Auth Failure Spike (PPI)", 2,
     lambda d: d["_is_failed"] & (d["_acct_vel_count_1hr"] > 3),
     "PPI-30,59,60", "FLAG"),
    # Rationale: Multiple failures in an hour — could be brute force or user error

    ("rule_geo_mismatch_2000km", "Device", "Geo Mismatch >2000km (PPI)", 2,
     lambda d: d["_is_ppi"]&d["_is_cross_border"]&(d["_amt"]>50000),
     "PPI-70", "STATIC"),
    # Rationale: Cross-border PPI with large amount — needs review

    # ═══ G12: OCCUPATION / PROFILE (9 rules) ═══
    ("rule_student_high_value", "Occupation", "Student High Value", 1,
     lambda d: d["_occupation"].str.contains("student",case=False,na=False)&(d["_amt"]>50000),
     "", "STATIC"),
    # Rationale: Students receive education loans, parental transfers — 50K is common

    ("rule_unemployed_large", "Occupation", "Unemployed Large Transfer", 1,
     lambda d: d["_occupation"].str.contains("unemployed|retired",case=False,na=False)&(d["_amt"]>20000),
     "", "STATIC"),
    # Rationale: Retired people have pensions, FD interest, property income — 20K is routine

    ("rule_freelancer_offshore", "Occupation", "Freelancer Offshore", 1,
     lambda d: d["_occupation"].str.contains("freelance",case=False,na=False)&d["_is_offshore"],
     "", "STATIC"),
    # Rationale: Freelancers routinely receive international payments — Upwork, Fiverr, etc.

    ("rule_low_income_large", "Occupation", "Low Income Large Txn", 1,
     lambda d: (d["_income"]>0)&(d["_income"]<300000)&(d["_amt"]>50000),
     "", "STATIC"),
    # Rationale: Low income + 50K could be loan disbursement, family gift, property sale

    ("rule_high_risk_industry", "Occupation", "High Risk Industry", 1,
     lambda d: d["_occupation"].str.contains("real estate|construction|unknown|mining|gambling",case=False,na=False),
     "", "FLAG"),
    # Rationale: Industry flag alone — no transaction anomaly, just a profile marker

    ("rule_pep_high_velocity", "Occupation", "PEP + High Velocity/Circular", 3,
     lambda d: d["_pep"]&((d["_acct_vel_count_24hr"]>10)|(d["_acct_vel_sum_24hr"]>1000000)),
     "PPI-71", "STATIC"),
    # Rationale: PEP with extreme velocity — mandatory PMLA enhanced monitoring

    ("rule_minor_anomalous", "Occupation", "Minor Anomalous Profile", 2,
     lambda d: d["_minor"]&((d["_acct_vel_sum_7d"]>50000)|(d["_amt"]>25000)),
     "PPI-73", "STATIC"),
    # Rationale: Minor with high value — needs guardian verification, potential misuse

    ("rule_nri_suspicious_domestic", "Occupation", "NRI Suspicious Domestic (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_residency"].str.contains("NRI|PIO|FOREIGN",case=False,na=False)&(d["_amt"]>10000),
     "PPI-68", "STATIC"),
    # Rationale: NRI using domestic PPI — increasingly common for NRIs visiting India

    ("rule_occupation_mcc_mismatch", "Occupation", "Occupation vs MCC Mismatch (PPI)", 2,
     lambda d: d["_is_ppi"] & d["_is_p2m"] & d["_occupation"].str.contains("agriculture|pension|farmer|retired|homemaker",case=False,na=False) & (d["_high_risk_mcc_pct"] > 0.50),
     "PPI-72", "FLAG"),
    # Rationale: Farmer spending 50%+ on gambling/crypto MCCs — meaningful mismatch

    # ═══ G13: PPI WALLET RULES (32 rules) ═══
    ("rule_ppi_load_then_spend", "PPI Wallet", "Load then P2M/P2P <30min", 2,
     lambda d: d["_is_ppi"]&(d["_is_p2p"]|d["_is_p2m"])&(d["_wallet_load_sum_7d"]>0)&(d["_amt"]>d["_wallet_load_sum_7d"]*0.50),
     "PPI-11", "STATIC"),
    # Rationale: Load and spend is normal PPI behavior — 50% threshold is generous

    ("rule_ppi_high_risk_mcc", "PPI Wallet", "High-Risk MCC >70% Monthly (PPI)", 3,
     lambda d: d["_is_ppi"] & d["_is_p2m"] & (d["_high_risk_mcc_pct"] > 0.70),
     "PPI-12", "FLAG"),
    # Rationale: 70%+ spend on gambling/crypto — strong addiction or laundering signal

    ("rule_ppi_multi_bene_day", "PPI Wallet", "P2P >15 Unique Beneficiaries/Day (PPI)", 3,
     lambda d: d["_is_ppi"]&d["_is_p2p"]&(d["_wallet_unique_bene_24hr"]>15),
     "PPI-14", "STATIC"),
    # Rationale: 15+ unique recipients in a day — distribution/mule behavior

    ("rule_ppi_single_source_load", "PPI Wallet", "Repeated Load Single Source (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_load"]&(d["_wallet_load_count_7d"]>10),
     "PPI-16", "STATIC"),
    # Rationale: Loading from same source repeatedly — could be salary/allowance

    ("rule_ppi_split_transfers", "PPI Wallet", "Split Transfer Pattern (PPI)", 2,
     lambda d: d["_is_ppi"]&d["_is_p2p"]&(d["_wallet_unique_bene_24hr"]>=3)&d["_is_round"],
     "PPI-17", "STATIC"),
    # Rationale: Round amounts to 3+ people — potential structuring or bill splitting

    ("rule_ppi_new_bene_burst", "PPI Wallet", "P2P >10 New Beneficiaries 7d (PPI)", 2,
     lambda d: d["_is_ppi"]&d["_is_p2p"]&(d["_wallet_unique_bene_7d"]>10),
     "PPI-18", "STATIC"),
    # Rationale: Many new recipients — could be gifting season or distribution

    ("rule_ppi_refund_abuse", "PPI Wallet", "Merchant Refund Abuse (PPI)", 3,
     lambda d: d["_is_ppi"] & d["_refund_flag"] & (d["_refund_count_merchant_30d"] > 5) & (d["_amt"] > 3000),
     "PPI-21", "FLAG"),
    # Rationale: 5+ refunds from same merchant — fraud ring indicator

    ("rule_ppi_kyc_expiry_limit", "PPI Wallet", "KYC Expiry + Limit 95% (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_small_kyc"]&(d["_acct_vel_sum_7d"]>d["_monthly_limit"]*0.90),
     "PPI-23", "STATIC"),
    # Rationale: Small KYC wallet near limit — common for active users, not alarming alone

    ("rule_ppi_load_source_change", "PPI Wallet", "Load Source Change >5/30d (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_load"]&(d["_wallet_load_sources_30d"]>5),
     "PPI-24", "STATIC"),
    # Rationale: Multiple funding sources — could be using different cards/accounts

    ("rule_ppi_bene_concentration", "PPI Wallet", "Single Beneficiary >60% (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_p2p"]&(d["_wallet_unique_bene_7d"]<=2)&(d["_acct_vel_sum_7d"]>10000),
     "PPI-25", "STATIC"),
    # Rationale: Sending to 1-2 people — landlord rent, family support — normal

    ("rule_ppi_multi_source_day", "PPI Wallet", "Multi Load Sources Same Day (PPI)", 2,
     lambda d: d["_is_ppi"]&d["_is_load"]&(d["_wallet_load_sources_30d"]>5),
     "PPI-29", "STATIC"),
    # Rationale: Multiple sources in one day is more suspicious than over a month

    ("rule_ppi_mcc_switch", "PPI Wallet", "Merchant MCC Switch (PPI)", 2,
     lambda d: d["_is_ppi"] & d["_is_p2m"] & (d["_merchant_mcc_change_count"] >= 2),
     "PPI-32", "FLAG"),
    # Rationale: Merchant changing category codes — potential MCC laundering

    ("rule_ppi_w2w_layering", "PPI Wallet", "Wallet-to-Wallet Layering >3 Hops (PPI)", 3,
     lambda d: d["_is_ppi"]&d["_is_p2p"]&(d["_wallet_unique_bene_24hr"]>3)&(d["_amt"]>5000),
     "PPI-33", "STATIC"),
    # Rationale: Multiple wallet hops with value — classic layering pattern

    ("rule_ppi_load_fail_alt", "PPI Wallet", "Load Fail then Alt Success (PPI)", 2,
     lambda d: d["_is_ppi"] & d["_is_load"] & d["_is_failed"] & (d["_acct_vel_count_1hr"] > 0),
     "PPI-38", "FLAG"),
    # Rationale: Failed then retry with different instrument — could be testing stolen cards

    ("rule_ppi_high_risk_mcc_cum", "PPI Wallet", "High-Risk MCC >30K/Month (PPI)", 2,
     lambda d: d["_is_ppi"] & d["_is_p2m"] & (d["_high_risk_mcc_sum_30d"] > 30000),
     "PPI-39", "FLAG"),
    # Rationale: Cumulative high-risk spend — pattern monitoring, not immediate alert

    ("rule_ppi_name_mismatch", "PPI Wallet", "Customer-Beneficiary Name Mismatch (PPI)", 1,
     lambda d: d["_is_ppi"] & d["_is_p2p"] & (d["_name_similarity"] < 0.40) & (d["_cp_name"] != ""),
     "PPI-40", "FLAG"),
    # Rationale: Name mismatch is common — nicknames, transliteration differences

    ("rule_ppi_kyc_behavior_breach", "PPI Wallet", "Small KYC Behavioral Breach (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_small_kyc"]&(d["_acct_vel_sum_7d"]>25000),
     "PPI-41", "STATIC"),
    # Rationale: 25K weekly for small KYC — may just be an active small wallet user

    ("rule_ppi_small_kyc_cross_state", "PPI Wallet", "Small KYC Cross-State P2P (PPI)", 1,
     lambda d: d["_is_ppi"]&d["_is_small_kyc"]&d["_is_p2p"]&d["_is_new_city"],
     "PPI-42", "STATIC"),
    # Rationale: Small KYC user in different city — could be traveling

    ("rule_ppi_multi_wallet_device", "PPI Wallet", "Multi-Wallet Same Device (PPI)", 3,
     lambda d: d["_is_ppi"]&(d["_distinct_devices_2h"]>=2)&(d["_acct_vel_sum_24hr"]>100000),
     "PPI-43", "STATIC"),
    # Rationale: Multiple wallets on same device + high value — strong fraud ring signal

    ("rule_ppi_multi_wallet_kyc", "PPI Wallet", "Multi-Wallet Same PAN/Aadhaar (PPI)", 3,
     lambda d: d["_is_ppi"] & (d["_pan_wallet_count"] > 3) & (d["_acct_vel_sum_7d"] > 50000),
     "PPI-44", "FLAG"),
    # Rationale: Same identity across 3+ wallets — circumventing KYC limits

    ("rule_ppi_shared_contact", "PPI Wallet", "Shared Contact Cluster (PPI)", 2,
     lambda d: d["_is_ppi"] & (d["_mobile_wallet_count"] > 5) & (d["_acct_vel_sum_7d"] > 50000),
     "PPI-45", "FLAG"),
    # Rationale: 5+ wallets on same mobile — potential but some families share phones

    ("rule_ppi_multi_wallet_funding", "PPI Wallet", "Multi-Wallet Same Funding (PPI)", 2,
     lambda d: d["_is_ppi"] & d["_is_load"] & (d["_load_src_wallet_count"] > 5) & (d["_amt"] > 10000),
     "PPI-46", "FLAG"),
    # Rationale: One funding source feeding many wallets — distribution network

    ("rule_ppi_shared_ip_cluster", "PPI Wallet", "Shared IP Wallet Cluster (PPI)", 2,
     lambda d: d["_is_ppi"] & (d["_ip_wallet_count"] > 10) & (d["_amt"] > 25000),
     "PPI-47", "FLAG"),
    # Rationale: 10+ wallets from same IP — could be office network or fraud farm

    ("rule_ppi_digital_content", "PPI Wallet", "High-Risk Digital Content (PPI)", 1,
     lambda d: d["_is_ppi"] & d["_is_p2m"] & (d["_high_risk_mcc_pct"] > 0.50) & (d["_high_risk_mcc_sum_30d"] > 25000),
     "PPI-50", "FLAG"),
    # Rationale: 50% on gaming/digital — could be legitimate gaming enthusiast

    ("rule_ppi_upi_carousel", "PPI Wallet", "UPI Handle Carousel (PPI)", 2,
     lambda d: d["_is_ppi"] & (d["_bene_vpa"] != "") & (d["_vpa_wallet_count"] > 5) & (d["_amt"] > 5000),
     "PPI-52", "FLAG"),
    # Rationale: Same VPA used across many wallets — potential layering

    ("rule_ppi_promo_referral", "PPI Wallet", "Promo Abuse Referral Loop (PPI)", 2,
     lambda d: d["_is_ppi"] & (d["_acct_open_days"] < 30) & (d["_amt"] < 500) & (d["_acct_vel_count_7d"] > 5) & (d["_pan_wallet_count"] > 2),
     "PPI-53", "FLAG"),
    # Rationale: New wallet + tiny amounts + multi-wallet — referral farming

    ("rule_ppi_cashback_cycling", "PPI Wallet", "Cashback Cycling (PPI)", 2,
     lambda d: d["_is_ppi"] & d["_is_p2p"] & (d["_amt"] < 500) & (d["_wallet_unique_bene_7d"] > 3) & (d["_acct_vel_count_7d"] > 5),
     "PPI-54", "FLAG"),
    # Rationale: Small P2P to many people — cashback abuse pattern

    ("rule_ppi_voucher_abuse", "PPI Wallet", "Voucher Abuse Low-Value Freq (PPI)", 1,
     lambda d: d["_is_ppi"]&(d["_amt"]<100)&(d["_acct_vel_count_24hr"]>20),
     "PPI-55", "STATIC"),
    # Rationale: 20+ tiny transactions could be testing or legitimate micro-payments

    ("rule_ppi_high_risk_region", "PPI Wallet", "High-Risk Domestic Region (PPI)", 1,
     lambda d: d["_is_ppi"] & d["_in_high_risk_region"] & (d["_amt"] > 5000),
     "PPI-69", "FLAG"),
    # Rationale: Region flag alone — profiling concern, low individual signal

    ("rule_ppi_cluster_alert", "PPI Wallet", "Cluster-Level Multi-Wallet Alert (PPI)", 3,
     lambda d: d["_is_ppi"] & ((d["_pan_wallet_count"] > 3) | (d["_mobile_wallet_count"] > 5) | (d["_ip_wallet_count"] > 10)) & (d["_high_risk_mcc_pct"] > 0.30) & (d["_acct_vel_sum_7d"] > 50000),
     "PPI-75", "FLAG"),
    # Rationale: Multi-signal convergence — wallets + high-risk MCC + velocity = strong fraud

]

print(f"Rules defined: {len(RULES)}")
sev_counts={1:0,2:0,3:0}; grp_counts=defaultdict(int); thresh_counts=defaultdict(int)
for item in RULES:
    sev_counts[item[3]]+=1; grp_counts[item[1]]+=1; thresh_counts[item[6]]+=1
print(f"  Severity 3 (High):   {sev_counts[3]}")
print(f"  Severity 2 (Medium): {sev_counts[2]}")
print(f"  Severity 1 (Low):    {sev_counts.get(1,0)}")
print(f"\nBy group:")
for g in sorted(grp_counts): print(f"  {g:<30s} {grp_counts[g]:>3} rules")
print(f"\nBy threshold type:")
for t in sorted(thresh_counts): print(f"  {t:<8s} {thresh_counts[t]:>3} rules")



print(f"\nAll 126 rules implemented (derived features computed in Cell 8)")
print(f"Implemented rules: {len(RULES)}")





# In[13]:


# Check which derived columns exist
required = ["_cust_baseline_freq_30d", "_cum_cash_30d", "_cum_foreign_30d",
            "_cum_credits_30d", "_cum_debits_30d", "_distinct_channels_24h",
            "_distinct_depositors_7d", "_is_new_device", "_is_new_city",
            "_distinct_devices_2h", "_high_risk_mcc_pct", "_high_risk_mcc_sum_30d",
            "_refund_flag", "_refund_count_merchant_30d", "_merchant_mcc_change_count",
            "_name_similarity", "_pan_wallet_count", "_mobile_wallet_count",
            "_ip_wallet_count", "_vpa_wallet_count", "_load_src_wallet_count",
            "_in_high_risk_region"]

present = [c for c in required if c in df.columns]
missing = [c for c in required if c not in df.columns]
print(f"Present: {len(present)}/{len(required)}")
print(f"Missing: {missing}")


# ## 7 — Execute All Rules
# 

# In[14]:


print(f"Executing all {len(RULES)} rules...")

rule_columns = []; severity_map = {}
for item in RULES:
    col_name, group, readable, severity, condition_fn = item[0], item[1], item[2], item[3], item[4]
    try:
        result = condition_fn(df)
        df[col_name] = result.astype(int)
    except Exception as e:
        print(f"  WARNING: {col_name} failed ({e}), defaulting to 0")
        df[col_name] = 0
    rule_columns.append(col_name)
    severity_map[col_name] = severity

print(f"\n{'Rule':<48s} {'Group':<20s} {'S':>1s} {'Type':<5s} {'Triggered':>10s} {'%':>7s} {'Ref':>12s}")
print("-"*108)
for item in RULES:
    col_name, group, readable, severity, _, reg, ttype = item
    trig = df[col_name].sum()
    pct = trig/len(df)*100
    si = {3:"H",2:"M",1:"L"}[severity]
    if trig > 0:
        print(f"  {col_name:<46s} {group:<20s} {si} {ttype:<5s} {trig:>10,} {pct:>6.2f}% {reg:>12s}")

total_trig = sum(df[c].sum() for c in rule_columns)
print(f"\n  Total triggers: {total_trig:,} across {len(rule_columns)} rules")
# Bank vs PPI trigger split
bank_trig = sum(df.loc[~df["_is_ppi"], c].sum() for c in rule_columns)
ppi_trig = sum(df.loc[df["_is_ppi"], c].sum() for c in rule_columns)
print(f"  Bank triggers: {bank_trig:,} | PPI triggers: {ppi_trig:,}")



# ## 8 — Composite Score & Alert Level
# 

# In[15]:


df["rule_score"] = sum(df[col]*severity_map[col] for col in rule_columns)
def triggered_list(row):
    return "; ".join([c for c in rule_columns if row[c]==1]) or ""
df["rules_triggered"] = df.apply(triggered_list, axis=1)
df["rules_triggered_count"] = sum(df[col] for col in rule_columns)

def alert_level(score):
    if score>=9: return "Critical"
    elif score>=6: return "High"
    elif score>=3: return "Medium"
    elif score>=1: return "Low"
    return "None"
df["alert_level"] = df["rule_score"].apply(alert_level)

print("COMPOSITE SCORING COMPLETE")
print(f"\n{'Alert Level':<12s} {'All':>10s} {'Bank':>10s} {'PPI':>10s}")
print("-"*47)
for level in ["Critical","High","Medium","Low","None"]:
    all_cnt = (df["alert_level"]==level).sum()
    bank_cnt = (df.loc[~df["_is_ppi"],"alert_level"]==level).sum()
    ppi_cnt = (df.loc[df["_is_ppi"],"alert_level"]==level).sum()
    print(f"  {level:<10s} {all_cnt:>10,} {bank_cnt:>10,} {ppi_cnt:>10,}")

max_possible = sum(severity_map[c] for c in rule_columns)
print(f"\nScore: min={df['rule_score'].min()} median={df['rule_score'].median():.0f} mean={df['rule_score'].mean():.2f} max={df['rule_score'].max()}")
print(f"Max possible: {max_possible} | Rules: {len(rule_columns)} | Sev3: {sev_counts[3]} Sev2: {sev_counts[2]} Sev1: {sev_counts.get(1,0)}")



# In[16]:


print("DONEEE")


# ## 9 — Cross-Reference with Typology Labels
# 

# In[ ]:


# # If typology labels exist, show how rules correlate with detected typologies
# if "is_aml" in df.columns:
#     aml = df[df["is_aml"] == 1]
#     clean = df[df["is_aml"] == 0]

#     print("Rule triggering: AML vs Clean transactions\n")
#     print(f"{'Rule':<42s} {'AML %':>7s} {'Clean %':>8s} {'Lift':>6s}")
#     print("-" * 68)

#     for col_name, _, _, sev, _ in RULES:
#         aml_rate = aml[col_name].mean() * 100 if len(aml) > 0 else 0
#         clean_rate = clean[col_name].mean() * 100 if len(clean) > 0 else 0
#         lift = aml_rate / max(clean_rate, 0.001)
#         if aml_rate > 0 or clean_rate > 0.1:
#             print(f"  {col_name:<40s} {aml_rate:>6.2f}% {clean_rate:>7.2f}% {lift:>5.1f}x")

#     print(f"\nAverage rule score:")
#     print(f"  AML transactions:   {aml['rule_score'].mean():.2f}")
#     print(f"  Clean transactions: {clean['rule_score'].mean():.2f}")

#     print(f"\nAlert level distribution for AML-labeled:")
#     for level in ["Critical","High","Medium","Low","None"]:
#         cnt = (aml["alert_level"] == level).sum()
#         print(f"  {level:<10s} {cnt:>8,} ({cnt/max(len(aml),1)*100:.1f}%)")
# else:
#     print("No 'is_aml' column found -- skipping typology cross-reference")



# In[ ]:





# ## 10 — Export Results
# 

# ## TO DELETE AFTER LOADING THE DATA INTO DATABASE

# In[ ]:


# import pandas as pd
# df = pd.read_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_rules.parquet")
# df.head()


# In[28]:


import io


# In[29]:


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


# In[18]:


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


# In[17]:


internal = [c for c in df.columns if c.startswith("_")]
df_out = df.drop(columns=internal)


# In[18]:


df_out.head()


# In[24]:


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


# In[21]:


for col in df_out.columns:

    dtype = str(df_out[col].dtype)

    if dtype in ["int64", "float64", "Int64"]:

        bad = df_out[
            df_out[col]
            .apply(lambda x: isinstance(x, str))
        ]

        if len(bad) > 0:

            print(f"\nInvalid strings in numeric column: {col}")

            print(
                bad[col]
                .head(10)
                .tolist()
            )


# In[26]:


df_out.head()


# In[ ]:


#f_out.to_parquet(r"C:\Users\VISHNUPRIYA\OneDrive\Desktop\Freelancing\AIGEN\smartsentry_aml_model\outputs_updated\stg_transactions_rules.parquet", index=False)


# In[33]:


# Write to PostgreSQL (full refresh each run)
from datetime import datetime
df_out = pd.DataFrame(df_out)
df_out["loaded_at"] = datetime.now()

write_table_fast(df_out, "stg_transactions_rules", mode="replace")
print(f"Rules output written: {len(df_out):,} rows x {len(df_out.columns)} cols")


# In[32]:


print(io)


# In[34]:


# ═══ Rule Trigger Diagnostics ═══
print("=" * 70)
print("RULE TRIGGER DIAGNOSTICS")
print("=" * 70)

# Find all rule columns
rule_cols = [c for c in df.columns if c.startswith("rule_") and c not in 
             {"rule_score", "rules_triggered", "rules_triggered_count"}]

print(f"\n  Total rule columns: {len(rule_cols)}")

# Per-transaction: how many rules triggered
df["_rules_fired"] = df[rule_cols].sum(axis=1).astype(int)

total = len(df)
has_rules = (df["_rules_fired"] > 0).sum()
no_rules = (df["_rules_fired"] == 0).sum()

print(f"\n  Transactions with ≥1 rule triggered:  {has_rules:>10,} ({has_rules/total*100:.1f}%)")
print(f"  Transactions with 0 rules triggered:  {no_rules:>10,} ({no_rules/total*100:.1f}%)")
print(f"  Total transactions:                   {total:>10,}")

# Break down by AML vs Clean
if "is_aml" in df.columns:
    aml = df["is_aml"] == 1
    clean = df["is_aml"] == 0

    aml_with_rules = (aml & (df["_rules_fired"] > 0)).sum()
    aml_no_rules = (aml & (df["_rules_fired"] == 0)).sum()
    clean_with_rules = (clean & (df["_rules_fired"] > 0)).sum()
    clean_no_rules = (clean & (df["_rules_fired"] == 0)).sum()

    print(f"\n  {'Category':<35s} {'Rules ≥1':>10s} {'Rules = 0':>10s} {'Total':>10s}")
    print(f"  {'─'*70}")
    print(f"  {'AML (is_aml=1)':<35s} {aml_with_rules:>10,} {aml_no_rules:>10,} {aml.sum():>10,}")
    print(f"  {'Clean (is_aml=0)':<35s} {clean_with_rules:>10,} {clean_no_rules:>10,} {clean.sum():>10,}")

    if aml.sum() > 0:
        print(f"\n  AML with rules:    {aml_with_rules/aml.sum()*100:.1f}% of AML transactions have ≥1 rule")
    if clean.sum() > 0:
        print(f"  Clean with rules:  {clean_with_rules/clean.sum()*100:.1f}% of clean transactions have ≥1 rule (false alarms)")

# Distribution of rules-per-transaction
print(f"\n  Rules-per-transaction distribution:")
print(f"    {'Rules Fired':<15s} {'Count':>10s} {'%':>8s}")
print(f"    {'─'*35}")
for n in range(0, min(df["_rules_fired"].max() + 1, 11)):
    cnt = (df["_rules_fired"] == n).sum()
    print(f"    {n:<15d} {cnt:>10,} {cnt/total*100:>7.1f}%")
if df["_rules_fired"].max() > 10:
    cnt = (df["_rules_fired"] > 10).sum()
    print(f"    {'>10':<15s} {cnt:>10,} {cnt/total*100:>7.1f}%")

# Top 20 most triggered rules
print(f"\n  Top 20 most triggered rules:")
print(f"    {'Rule':<45s} {'Triggered':>10s} {'% of Txns':>10s}")
print(f"    {'─'*68}")
rule_triggers = {r: df[r].sum() for r in rule_cols}
for rule, cnt in sorted(rule_triggers.items(), key=lambda x: -x[1])[:20]:
    print(f"    {rule:<45s} {cnt:>10,} {cnt/total*100:>9.2f}%")

# Rules that NEVER triggered
never_fired = [r for r, cnt in rule_triggers.items() if cnt == 0]
print(f"\n  Rules that never triggered: {len(never_fired)}")
if never_fired:
    for r in never_fired[:10]:
        print(f"    {r}")
    if len(never_fired) > 10:
        print(f"    ... and {len(never_fired)-10} more")

# Per-typology rule coverage
if "aml_typology" in df.columns:
    print(f"\n  Rule coverage by typology:")
    print(f"    {'Typology':<40s} {'Total':>7s} {'With Rules':>11s} {'Coverage':>9s}")
    print(f"    {'─'*70}")
    for typ in sorted(df["aml_typology"].dropna().unique()):
        if str(typ) in ("", "nan", "None"): continue
        mask = df["aml_typology"] == typ
        typ_total = mask.sum()
        typ_rules = (mask & (df["_rules_fired"] > 0)).sum()
        pct = typ_rules / typ_total * 100 if typ_total > 0 else 0
        status = "✓" if pct > 70 else ("⚡" if pct > 40 else "⚠")
        print(f"    {str(typ)[:38]:<40s} {typ_total:>7,} {typ_rules:>11,} {pct:>7.1f}% {status}")

df.drop(columns=["_rules_fired"], inplace=True)


# In[35]:


# Top 20 most-triggered rules
rule_cols = [c for c in df.columns if c.startswith("rule_") and c not in 
             {"rule_score", "rules_triggered", "rules_triggered_count"}]

print(f"\n  Top 20 Rules by Trigger Rate (Clean transactions only):")
print(f"  {'Rule':<50s} {'Clean Triggers':>15s} {'% of Clean':>12s}")
print(f"  {'─'*80}")

clean_mask = df["is_aml"] == 0
clean_total = clean_mask.sum()

triggers = []
for rc in rule_cols:
    ct = df.loc[clean_mask, rc].sum()
    triggers.append((rc, ct, ct/max(clean_total,1)*100))

triggers.sort(key=lambda x: -x[1])
for rule, cnt, pct in triggers[:20]:
    print(f"  {rule:<50s} {cnt:>15,} {pct:>11.2f}%")

