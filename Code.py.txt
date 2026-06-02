# BLOCK 1 — Install & Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# BLOCK 2 — Load the datasets
claims_path = '/content/claims_10000.xlsx'
policies_path = '/content/policies_1000.xlsx'
claims = pd.read_excel(claims_path)
policies = pd.read_excel(policies_path)

# BLOCK 3 — Clean Policies Dataset (fixed date logic + gender as male/female only)
pol = policies.copy()
# Standardize column names
pol.columns = [c.strip().lower().replace(" ", "_") for c in pol.columns]
# -----------------------------
# Smoking Status Cleaning
# -----------------------------
if "smoking_status" in pol.columns:
    pol["smoking_status"] = (
        pol["smoking_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "true": 1,
            "false": 0,
        })
    )
    pol.loc[~pol["smoking_status"].isin([0, 1]), "smoking_status"] = np.nan
# -----------------------------
# FIXED: Convert only real date columns
# -----------------------------
for col in pol.columns:
    col_l = col.lower()
    # Only treat as date if the name clearly has 'date'
    # (so 'gender' will NOT match anymore)
    if "date" in col_l:
        pol[col] = pd.to_datetime(pol[col], errors="coerce")
# -----------------------------
# Gender Cleaning → only 'male' / 'female'
# -----------------------------
if "gender" in pol.columns:
    pol["gender"] = (
        pol["gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "m": "male",
            "male": "male",
            "man": "male",
            "f": "female",
            "female": "female",
            "woman": "female",
        })
    )
    # Any weird values → NaN
    pol.loc[~pol["gender"].isin(["male", "female"]), "gender"] = np.nan
# -----------------------------
# Trim spaces for remaining text columns
# -----------------------------
for col in pol.select_dtypes(include="object").columns:
    pol[col] = pol[col].str.strip()
# Remove duplicate rows
pol = pol.drop_duplicates()
print("Policies cleaned shape:", pol.shape)
pol.head()

# BLOCK 4 — Clean Claims Dataset

clm = claims.copy()
clm.columns = [c.strip().lower().replace(" ", "_") for c in clm.columns]
# Convert dates
for col in clm.columns:
    if "date" in col or "loss" in col or "report" in col:
        clm[col] = pd.to_datetime(clm[col], errors="coerce")
# Convert amount columns
for col in clm.columns:
    if ("amount" in col) or ("paid" in col) or ("reserve" in col):
        clm[col] = pd.to_numeric(clm[col], errors="coerce").fillna(0)
# Trim spaces
for col in clm.select_dtypes(include="object").columns:
    clm[col] = clm[col].str.strip()
clm = clm.drop_duplicates()

# BLOCK 5 — Feature Engineering

# Ensure ID types match
if "policy_id" in clm.columns and "policy_id" in pol.columns:
    clm["policy_id"] = clm["policy_id"].astype(str).str.strip()
    pol["policy_id"] = pol["policy_id"].astype(str).str.strip()

# Age buckets
if "age" in pol.columns:
    pol["age"] = pd.to_numeric(pol["age"], errors="coerce")
    pol["age_group"] = pd.cut(
        pol["age"],
        bins=[0, 18, 30, 45, 60, 75, 120],
        labels=["<18", "18-29", "30-44", "45-59", "60-74", "75+"],
        right=False,
    )

# Policy tenure
start = next((c for c in pol.columns if "start" in c), None)
end = next((c for c in pol.columns if "end" in c), None)
if start and end:
    pol[start] = pd.to_datetime(pol[start], errors="coerce")
    pol[end] = pd.to_datetime(pol[end], errors="coerce")
    pol["policy_tenure_days"] = (pol[end] - pol[start]).dt.days
    pol["policy_tenure_years"] = pol["policy_tenure_days"] / 365.25

# Determine claim amount
if "claim_amount" in clm.columns:
    clm["claim_amount"] = pd.to_numeric(clm["claim_amount"], errors="coerce").fillna(0)
else:
    amt_col = next((c for c in clm.columns if "amount" in c), None)
    clm["claim_amount"] = clm[amt_col] if amt_col else 0

# Year/Month fields
date_col = next((c for c in clm.columns if "date" in c), None)
if date_col:
    clm["claim_year"] = clm[date_col].dt.year
    clm["claim_month"] = clm[date_col].dt.month
print("Feature engineering complete.")
pol.head(), clm.head()

# BLOCK 6 — Merge Claims and Policies

if "policy_id" in clm.columns and "policy_id" in pol.columns:
    merged = clm.merge(pol, on="policy_id", how="left")
else:
    merged = clm.copy()
print("Merged dataset shape:", merged.shape)
merged.head()

# BLOCK 7 — Policy Aggregations

agg = (
    merged.groupby("policy_id")
    .agg(
        total_claims=("claim_amount", "count"),
        total_claim_amount=("claim_amount", "sum"),
        avg_claim_amount=("claim_amount", "mean"),
    )
    .reset_index()
)
pol = pol.merge(agg, on="policy_id", how="left")
pol["total_claims"] = pol["total_claims"].fillna(0).astype(int)
pol["total_claim_amount"] = pol["total_claim_amount"].fillna(0)
pol["avg_claim_amount"] = pol["avg_claim_amount"].fillna(0)

# Loss ratio
if "annual_premium" in pol.columns:
    pol["loss_ratio"] = pol["total_claim_amount"] / pol["annual_premium"]
else:
    pol["loss_ratio"] = np.nan
pol.head()

# BLOCK 8 — Save cleaned files

from google.colab import files

output_files = [
    "/content/cleaned_outputs/policies_1000_cleaned.csv",
    "/content/cleaned_outputs/claims_10000_cleaned.csv",
    "/content/cleaned_outputs/merged_insurance.csv"
]
for file_path in output_files:
    try:
        files.download(file_path)
        print(f"Downloaded: {file_path}")
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")