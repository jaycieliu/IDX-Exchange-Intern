from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Week 6 – Feature Engineering and Market Metrics
# IDX Exchange Data Analyst Internship
#
# Purpose:
# Create sold-side and listing-side market metrics for Tableau.
# ============================================================


# -----------------------------
# 1. Define file paths
# -----------------------------

BASE_DIR = Path("/Users/amyliu/Desktop/IDX")

SOLD_FILE = BASE_DIR / "data" / "generated" / "sold_with_rates_week4-5.csv"
LISTINGS_FILE = BASE_DIR / "data" / "generated" / "listing_with_rates_week4-5.csv"

OUTPUT_DIR = BASE_DIR / "data" / "week6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Sold file:", SOLD_FILE)
print("Listing file:", LISTINGS_FILE)
print("Output folder:", OUTPUT_DIR)

if not SOLD_FILE.exists():
    raise FileNotFoundError(f"Sold file not found: {SOLD_FILE}")

if not LISTINGS_FILE.exists():
    raise FileNotFoundError(f"Listing file not found: {LISTINGS_FILE}")


# -----------------------------
# 2. Load datasets
# -----------------------------

sold = pd.read_csv(SOLD_FILE, low_memory=False)
listings = pd.read_csv(LISTINGS_FILE, low_memory=False)

# Filter to Residential only
sold = sold[sold["PropertyType"] == "Residential"].copy()
listings = listings[listings["PropertyType"] == "Residential"].copy()

print(f"Sold rows loaded: {len(sold):,}")
print(f"Listing rows loaded: {len(listings):,}")


# -----------------------------
# 3. Convert sold date fields
# -----------------------------

sold_date_cols = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

for col in sold_date_cols:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")
        print(f"Converted sold date column: {col}")
    else:
        print(f"Missing sold date column: {col}")


# -----------------------------
# 4. Convert sold numeric fields
# -----------------------------

sold_numeric_cols = [
    "ClosePrice",
    "OriginalListPrice",
    "ListPrice",
    "LivingArea",
    "DaysOnMarket"
]

for col in sold_numeric_cols:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors="coerce")
        print(f"Converted sold numeric column: {col}")
    else:
        print(f"Missing sold numeric column: {col}")


# -----------------------------
# 5. Create sold-side market metrics
# -----------------------------

sold["price_ratio"] = np.where(
    sold["OriginalListPrice"].notna() & (sold["OriginalListPrice"] != 0),
    sold["ClosePrice"] / sold["OriginalListPrice"],
    np.nan
)

sold["close_to_original_list_ratio"] = np.where(
    sold["OriginalListPrice"].notna() & (sold["OriginalListPrice"] != 0),
    sold["ClosePrice"] / sold["OriginalListPrice"],
    np.nan
)

sold["price_per_sqft"] = np.where(
    sold["LivingArea"].notna() & (sold["LivingArea"] != 0),
    sold["ClosePrice"] / sold["LivingArea"],
    np.nan
)

sold["days_on_market"] = sold["DaysOnMarket"]

sold["close_year"] = sold["CloseDate"].dt.year
sold["close_month"] = sold["CloseDate"].dt.month
sold["yrmo"] = sold["CloseDate"].dt.to_period("M").astype(str)

sold["listing_to_contract_days"] = (
    sold["PurchaseContractDate"] - sold["ListingContractDate"]
).dt.days

sold["contract_to_close_days"] = (
    sold["CloseDate"] - sold["PurchaseContractDate"]
).dt.days

print("Sold-side Week 6 metrics created.")


# -----------------------------
# 6. Validate sold engineered metrics
# -----------------------------

sold_engineered_cols = [
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "days_on_market",
    "close_year",
    "close_month",
    "yrmo",
    "listing_to_contract_days",
    "contract_to_close_days"
]

sold_metric_nulls = (
    sold[sold_engineered_cols]
    .isna()
    .sum()
    .reset_index()
)

sold_metric_nulls.columns = ["engineered_column", "null_count"]
sold_metric_nulls["null_pct"] = sold_metric_nulls["null_count"] / len(sold)

sold_metric_summary = (
    sold[
        [
            "price_ratio",
            "close_to_original_list_ratio",
            "price_per_sqft",
            "days_on_market",
            "listing_to_contract_days",
            "contract_to_close_days"
        ]
    ]
    .describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    .T
)

print("Sold metric null summary:")
print(sold_metric_nulls)

print("Sold metric summary:")
print(sold_metric_summary)


# -----------------------------
# 7. Create sold sample output
# -----------------------------

sold_sample_cols = [
    "CloseDate",
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "DaysOnMarket",
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "close_year",
    "close_month",
    "yrmo",
    "ListingContractDate",
    "PurchaseContractDate",
    "listing_to_contract_days",
    "contract_to_close_days",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "MLSAreaMajor"
]

sold_sample_cols = [col for col in sold_sample_cols if col in sold.columns]
sold_sample_output = sold[sold_sample_cols].head(20)


# -----------------------------
# 8. Create sold segmented summaries
# -----------------------------

sold_county_summary = (
    sold.groupby("CountyOrParish", dropna=False)
    .agg(
        closed_sales=("ClosePrice", "count"),
        median_close_price=("ClosePrice", "median"),
        average_close_price=("ClosePrice", "mean"),
        median_price_per_sqft=("price_per_sqft", "median"),
        average_days_on_market=("days_on_market", "mean"),
        median_days_on_market=("days_on_market", "median"),
        average_close_to_original_list_ratio=("close_to_original_list_ratio", "mean"),
        median_listing_to_contract_days=("listing_to_contract_days", "median"),
        median_contract_to_close_days=("contract_to_close_days", "median")
    )
    .reset_index()
    .sort_values("closed_sales", ascending=False)
)

sold_property_summary = (
    sold.groupby(["PropertyType", "PropertySubType"], dropna=False)
    .agg(
        closed_sales=("ClosePrice", "count"),
        median_close_price=("ClosePrice", "median"),
        average_close_price=("ClosePrice", "mean"),
        median_price_per_sqft=("price_per_sqft", "median"),
        average_days_on_market=("days_on_market", "mean"),
        average_close_to_original_list_ratio=("close_to_original_list_ratio", "mean")
    )
    .reset_index()
    .sort_values("closed_sales", ascending=False)
)

sold_mls_area_summary = (
    sold.groupby("MLSAreaMajor", dropna=False)
    .agg(
        closed_sales=("ClosePrice", "count"),
        median_close_price=("ClosePrice", "median"),
        average_close_price=("ClosePrice", "mean"),
        median_price_per_sqft=("price_per_sqft", "median"),
        average_days_on_market=("days_on_market", "mean"),
        average_close_to_original_list_ratio=("close_to_original_list_ratio", "mean")
    )
    .reset_index()
    .sort_values("closed_sales", ascending=False)
)

sold_office_summary = (
    sold.groupby(["ListOfficeName", "BuyerOfficeName"], dropna=False)
    .agg(
        closed_sales=("ClosePrice", "count"),
        total_sales_volume=("ClosePrice", "sum"),
        median_close_price=("ClosePrice", "median"),
        average_days_on_market=("days_on_market", "mean"),
        average_close_to_original_list_ratio=("close_to_original_list_ratio", "mean")
    )
    .reset_index()
    .sort_values("total_sales_volume", ascending=False)
)

print("Sold segmented summaries created.")


# -----------------------------
# 9. Convert listing date fields
# -----------------------------

if "ListingContractDate" in listings.columns:
    listings["ListingContractDate"] = pd.to_datetime(
        listings["ListingContractDate"],
        errors="coerce"
    )
    print("Converted listing date column: ListingContractDate")
else:
    print("Missing listing date column: ListingContractDate")


# -----------------------------
# 10. Convert listing numeric fields
# -----------------------------

listing_numeric_cols = [
    "ListPrice",
    "LivingArea",
    "DaysOnMarket"
]

for col in listing_numeric_cols:
    if col in listings.columns:
        listings[col] = pd.to_numeric(listings[col], errors="coerce")
        print(f"Converted listing numeric column: {col}")
    else:
        print(f"Missing listing numeric column: {col}")


# -----------------------------
# 11. Create listing-side market metrics
# -----------------------------

listings["list_year"] = listings["ListingContractDate"].dt.year
listings["list_month"] = listings["ListingContractDate"].dt.month
listings["list_yrmo"] = listings["ListingContractDate"].dt.to_period("M").astype(str)

listings["list_price_per_sqft"] = np.where(
    listings["LivingArea"].notna() & (listings["LivingArea"] != 0),
    listings["ListPrice"] / listings["LivingArea"],
    np.nan
)

listings["listing_days_on_market"] = listings["DaysOnMarket"]

print("Listing-side Week 6 metrics created.")


# -----------------------------
# 12. Create listing sample output
# -----------------------------

listing_sample_cols = [
    "ListingContractDate",
    "ListPrice",
    "LivingArea",
    "DaysOnMarket",
    "list_year",
    "list_month",
    "list_yrmo",
    "list_price_per_sqft",
    "listing_days_on_market",
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "MLSAreaMajor",
    "City",
    "PostalCode",
    "ListOfficeName"
]

listing_sample_cols = [
    col for col in listing_sample_cols if col in listings.columns
]

listing_sample_output = listings[listing_sample_cols].head(20)


# -----------------------------
# 13. Create listing segmented summaries
# -----------------------------

monthly_new_listings = (
    listings.groupby("list_yrmo", dropna=False)
    .agg(
        new_listings=("ListingContractDate", "count"),
        median_list_price=("ListPrice", "median"),
        average_list_price=("ListPrice", "mean"),
        median_list_price_per_sqft=("list_price_per_sqft", "median"),
        average_listing_days_on_market=("listing_days_on_market", "mean")
    )
    .reset_index()
    .sort_values("list_yrmo")
)

listing_county_summary = (
    listings.groupby("CountyOrParish", dropna=False)
    .agg(
        new_listings=("ListingContractDate", "count"),
        median_list_price=("ListPrice", "median"),
        average_list_price=("ListPrice", "mean"),
        median_list_price_per_sqft=("list_price_per_sqft", "median"),
        average_listing_days_on_market=("listing_days_on_market", "mean")
    )
    .reset_index()
    .sort_values("new_listings", ascending=False)
)

listing_property_summary = (
    listings.groupby(["PropertyType", "PropertySubType"], dropna=False)
    .agg(
        new_listings=("ListingContractDate", "count"),
        median_list_price=("ListPrice", "median"),
        average_list_price=("ListPrice", "mean"),
        median_list_price_per_sqft=("list_price_per_sqft", "median"),
        average_listing_days_on_market=("listing_days_on_market", "mean")
    )
    .reset_index()
    .sort_values("new_listings", ascending=False)
)

print("Listing segmented summaries created.")


# -----------------------------
# 14. Save Week 6 outputs
# -----------------------------
'''
sold.to_csv(
    OUTPUT_DIR / "sold_week6_engineered_metrics.csv",
    index=False
)

listings.to_csv(
    OUTPUT_DIR / "listing_week6_engineered_metrics.csv",
    index=False
)

sold_metric_nulls.to_csv(
    OUTPUT_DIR / "week6_sold_metric_null_summary.csv",
    index=False
)

sold_metric_summary.to_csv(
    OUTPUT_DIR / "week6_sold_metric_summary.csv"
)

sold_sample_output.to_csv(
    OUTPUT_DIR / "week6_sold_sample_output.csv",
    index=False
)

sold_county_summary.to_csv(
    OUTPUT_DIR / "week6_sold_county_summary.csv",
    index=False
)

sold_property_summary.to_csv(
    OUTPUT_DIR / "week6_sold_property_summary.csv",
    index=False
)

sold_mls_area_summary.to_csv(
    OUTPUT_DIR / "week6_sold_mls_area_summary.csv",
    index=False
)

sold_office_summary.to_csv(
    OUTPUT_DIR / "week6_sold_office_summary.csv",
    index=False
)

listing_sample_output.to_csv(
    OUTPUT_DIR / "week6_listing_sample_output.csv",
    index=False
)

monthly_new_listings.to_csv(
    OUTPUT_DIR / "week6_monthly_new_listings.csv",
    index=False
)

listing_county_summary.to_csv(
    OUTPUT_DIR / "week6_listing_county_summary.csv",
    index=False
)

listing_property_summary.to_csv(
    OUTPUT_DIR / "week6_listing_property_summary.csv",
    index=False
)

print(f"All Week 6 outputs saved to: {OUTPUT_DIR}")
print("Week 6 feature engineering completed successfully.")
'''