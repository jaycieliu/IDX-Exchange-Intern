import pandas as pd
from pathlib import Path


# ============================================================
# Mortgage Rate Enrichment
# Purpose:
# 1. Fetch FRED MORTGAGE30US data
# 2. Resample weekly mortgage rates to monthly averages
# 3. Create year_month keys in sold and listings datasets
# 4. Merge mortgage rates into both datasets
# 5. Validate merge completeness
# 6. Save enriched datasets as new CSV files
# ============================================================


# ------------------------------------------------------------
# 1. Set file paths
# ------------------------------------------------------------

# Change these file names based on your actual combined datasets
SOLD_FILE = "/Users/amyliu/Desktop/IDX/data/generated/sold_cleaned.csv"
LISTINGS_FILE = "/Users/amyliu/Desktop/IDX/data/generated/listing_cleaned.csv"

# Output files
SOLD_OUTPUT_FILE = "/Users/amyliu/Desktop/IDX/data/generated/sold_combined_residential_with_mortgage_rates.csv"
LISTINGS_OUTPUT_FILE = "/Users/amyliu/Desktop/IDX/data/generated/listings_combined_residential_with_mortgage_rates.csv"


# ------------------------------------------------------------
# 2. Load combined MLS datasets
# ------------------------------------------------------------

sold = pd.read_csv(SOLD_FILE, low_memory=False)
listings = pd.read_csv(LISTINGS_FILE, low_memory=False)

print("Sold dataset loaded:", sold.shape)
print("Listings dataset loaded:", listings.shape)


# ------------------------------------------------------------
# 3. Fetch mortgage rate data from FRED
# ------------------------------------------------------------

fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

mortgage = pd.read_csv(fred_url)

# Rename columns for clarity
mortgage.columns = ["date", "rate_30yr_fixed"]

# Convert date column to datetime
mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")

# Convert mortgage rate to numeric
mortgage["rate_30yr_fixed"] = pd.to_numeric(
    mortgage["rate_30yr_fixed"],
    errors="coerce"
)

# Drop rows with missing dates or rates
mortgage = mortgage.dropna(subset=["date", "rate_30yr_fixed"])

print("Mortgage data loaded:", mortgage.shape)
print(mortgage.head())


# ------------------------------------------------------------
# 4. Resample weekly mortgage rates to monthly averages
# ------------------------------------------------------------

mortgage["year_month"] = mortgage["date"].dt.to_period("M")

mortgage_monthly = (
    mortgage
    .groupby("year_month", as_index=False)["rate_30yr_fixed"]
    .mean()
)

# Optional: round mortgage rate to 3 decimals
mortgage_monthly["rate_30yr_fixed"] = mortgage_monthly["rate_30yr_fixed"].round(3)

print("Monthly mortgage data:")
print(mortgage_monthly.head())


# ------------------------------------------------------------
# 5. Create matching year_month key in MLS datasets
# ------------------------------------------------------------

# Sold dataset: use CloseDate
sold["CloseDate"] = pd.to_datetime(sold["CloseDate"], errors="coerce")
sold["year_month"] = sold["CloseDate"].dt.to_period("M")

# Listings dataset: use ListingContractDate
listings["ListingContractDate"] = pd.to_datetime(
    listings["ListingContractDate"],
    errors="coerce"
)
listings["year_month"] = listings["ListingContractDate"].dt.to_period("M")


# ------------------------------------------------------------
# 6. Merge mortgage rates into both datasets
# ------------------------------------------------------------

sold_with_rates = sold.merge(
    mortgage_monthly,
    on="year_month",
    how="left"
)

listings_with_rates = listings.merge(
    mortgage_monthly,
    on="year_month",
    how="left"
)

print("Sold dataset after merge:", sold_with_rates.shape)
print("Listings dataset after merge:", listings_with_rates.shape)


# ------------------------------------------------------------
# 7. Validate merge completeness
# ------------------------------------------------------------

sold_missing_rates = sold_with_rates["rate_30yr_fixed"].isna().sum()
listings_missing_rates = listings_with_rates["rate_30yr_fixed"].isna().sum()

print("\nValidation Results")
print("------------------")
print(f"Missing mortgage rates in sold dataset: {sold_missing_rates}")
print(f"Missing mortgage rates in listings dataset: {listings_missing_rates}")

# Show rows with missing mortgage rates if any exist
if sold_missing_rates > 0:
    print("\nSold rows with missing mortgage rates:")
    print(
        sold_with_rates.loc[
            sold_with_rates["rate_30yr_fixed"].isna(),
            ["CloseDate", "year_month", "rate_30yr_fixed"]
        ].head()
    )

if listings_missing_rates > 0:
    print("\nListings rows with missing mortgage rates:")
    print(
        listings_with_rates.loc[
            listings_with_rates["rate_30yr_fixed"].isna(),
            ["ListingContractDate", "year_month", "rate_30yr_fixed"]
        ].head()
    )


# ------------------------------------------------------------
# 8. Preview enriched datasets
# ------------------------------------------------------------

print("\nSold preview:")
sold_preview_cols = [
    "CloseDate",
    "year_month",
    "ClosePrice",
    "rate_30yr_fixed"
]

existing_sold_preview_cols = [
    col for col in sold_preview_cols if col in sold_with_rates.columns
]

print(sold_with_rates[existing_sold_preview_cols].head())


print("\nListings preview:")
listings_preview_cols = [
    "ListingContractDate",
    "year_month",
    "ListPrice",
    "rate_30yr_fixed"
]

existing_listings_preview_cols = [
    col for col in listings_preview_cols if col in listings_with_rates.columns
]

print(listings_with_rates[existing_listings_preview_cols].head())


# ------------------------------------------------------------
# 9. Convert year_month to string before saving
# ------------------------------------------------------------

sold_with_rates["year_month"] = sold_with_rates["year_month"].astype(str)
listings_with_rates["year_month"] = listings_with_rates["year_month"].astype(str)


# ------------------------------------------------------------
# 10. Save enriched datasets
# ------------------------------------------------------------

sold_with_rates.to_csv(SOLD_OUTPUT_FILE, index=False)
listings_with_rates.to_csv(LISTINGS_OUTPUT_FILE, index=False)

print("\nFiles saved successfully:")
print(f"- {SOLD_OUTPUT_FILE}")
print(f"- {LISTINGS_OUTPUT_FILE}")