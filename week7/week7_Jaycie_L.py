from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Week 7 – Outlier Detection and Data Quality
# IDX Exchange Data Analyst Internship
#
# Purpose:
# Apply business-rule checks and IQR outlier detection to the
# Residential sold dataset. An optional listing-side check is
# included for listing market activity fields.
# ============================================================


# -----------------------------
# 1. Define file paths
# -----------------------------

BASE_DIR = Path("/Users/amyliu/Desktop/IDX")

sold_candidates = [
    BASE_DIR / "data" / "week6" / "sold_week6_engineered_metrics.csv",
    BASE_DIR / "data" / "generated" / "week6" / "sold_week6_engineered_metrics.csv",
]

listing_candidates = [
    BASE_DIR / "data" / "week6" / "listing_week6_engineered_metrics.csv",
    BASE_DIR / "data" / "generated" / "week6" / "listing_week6_engineered_metrics.csv",
]

SOLD_FILE = next((path for path in sold_candidates if path.exists()), None)
LISTINGS_FILE = next((path for path in listing_candidates if path.exists()), None)

OUTPUT_DIR = BASE_DIR / "data" / "week7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if SOLD_FILE is None:
    raise FileNotFoundError(
        "Could not find sold_week6_engineered_metrics.csv. "
        "Check data/week6 or data/generated/week6."
    )

print("Sold input file:", SOLD_FILE)
print("Listing input file:", LISTINGS_FILE if LISTINGS_FILE is not None else "Not found / optional")
print("Output folder:", OUTPUT_DIR)


# -----------------------------
# 2. Load Week 6 sold dataset
# -----------------------------

sold = pd.read_csv(SOLD_FILE, low_memory=False)

print(f"\nSold rows loaded: {len(sold):,}")
print(f"Sold columns loaded: {sold.shape[1]:,}")


# -----------------------------
# 3. Filter sold data to Residential only
# -----------------------------

if "PropertyType" not in sold.columns:
    raise KeyError("PropertyType column is required for the Residential filter.")

print("\nPropertyType distribution before Residential filter:")
print(sold["PropertyType"].value_counts(dropna=False))

before_residential_filter = len(sold)
sold = sold[sold["PropertyType"] == "Residential"].copy()
after_residential_filter = len(sold)

print(f"\nRows before Residential filter: {before_residential_filter:,}")
print(f"Rows after Residential filter: {after_residential_filter:,}")
print(f"Rows removed by Residential filter: {before_residential_filter - after_residential_filter:,}")

print("\nPropertyType distribution after Residential filter:")
print(sold["PropertyType"].value_counts(dropna=False))


# -----------------------------
# 4. Check and convert key numeric fields
# -----------------------------

outlier_fields = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
]

missing_fields = [col for col in outlier_fields if col not in sold.columns]
if missing_fields:
    raise KeyError(f"Missing required sold outlier fields: {missing_fields}")

for col in outlier_fields:
    sold[col] = pd.to_numeric(sold[col], errors="coerce")

before_numeric_summary = sold[outlier_fields].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

print("\nSold numeric summary before filtering:")
print(before_numeric_summary)


# -----------------------------
# 5. Apply business-rule invalid flags
# -----------------------------

sold["invalid_close_price_flag"] = sold["ClosePrice"].isna() | (sold["ClosePrice"] <= 0)
sold["invalid_living_area_flag"] = sold["LivingArea"].isna() | (sold["LivingArea"] <= 0)
sold["invalid_days_on_market_flag"] = sold["DaysOnMarket"].isna() | (sold["DaysOnMarket"] < 0)

business_rule_summary = pd.DataFrame({
    "flag": [
        "invalid_close_price_flag",
        "invalid_living_area_flag",
        "invalid_days_on_market_flag",
    ],
    "flagged_records": [
        sold["invalid_close_price_flag"].sum(),
        sold["invalid_living_area_flag"].sum(),
        sold["invalid_days_on_market_flag"].sum(),
    ],
})

business_rule_summary["flagged_pct"] = business_rule_summary["flagged_records"] / len(sold)

print("\nBusiness-rule summary:")
print(business_rule_summary)


# -----------------------------
# 6. Calculate IQR thresholds
# -----------------------------

iqr_thresholds = []

for col in outlier_fields:
    valid_series = sold[col].dropna()

    q1 = valid_series.quantile(0.25)
    q3 = valid_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    iqr_thresholds.append({
        "field": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    })

iqr_thresholds_df = pd.DataFrame(iqr_thresholds)

print("\nIQR thresholds:")
print(iqr_thresholds_df)


# -----------------------------
# 7. Add IQR outlier flags
# -----------------------------

for _, row in iqr_thresholds_df.iterrows():
    col = row["field"]
    lower = row["lower_bound"]
    upper = row["upper_bound"]

    flag_col = f"{col.lower()}_iqr_outlier_flag"

    sold[flag_col] = (
        sold[col].notna()
        & ((sold[col] < lower) | (sold[col] > upper))
    )

iqr_flag_cols = [
    "closeprice_iqr_outlier_flag",
    "livingarea_iqr_outlier_flag",
    "daysonmarket_iqr_outlier_flag",
]

iqr_flag_summary = sold[iqr_flag_cols].sum().reset_index()
iqr_flag_summary.columns = ["flag", "flagged_records"]
iqr_flag_summary["flagged_pct"] = iqr_flag_summary["flagged_records"] / len(sold)

print("\nIQR flag summary:")
print(iqr_flag_summary)


# -----------------------------
# 8. Create combined removal flag
# -----------------------------

sold["any_business_rule_invalid_flag"] = (
    sold["invalid_close_price_flag"]
    | sold["invalid_living_area_flag"]
    | sold["invalid_days_on_market_flag"]
)

sold["any_iqr_outlier_flag"] = (
    sold["closeprice_iqr_outlier_flag"]
    | sold["livingarea_iqr_outlier_flag"]
    | sold["daysonmarket_iqr_outlier_flag"]
)

sold["remove_from_clean_analysis_flag"] = (
    sold["any_business_rule_invalid_flag"]
    | sold["any_iqr_outlier_flag"]
)

overall_flag_summary = pd.DataFrame({
    "category": [
        "Business-rule invalid records",
        "IQR outlier records",
        "Total records removed from clean analysis",
    ],
    "record_count": [
        sold["any_business_rule_invalid_flag"].sum(),
        sold["any_iqr_outlier_flag"].sum(),
        sold["remove_from_clean_analysis_flag"].sum(),
    ],
})

overall_flag_summary["record_pct"] = overall_flag_summary["record_count"] / len(sold)

print("\nOverall flag summary:")
print(overall_flag_summary)


# -----------------------------
# 9. Create clean filtered sold dataset
# -----------------------------

sold_clean_filtered = sold[~sold["remove_from_clean_analysis_flag"]].copy()

print(f"\nFull flagged sold dataset rows: {len(sold):,}")
print(f"Clean filtered sold dataset rows: {len(sold_clean_filtered):,}")
print(f"Rows removed: {len(sold) - len(sold_clean_filtered):,}")
print(f"Removal percentage: {(len(sold) - len(sold_clean_filtered)) / len(sold):.2%}")


# -----------------------------
# 10. Compare before and after filtering
# -----------------------------

comparison_rows = []

for col in outlier_fields:
    comparison_rows.append({
        "field": col,
        "before_row_count": sold[col].notna().sum(),
        "after_row_count": sold_clean_filtered[col].notna().sum(),
        "before_median": sold[col].median(),
        "after_median": sold_clean_filtered[col].median(),
        "median_change": sold_clean_filtered[col].median() - sold[col].median(),
        "before_mean": sold[col].mean(),
        "after_mean": sold_clean_filtered[col].mean(),
        "mean_change": sold_clean_filtered[col].mean() - sold[col].mean(),
    })

before_after_comparison = pd.DataFrame(comparison_rows)

print("\nSold before/after comparison:")
print(before_after_comparison)


# -----------------------------
# 11. Percentile comparison
# -----------------------------

before_percentiles = sold[outlier_fields].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

after_percentiles = sold_clean_filtered[outlier_fields].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

before_percentiles["dataset"] = "before_filtering"
after_percentiles["dataset"] = "after_filtering"

percentile_comparison = pd.concat([
    before_percentiles,
    after_percentiles,
]).reset_index().rename(columns={"index": "field"})

print("\nPercentile comparison:")
print(percentile_comparison)


# -----------------------------
# 12. Optional listing-side outlier check
# -----------------------------

if LISTINGS_FILE is not None:
    listings = pd.read_csv(LISTINGS_FILE, low_memory=False)

    print(f"\nListing rows loaded: {len(listings):,}")

    if "PropertyType" not in listings.columns:
        raise KeyError("PropertyType column is required for the listing Residential filter.")

    print("Listing PropertyType distribution before Residential filter:")
    print(listings["PropertyType"].value_counts(dropna=False))

    listings = listings[listings["PropertyType"] == "Residential"].copy()

    print(f"\nListing rows after Residential filter: {len(listings):,}")
    print("Listing PropertyType distribution after Residential filter:")
    print(listings["PropertyType"].value_counts(dropna=False))

    listing_outlier_fields = [
        "ListPrice",
        "LivingArea",
        "DaysOnMarket",
    ]

    missing_listing_fields = [
        col for col in listing_outlier_fields if col not in listings.columns
    ]
    if missing_listing_fields:
        raise KeyError(f"Missing required listing outlier fields: {missing_listing_fields}")

    for col in listing_outlier_fields:
        listings[col] = pd.to_numeric(listings[col], errors="coerce")

    listings["invalid_list_price_flag"] = (
        listings["ListPrice"].isna() | (listings["ListPrice"] <= 0)
    )
    listings["invalid_living_area_flag"] = (
        listings["LivingArea"].isna() | (listings["LivingArea"] <= 0)
    )
    listings["invalid_days_on_market_flag"] = (
        listings["DaysOnMarket"].isna() | (listings["DaysOnMarket"] < 0)
    )

    listing_iqr_thresholds = []

    for col in listing_outlier_fields:
        valid_series = listings[col].dropna()

        q1 = valid_series.quantile(0.25)
        q3 = valid_series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        listing_iqr_thresholds.append({
            "field": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })

    listing_iqr_thresholds_df = pd.DataFrame(listing_iqr_thresholds)

    for _, row in listing_iqr_thresholds_df.iterrows():
        col = row["field"]
        lower = row["lower_bound"]
        upper = row["upper_bound"]

        flag_col = f"{col.lower()}_iqr_outlier_flag"

        listings[flag_col] = (
            listings[col].notna()
            & ((listings[col] < lower) | (listings[col] > upper))
        )

    listings["any_listing_business_rule_invalid_flag"] = (
        listings["invalid_list_price_flag"]
        | listings["invalid_living_area_flag"]
        | listings["invalid_days_on_market_flag"]
    )

    listings["any_listing_iqr_outlier_flag"] = (
        listings["listprice_iqr_outlier_flag"]
        | listings["livingarea_iqr_outlier_flag"]
        | listings["daysonmarket_iqr_outlier_flag"]
    )

    listings["remove_from_listing_clean_analysis_flag"] = (
        listings["any_listing_business_rule_invalid_flag"]
        | listings["any_listing_iqr_outlier_flag"]
    )

    listings_clean_filtered = listings[
        ~listings["remove_from_listing_clean_analysis_flag"]
    ].copy()

    listing_before_after_comparison = pd.DataFrame([
        {
            "field": col,
            "before_row_count": listings[col].notna().sum(),
            "after_row_count": listings_clean_filtered[col].notna().sum(),
            "before_median": listings[col].median(),
            "after_median": listings_clean_filtered[col].median(),
            "median_change": (
                listings_clean_filtered[col].median() - listings[col].median()
            ),
            "before_mean": listings[col].mean(),
            "after_mean": listings_clean_filtered[col].mean(),
            "mean_change": (
                listings_clean_filtered[col].mean() - listings[col].mean()
            ),
        }
        for col in listing_outlier_fields
    ])

    print("\nListing-side outlier check:")
    print(f"Original listing rows: {len(listings):,}")
    print(f"Clean filtered listing rows: {len(listings_clean_filtered):,}")
    print(f"Rows removed: {len(listings) - len(listings_clean_filtered):,}")
    print(
        "Removal percentage: "
        f"{(len(listings) - len(listings_clean_filtered)) / len(listings):.2%}"
    )

    print("\nListing IQR thresholds:")
    print(listing_iqr_thresholds_df)

    print("\nListing before/after comparison:")
    print(listing_before_after_comparison)

else:
    listings = None
    listing_iqr_thresholds_df = pd.DataFrame()
    listings_clean_filtered = pd.DataFrame()
    listing_before_after_comparison = pd.DataFrame()

    print("\nListing file not found. Optional listing-side check skipped.")


# -----------------------------
# 13. Save Week 7 outputs
# -----------------------------

full_flagged_output = OUTPUT_DIR / "sold_week7_full_flagged_dataset.csv"
clean_filtered_output = OUTPUT_DIR / "sold_week7_clean_filtered_dataset.csv"
iqr_thresholds_output = OUTPUT_DIR / "week7_iqr_thresholds.csv"
business_rule_output = OUTPUT_DIR / "week7_business_rule_summary.csv"
iqr_flag_output = OUTPUT_DIR / "week7_iqr_flag_summary.csv"
overall_flag_output = OUTPUT_DIR / "week7_overall_flag_summary.csv"
before_after_output = OUTPUT_DIR / "week7_before_after_comparison.csv"
percentile_comparison_output = OUTPUT_DIR / "week7_percentile_comparison.csv"

sold.to_csv(full_flagged_output, index=False)
sold_clean_filtered.to_csv(clean_filtered_output, index=False)
iqr_thresholds_df.to_csv(iqr_thresholds_output, index=False)
business_rule_summary.to_csv(business_rule_output, index=False)
iqr_flag_summary.to_csv(iqr_flag_output, index=False)
overall_flag_summary.to_csv(overall_flag_output, index=False)
before_after_comparison.to_csv(before_after_output, index=False)
percentile_comparison.to_csv(percentile_comparison_output, index=False)

print("\nSaved required Week 7 sold-side outputs:")
print(full_flagged_output)
print(clean_filtered_output)
print(iqr_thresholds_output)
print(business_rule_output)
print(iqr_flag_output)
print(overall_flag_output)
print(before_after_output)
print(percentile_comparison_output)

if listings is not None:
    listing_full_flagged_output = OUTPUT_DIR / "listing_week7_full_flagged_dataset.csv"
    listing_clean_filtered_output = OUTPUT_DIR / "listing_week7_clean_filtered_dataset.csv"
    listing_iqr_thresholds_output = OUTPUT_DIR / "week7_listing_iqr_thresholds.csv"
    listing_before_after_output = OUTPUT_DIR / "week7_listing_before_after_comparison.csv"

    listings.to_csv(listing_full_flagged_output, index=False)
    listings_clean_filtered.to_csv(listing_clean_filtered_output, index=False)
    listing_iqr_thresholds_df.to_csv(listing_iqr_thresholds_output, index=False)
    listing_before_after_comparison.to_csv(listing_before_after_output, index=False)

    print("\nSaved optional listing-side outputs:")
    print(listing_full_flagged_output)
    print(listing_clean_filtered_output)
    print(listing_iqr_thresholds_output)
    print(listing_before_after_output)

print("\nWeek 7 outlier detection completed successfully.")
