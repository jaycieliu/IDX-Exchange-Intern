import pandas as pd
from pathlib import Path

# -----------------------------
# Load Week 3 mortgage-enriched datasets
# -----------------------------
# Update these paths if needed.
sold_input = Path('/Users/amyliu/Desktop/IDX/data/generated/sold_with_mortgage_rates.csv')
listing_input = Path('/Users/amyliu/Desktop/IDX/data/generated/listing_with_mortgage_rates.csv')

sold_with_mortgage_rates = pd.read_csv(sold_input)
listings_with_mortgage_rates = pd.read_csv(listing_input)

# Working copies
sold_with_rates = sold_with_mortgage_rates.copy()
listing_with_rates = listings_with_mortgage_rates.copy()

# -----------------------------
# Step 1: Convert date fields to datetime
# -----------------------------
sold_date_cols = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
for col in sold_date_cols:
    if col in sold_with_rates.columns:
        sold_with_rates[col] = pd.to_datetime(sold_with_rates[col], errors='coerce')

listing_date_cols = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
for col in listing_date_cols:
    if col in listing_with_rates.columns:
        listing_with_rates[col] = pd.to_datetime(listing_with_rates[col], errors='coerce')

# -----------------------------
# Step 2: Remove redundant / unnecessary columns
# -----------------------------
# 2.1 Drop non-core columns with >90% missing
core_fields_sold = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate',
    'LivingArea', 'LotSizeAcres', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
    'CountyOrParish', 'City', 'PostalCode', 'Latitude', 'Longitude',
    'PropertyType', 'PropertySubType', 'MLSAreaMajor',
    'ListOfficeName', 'BuyerOfficeName',
    'year_month', 'rate_30yr_fixed'
]

high_missing_cols_sold = sold_with_rates.isnull().mean().loc[lambda x: x > 0.90].index.tolist()
drop_cols_sold = [col for col in high_missing_cols_sold if col not in core_fields_sold]
sold_with_rates = sold_with_rates.drop(columns=drop_cols_sold)

print('Columns dropped from sold dataset:')
print(drop_cols_sold)
print('New sold dataset shape:', sold_with_rates.shape)

core_fields_listing = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate',
    'LivingArea', 'LotSizeAcres', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt',
    'CountyOrParish', 'City', 'PostalCode', 'Latitude', 'Longitude',
    'PropertyType', 'PropertySubType', 'MLSAreaMajor',
    'ListOfficeName', 'year_month', 'rate_30yr_fixed'
]

high_missing_cols_listing = listing_with_rates.isnull().mean().loc[lambda x: x > 0.90].index.tolist()
drop_cols_listing = [col for col in high_missing_cols_listing if col not in core_fields_listing]
listing_with_rates = listing_with_rates.drop(columns=drop_cols_listing)

print('Columns dropped from listing dataset:')
print(drop_cols_listing)
print('New listing dataset shape:', listing_with_rates.shape)

# 2.2 Drop obvious duplicate .1 columns in listings if they are true duplicates
duplicate_pairs = [
    ('PropertyType', 'PropertyType.1'),
    ('ListAgentFirstName', 'ListAgentFirstName.1'),
    ('DaysOnMarket', 'DaysOnMarket.1'),
    ('LivingArea', 'LivingArea.1'),
    ('Longitude', 'Longitude.1'),
    ('Latitude', 'Latitude.1'),
    ('ListPrice', 'ListPrice.1'),
    ('ListAgentLastName', 'ListAgentLastName.1'),
    ('CloseDate', 'CloseDate.1'),
    ('BuyerOfficeName', 'BuyerOfficeName.1'),
    ('UnparsedAddress', 'UnparsedAddress.1'),
]

true_duplicate_cols = []
for col1, col2 in duplicate_pairs:
    if col1 in listing_with_rates.columns and col2 in listing_with_rates.columns:
        same = listing_with_rates[col1].equals(listing_with_rates[col2])
        print(f'{col1} vs {col2}: {same}')
        if same:
            true_duplicate_cols.append(col2)

print('True duplicate .1 columns to drop:')
print(true_duplicate_cols)
listing_with_rates = listing_with_rates.drop(columns=true_duplicate_cols)

# -----------------------------
# Step 3: Confirm numeric fields are numeric
# -----------------------------
sold_numeric_cols = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket',
    'YearBuilt', 'Latitude', 'Longitude', 'rate_30yr_fixed'
]

listing_numeric_cols = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice',
    'LivingArea', 'LotSizeAcres', 'LotSizeSquareFeet',
    'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket',
    'YearBuilt', 'Latitude', 'Longitude', 'rate_30yr_fixed'
]

for col in sold_numeric_cols:
    if col in sold_with_rates.columns:
        sold_with_rates[col] = pd.to_numeric(sold_with_rates[col], errors='coerce')

for col in listing_numeric_cols:
    if col in listing_with_rates.columns:
        listing_with_rates[col] = pd.to_numeric(listing_with_rates[col], errors='coerce')

print('Sold dataset numeric field types:')
print(sold_with_rates[[c for c in sold_numeric_cols if c in sold_with_rates.columns]].dtypes)

print('\nListing dataset numeric field types:')
print(listing_with_rates[[c for c in listing_numeric_cols if c in listing_with_rates.columns]].dtypes)

# -----------------------------
# Step 4: Create invalid-value flags
# -----------------------------
sold_with_rates['invalid_closeprice_flag'] = sold_with_rates['ClosePrice'] <= 0
sold_with_rates['invalid_livingarea_flag'] = sold_with_rates['LivingArea'] <= 0
sold_with_rates['invalid_dom_flag'] = sold_with_rates['DaysOnMarket'] < 0
sold_with_rates['invalid_bedrooms_flag'] = sold_with_rates['BedroomsTotal'] < 0
sold_with_rates['invalid_bathrooms_flag'] = sold_with_rates['BathroomsTotalInteger'] < 0

listing_with_rates['invalid_closeprice_flag'] = listing_with_rates['ClosePrice'] <= 0
listing_with_rates['invalid_livingarea_flag'] = listing_with_rates['LivingArea'] <= 0
listing_with_rates['invalid_dom_flag'] = listing_with_rates['DaysOnMarket'] < 0
listing_with_rates['invalid_bedrooms_flag'] = listing_with_rates['BedroomsTotal'] < 0
listing_with_rates['invalid_bathrooms_flag'] = listing_with_rates['BathroomsTotalInteger'] < 0

# -----------------------------
# Step 5: Create date consistency flags
# -----------------------------
for df in [sold_with_rates, listing_with_rates]:
    df['listing_after_close_flag'] = df['ListingContractDate'] > df['CloseDate']
    df['purchase_after_close_flag'] = df['PurchaseContractDate'] > df['CloseDate']
    df['negative_timeline_flag'] = df['PurchaseContractDate'] < df['ListingContractDate']

# -----------------------------
# Step 6: Create geographic data quality flags
# -----------------------------
for df in [sold_with_rates, listing_with_rates]:
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    df['missing_coord_flag'] = df['Latitude'].isna() | df['Longitude'].isna()
    df['zero_coord_flag'] = (df['Latitude'] == 0) | (df['Longitude'] == 0)
    df['positive_longitude_flag'] = df['Longitude'] > 0
    df['implausible_coord_flag'] = (
        (df['Latitude'] < 30) | (df['Latitude'] > 43) |
        (df['Longitude'] < -125) | (df['Longitude'] > -114)
    )

# -----------------------------
# Summary outputs required by the deliverable
# -----------------------------
print('\nSold dataset flag summary')
print('Invalid numeric flags:')
print('ClosePrice <= 0:', sold_with_rates['invalid_closeprice_flag'].sum())
print('LivingArea <= 0:', sold_with_rates['invalid_livingarea_flag'].sum())
print('DaysOnMarket < 0:', sold_with_rates['invalid_dom_flag'].sum())
print('BedroomsTotal < 0:', sold_with_rates['invalid_bedrooms_flag'].sum())
print('BathroomsTotalInteger < 0:', sold_with_rates['invalid_bathrooms_flag'].sum())
print('\nDate consistency flags:')
print('Listing after close:', sold_with_rates['listing_after_close_flag'].sum())
print('Purchase after close:', sold_with_rates['purchase_after_close_flag'].sum())
print('Negative timeline:', sold_with_rates['negative_timeline_flag'].sum())
print('\nGeographic data quality flags:')
print('Missing coordinates:', sold_with_rates['missing_coord_flag'].sum())
print('Zero coordinates:', sold_with_rates['zero_coord_flag'].sum())
print('Positive longitude:', sold_with_rates['positive_longitude_flag'].sum())
print('Implausible coordinates:', sold_with_rates['implausible_coord_flag'].sum())

print('\nListing dataset flag summary')
print('Invalid numeric flags:')
print('ClosePrice <= 0:', listing_with_rates['invalid_closeprice_flag'].sum())
print('LivingArea <= 0:', listing_with_rates['invalid_livingarea_flag'].sum())
print('DaysOnMarket < 0:', listing_with_rates['invalid_dom_flag'].sum())
print('BedroomsTotal < 0:', listing_with_rates['invalid_bedrooms_flag'].sum())
print('BathroomsTotalInteger < 0:', listing_with_rates['invalid_bathrooms_flag'].sum())
print('\nDate consistency flags:')
print('Listing after close:', listing_with_rates['listing_after_close_flag'].sum())
print('Purchase after close:', listing_with_rates['purchase_after_close_flag'].sum())
print('Negative timeline:', listing_with_rates['negative_timeline_flag'].sum())
print('\nGeographic data quality flags:')
print('Missing coordinates:', listing_with_rates['missing_coord_flag'].sum())
print('Zero coordinates:', listing_with_rates['zero_coord_flag'].sum())
print('Positive longitude:', listing_with_rates['positive_longitude_flag'].sum())
print('Implausible coordinates:', listing_with_rates['implausible_coord_flag'].sum())

print('\nBefore/after row counts')
print('Sold rows before cleaning:', len(sold_with_mortgage_rates))
print('Sold rows after cleaning:', len(sold_with_rates))
print('Listing rows before cleaning:', len(listings_with_mortgage_rates))
print('Listing rows after cleaning:', len(listing_with_rates))

print('\nBefore/after shapes')
print('Sold shape before cleaning:', sold_with_mortgage_rates.shape)
print('Sold shape after cleaning:', sold_with_rates.shape)
print('Listing shape before cleaning:', listings_with_mortgage_rates.shape)
print('Listing shape after cleaning:', listing_with_rates.shape)

# -----------------------------
# Save cleaned datasets
# -----------------------------
sold_with_rates.to_csv('sold_with_rates_week4-5.csv', index=False)
listing_with_rates.to_csv('listing_with_rates_week4-5.csv', index=False)
