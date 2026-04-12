import pandas as pd
from glob import glob
import os

# define file paths
sold_folder = "../data/raw/sold"
sold_files = sorted(glob(os.path.join(sold_folder, "*.csv")))

listing_folder = "../data/raw/listing"
listing_files = sorted(glob(os.path.join(listing_folder, "*.csv")))

#-------Sold data processing-------
# load and combine Sold data
sold_dfs = []
sold_total_rows_before = 0
for file in sold_files:
    df = pd.read_csv(file)
    sold_total_rows_before += len(df)
    sold_dfs.append(df)

print(f"Total Sold rows before concatenation: {sold_total_rows_before}")

# combine
sold_combined = pd.concat(sold_dfs, ignore_index=True)
print(f"Total Sold rows after concatenation: {len(sold_combined)}")

# 'Residential' filter 
sold_residential = sold_combined[sold_combined["PropertyType"] == "Residential"]

print(f"Sold rows after Residential filter: {len(sold_residential)}")

#save Sold output
sold_residential.to_csv("../data/generated/combined_sold_residential.csv", index=False)


#-------Listing data processing-------
# load and combine Listing data
listing_dfs = []
listing_total_rows_before = 0
for file in listing_files:
    df = pd.read_csv(file)
    listing_total_rows_before += len(df)
    listing_dfs.append(df)

print(f"Total Listing rows before concatenation: {listing_total_rows_before}")

# combine
listing_combined = pd.concat(listing_dfs, ignore_index=True)
print(f"Total Listing rows after concatenation: {len(listing_combined)}")

# 'Residential' filter 
listing_residential = listing_combined[listing_combined["PropertyType"] == "Residential"]

print(f"Listing rows after Residential filter: {len(listing_residential)}")

#save Listing output
listing_residential.to_csv("../data/generated/combined_listing_residential.csv", index=False)






