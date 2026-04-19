# %%
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# # Part 1: For sold dataset

# %%
sold_combined = pd.read_csv('../data/generated/combined_sold.csv')

# %% [markdown]
# ## 1. Inspect structure

# %%
sold_combined.columns

# %%
sold_combined.shape

# %%
sold_combined.head()

# %%
sold_combined.info()

# %% [markdown]
# Sold rows before Residential filter: 591733

# %% [markdown]
# ## 2. Property type validation

# %% [markdown]
# **Q1: What property types exist?**

# %%
sold_combined['PropertyType'].unique()

# %%
sold_combined['PropertyType'].value_counts(dropna=False)

# %%
sold_property = sold_combined.copy()

sold_property_count = sold_property['PropertyType'].value_counts().reset_index()

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=sold_property_count,
    x='PropertyType',
    y='count',
    hue='PropertyType'
)

for bar in ax.patches:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    plt.text(
        x,
        height + 2000,   
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.title('Property Type Distribution — Sold Dataset')
plt.show()

# %% [markdown]
# The property type distribution shows that the raw sold dataset is not limited to standard residential transactions. Although Residential is the primary category, the dataset also includes lease, land, manufactured housing, income, and commercial-related records. Therefore, filtering to Residential only is a necessary validation step to ensure the analysis population is consistent with the intended residential market focus.

# %% [markdown]
# **Q2: What is the Residential vs. other property type share?**

# %%
sold_property_count

# %%
residential_count = sold_property_count.loc[
    sold_property_count['PropertyType'] == 'Residential', 'count'
].iloc[0]
total_count = len(sold_combined)
other_count = total_count - residential_count

residential_pct = residential_count / total_count * 100
other_pct = other_count / total_count * 100

print(f"Residential share on Sold Dataset: {residential_pct:.2f}%")
print(f"Other share on Sold Dataset: {other_pct:.2f}%")

# %% [markdown]
# Residential properties make up 67.19% of the combined sold dataset, compared with 32.81% for all other property types. 

# %% [markdown]
# ## 3. Structure inspection on filtered residential data

# %%
#use filtered dataset
sold_combined_residential = pd.read_csv('../data/generated/combined_sold_residential.csv')

# %%
sold_combined_residential.info()

# %% [markdown]
# Sold rows after Residential filter: 397603

# %%
sold_combined_residential.shape

# %%
sold_combined_residential.columns

# %%
sold_combined_residential.info()

# %%
sold_combined_residential.head()

# %% [markdown]
# The filtered residential sold dataset retains 397,603 observations across 84 columns, showing that a substantial residential analysis sample remains after applying the property type filter. Core market-analysis variables such as ClosePrice, ListPrice, OriginalListPrice, LivingArea, DaysOnMarket, CountyOrParish, City, and the major transaction date fields are all present, so the dataset contains the key information needed for subsequent EDA. Most major pricing and housing characteristic fields already have appropriate numeric data types, but several date variables are still stored as strings and will need to be converted to datetime in later cleaning.

# %% [markdown]
# ## 4. Missing value analysis

# %%
sold_combined_residential.isnull().sum()

# %%
missing_report = pd.DataFrame({
    'column': sold_combined_residential.columns,
    'null_count': sold_combined_residential.isnull().sum().values,
    'null_pct': sold_combined_residential.isnull().sum().values / len(sold_combined_residential) * 100
})

missing_report['gt_90pct_missing'] = missing_report['null_pct'] > 90
missing_report = missing_report.sort_values('null_pct', ascending=False)
missing_report

# %%
missing_report[missing_report['gt_90pct_missing'] == True][['column','null_pct']]

# %% [markdown]
# For the flagged columns(missing percentage greater than 90%), there are no clear must-keep core variables. Thus, Those high-missing, low-value columns should be dropped.

# %%
sold_cleaned = sold_combined_residential.drop(columns=missing_report[missing_report['gt_90pct_missing'] == True]['column'].tolist()).copy()

# %%
sold_cleaned.info()

# %%
sold_cleaned.to_csv('../data/generated/sold_cleaned.csv', index=False)

# %% [markdown]
# ## 5. Numeric distribution review

# %%
# identify numeric columns
numeric_keys = [
    'ClosePrice',
    'ListPrice',
    'OriginalListPrice',
    'LivingArea',
    'LotSizeAcres',
    'BedroomsTotal',
    'BathroomsTotalInteger',
    'DaysOnMarket',
    'YearBuilt'
]

# %%
#summary statistics for key numeric columns
sold_cleaned[numeric_keys].describe()

# %% [markdown]
# The numeric distribution review indicates that the residential sold dataset contains both invalid values and extreme right-tail outliers. *DaysOnMarket* includes **negative observations**, which are not logically valid. And several core variables such as *ClosePrice, OriginalListPrice, LivingArea, and LotSizeAcres* contain **zero values** that may represent invalid or placeholder entries. In addition, the large gaps between medians and maximum values — for example in *ClosePrice, LivingArea, LotSizeAcres, and BathroomsTotalInteger* — show that the data is **highly skewed** and includes unusually large observations that will need to be flagged or addressed in later cleaning steps.

# %%
#percentile summary for key numeric columns
percentile_summary_sold = (sold_cleaned[numeric_keys].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T)[
    ['min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max']
]

percentile_summary_sold

# %% [markdown]
# ### **5.1 Histogram for each key numeric variables**

# %% [markdown]
# Close Price is behaving exactly the way a highly right-skewed variable with extreme outliers behaves.

# %%
for col in numeric_keys:
    plt.figure(figsize=(6,4))
    sns.histplot(data = sold_cleaned,
             x = col, 
             edgecolor = 'blue',
             color = 'blue',
             stat='density',
             kde=True
             )
    mean_value = sold_cleaned[col].mean()
    plt.axvline(x = mean_value,
                         color = 'red',
                         linestyle = '--',
                         linewidth = 2) #vertical line
    plt.title(f'Distribution of {col} — Sold Dataset')

# %% [markdown]
# The histogram review shows that several key residential variables — including *ClosePrice, ListPrice, OriginalListPrice, LivingArea, LotSizeAcres, and DaysOnMarket* — are **highly right-skewed** and contain **extreme upper-tail** observations. In contrast, *BedroomsTotal and BathroomsTotalInteger* are concentrated around typical residential values but still contain unusually large observations, while *YearBuilt* has a more interpretable historical distribution centered in the modern period. Overall, the plots confirm that the dataset contains both implausible values and substantial outliers, indicating that later cleaning steps should include invalid-value checks and outlier treatment before final market analysis.

# %% [markdown]
# ### **5.2 Boxplot**

# %%
for col in numeric_keys:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=sold_cleaned[col])
    plt.title(f'Boxplot of {col} - Sold Dataset')

# %% [markdown]
# The boxplots confirm the conclusions from the histograms by showing that several key numeric variables are highly right-skewed and contain substantial outliers.

# %% [markdown]
# ## 6. Outlier and strange values

# %% [markdown]
# ### 6.1. Outlier values applied IQR method

# %%
for col in numeric_keys:
    #Identifying outliers using boxplot method for each key numeric column
    Q1 = sold_cleaned[col].quantile(0.25)
    Q3 = sold_cleaned[col].quantile(0.75)


    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate the upper bound for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5*IQR

    print(f'Outlier bounds for {col} - Sold Dataset: [{lower_bound:.2f}, {upper_bound:.2f}]')

    outliers = sold_cleaned[(sold_cleaned[col] > upper_bound) | (sold_cleaned[col] < lower_bound)]
    print(f'Percentage of {col} outliers - Sold Dataset:', round(len(outliers) / len(sold_cleaned) * 100,2),'%\n')


# %% [markdown]
# ### 6.2. Strange values
# -  Invalid : 
#     * DaysOnMarket < 0
# 	* ClosePrice = 0
# 	* OriginalListPrice = 0
# 	* LivingArea = 0
# 	* LotSizeAcres = 0
# 	* BedroomsTotal = 0
# 	* BathroomsTotalInteger = 0
# - Very large:
# 	* BedroomsTotal = 45
# 	* BathroomsTotalInteger = 175
# 	* LivingArea = 17,021,320
# 	* LotSizeAcres = 7,810,698
# 
# - Unusual but may be possible:
# 	* YearBuilt = 1776
# 

# %% [markdown]
# ## 7. EDA question

# %% [markdown]
# 1.  What are the median and average close prices?

# %%
median_close_price = sold_cleaned['ClosePrice'].median()
mean_close_price = sold_cleaned['ClosePrice'].mean()

print("Median ClosePrice:", median_close_price)
print("Average ClosePrice:", mean_close_price)

# %% [markdown]
# 2. What does the Days on Market distribution look like?

# %%
sold_cleaned['DaysOnMarket'].describe()

# %% [markdown]
# *Days on Market* is strongly right-skewed, with most observations clustered at lower values and a long upper tail of extreme observations. It also includes negative values, which are invalid, and the large gap between the median and maximum indicates substantial upper-tail outliers.

# %% [markdown]
# 3. What percentage of homes sold above vs. below list price?

# %%
above_list_price = sold_cleaned[sold_cleaned['ClosePrice'] > sold_cleaned['ListPrice']]
below_list_price = sold_cleaned[sold_cleaned['ClosePrice'] < sold_cleaned['ListPrice']]
equal_list_price = sold_cleaned[sold_cleaned['ClosePrice'] == sold_cleaned['ListPrice']]

# since some rows may have missing values in either ClosePrice or ListPrice, we should only consider rows where both are present for percentage calculations

total_valid = len(sold_cleaned[['ClosePrice', 'ListPrice']].dropna())

above_pct = len(above_list_price) / total_valid * 100
below_pct = len(below_list_price) / total_valid * 100
at_pct = len(equal_list_price) / total_valid * 100

print("Above list percentage:", f"{above_pct:.2f}%")
print("Below list percentage:", f"{below_pct:.2f}%")

# %% [markdown]
# 4. Are there any apparent date consistency issues (e.g., close date before listing date)?

# %%
#convert to datetime
sold_cleaned['CloseDate'] = pd.to_datetime(sold_cleaned['CloseDate'], errors='coerce')
sold_cleaned['ListingContractDate'] = pd.to_datetime(sold_cleaned['ListingContractDate'], errors='coerce')
sold_cleaned['PurchaseContractDate'] = pd.to_datetime(sold_cleaned['PurchaseContractDate'], errors='coerce')


close_before_listing = (sold_cleaned['CloseDate'] < sold_cleaned['ListingContractDate']).sum()
close_before_purchase = (sold_cleaned['CloseDate'] < sold_cleaned['PurchaseContractDate']).sum()
purchase_before_listing = (sold_cleaned['PurchaseContractDate'] < sold_cleaned['ListingContractDate']).sum()

print("Close date before listing date - sold dataset:", close_before_listing)
print("Close date before purchase contract date - sold dataset:", close_before_purchase)
print("Purchase contract date before listing date - sold dataset:", purchase_before_listing)

# %% [markdown]
# 5. Which counties have the highest median prices?

# %%
county_median_prices = (
    sold_cleaned
    .groupby('CountyOrParish')['ClosePrice']
    .median()
    .sort_values(ascending=False)
)

county_median_prices.head(10)

# %% [markdown]
# The counties with the highest median close prices are Del Norte. 

# %% [markdown]
# # Part 2: For listing dataset

# %%
listing_combined = pd.read_csv('../data/generated/combined_listing.csv')

# %%
listing_combined.head()

# %% [markdown]
# ## 1. Structure Inspection

# %%
listing_combined.shape

# %%
listing_combined.columns

# %%
listing_combined.info()

# %% [markdown]
# Listing rows before Residential filter: 852963

# %% [markdown]
# ## 2.Property Type Validation

# %% [markdown]
# **Q1: What property types exist?**

# %%
listing_combined['PropertyType'].unique()

# %%
listing_combined['PropertyType'].value_counts(dropna=False)

# %%
listing_property = listing_combined.copy()

listing_property_count = listing_property['PropertyType'].value_counts().reset_index()

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=listing_property_count,
    x='PropertyType',
    y='count',
    hue='PropertyType'
)

for bar in ax.patches:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    plt.text(
        x,
        height + 2000,   
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.ylabel('Count')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.title('Property Type Distribution — Listing Dataset')
plt.show()

# %% [markdown]
# The listing dataset includes several property types, but Residential is by far the dominant category with 540,183 records, substantially exceeding all other groups. ResidentialLease is the second-largest category, while the remaining property types make up much smaller portions of the data. This supports applying a only Residential filter so that subsequent analysis focuses only on standard residential listings.

# %% [markdown]
# **Q2: What is the Residential vs. other property type share?**

# %%
listing_property_count

# %%
residential_count = listing_property_count.loc[
    listing_property_count['PropertyType'] == 'Residential', 'count'
].iloc[0]
total_count = len(listing_combined)
other_count = total_count - residential_count

residential_pct = residential_count / total_count * 100
other_pct = other_count / total_count * 100

print(f"Residential share on Listing Dataset: {residential_pct:.2f}%")
print(f"Other share on Listing Dataset: {other_pct:.2f}%")

# %% [markdown]
# Residential properties make up 63.33% of the combined listing dataset, compared with 36.67% for all other property types. 

# %% [markdown]
# ## 3. Structure inspection on filtered residential data

# %%
#use filtered dataset
listing_combined_residential = pd.read_csv('../data/generated/combined_listing_residential.csv')

# %%
listing_combined_residential.info()

# %% [markdown]
# Listing rows after Residential filter: 540183

# %%
listing_combined_residential.shape

# %%
listing_combined_residential.head()

# %% [markdown]
# ## 4. Missing value Analysis

# %%
listing_combined_residential.isnull().sum()

# %%
missing_report_listing = pd.DataFrame({
    'column': listing_combined_residential.columns,
    'null_count': listing_combined_residential.isnull().sum().values,
    'null_pct': listing_combined_residential.isnull().sum().values / len(listing_combined_residential) * 100
})

missing_report_listing['gt_90pct_missing'] = missing_report_listing['null_pct'] > 90
missing_report_listing = missing_report_listing.sort_values('null_pct', ascending=False)
missing_report_listing

# %%
missing_report_listing[missing_report_listing['gt_90pct_missing'] == True][['column','null_pct']]

# %% [markdown]
# For the flagged columns(missing percentage greater than 90%), there are no clear must-keep core variables. Thus, Those high-missing, low-value columns should be dropped.

# %%
listing_combined_cleaned = listing_combined_residential.drop(columns=missing_report_listing[missing_report_listing['gt_90pct_missing'] == True]['column'].tolist()).copy()

# %%
listing_combined_cleaned.info()

# %%
listing_combined_cleaned.to_csv('../data/generated/listing_cleaned.csv', index=False)

# %% [markdown]
# The filtered residential listings dataset contains 540,183 rows and 71 columns, providing a large sample for listing-side analysis. Core fields such as *ListPrice, OriginalListPrice, LivingArea, DaysOnMarket, ListingContractDate, CountyOrParish, City, PropertySubType, BedroomsTotal, BathroomsTotalInteger*, and *YearBuilt* are present, so the dataset includes the main variables needed for pricing, timing, geographic, and property-level analysis. Most major analytic fields already have appropriate numeric types, but key date columns are still stored as strings and should be converted to datetime. The structure review also shows some duplicated columns with *.1 suffixes* and several fields with substantial missingness, so duplicate-column review and missing-value analysis are needed next.

# %% [markdown]
# ## 5. Numeric Distribution Review

# %%
# identify numeric columns
listing_numeric_focus = [
    'ClosePrice',
    'ListPrice',
    'OriginalListPrice',
    'LivingArea',
    'LotSizeAcres',
    'BedroomsTotal',
    'BathroomsTotalInteger',
    'DaysOnMarket',
    'YearBuilt'
]

# %%
#summary statistics for key numeric columns
listing_combined_cleaned[listing_numeric_focus].describe()

# %% [markdown]
# The listings numeric summary shows that most key variables are highly right-skewed and contain extreme upper-tail values, especially *ListPrice, OriginalListPrice, LivingArea, and LotSizeAcres*. The data also includes suspicious values such as negative *DaysOnMarket*, zero values in several price and size fields, and implausibly large bedroom and bathroom counts. Overall, these results suggest that the listings dataset contains both outliers and likely data-quality issues that should be flagged before later analysis.

# %%
#percentile summary for key numeric columns
percentile_summary_listing = (listing_combined_cleaned[listing_numeric_focus].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T)[
    ['min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max']
]

percentile_summary_listing

# %% [markdown]
# ### **5.1 Histogram for each key numeric variables**

# %% [markdown]
# Close Price is behaving exactly the way a highly right-skewed variable with extreme outliers behaves.

# %%
for col in listing_numeric_focus:
    plt.figure(figsize=(6,4))
    sns.histplot(data = listing_combined_cleaned,
             x = col, 
             edgecolor = 'blue',
             color = 'blue',
             stat='density',
             kde=True
             )
    mean_value = listing_combined_cleaned[col].mean()
    plt.axvline(x = mean_value,
                         color = 'red',
                         linestyle = '--',
                         linewidth = 2) #vertical line
    plt.title(f'Distribution of {col} — Listing Dataset')

# %% [markdown]
# The histograms show that most key numeric variables in the listings dataset are **highly right-skewed**, especially *ClosePrice, ListPrice, OriginalListPrice, LivingArea, and LotSizeAcres*, where a small number of extreme values stretch the upper tail. *BedroomsTotal and BathroomsTotalInteger* are centered in more typical residential ranges but still include unusual large values, while *DaysOnMarket* has a right-skewed distribution with negative observations. *YearBuilt* is comparatively stable, though a few unusual values remain. Overall, the plots confirm the presence of strong skewness, outliers, and some suspicious records in the listings data. 

# %% [markdown]
# ### **5.2 Boxplot**

# %%
for col in listing_numeric_focus:
    plt.figure(figsize=(6,4))
    sns.boxplot(y=listing_combined_cleaned[col])
    plt.title(f'Boxplot of {col} — Listing Dataset')

# %% [markdown]
# The boxplots confirm the conclusions from the histograms by showing that several key numeric variables are highly right-skewed and contain substantial outliers.

# %% [markdown]
# ## 6. Outlier and strange values

# %% [markdown]
# #### 6.1. Outlier values applied IQR method

# %%
for col in listing_numeric_focus:
    #Identifying outliers using boxplot method for each key numeric column
    Q1 = listing_combined_cleaned[col].quantile(0.25)
    Q3 = listing_combined_cleaned[col].quantile(0.75)


    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Calculate the upper bound for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5*IQR

    print(f'Outlier bounds for {col} — Listing Dataset: [{lower_bound:.2f}, {upper_bound:.2f}]')

    outliers = listing_combined_cleaned[(listing_combined_cleaned[col] > upper_bound) | (listing_combined_cleaned[col] < lower_bound)]
    print(f'Percentage of {col} outliers — Listing Dataset:', round(len(outliers) / len(listing_combined_cleaned) * 100,2),'%\n')


# %% [markdown]
# #### 6.2. Strange values
# -  Invalid : 
#     * DaysOnMarket < 0
# 	* OriginalListPrice = 0
# 	* LivingArea = 0
# 	* LotSizeAcres = 0
# 	* BedroomsTotal = 0
# 	* BathroomsTotalInteger = 0
# - Very large:
# 	* BedroomsTotal = 94
# 	* BathroomsTotalInteger = 2208
# 	* LivingArea = 17,021,320
# 	* LotSizeAcres = 4,187,292
# 
# - Unusual but may be possible:
# 	* YearBuilt = 1776, 2028
# 

# %% [markdown]
# ## 7. EDA

# %% [markdown]
# 1. What are the median and average list prices?

# %%
median_list_price = listing_combined_cleaned['ListPrice'].median()
mean_list_price = listing_combined_cleaned['ListPrice'].mean()

print("Median ListPrice - Listing dataset:", median_list_price)
print("Average ListPrice - Listing dataset:", f"{mean_list_price:.2f}")

# %% [markdown]
# 2. What does the Days on Market distribution look like?

# %%
listing_combined_cleaned['DaysOnMarket'].describe()


# %% [markdown]
# 3. What percentage of listings are currently above, below, or at their original list price?
# 

# %%
above_original_pct = (listing_combined_cleaned['ListPrice'] > listing_combined_cleaned['OriginalListPrice']).mean() * 100
below_original_pct = (listing_combined_cleaned['ListPrice'] < listing_combined_cleaned['OriginalListPrice']).mean() * 100
at_original_pct = (listing_combined_cleaned['ListPrice'] == listing_combined_cleaned['OriginalListPrice']).mean() * 100


# since some rows may have missing values in either ClosePrice or ListPrice, we should only consider rows where both are present for percentage calculations

total_valid = len(listing_combined_cleaned[['OriginalListPrice', 'ListPrice']].dropna())

above_pct = len(above_list_price) / total_valid * 100
below_pct = len(below_list_price) / total_valid * 100
at_pct = len(equal_list_price) / total_valid * 100

print("Above original listing price percentage - listing dataset:", f"{above_pct:.2f}%")
print("Below original listing price percentage - listing dataset:", f"{below_pct:.2f}%")

# %% [markdown]
# 4. Are there any apparent date consistency issues?

# %%
listing_combined_cleaned['CloseDate'] = pd.to_datetime(listing_combined_cleaned['CloseDate'], errors='coerce')
listing_combined_cleaned['ListingContractDate'] = pd.to_datetime(listing_combined_cleaned['ListingContractDate'], errors='coerce')
listing_combined_cleaned['PurchaseContractDate'] = pd.to_datetime(listing_combined_cleaned['PurchaseContractDate'], errors='coerce')


close_before_listing = (listing_combined_cleaned['CloseDate'] < listing_combined_cleaned['ListingContractDate']).sum()
close_before_purchase = (listing_combined_cleaned['CloseDate'] < listing_combined_cleaned['PurchaseContractDate']).sum()
purchase_before_listing = (listing_combined_cleaned['PurchaseContractDate'] < listing_combined_cleaned['ListingContractDate']).sum()

print("Close date before listing date - listing combined cleaned:", close_before_listing)
print("Close date before purchase contract date - listing combined cleaned:", close_before_purchase)
print("Purchase contract date before listing date - listing combined cleaned:", purchase_before_listing)

# %% [markdown]
# 5. Which counties have the highest median list prices?

# %%
county_median_prices_listing = (
    listing_combined_cleaned
    .groupby('CountyOrParish')['ListPrice']
    .median()
    .sort_values(ascending=False)
)

county_median_prices_listing.head(10)

# %% [markdown]
# The counties with the highest median listing prices are San Mateo. 

# %%



