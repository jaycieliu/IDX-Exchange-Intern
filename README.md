# IDX-Exchange-Intern

## Week 0 Progress

- Set up the local development environment in **VS Code**.
- Downloaded all raw MLS CSV files from the FTP **`/raw`** directory.
- Reviewed and organized the two provided extraction scripts
- Ran the provided scripts successfully to generate the **February 2026** outputs
- Updated both extraction scripts 
- Ran the updated scripts successfully to also generate the **March 2026** outputs
- Performed a basic validation check on the generated CSV files.
- Organized the project structure so the code, raw files, and generated outputs are clearly separated for the next phase of work.

## Week 1 Progress

- Concatenated monthly listing and sold MLS data from 01-2024 to 03-2026, using the official centralized files provided in the Spring 2026 group folder:
  -  Sold with 591733 rows and Listing with 852963 rows.
- Filtered to only **Residential** property type:
  -  Sold with 397603 rows and Listing with 540183 rows.
- Validated the workflow: compared total rows counts before and after concatenation and filtering
- Codes:
  - [Week1 Notebook](/Week1/week1_Jaycie_L.ipynb)
  - [Week1 Python script](/Week1/week1_Jaycie_L.py)

## Week 2 Progress

-	Inspected dataset structure, key fields, and data types for both sold and listing datasets.
-	Validated PropertyType distributions and filtered both datasets to Residential only.
  - PropertyType distributions:
    - Sold dataset:
     <img width="790" height="510" alt="image" src="https://github.com/user-attachments/assets/4f4a25a7-49a9-4432-a22e-347781f6b896" />
    - Listing dataset:
      <img width="790" height="510" alt="image" src="https://github.com/user-attachments/assets/f5507914-74e7-4638-add7-db880c59253d" />

-	Built null-count and missing-value reports, including >90% missing flags.
  - Sold dataset:
    <img width="596" height="363" alt="Screenshot 2026-04-19 at 4 05 38 PM" src="https://github.com/user-attachments/assets/1c610bd2-31d8-47a6-9d7c-942006f23eaa" />
  - Listing dataset:
    <img width="588" height="368" alt="image" src="https://github.com/user-attachments/assets/ffeae135-2d81-4f05-b6ba-29fefc842750" />
    
-	Reviewed required numeric fields with summary stats, percentiles, histograms, and boxplots.
  - Sold dataset:
    <img width="1272" height="255" alt="image" src="https://github.com/user-attachments/assets/f00a3547-5a5a-456a-afff-e4c6327a666e" />

  - Listing dataset:
    <img width="1269" height="245" alt="image" src="https://github.com/user-attachments/assets/b223b384-d8ee-4af0-aaee-2d5a0e6a7c8e" />

-	Identified strange values, invalid records, and extreme outliers.
-	Completed initial EDA on pricing, Days on Market, date consistency, and county-level median prices.
- Saved filtered residential datasets and removed non-core variables with more than 90% missing values for the next phase.
- Codes:
  - [Week2 Notebook](/week2/week2_Jaycie_L.ipynb)
  - [Week2 Python script](/week2/week2_Jaycie_L.py)
