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





