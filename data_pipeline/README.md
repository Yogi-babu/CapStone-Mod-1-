# Data Pipeline Module

This module scrapes catalog data from `books.toscrape.com`, cleans it, converts the baseline price using the project-defined fixed rate, loads it into a normalized SQLite database, and verifies the results with SQL and pandas.

## Fixed baseline conversion rate

The required project baseline is:

1 GBP = 105.50 INR

This is a fixed, keyless project constant for the assignment and is intentionally not a live market lookup or date-dependent rate.

## Install

```bash
cd /path/to/CAPSTONE
python -m venv .venv
.venv\Scripts\activate
pip install -r data_pipeline/requirements.txt
```

## Run

```bash
python data_pipeline/etl_pipeline.py
```

The script will:

1. Scrape books from at least three categories on `books.toscrape.com`.
2. Clean price, rating, and availability fields.
3. Convert to `price_inr` using the fixed 1 GBP = 105.50 INR rate.
4. Create a normalized SQLite schema with `categories` and `books` tables.
5. Execute and save SQL queries with outputs.
6. Compare SQL and pandas join results for validation.

## Cleaning and parsing decisions

- `price_gbp`: currency symbol stripped and converted to `float`.
- `rating`: text like `One`/`Five` converted to integer 1–5.
- `in_stock`: parsed from the availability text and stored as integer 0/1 in SQLite.
- For messy or missing numeric values, the pipeline uses median imputation instead of crashing the run. This keeps the ETL resilient while preserving a consistent numeric distribution. Non-critical textual fields that fail parsing are dropped or flagged as missing rather than causing the pipeline to abort.

## Output artifacts

- SQLite database: `data_pipeline/books_catalog.db`
- Query log: `data_pipeline/sql_queries_and_results.txt`
- Script: `data_pipeline/etl_pipeline.py`
