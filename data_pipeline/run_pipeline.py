"""
Zepto Data & AI Platform - Module 1: Data Pipeline
Execution Script: End-to-end scraper -> cleaner -> DB loader -> query runner -> merge verifier
"""

import os
import sys
import logging
from scraper import (
    fetch_exchange_rate_with_fallback,
    scrape_all_categories,
    clean_book_data,
    FIXED_GBP_TO_INR_RATE
)
from db_loader import (
    load_data_to_db,
    run_sql_queries,
    verify_sql_vs_pandas_merge
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("\n--- Zepto Catalog Pipeline: Scraping & Loading ---")

    # 1. Currency Conversion Rate Check
    print("\n[1/5] Verifying currency conversion baseline...")
    rate = fetch_exchange_rate_with_fallback(FIXED_GBP_TO_INR_RATE)
    print(f"Using fixed conversion rate: 1 GBP = {rate} INR")

    # 2. Scraping Catalog Data
    print("\n[2/5] Scraping catalog categories from books.toscrape.com...")
    raw_books = scrape_all_categories()
    print(f"Total raw records scraped: {len(raw_books)}")

    # 3. Cleaning & Enrichment
    print("\n[3/5] Cleaning records & converting currency to INR...")
    df_cleaned = clean_book_data(raw_books, conversion_rate=rate)
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Categories captured: {df_cleaned['category'].unique().tolist()}")
    print("\nFirst 5 cleaned rows:")
    print(df_cleaned.head(5).to_string())

    # Check minimum requirements
    assert len(df_cleaned) >= 60, f"Expected at least 60 books, got {len(df_cleaned)}"
    assert df_cleaned['category'].nunique() >= 3, f"Expected at least 3 categories, got {df_cleaned['category'].nunique()}"

    # 4. Loading to SQLite Normalized Schema
    db_path = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")
    print(f"\n[4/5] Ingesting into SQLite database: {db_path}...")
    load_data_to_db(df_cleaned, db_path=db_path)
    print("Successfully populated categories and books tables.")

    # 5. Executing SQL Queries & Pandas Comparison
    print("\n[5/5] Executing SQL queries and verifying pandas merge equivalence...")
    query_results = run_sql_queries(db_path=db_path)
    df_sql, df_merge, is_equiv = verify_sql_vs_pandas_merge(db_path=db_path)
    print(f"Pandas merge vs SQL join match: {'PASS' if is_equiv else 'FAIL'}")

    print("\n--- Pipeline run complete: all 5 steps finished successfully ---\n")


if __name__ == "__main__":
    main()
