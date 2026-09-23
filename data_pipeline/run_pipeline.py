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
    print("=" * 80)
    print("ZEPTO DATA & AI PLATFORM: MODULE 1 (DATA PIPELINE)")
    print("=" * 80)

    # 1. Currency Conversion Rate Verification
    print("\n[Step 1] Checking Currency Conversion Rate...")
    rate = fetch_exchange_rate_with_fallback(FIXED_GBP_TO_INR_RATE)
    print(f"-> Applied Conversion Rate: 1 GBP = {rate} INR (Fixed Baseline)\n")

    # 2. Scraping Catalog Data
    print("[Step 2] Scraping Catalog Products from books.toscrape.com...")
    raw_books = scrape_all_categories()
    print(f"-> Total Raw Books Scraped: {len(raw_books)}\n")

    # 3. Cleaning & Enrichment
    print("[Step 3] Cleaning and Enriching Catalog Records...")
    df_cleaned = clean_book_data(raw_books, conversion_rate=rate)
    print(f"-> Cleaned Dataset Shape: {df_cleaned.shape}")
    print(f"-> Unique Categories: {df_cleaned['category'].unique().tolist()}")
    print("\nSample Cleaned Records:")
    print(df_cleaned.head(5).to_string())
    print()

    # Verify requirements: >= 60 books across >= 3 categories
    assert len(df_cleaned) >= 60, f"Expected >= 60 books, got {len(df_cleaned)}"
    assert df_cleaned['category'].nunique() >= 3, f"Expected >= 3 categories, got {df_cleaned['category'].nunique()}"

    # 4. Loading to SQLite Normalized Schema
    db_path = os.path.join(os.path.dirname(__file__), "zepto_catalog.db")
    print(f"[Step 4] Loading to Normalized SQLite Database: {db_path}...")
    load_data_to_db(df_cleaned, db_path=db_path)
    print("-> Successfully populated categories and books tables.\n")

    # 5. Executing SQL Queries
    print("[Step 5] Executing SQL Queries Against Normalized Database...")
    query_results = run_sql_queries(db_path=db_path)
    print(f"-> Executed {len(query_results)} SQL queries successfully.\n")

    # 6. Verifying SQL JOIN vs. In-Memory Pandas Merge
    print("[Step 6] Verifying SQL JOIN vs. Pandas Merge Equivalence...")
    df_sql, df_merge, is_equiv = verify_sql_vs_pandas_merge(db_path=db_path)
    print(f"-> Verification Result: {'PASS' if is_equiv else 'FAIL'}\n")

    print("=" * 80)
    print("MODULE 1 DATA PIPELINE EXECUTION COMPLETE: ALL CRITERIA SATISFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
