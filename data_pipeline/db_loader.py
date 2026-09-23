"""
Zepto Data & AI Platform - Module 1: Data Pipeline
Database Loader & SQL Query Engine:
Populates normalized SQLite database and runs analytics queries.
"""

import sqlite3
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CREATE_CATEGORIES_TABLE = """
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);
"""

CREATE_BOOKS_TABLE = """
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
"""


def init_database(db_path: str = "zepto_catalog.db") -> sqlite3.Connection:
    """Initializes normalized SQLite database with tables and foreign key constraints enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    cursor.execute(CREATE_CATEGORIES_TABLE)
    cursor.execute(CREATE_BOOKS_TABLE)
    conn.commit()
    logger.info(f"Initialized normalized schema in {db_path}.")
    return conn


def load_data_to_db(df: pd.DataFrame, db_path: str = "zepto_catalog.db") -> None:
    """Inserts cleaned catalog data into categories and books tables."""
    conn = init_database(db_path)
    cursor = conn.cursor()

    # Clear existing rows to allow idempotent regeneration
    cursor.execute("DELETE FROM books;")
    cursor.execute("DELETE FROM categories;")
    conn.commit()

    # 1. Populate categories table
    unique_categories = sorted(df["category"].unique())
    category_map: Dict[str, int] = {}

    for cat_name in unique_categories:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat_name,))
        cat_id = cursor.lastrowid
        category_map[cat_name] = cat_id

    conn.commit()
    logger.info(f"Inserted {len(category_map)} distinct categories.")

    # 2. Populate books table with foreign key reference
    book_records = []
    for _, row in df.iterrows():
        cat_id = category_map[row["category"]]
        book_records.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            cat_id,
        ))

    cursor.executemany(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        book_records,
    )
    conn.commit()
    logger.info(f"Inserted {len(book_records)} books referencing categories.")
    conn.close()


def get_required_queries() -> List[Dict[str, str]]:
    """Returns a list of SQL queries collectively covering required clauses and JOIN."""
    return [
        {
            "id": "query_1_select_where",
            "title": "Query 1: SELECT / WHERE (Filter high-rated books)",
            "sql": "SELECT title, price_inr, rating FROM books WHERE rating >= 4;",
            "description": "Selects all books with a rating of 4 or 5 stars."
        },
        {
            "id": "query_2_order_by_limit",
            "title": "Query 2: ORDER BY & LIMIT (Top 5 most expensive books in INR)",
            "sql": "SELECT title, price_inr, rating FROM books ORDER BY price_inr DESC LIMIT 5;",
            "description": "Returns the 5 highest-priced catalog books sorted descending."
        },
        {
            "id": "query_3_distinct",
            "title": "Query 3: DISTINCT (Unique rating levels in catalog)",
            "sql": "SELECT DISTINCT rating FROM books ORDER BY rating ASC;",
            "description": "Identifies all distinct star ratings present in catalog."
        },
        {
            "id": "query_4_between_in",
            "title": "Query 4: BETWEEN & IN (Mid-range books with rating in 3 or 5)",
            "sql": "SELECT title, price_inr, rating FROM books WHERE price_inr BETWEEN 2000 AND 4500 AND rating IN (3, 5);",
            "description": "Filters books with price between 2000 and 4500 INR and rating in (3, 5)."
        },
        {
            "id": "query_5_join",
            "title": "Query 5: JOIN (Books with their Category Name)",
            "sql": """
            SELECT 
                b.book_id,
                b.title,
                c.category_name,
                b.price_gbp,
                b.price_inr,
                b.rating,
                b.in_stock
            FROM books b
            INNER JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.book_id ASC;
            """,
            "description": "Joins books and categories on foreign key category_id to enrich records."
        },
        {
            "id": "query_6_category_aggregates",
            "title": "Query 6: JOIN with GROUP BY / Aggregate (Catalog benchmarks)",
            "sql": """
            SELECT 
                c.category_name,
                COUNT(b.book_id) AS total_books,
                ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
                ROUND(AVG(b.rating), 2) AS avg_rating
            FROM categories c
            LEFT JOIN books b ON c.category_id = b.category_id
            GROUP BY c.category_name
            ORDER BY avg_price_inr DESC;
            """,
            "description": "Demonstrates relational aggregation per category."
        }
    ]


def run_sql_queries(db_path: str = "zepto_catalog.db") -> Dict[str, pd.DataFrame]:
    """Executes all catalog queries and prints formatted results."""
    conn = sqlite3.connect(db_path)
    results = {}
    queries = get_required_queries()

    for q in queries:
        print("=" * 80)
        print(f"{q['title']}")
        print(f"Description: {q['description']}")
        print(f"SQL:\n{q['sql'].strip()}\n")
        df_res = pd.read_sql_query(q["sql"], conn)
        results[q["id"]] = df_res
        print(f"Output ({len(df_res)} rows):")
        print(df_res.head(10).to_string(index=False))
        if len(df_res) > 10:
            print(f"... and {len(df_res) - 10} more rows.")
        print()

    conn.close()
    return results


def verify_sql_vs_pandas_merge(db_path: str = "zepto_catalog.db") -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Reads categories and books into pandas DataFrames via pd.read_sql,
    reproduces Query 5's join using in-memory pd.merge (no SQL),
    and validates that both produce strictly equivalent output.
    """
    conn = sqlite3.connect(db_path)

    # 1. SQL Join approach
    join_sql = """
    SELECT 
        b.book_id,
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock
    FROM books b
    INNER JOIN categories c ON b.category_id = c.category_id
    ORDER BY b.book_id ASC;
    """
    df_sql = pd.read_sql_query(join_sql, conn)

    # 2. In-memory pd.merge approach
    df_books = pd.read_sql_query("SELECT * FROM books;", conn)
    df_categories = pd.read_sql_query("SELECT * FROM categories;", conn)
    conn.close()

    df_merged = pd.merge(
        df_books,
        df_categories,
        on="category_id",
        how="inner"
    )

    # Select and order matching columns
    cols = ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]
    df_merged = df_merged[cols].sort_values("book_id").reset_index(drop=True)
    df_sql = df_sql.reset_index(drop=True)

    # Validate equivalence
    is_equivalent = df_sql.equals(df_merged)

    print("=" * 80)
    print("SIDE-BY-SIDE EQUIVALENCE: pd.read_sql (SQL JOIN) vs. pd.merge (IN-MEMORY PANDAS)")
    print("=" * 80)
    print("\n--- pd.read_sql (First 5 Rows) ---")
    print(df_sql.head(5).to_string(index=False))
    print("\n--- pd.merge (First 5 Rows) ---")
    print(df_merged.head(5).to_string(index=False))
    print(f"\nExact Equivalence Verified: {is_equivalent}")
    assert is_equivalent, "DataFrames from SQL JOIN and pd.merge are not identical!"

    return df_sql, df_merged, is_equivalent
