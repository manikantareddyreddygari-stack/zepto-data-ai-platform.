# Module 1: Data Pipeline (`/data_pipeline`)

## Overview
Zepto's competitive catalog analytics workflow requires reliable extraction of external pricing and product availability data before downstream ingestion into analytical warehouses. This module implements a robust, end-to-end data pipeline that:
1. Scrapes live catalog data across multiple product categories from [books.toscrape.com](http://books.toscrape.com/).
2. Cleans raw strings into typed representations (`price_gbp`, `rating`, `in_stock`).
3. Enriches prices with the project baseline conversion rate ($1\text{ GBP} = 105.50\text{ INR}$).
4. Ingests cleaned data into a normalized SQLite relational database (`zepto_catalog.db`) with primary/foreign key integrity.
5. Executes analytical SQL queries and demonstrates mathematical equivalence between relational SQL `JOIN` and in-memory `pd.merge`.

---

## Currency Conversion Baseline
- **Required Fixed Baseline**: $1\text{ GBP} = 105.50\text{ INR}$.
- This is a project-defined fixed constant without date reference or live external API dependency, guaranteeing deterministic and reproducible evaluation.
- *Optional keyless stretch*: An HTTP check against `open.er-api.com` is implemented with explicit status-code validation; upon any failure or discrepancy, it strictly falls back to the fixed $105.50$ baseline.

---

## Data Cleaning & Parsing Decisions
| Field | Raw Source | Cleaned Type | Parsing Logic & Handling |
| :--- | :--- | :--- | :--- |
| **Title** | HTML `<h3><a>` title / text | `TEXT` | Extracted and trimmed. If title is completely absent, the row is dropped as a malformed catalog entity. |
| **Category** | Page navigation hierarchy | `TEXT` | Extracted from category listing URLs (e.g. Mystery, Historical Fiction, Travel, Classics, Philosophy). |
| **Price (GBP)** | `£51.77` | `REAL` (`float`) | Stripped currency symbol (`£`) using regex `[^\d.]`. If a value fails numeric conversion, **median imputation** is applied to preserve row volume without skewing the distribution. |
| **Rating** | Text class: `One` ... `Five` | `INTEGER` ($1$–$5$) | Mapped via explicit lookup table: `{"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}`. |
| **Availability**| `In stock` | `INTEGER` (Boolean $0$/$1$) | Parsed as `1` if `"in stock"` is present in the lowercased string, else `0`. |
| **Price (INR)** | Derived | `REAL` (`float`) | Computed as $\text{round}(\text{price\_gbp} \times 105.50, 2)$. |

---

## Relational Schema Design
Normalized into two tables to eliminate category redundancy and maintain referential integrity:

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (category_id)
);
```

---

## Executed SQL Queries & Demonstrated Clauses

The pipeline executes 6 SQL queries covering all assignment requirements:

1. **`SELECT / WHERE`** (`query_1_select_where`):
   ```sql
   SELECT title, price_inr, rating FROM books WHERE rating >= 4;
   ```
   *Filters catalog items that meet Zepto's premium 4-star and 5-star quality bar.*

2. **`ORDER BY & LIMIT`** (`query_2_order_by_limit`):
   ```sql
   SELECT title, price_inr, rating FROM books ORDER BY price_inr DESC LIMIT 5;
   ```
   *Returns the top 5 highest-priced catalog offerings.*

3. **`DISTINCT`** (`query_3_distinct`):
   ```sql
   SELECT DISTINCT rating FROM books ORDER BY rating ASC;
   ```
   *Retrieves all distinct rating scores present across catalog inventory.*

4. **`BETWEEN & IN`** (`query_4_between_in`):
   ```sql
   SELECT title, price_inr, rating FROM books 
   WHERE price_inr BETWEEN 2000 AND 4500 AND rating IN (3, 5);
   ```
   *Selects mid-tier priced products having specific ratings.*

5. **`JOIN`** (`query_5_join`):
   ```sql
   SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
   FROM books b
   INNER JOIN categories c ON b.category_id = c.category_id
   ORDER BY b.book_id ASC;
   ```
   *Denormalizes catalog items with their human-readable category name.*

6. **Relational Category Aggregation** (`query_6_category_aggregates`):
   ```sql
   SELECT c.category_name, COUNT(b.book_id) AS total_books,
          ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
          ROUND(AVG(b.rating), 2) AS avg_rating
   FROM categories c
   LEFT JOIN books b ON c.category_id = b.category_id
   GROUP BY c.category_name
   ORDER BY avg_price_inr DESC;
   ```
   *Computes operational category benchmarks.*

---

## Equivalence Verification: `pd.read_sql` vs. `pd.merge`
To ensure data fidelity between warehouse SQL queries and Python data frames:
1. The join query is executed via `pd.read_sql_query(join_sql, conn)`.
2. The raw tables `books` and `categories` are read independently and merged in-memory using `pd.merge(df_books, df_categories, on="category_id", how="inner")`.
3. Columns are aligned, sorted by `book_id`, and verified via `df_sql.equals(df_merged)`.
4. Result: **Strict mathematical equivalence confirmed (`True`)**.

---

## How to Run

### Command Line
```bash
python data_pipeline/run_pipeline.py
```

### Interactive Notebook
Open and run all cells in:
`data_pipeline/data_pipeline.ipynb`
