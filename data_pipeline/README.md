# Module 1: Data Pipeline (`/data_pipeline`)

**Author**: Manikanta Reddy Reddygari  
**Target Site**: [books.toscrape.com](http://books.toscrape.com/)  
**Database**: `zepto_catalog.db` (SQLite)

---

## Overview

In this module, I built an automated catalog scraping and data-loading pipeline that mirrors how competitive intelligence and product benchmarking data is ingested before dashboarding:
1. Scrapes book listings across 5 distinct categories from `books.toscrape.com`.
2. Cleans raw strings into proper data types (`price_gbp`, `rating`, `in_stock`).
3. Enriches prices using the fixed baseline currency exchange rate ($1\text{ GBP} = 105.50\text{ INR}$).
4. Inserts the cleaned data into a normalized SQLite relational schema (`categories` and `books` tables with a foreign key).
5. Executes 6 SQL analytical queries and proves that `pd.read_sql` and in-memory `pd.merge` yield equivalent results.

---

## Currency Conversion Rate

- **Required Baseline Rate**: **$1\text{ GBP} = 105.50\text{ INR}$**
- Per the assignment instructions, this is a fixed, project-defined constant with no date reference or live API requirement.
- *Stretch note*: I also wrote an optional helper function in `scraper.py` that tests an HTTP status code check against `open.er-api.com` and gracefully falls back to $105.50$ on any error or discrepancy.

---

## Data Cleaning & Parsing Choices

| Field | Raw Format | Cleaned Type | How I Handled It |
| :--- | :--- | :--- | :--- |
| **Title** | `<h3><a>` title attribute | `TEXT` | Extracted the full title text and stripped whitespace. Dropped records if title was empty. |
| **Category** | Breadcrumb category path | `TEXT` | Extracted from the category URL (Mystery, Historical Fiction, Travel, Classics, Philosophy). |
| **Price (GBP)** | `£51.77` | `REAL` (`float`) | Stripped `£` and non-numeric characters using regex. Used **median imputation** as a safeguard if any price failed numeric parsing. |
| **Rating** | Text class (`"One"` ... `"Five"`) | `INTEGER` ($1$–$5$) | Mapped using a Python dictionary: `{"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}`. |
| **Availability**| `"In stock (22 available)"` | `INTEGER` (Boolean $0$/$1$) | Parsed as `1` if `"in stock"` is in the string, else `0`. |
| **Price (INR)** | Calculated | `REAL` (`float`) | Computed as $\text{round}(\text{price\_gbp} \times 105.50, 2)$. |

---

## Relational Schema Design

I normalized the catalog into two tables to eliminate redundant category strings on book rows:

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

## Executed SQL Queries

The pipeline runs 6 SQL queries covering all assignment requirements:

1. **`SELECT / WHERE`**:
   ```sql
   SELECT title, price_inr, rating FROM books WHERE rating >= 4;
   ```
   *Filters catalog books with ratings of 4 or 5 stars.*

2. **`ORDER BY & LIMIT`**:
   ```sql
   SELECT title, price_inr, rating FROM books ORDER BY price_inr DESC LIMIT 5;
   ```
   *Finds the top 5 most expensive catalog items in INR.*

3. **`DISTINCT`**:
   ```sql
   SELECT DISTINCT rating FROM books ORDER BY rating ASC;
   ```
   *Retrieves all unique rating levels present in the catalog.*

4. **`BETWEEN & IN`**:
   ```sql
   SELECT title, price_inr, rating FROM books 
   WHERE price_inr BETWEEN 2000 AND 4500 AND rating IN (3, 5);
   ```
   *Selects mid-tier priced products with 3 or 5 star ratings.*

5. **`JOIN`**:
   ```sql
   SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
   FROM books b
   INNER JOIN categories c ON b.category_id = c.category_id
   ORDER BY b.book_id ASC;
   ```
   *Enriches each book row with its human-readable category name.*

6. **Relational Category Aggregation**:
   ```sql
   SELECT c.category_name, COUNT(b.book_id) AS total_books,
          ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
          ROUND(AVG(b.rating), 2) AS avg_rating
   FROM categories c
   LEFT JOIN books b ON c.category_id = b.category_id
   GROUP BY c.category_name
   ORDER BY avg_price_inr DESC;
   ```
   *Computes catalog counts, average pricing, and ratings per category.*

---

## SQL vs. Pandas Merge Verification

To confirm data consistency:
1. I executed the SQL join query using `pd.read_sql_query(join_sql, conn)`.
2. I loaded the raw `books` and `categories` tables into memory and merged them using `pd.merge(df_books, df_categories, on="category_id", how="inner")`.
3. I checked `df_sql.equals(df_merged)`.
4. Result: **`True` — outputs are identical**.

---

## How to Run

```bash
# Run script start to finish
python data_pipeline/run_pipeline.py

# Or explore via notebook
jupyter notebook data_pipeline/data_pipeline.ipynb
```
