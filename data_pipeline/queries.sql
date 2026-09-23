-- Zepto Data & AI Platform - Module 1: SQL Analytics Queries
-- Database: zepto_catalog.db

-- 1. SELECT / WHERE: High-rated catalog items
SELECT title, price_inr, rating 
FROM books 
WHERE rating >= 4;

-- 2. ORDER BY & LIMIT: Top 5 most premium catalog items
SELECT title, price_inr, rating 
FROM books 
ORDER BY price_inr DESC 
LIMIT 5;

-- 3. DISTINCT: Unique rating levels across the catalog
SELECT DISTINCT rating 
FROM books 
ORDER BY rating ASC;

-- 4. BETWEEN & IN: Mid-range products with rating in (3, 5)
SELECT title, price_inr, rating 
FROM books 
WHERE price_inr BETWEEN 2000 AND 4500 
  AND rating IN (3, 5);

-- 5. JOIN: Enrich books with normalized category names
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

-- 6. Relational Aggregation per category
SELECT 
    c.category_name,
    COUNT(b.book_id) AS total_books,
    ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
    ROUND(AVG(b.rating), 2) AS avg_rating
FROM categories c
LEFT JOIN books b ON c.category_id = b.category_id
GROUP BY c.category_name
ORDER BY avg_price_inr DESC;
