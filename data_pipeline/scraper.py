"""
Zepto Data & AI Platform - Module 1: Data Pipeline
Catalog Scraper: Scrapes book products from books.toscrape.com
"""

import re
import logging
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://books.toscrape.com"
FIXED_GBP_TO_INR_RATE = 105.50  # Required project fixed baseline conversion constant

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

CATEGORIES_TO_SCRAPE = [
    {"name": "Mystery", "url": f"{BASE_URL}/catalogue/category/books/mystery_3/index.html"},
    {"name": "Historical Fiction", "url": f"{BASE_URL}/catalogue/category/books/historical-fiction_4/index.html"},
    {"name": "Travel", "url": f"{BASE_URL}/catalogue/category/books/travel_2/index.html"},
    {"name": "Classics", "url": f"{BASE_URL}/catalogue/category/books/classics_6/index.html"},
    {"name": "Philosophy", "url": f"{BASE_URL}/catalogue/category/books/philosophy_7/index.html"},
]


def fetch_exchange_rate_with_fallback(fallback_rate: float = FIXED_GBP_TO_INR_RATE) -> float:
    """
    Optional stretch: Attempt keyless API lookup with explicit status code handling,
    falling back to the required fixed baseline rate (105.50) upon any failure.
    """
    api_url = "https://open.er-api.com/v6/latest/GBP"
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            rate = data.get("rates", {}).get("INR")
            if rate is not None:
                logger.info(f"Retrieved live rate from API: 1 GBP = {rate} INR (optional check).")
                # We log the live rate for demonstration, but project requirements mandate fixed rate:
                logger.info(f"Using required fixed project baseline: 1 GBP = {fallback_rate} INR.")
                return fallback_rate
        else:
            logger.warning(f"Exchange API returned HTTP {response.status_code}. Using fallback {fallback_rate}.")
    except Exception as e:
        logger.warning(f"Exchange rate lookup failed ({e}). Using fixed fallback baseline {fallback_rate}.")
    return fallback_rate


def scrape_category_page(category_name: str, page_url: str) -> List[Dict[str, Any]]:
    """Scrapes book records from a specific category page."""
    books = []
    current_url: Optional[str] = page_url

    while current_url:
        logger.info(f"Scraping category '{category_name}' from: {current_url}")
        resp = requests.get(current_url, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Failed to fetch {current_url}: status {resp.status_code}")
            break

        soup = BeautifulSoup(resp.content, "html.parser")
        product_pods = soup.select("article.product_pod")

        for pod in product_pods:
            # Title
            title_tag = pod.select_one("h3 a")
            title = title_tag["title"] if title_tag and title_tag.has_attr("title") else (title_tag.text.strip() if title_tag else "Unknown Title")

            # Price text
            price_tag = pod.select_one("p.price_color")
            price_text = price_tag.text.strip() if price_tag else ""

            # Rating
            rating_tag = pod.select_one("p.star-rating")
            rating_classes = rating_tag.get("class", []) if rating_tag else []
            rating_str = next((c for c in rating_classes if c != "star-rating"), "One")

            # Availability
            avail_tag = pod.select_one("p.instock.availability")
            avail_text = avail_tag.text.strip() if avail_tag else ""

            books.append({
                "title": title,
                "price_raw": price_text,
                "star_rating_raw": rating_str,
                "availability_raw": avail_text,
                "category": category_name,
            })

        # Check for pagination (next button)
        next_button = soup.select_one("li.next a")
        if next_button:
            next_page = next_button["href"]
            current_url = current_url.rsplit("/", 1)[0] + "/" + next_page
        else:
            current_url = None

    return books


def scrape_all_categories() -> List[Dict[str, Any]]:
    """Scrapes books across all configured categories to ensure >= 60 items across >= 3 categories."""
    all_books = []
    for cat in CATEGORIES_TO_SCRAPE:
        cat_books = scrape_category_page(cat["name"], cat["url"])
        logger.info(f"Scraped {len(cat_books)} books from '{cat['name']}'.")
        all_books.extend(cat_books)
    logger.info(f"Total raw books scraped across {len(CATEGORIES_TO_SCRAPE)} categories: {len(all_books)}")
    return all_books


def clean_book_data(raw_books: List[Dict[str, Any]], conversion_rate: float = FIXED_GBP_TO_INR_RATE) -> pd.DataFrame:
    """
    Cleans raw scraped book data:
    1. Strips currency symbol from price and converts to float price_gbp.
    2. Converts star_rating text into integer rating (1-5).
    3. Parses availability text into boolean in_stock (1 or 0).
    4. Handles messy/missing values: drops corrupt records without title, applies median imputation for price if corrupted.
    5. Enriches with price_inr = price_gbp * conversion_rate (105.50 INR/GBP).
    """
    records = []
    for row in raw_books:
        title = row.get("title", "").strip()
        if not title:
            # Drop records with invalid or missing titles
            continue

        category = row.get("category", "General").strip()

        # Clean price
        price_raw = row.get("price_raw", "")
        # Remove non-numeric characters except decimal point
        cleaned_price = re.sub(r"[^\d.]", "", price_raw)
        try:
            price_gbp = float(cleaned_price) if cleaned_price else None
        except ValueError:
            price_gbp = None

        # Clean rating
        rating_text = row.get("star_rating_raw", "One")
        rating = RATING_MAP.get(rating_text, 1)

        # Clean availability
        avail_raw = row.get("availability_raw", "").lower()
        in_stock = 1 if "in stock" in avail_raw else 0

        records.append({
            "title": title,
            "category": category,
            "price_gbp": price_gbp,
            "rating": rating,
            "in_stock": in_stock,
        })

    df = pd.DataFrame(records)

    # Missing value handling:
    # If any price_gbp is missing/null, use median imputation
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        logger.warning(f"Imputing missing price_gbp with median value: {median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # Compute required fixed price_inr column
    df["price_inr"] = (df["price_gbp"] * conversion_rate).round(2)

    logger.info(f"Cleaned dataset ready: {len(df)} rows across {df['category'].nunique()} categories.")
    return df
