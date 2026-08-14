import os
import re
import sqlite3
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
DB_PATH = os.path.join(os.path.dirname(__file__), "books_catalog.db")
QUERY_LOG_PATH = os.path.join(os.path.dirname(__file__), "sql_queries_and_results.txt")
GBP_TO_INR = 105.50
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def fetch_html(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {url}")
    return response.text


def get_category_links() -> list[tuple[str, str]]:
    soup = BeautifulSoup(fetch_html(BASE_URL), "html.parser")
    seen = set()
    links = []
    for anchor in soup.select("ul.nav li a"):
        label = anchor.get_text(" ", strip=True)
        href = anchor.get("href")
        if not href or not label or label.lower() in {"books", "all products"}:
            continue
        normalized_label = label.strip()
        if normalized_label.lower() in seen:
            continue
        seen.add(normalized_label.lower())
        links.append((normalized_label, urljoin(BASE_URL, href)))
    return links[:5]


def parse_price(raw_price: str) -> float | None:
    if raw_price is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", raw_price)
    if cleaned in {"", ".", "-", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_rating(raw_rating: str) -> int | None:
    if raw_rating is None:
        return None
    rating_text = raw_rating.strip()
    return RATING_MAP.get(rating_text, None)


def parse_availability(raw_availability: str) -> bool:
    if raw_availability is None:
        return False
    return "in stock" in raw_availability.lower()


def scrape_category(category_name: str, category_url: str) -> list[dict]:
    all_books = []
    seen_pages = set()
    next_url = category_url

    while next_url and len(all_books) < 150:
        if next_url in seen_pages:
            break
        seen_pages.add(next_url)
        soup = BeautifulSoup(fetch_html(next_url), "html.parser")
        for card in soup.select("article.product_pod"):
            title_tag = card.select_one("h3 > a")
            if title_tag is None:
                continue
            title = title_tag.get("title") or title_tag.get_text(" ", strip=True)
            price_elem = card.select_one("p.price_color")
            rating_elem = card.select_one("p.star-rating")
            availability_elem = card.select_one("p.instock.availability")

            price_text = price_elem.get_text(" ", strip=True) if price_elem else None
            rating_text = rating_elem.get("class", [])[-1] if rating_elem else None
            availability_text = availability_elem.get_text(" ", strip=True) if availability_elem else None

            record = {
                "title": title,
                "price_gbp_raw": price_text,
                "rating_raw": rating_text,
                "availability_raw": availability_text,
                "category": category_name,
            }
            all_books.append(record)

        next_link = soup.select_one("li.next > a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None

    return all_books


def robust_clean(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["price_gbp"] = df["price_gbp_raw"].apply(parse_price)
    df["rating"] = df["rating_raw"].apply(parse_rating)
    df["availability_text"] = df["availability_raw"].fillna("")
    df["in_stock"] = df["availability_text"].apply(parse_availability).astype(bool)

    if not df["price_gbp"].dropna().empty:
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
    else:
        df["price_gbp"] = 0.0

    if not df["rating"].dropna().empty:
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating).round().astype(int)
    else:
        df["rating"] = 3

    df["rating"] = df["rating"].clip(lower=1, upper=5)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    df["category"] = df["category"].fillna("Unknown")
    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]


def create_database_and_load(df: pd.DataFrame, category_names: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DROP TABLE IF EXISTS books")
    conn.execute("DROP TABLE IF EXISTS categories")

    conn.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
        """
    )

    category_map = {name: idx + 1 for idx, name in enumerate(category_names)}
    for category_name, category_id in category_map.items():
        conn.execute(
            "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
            (category_id, category_name),
        )

    for _, row in df.iterrows():
        category_id = category_map.get(row["category"], 1)
        conn.execute(
            """
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                int(row["rating"]),
                int(row["in_stock"]),
                category_id,
            ),
        )

    conn.commit()
    return conn


def build_queries() -> dict[str, str]:
    return {
        "select_where": """
            SELECT title, rating, price_inr
            FROM books
            WHERE in_stock = 1 AND rating >= 4
            ORDER BY price_inr DESC
        """,
        "order_limit": """
            SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 10
        """,
        "distinct": """
            SELECT DISTINCT category_id
            FROM books
            WHERE rating >= 3
            ORDER BY category_id
        """,
        "between": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 20.00 AND 35.00
            ORDER BY price_gbp DESC
        """,
        "in_clause": """
            SELECT b.title, c.category_name, b.price_inr
            FROM books b
            JOIN categories c ON c.category_id = b.category_id
            WHERE c.category_name IN ('Travel', 'Mystery', 'Historical Fiction')
            ORDER BY b.price_inr DESC
            LIMIT 10
        """,
        "join_top_by_category": """
            SELECT c.category_name, b.title, b.rating, b.price_inr
            FROM books b
            JOIN categories c ON c.category_id = b.category_id
            WHERE b.book_id IN (
                SELECT b2.book_id
                FROM books b2
                WHERE b2.category_id = b.category_id
                ORDER BY b2.rating DESC, b2.price_inr DESC
                LIMIT 1
            )
            ORDER BY c.category_name, b.rating DESC, b.price_inr DESC
        """,
    }


def save_query_log(query_results: dict[str, pd.DataFrame]) -> None:
    lines = []
    for name, df in query_results.items():
        lines.append(f"Query: {name}")
        lines.append(df.to_string(index=False))
        lines.append("")
        lines.append("-" * 80)
        lines.append("")
    with open(QUERY_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    category_links = get_category_links()
    if len(category_links) < 3:
        raise RuntimeError("Fewer than three categories found on the source site.")

    raw_rows = []
    selected_categories = []
    for category_name, category_url in category_links[:3]:
        selected_categories.append(category_name)
        raw_rows.extend(scrape_category(category_name, category_url))

    cleaned_df = robust_clean(raw_rows)
    if len(cleaned_df) < 60:
        raise RuntimeError(f"Only {len(cleaned_df)} rows collected. Need at least 60 books across three categories.")

    conn = create_database_and_load(cleaned_df, selected_categories)
    queries = build_queries()
    query_results = {}
    for name, sql in queries.items():
        df = pd.read_sql_query(sql, conn)
        query_results[name] = df

    save_query_log(query_results)

    sql_top_books = pd.read_sql_query(queries["order_limit"], conn)
    sql_distinct = pd.read_sql_query(queries["distinct"], conn)
    books_df = pd.read_sql_query("SELECT * FROM books", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories", conn)

    merged_join = books_df.merge(categories_df, on="category_id", how="left")
    merged_join = (
        merged_join.sort_values(["category_name", "rating", "price_inr"], ascending=[True, False, False])
        .groupby("category_name", as_index=False)
        .head(1)
        .loc[:, ["category_name", "title", "rating", "price_inr"]]
        .sort_values(["category_name", "rating", "price_inr"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    sql_join = pd.read_sql_query(queries["join_top_by_category"], conn).reset_index(drop=True)
    pd.testing.assert_frame_equal(sql_join, merged_join)

    print("Scraped rows:", len(cleaned_df))
    print("Distinct categories:", cleaned_df["category"].nunique())
    print("\nTop 10 books by INR:\n", sql_top_books.to_string(index=False))
    print("\nDistinct categories with rating >= 3:\n", sql_distinct.to_string(index=False))
    comparison = pd.concat(
        [
            sql_join.rename(columns={"category_name": "category_name_sql", "title": "title_sql", "rating": "rating_sql", "price_inr": "price_inr_sql"}),
            merged_join.rename(columns={"category_name": "category_name_pd", "title": "title_pd", "rating": "rating_pd", "price_inr": "price_inr_pd"}),
        ],
        axis=1,
    )
    print("\nSQL vs pandas join comparison (side by side):")
    print(comparison.to_string(index=False))
    print("\nSQL join result matches pandas merge output.")
    print("\nQuery log saved to:", QUERY_LOG_PATH)

    conn.close()


if __name__ == "__main__":
    main()
