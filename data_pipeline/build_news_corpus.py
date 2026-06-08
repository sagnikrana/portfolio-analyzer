"""Refresh the news layer the diagnosis engine reads, via the GDELT DOC 2.0 API.

The app surfaces "narrative evidence" from two files:

  data/processed/structured/news_metadata.csv
      url, url_mobile, title, seendate, socialimage, domain, language,
      sourcecountry, ticker
  data/processed/unstructured/document_corpus.csv
      document_id, source_type, ticker, domain, title, document_date, url,
      body_text, metadata_json

GDELT's DOC 2.0 ArtList endpoint returns exactly the news_metadata fields, so we
query it per ticker (company name) for recent articles and rebuild both files.

Scope & honesty:
  - This refreshes news *metadata + titles*, which is what the diagnosis surface
    actually displays (NarrativeEvidence uses titles/dates). Full article-body
    scraping and the Chroma vector re-embedding from the original notebook are
    intentionally NOT reproduced here (heavy + fragile); body_text is set to the
    title so the corpus stays consistent.
  - SEC-filing rows already in document_corpus.csv are preserved; only the
    news_article rows are replaced.
  - GDELT rate-limits aggressively (HTTP 429), so requests are spaced out and
    retried with backoff. If everything fails, existing files are left untouched.

Run:  python data_pipeline/build_news_corpus.py [--max-tickers 40] [--tickers AAPL MSFT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
STRUCTURED_DIR = ROOT / "data" / "processed" / "structured"
UNSTRUCTURED_DIR = ROOT / "data" / "processed" / "unstructured"
NEWS_METADATA_PATH = STRUCTURED_DIR / "news_metadata.csv"
DOCUMENT_CORPUS_PATH = UNSTRUCTURED_DIR / "document_corpus.csv"
UNIVERSE_PATH = RAW_DIR / "buy_candidate_universe.csv"

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-analyzer news refresh)"}
REQUEST_SPACING_SECONDS = 6.0  # GDELT throttles ~1 req / 5s
ARTICLES_PER_TICKER = 8
NEWS_TIMESPAN = "21d"

NEWS_METADATA_COLUMNS = [
    "url", "url_mobile", "title", "seendate", "socialimage",
    "domain", "language", "sourcecountry", "ticker",
]
CORPUS_COLUMNS = [
    "document_id", "source_type", "ticker", "domain", "title",
    "document_date", "url", "body_text", "metadata_json",
]


def _ticker_names() -> dict[str, str]:
    """ticker -> company name, from the buy universe (for cleaner GDELT queries)."""
    if not UNIVERSE_PATH.exists():
        return {}
    df = pd.read_csv(UNIVERSE_PATH)
    if "ticker" not in df.columns or "security_name" not in df.columns:
        return {}
    return {
        str(t).strip().upper(): str(n).strip()
        for t, n in zip(df["ticker"], df["security_name"])
        if str(t).strip()
    }


def _holdings_tickers() -> list[str]:
    """Distinct holdings tickers from the most recent activity CSV in data/raw."""
    candidates = sorted(
        RAW_DIR.glob("robinhood_activity_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) or [RAW_DIR / "fake_mantis_invest.csv"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            continue
        col = next((c for c in ("Instrument", "instrument", "ticker", "Ticker") if c in df.columns), None)
        if not col:
            continue
        tickers = sorted({str(t).strip().upper() for t in df[col].dropna() if str(t).strip()})
        if tickers:
            return tickers
    return []


def _get_articles(query: str) -> list[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": ARTICLES_PER_TICKER,
        "timespan": NEWS_TIMESPAN,
        "sort": "datedesc",
    }
    for attempt in range(4):
        try:
            resp = requests.get(GDELT_URL, params=params, headers=REQUEST_HEADERS, timeout=45)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json().get("articles", []) or []
        except Exception:
            time.sleep(5 * (attempt + 1))
    return []


def fetch_news(tickers: list[str], names: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        name = names.get(ticker, ticker)
        # Quote the company name; add the ticker to disambiguate common words.
        query = f'"{name}" sourcelang:english' if name and name != ticker else f"{ticker} stock sourcelang:english"
        articles = _get_articles(query)
        for art in articles:
            rows.append({
                "url": art.get("url", ""),
                "url_mobile": art.get("url_mobile", ""),
                "title": (art.get("title") or "").strip(),
                "seendate": art.get("seendate", ""),
                "socialimage": art.get("socialimage", ""),
                "domain": art.get("domain", ""),
                "language": art.get("language", ""),
                "sourcecountry": art.get("sourcecountry", ""),
                "ticker": ticker,
            })
        print(f"  [{i + 1}/{len(tickers)}] {ticker}: {len(articles)} articles")
        if i < len(tickers) - 1:
            time.sleep(REQUEST_SPACING_SECONDS)
    return pd.DataFrame(rows, columns=NEWS_METADATA_COLUMNS)


def _seendate_to_date(seendate: str) -> str:
    s = str(seendate)
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def news_to_corpus(news: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for art in news.itertuples(index=False):
        url = art.url or ""
        doc_hash = hashlib.sha1(f"{art.ticker}|{url}".encode()).hexdigest()[:10]
        title = (art.title or "").strip()
        rows.append({
            "document_id": f"news::{art.ticker}::{doc_hash}",
            "source_type": "news_article",
            "ticker": art.ticker,
            "domain": art.domain,
            "title": title,
            "document_date": _seendate_to_date(art.seendate),
            "url": url,
            # GDELT does not provide article bodies; title is what the app surfaces.
            "body_text": title,
            "metadata_json": json.dumps({
                "ticker": art.ticker, "source_type": "news_article",
                "domain": art.domain, "seendate": art.seendate, "url": url,
            }),
        })
    return pd.DataFrame(rows, columns=CORPUS_COLUMNS)


def build_news_corpus(tickers: list[str] | None = None, max_tickers: int = 40) -> dict:
    names = _ticker_names()
    if not tickers:
        tickers = _holdings_tickers()
    tickers = [t for t in tickers if t][:max_tickers]
    if not tickers:
        raise RuntimeError("No tickers to query; aborting.")
    print(f"Querying GDELT for {len(tickers)} tickers (~{REQUEST_SPACING_SECONDS:.0f}s apart) ...")

    news = fetch_news(tickers, names)
    if news.empty:
        print("GDELT returned nothing (rate-limited or down); leaving existing files untouched.")
        return {"news_rows": 0, "corpus_news_rows": 0, "skipped": True}

    # Carry forward prior articles for tickers that returned nothing this run
    # (GDELT throttling shouldn't silently shrink coverage to zero for a ticker).
    fetched_tickers = set(news["ticker"].unique())
    if NEWS_METADATA_PATH.exists():
        try:
            prior = pd.read_csv(NEWS_METADATA_PATH)
            requested = set(tickers)
            carry = prior[
                prior["ticker"].isin(requested) & ~prior["ticker"].isin(fetched_tickers)
            ]
            if not carry.empty:
                print(f"  carried forward prior news for {carry['ticker'].nunique()} throttled ticker(s).")
                news = pd.concat([news, carry[NEWS_METADATA_COLUMNS]], ignore_index=True)
        except Exception:
            pass

    # news_metadata.csv: fresh pull (+ carried-forward rows for throttled tickers).
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    news = news.drop_duplicates(subset=["ticker", "url"]).reset_index(drop=True)
    news.to_csv(NEWS_METADATA_PATH, index=False)

    # document_corpus.csv: preserve non-news rows (e.g. sec_filing), replace news.
    fresh_corpus = news_to_corpus(news)
    if DOCUMENT_CORPUS_PATH.exists():
        existing = pd.read_csv(DOCUMENT_CORPUS_PATH)
        preserved = existing[existing.get("source_type") != "news_article"]
        combined = pd.concat([preserved, fresh_corpus], ignore_index=True)
    else:
        combined = fresh_corpus
    UNSTRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    combined[CORPUS_COLUMNS].to_csv(DOCUMENT_CORPUS_PATH, index=False)

    return {
        "news_rows": len(news),
        "corpus_news_rows": len(fresh_corpus),
        "tickers": len(tickers),
        "skipped": False,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refresh news_metadata + document_corpus from GDELT.")
    ap.add_argument("--tickers", nargs="*", default=None, help="Explicit tickers (default: holdings).")
    ap.add_argument("--max-tickers", type=int, default=40, help="Cap to bound runtime/rate limits.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = build_news_corpus(tickers=args.tickers, max_tickers=args.max_tickers)
    if summary["skipped"]:
        print("No update written.")
    else:
        print(f"Wrote {summary['news_rows']} news rows ({summary['tickers']} tickers) -> {NEWS_METADATA_PATH.name}")
        print(f"Updated corpus with {summary['corpus_news_rows']} news rows -> {DOCUMENT_CORPUS_PATH.name}")
