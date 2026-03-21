"""
agents/tools/news_tools.py
---------------------------
LangChain tool wrappers around NewsResearcher.

Single public function:
    make_news_tools(researcher: NewsResearcher) -> list[BaseTool]

Returns four tools:
    fetch_breaking_news   — recent high-impact news for a list of tickers
    fetch_macro_news      — macro-economic and market-wide news
    fetch_catalyst_news   — catalyst events for a specific ticker
    fetch_daily_digest    — structured daily market digest (NewsDigest model)

Design invariants:
- Lists of NewsItem are serialised as indented JSON arrays.
- NewsDigest is serialised as indented JSON.
- Tickers and sectors are accepted as comma-separated strings for
  reliable LangChain argument parsing.
- All tools catch exceptions and return "ERROR: …" strings so the
  orchestrator can surface them gracefully.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from webai import NewsResearcher


def make_news_tools(researcher: NewsResearcher) -> list[BaseTool]:
    """Return tools backed by *researcher*."""

    @tool
    def fetch_breaking_news(
        tickers: str,
        days_back: int = 1,
    ) -> str:
        """Fetch the latest breaking news for a set of tickers.

        Results are sorted by relevance_score descending.

        Args:
            tickers: Comma-separated ticker symbols, e.g. "NVDA, AAPL, MSFT".
            days_back: How many days back to search (default 1).

        Returns:
            Indented JSON array of NewsItem objects, or "ERROR: …".
        """
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            items = researcher.fetch_breaking_news(ticker_list, days_back=days_back)
            return json.dumps([i.model_dump() for i in items], indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    @tool
    def fetch_macro_news(days_back: int = 1) -> str:
        """Fetch macro-economic and broad market news.

        Results are sorted by relevance_score descending.

        Args:
            days_back: How many days back to search (default 1).

        Returns:
            Indented JSON array of NewsItem objects, or "ERROR: …".
        """
        try:
            items = researcher.fetch_macro_news(days_back=days_back)
            return json.dumps([i.model_dump() for i in items], indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    @tool
    def fetch_catalyst_news(
        ticker: str,
        days_back: int = 7,
    ) -> str:
        """Fetch catalyst events for a specific ticker.

        Post-filtered: only items where the ticker appears in
        tickers_mentioned OR relevance_score >= 0.5 are returned.

        Args:
            ticker: Single equity ticker, e.g. "NVDA".
            days_back: How many days back to search (default 7).

        Returns:
            Indented JSON array of NewsItem objects, or "ERROR: …".
        """
        try:
            items = researcher.fetch_catalyst_news(
                ticker.strip().upper(), days_back=days_back
            )
            return json.dumps([i.model_dump() for i in items], indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    @tool
    def fetch_daily_digest(
        tickers: str,
        sectors: str,
        date: str,
        days_back: int = 1,
    ) -> str:
        """Produce a structured daily market digest (NewsDigest).

        Raises RuntimeError internally on LLM synthesis failure; this tool
        converts that to an "ERROR: …" string.

        Args:
            tickers: Comma-separated ticker symbols to watch,
                     e.g. "NVDA, AAPL, TSLA".
            sectors: Comma-separated sector names to include,
                     e.g. "Technology, Healthcare".
            date: Digest date as ISO string, e.g. "2026-03-21".
            days_back: How many days back to search (default 1).

        Returns:
            Indented JSON of NewsDigest (top_stories, market_theme,
            sector_movers, macro_events, watchlist_alerts), or "ERROR: …".
        """
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            sector_list = [s.strip() for s in sectors.split(",") if s.strip()]
            digest = researcher.fetch_daily_digest(
                ticker_list, sector_list, date, days_back=days_back
            )
            return digest.model_dump_json(indent=2)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    return [fetch_breaking_news, fetch_macro_news, fetch_catalyst_news, fetch_daily_digest]
