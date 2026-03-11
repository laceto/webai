"""
webai.news — Breaking-news extraction and daily digest synthesis.

Public API
----------
- ``NewsResearcher.fetch_breaking_news(tickers, days_back=1)``  → ``list[NewsItem]``
- ``NewsResearcher.fetch_macro_news(days_back=1)``              → ``list[NewsItem]``
- ``NewsResearcher.fetch_catalyst_news(ticker, days_back=7)``   → ``list[NewsItem]``
- ``NewsResearcher.fetch_daily_digest(...)``                     → ``NewsDigest``

Data models (all exported)
--------------------------
- ``NewsItem``    — individual news event with sentiment, event_type, relevance_score
- ``SectorMover`` — sector-level movement note for digest summaries
- ``NewsDigest``  — structured daily market digest

Internal
--------
- ``_NewsResults`` — structured-output wrapper; not exported

Invariants
----------
- ``_extract_news_items`` never raises; returns ``[]`` on failure (logged WARNING).
- ``fetch_daily_digest`` raises ``RuntimeError`` on LLM synthesis failure.
- Domain filter falls back to unfiltered if all results are removed (logged WARNING).
- Fan-out and parallel search follow the same invariants as ``webai.utils``.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from webai import utils
from webai.ticker import TRUSTED_DOMAINS as _DEFAULT_TRUSTED_DOMAINS
from webai.tools import SearchProvider, SearchResult, WebSearcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# News-domain few-shot examples for query_translation
# ---------------------------------------------------------------------------
# These are intentionally shorter / more concrete than the finance fan-out
# sets because `days_back=1` searches don't benefit from broad step-back
# queries — we want recent, specific results, not macro context.

_NEWS_EXPAND_EXAMPLES: list[dict] = [
    {
        "original_query": "NVDA breaking news",
        "new_queries": [
            "Nvidia latest headlines and announcements today",
            "NVDA stock-moving news and press releases",
        ],
    },
    {
        "original_query": "market macro news Federal Reserve interest rates inflation",
        "new_queries": [
            "Federal Reserve latest rate decision and economic outlook",
            "US inflation CPI data and market impact today",
        ],
    },
]

_NEWS_DECOMPOSE_EXAMPLES: list[dict] = [
    {
        "original_query": "AAPL catalyst earnings analyst upgrade downgrade",
        "new_queries": [
            "What are Apple's most recent earnings results and guidance?",
            "Which analysts have upgraded or downgraded AAPL recently?",
            "What product launches or partnerships could move AAPL stock?",
        ],
    },
]

_NEWS_STEP_BACK_EXAMPLES: list[dict] = [
    {
        "original_query": "TSLA breaking news",
        "new_queries": [
            "Broader EV sector news and regulatory environment",
            "Macro economic conditions affecting growth and technology stocks",
        ],
    },
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """
    A single, standalone financial news event extracted from search results.

    ``relevance_score`` reflects actionability: 0.9+ is reserved for clearly
    market-moving breaking news; lower scores indicate background or routine
    items.  This field is the primary sort key for all fetch methods.
    """

    headline: str = Field(..., description="Exact or paraphrased headline.")
    summary: str = Field(
        ...,
        description="2–4 sentence market-relevant summary of what happened and why it matters.",
    )
    source_url: str = Field(
        default="",
        description="Article URL. Empty string if unavailable.",
    )
    published_date: str = Field(
        default="",
        description="Publication date in YYYY-MM-DD format, or empty string if not determinable.",
    )
    tickers_mentioned: list[str] = Field(
        default_factory=list,
        description="Uppercase ticker symbols mentioned or clearly implied by the article.",
    )
    event_type: Literal[
        "earnings",
        "guidance",
        "analyst_action",
        "macro",
        "regulatory",
        "m_and_a",
        "insider",
        "technical",
        "other",
    ] = Field(..., description="Single best-fitting event category.")
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        ...,
        description="Market sentiment implied by the news item.",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "0–1 actionability score. "
            "0.9+ reserved for clearly market-moving breaking news."
        ),
    )


class _NewsResults(BaseModel):
    """
    Internal structured-output wrapper returned by the news-extraction chain.

    Not exported — only ``NewsItem`` is part of the public API.  Using a
    wrapper model avoids asking the LLM to return a bare array, which some
    providers handle poorly.
    """

    items: list[NewsItem] = Field(default_factory=list)


class SectorMover(BaseModel):
    """
    A market sector showing notable movement, included in a ``NewsDigest``.
    """

    sector: str = Field(..., description="Sector name.")
    note: str = Field(
        ...,
        description="1–2 sentence note on why this sector is moving.",
    )


class NewsDigest(BaseModel):
    """
    Structured daily market news digest.

    Aggregates the highest-relevance stories, dominant market theme, notable
    sector movers, macro events, and optional watchlist alerts for a single
    trading day.
    """

    date: str = Field(..., description="YYYY-MM-DD date this digest covers.")
    top_stories: list[NewsItem] = Field(
        default_factory=list,
        description="3–7 highest-relevance news items for the day.",
    )
    market_theme: str = Field(
        default="",
        description="1–3 sentences on the dominant market narrative.",
    )
    sector_movers: list[SectorMover] = Field(
        default_factory=list,
        description="Up to 5 notable sectors with movement context.",
    )
    macro_events: list[str] = Field(
        default_factory=list,
        description=(
            "Fed decisions, CPI releases, geopolitical events, and other macro items."
        ),
    )
    watchlist_alerts: list[str] = Field(
        default_factory=list,
        description=(
            "Short alert strings for watchlist tickers/themes with significant news. "
            "Empty when no watchlist is provided."
        ),
    )


# ---------------------------------------------------------------------------
# Synthesis prompt templates
# ---------------------------------------------------------------------------

_NEWS_EXTRACTION_PROMPT_TEMPLATE = (
    "You are a financial news analyst. Extract all distinct news items from the "
    "search results below. Each item must be a separate, standalone piece of news.\n\n"
    "Context: {context_hint}\n\n"
    "SEARCH RESULTS:\n{sources_section}\n\n"
    "Instructions for each NewsItem:\n"
    "- headline: exact or paraphrased headline\n"
    "- summary: 2–4 sentences, what happened and why it matters to markets\n"
    "- source_url: from the Source field above; empty string if unavailable\n"
    "- published_date: YYYY-MM-DD if determinable; else empty string\n"
    "- tickers_mentioned: uppercase symbols mentioned or clearly implied\n"
    "- event_type: single best category (earnings / guidance / analyst_action / "
    "macro / regulatory / m_and_a / insider / technical / other)\n"
    "- sentiment: bullish / bearish / neutral\n"
    "- relevance_score: 0.0–1.0; 0.9+ only for clearly market-moving breaking news\n\n"
    "Return ALL distinct news items found. Skip sources with no news content."
)

_DIGEST_PROMPT_TEMPLATE = (
    "You are a senior market strategist. Produce a daily news digest for {date}.\n\n"
    "WATCHLIST: {watchlist_section}\n\n"
    "SEARCH RESULTS:\n{sources_section}\n\n"
    "Instructions:\n"
    "- date: '{date}'\n"
    "- top_stories: 3–7 highest-relevance NewsItem objects (fully populated, "
    "including all fields)\n"
    "- market_theme: 1–3 sentences on the dominant market narrative\n"
    "- sector_movers: up to 5 SectorMover objects for the most notable sectors\n"
    "- macro_events: list all macro events (Fed decisions, CPI, earnings season "
    "milestones, geopolitical)\n"
    "- watchlist_alerts: only if a watchlist is provided — short alert strings for "
    "items with significant news; otherwise leave empty"
)


# ---------------------------------------------------------------------------
# NewsResearcher
# ---------------------------------------------------------------------------


class NewsResearcher:
    """
    Fetches and synthesises financial news using a fan-out Tavily pipeline.

    All search operations target Tavily with ``topic="news"`` (or
    ``topic="finance"`` for :meth:`fetch_catalyst_news`) and support a
    ``days_back`` time filter.  Structured-output chains are built once at
    construction time and reused across all calls.

    Args:
        model: Any LangChain ``BaseChatModel`` that supports
            ``with_structured_output``.
        tavily_api_key: Tavily API key. Falls back to ``TAVILY_API_KEY`` env var.
            **Required** — raises ``ValueError`` if absent.
        openai_api_key: Optional OpenAI key passed through to ``WebSearcher``.
        max_results_per_query: Tavily results per query (default 5).
        trusted_domains: Override the default ``TRUSTED_DOMAINS`` list imported
            from ``webai.ticker``.
        filter_by_domain: Enable/disable domain filtering (default ``True``).
            When ``True``, results not matching ``trusted_domains`` are dropped;
            falls back to unfiltered if filtering removes everything.
        max_workers: Thread-pool size for parallel searches (default 8).
        debug: When ``True``, sets this module's logger to ``DEBUG`` level and
            attaches a ``StreamHandler`` if none is already configured.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tavily_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        max_results_per_query: int = 5,
        trusted_domains: Optional[list[str]] = None,
        filter_by_domain: bool = True,
        max_workers: int = 8,
        debug: bool = False,
    ) -> None:
        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                logger.addHandler(logging.StreamHandler())

        # Fail-fast: Tavily is the sole search backend for this class.
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError(
                "NewsResearcher requires a Tavily API key. "
                "Pass tavily_api_key= or set the TAVILY_API_KEY environment variable."
            )

        self._model = model
        self.max_workers = max_workers
        self.filter_by_domain = filter_by_domain
        # Copy so callers cannot mutate the module-level list through this instance.
        self.trusted_domains: list[str] = list(
            trusted_domains if trusted_domains is not None else _DEFAULT_TRUSTED_DOMAINS
        )

        self._searcher = WebSearcher(
            provider=SearchProvider.TAVILY,
            tavily_api_key=tavily_key,
            openai_api_key=openai_api_key,
            max_results=max_results_per_query,
            include_raw_content=True,
            debug=debug,
        )

        # Build structured-output chains once; reuse across all calls.
        self._news_chain = model.with_structured_output(_NewsResults)
        self._digest_chain = model.with_structured_output(NewsDigest)

        logger.debug(
            "NewsResearcher initialized | filter_by_domain=%s trusted_domains=%d",
            filter_by_domain,
            len(self.trusted_domains),
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_breaking_news(
        self,
        tickers: list[str],
        days_back: int = 1,
    ) -> list[NewsItem]:
        """
        Fetch breaking news for one or more ticker symbols.

        For each ticker a dedicated base query is built and fanned out into
        multiple variants, then all variants are searched against Tavily's
        ``topic="news"`` backend with a ``days_back`` time filter.  Results
        from all tickers are pooled, deduplicated, domain-filtered, news-
        extracted, and sorted by ``relevance_score`` descending.

        Args:
            tickers: Ticker symbols (e.g. ``["NVDA", "AAPL"]``).
            days_back: Restrict search to the last N days (default 1).

        Returns:
            List of ``NewsItem`` objects sorted by ``relevance_score`` desc.
            Returns ``[]`` if no results survive the pipeline.
        """
        if not tickers:
            return []

        # Build and fan out one query set per ticker; deduplicate across tickers.
        all_queries: list[str] = []
        seen_queries: dict[str, None] = {}
        for ticker in tickers:
            base = f"{ticker.upper()} breaking news"
            for q in utils.fan_out_queries(
                self._model,
                base,
                _NEWS_EXPAND_EXAMPLES,
                _NEWS_DECOMPOSE_EXAMPLES,
                _NEWS_STEP_BACK_EXAMPLES,
            ):
                if q not in seen_queries:
                    seen_queries[q] = None
                    all_queries.append(q)

        logger.debug(
            "fetch_breaking_news | %d tickers → %d queries",
            len(tickers),
            len(all_queries),
        )

        raw = utils.run_searches_parallel(
            self._searcher,
            all_queries,
            topic="news",
            max_workers=self.max_workers,
            days=days_back,
        )
        deduped = utils.deduplicate(raw)
        filtered = self._apply_domain_filter(deduped)

        context_hint = f"Breaking news for tickers: {', '.join(t.upper() for t in tickers)}"
        items = self._extract_news_items(filtered, context_hint=context_hint)
        items.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.debug("fetch_breaking_news | extracted %d items", len(items))
        return items

    def fetch_macro_news(self, days_back: int = 1) -> list[NewsItem]:
        """
        Fetch broad macro-economic news (Fed, inflation, interest rates, etc.).

        Args:
            days_back: Restrict search to the last N days (default 1).

        Returns:
            List of ``NewsItem`` objects sorted by ``relevance_score`` desc.
        """
        base = "market macro news Federal Reserve interest rates inflation"
        all_queries = utils.fan_out_queries(
            self._model,
            base,
            _NEWS_EXPAND_EXAMPLES,
            _NEWS_DECOMPOSE_EXAMPLES,
            _NEWS_STEP_BACK_EXAMPLES,
        )
        logger.debug("fetch_macro_news | %d queries", len(all_queries))

        raw = utils.run_searches_parallel(
            self._searcher,
            all_queries,
            topic="news",
            max_workers=self.max_workers,
            days=days_back,
        )
        deduped = utils.deduplicate(raw)
        filtered = self._apply_domain_filter(deduped)

        items = self._extract_news_items(filtered, context_hint="Macro-economic news")
        items.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.debug("fetch_macro_news | extracted %d items", len(items))
        return items

    def fetch_catalyst_news(
        self,
        ticker: str,
        days_back: int = 7,
    ) -> list[NewsItem]:
        """
        Fetch catalyst-specific news for a single ticker.

        Uses ``topic="finance"`` (better for structured financial content than
        ``"news"``) and applies a post-filter to keep only items where the
        ticker is explicitly mentioned OR the ``relevance_score`` is at least
        0.5.

        Args:
            ticker: Ticker symbol (e.g. ``"NVDA"``).  Whitespace is stripped.
            days_back: Restrict search to the last N days (default 7).

        Returns:
            List of ``NewsItem`` objects sorted by ``relevance_score`` desc.
        """
        ticker = ticker.strip().upper()
        base = f"{ticker} catalyst earnings analyst upgrade downgrade"
        all_queries = utils.fan_out_queries(
            self._model,
            base,
            _NEWS_EXPAND_EXAMPLES,
            _NEWS_DECOMPOSE_EXAMPLES,
            _NEWS_STEP_BACK_EXAMPLES,
        )
        logger.debug("fetch_catalyst_news | %s → %d queries", ticker, len(all_queries))

        raw = utils.run_searches_parallel(
            self._searcher,
            all_queries,
            topic="finance",
            max_workers=self.max_workers,
            days=days_back,
        )
        deduped = utils.deduplicate(raw)
        filtered = self._apply_domain_filter(deduped)

        items = self._extract_news_items(
            filtered,
            context_hint=f"Catalyst and earnings news for {ticker}",
        )

        # Post-filter: only keep items explicitly about this ticker or high-relevance.
        # This prevents macro noise from dominating a ticker-specific query.
        relevant = [
            item for item in items
            if ticker in item.tickers_mentioned or item.relevance_score >= 0.5
        ]
        relevant.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.debug(
            "fetch_catalyst_news | %s: %d items after post-filter", ticker, len(relevant)
        )
        return relevant

    def fetch_daily_digest(
        self,
        tickers: Optional[list[str]] = None,
        sectors: Optional[list[str]] = None,
        date: Optional[str] = None,
        days_back: int = 1,
    ) -> NewsDigest:
        """
        Produce a structured daily market digest.

        Builds a combined query set from a macro anchor, one query per ticker,
        and one query per sector.  All queries run in parallel against
        ``topic="news"``.  Domain filtering falls back to unfiltered if all
        results are removed.

        Args:
            tickers: Optional watchlist ticker symbols (e.g. ``["NVDA", "AAPL"]``).
            sectors: Optional sector names (e.g. ``["Technology", "Energy"]``).
            date: Digest date in YYYY-MM-DD format.  Defaults to today (UTC).
            days_back: Restrict search to the last N days (default 1).

        Returns:
            A validated ``NewsDigest`` instance.

        Raises:
            RuntimeError: If LLM synthesis fails.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        tickers = tickers or []
        sectors = sectors or []

        # Build a deduplicated query set: macro anchor + per-ticker + per-sector.
        macro_anchor = "market macro news Federal Reserve interest rates inflation"
        seen: dict[str, None] = {macro_anchor: None}
        queries: list[str] = [macro_anchor]

        for ticker in tickers:
            q = f"{ticker.upper()} latest news"
            if q not in seen:
                seen[q] = None
                queries.append(q)

        for sector in sectors:
            q = f"{sector} sector news outlook"
            if q not in seen:
                seen[q] = None
                queries.append(q)

        logger.debug("fetch_daily_digest | date=%s  %d queries", date, len(queries))

        raw = utils.run_searches_parallel(
            self._searcher,
            queries,
            topic="news",
            max_workers=self.max_workers,
            days=days_back,
        )
        deduped = utils.deduplicate(raw)
        filtered = self._apply_domain_filter(deduped)

        # _apply_domain_filter already handles fallback; this is a safety net
        # in case both filtered and deduped are empty.
        if not filtered:
            filtered = deduped

        source_blocks = self._format_source_blocks(filtered)

        if tickers or sectors:
            parts = []
            if tickers:
                parts.append("Tickers: " + ", ".join(t.upper() for t in tickers))
            if sectors:
                parts.append("Sectors: " + ", ".join(sectors))
            watchlist_section = "; ".join(parts)
        else:
            watchlist_section = "(none)"

        prompt = _DIGEST_PROMPT_TEMPLATE.format(
            date=date,
            watchlist_section=watchlist_section,
            sources_section=source_blocks,
        )

        logger.debug(
            "fetch_daily_digest | invoking digest chain with %d sources", len(filtered)
        )

        try:
            result_obj = self._digest_chain.invoke(prompt)
        except Exception as exc:
            logger.error(
                "fetch_daily_digest | chain invocation failed: %s", exc, exc_info=True
            )
            raise RuntimeError(
                f"News digest synthesis failed for date={date!r}: {exc}"
            ) from exc

        if not isinstance(result_obj, NewsDigest):
            raise RuntimeError(
                f"Digest chain returned unexpected type {type(result_obj).__name__} "
                f"instead of NewsDigest."
            )

        logger.debug(
            "fetch_daily_digest | done: top_stories=%d macro_events=%d",
            len(result_obj.top_stories),
            len(result_obj.macro_events),
        )
        return result_obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_domain_filter(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Apply the trusted-domain filter with automatic fallback.

        If filtering is disabled or removes all results, the unfiltered list is
        returned and a WARNING is logged — matching the fallback pattern used in
        ``TickerResearcher``.
        """
        if not self.filter_by_domain:
            return results
        filtered = utils.filter_by_domain(results, self.trusted_domains)
        if not filtered and results:
            logger.warning(
                "_apply_domain_filter | filter removed all %d results — "
                "falling back to unfiltered.",
                len(results),
            )
            return results
        return filtered

    def _extract_news_items(
        self,
        results: list[SearchResult],
        context_hint: str = "",
    ) -> list[NewsItem]:
        """
        Invoke the news-extraction chain on *results*.

        Returns an empty list on any failure; **never raises**.  This is the
        only synthesis method in this class allowed to swallow exceptions — all
        public methods that need guaranteed output use ``fetch_daily_digest``
        (which does raise) or post-process the list themselves.

        Args:
            results: Search results to extract news events from.
            context_hint: Short description of the expected content type,
                included in the prompt to help the LLM focus (e.g.
                ``"Breaking news for tickers: NVDA, AAPL"``).

        Returns:
            List of ``NewsItem`` objects; empty if extraction fails.
        """
        if not results:
            return []

        source_blocks = self._format_source_blocks(results)
        prompt = _NEWS_EXTRACTION_PROMPT_TEMPLATE.format(
            context_hint=context_hint,
            sources_section=source_blocks,
        )

        try:
            result_obj = self._news_chain.invoke(prompt)
            if not isinstance(result_obj, _NewsResults):
                logger.warning(
                    "_extract_news_items | unexpected return type: %s",
                    type(result_obj).__name__,
                )
                return []
            return result_obj.items
        except Exception as exc:
            logger.warning(
                "_extract_news_items | extraction chain failed: %s", exc, exc_info=True
            )
            return []

    @staticmethod
    def _format_source_blocks(results: list[SearchResult]) -> str:
        """
        Format search results into numbered source blocks for prompt inclusion.

        Each block::

            [N] Title
                Source: URL
                Content text...
        """
        blocks: list[str] = []
        for i, result in enumerate(results, 1):
            blocks.append(
                f"[{i}] {result.title}\n"
                f"    Source: {result.source}\n"
                f"    {result.content}"
            )
        return "\n\n".join(blocks)
