"""
webai.ticker — Structured equity and sector research via Tavily fan-out.

Public API
----------
- ``TickerResearcher.research_ticker(symbol)``  → ``TickerResearch``
- ``TickerResearcher.research_sector(sector)``  → ``SectorResearch``

Both methods share the same pipeline:
1. Build a base query (ticker anchor or sector phrase).
2. Fan-out to multiple query variants using ``query_translation`` (expand,
   decompose, step-back) with domain-specific few-shot examples passed
   explicitly so there is no hidden coupling between the two flows.
3. Execute all queries in parallel against Tavily with ``topic="finance"``.
4. Deduplicate by URL, then optionally filter to a trusted-domain allowlist.
5. Synthesise a validated Pydantic object via the LLM's structured-output chain.

Invariants
----------
- ``_fan_out_queries`` always returns at least ``[base_query]``.
- ``_run_searches_parallel`` returns an empty list (not raises) if all queries fail.
- ``_filter_by_domain`` returns an empty list (not raises) if no results match.
- ``research_ticker`` / ``research_sector`` fall back to unfiltered results if
  domain filtering empties the list; they raise ``RuntimeError`` only if no
  results survive at all.
- ``_synthesize`` / ``_synthesize_sector`` always raise on failure; they never
  return partial/None.
"""
from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from webai.tools import SearchProvider, SearchResult, WebSearcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trusted-domain allowlist (Requirement 4)
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS: list[str] = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "fool.com",
    "seekingalpha.com",
    "finance.yahoo.com",
    "barrons.com",
    "investopedia.com",
    "sec.gov",
    "morningstar.com",
    "zacks.com",
    "thestreet.com",
    "benzinga.com",
    "businesswire.com",
    "prnewswire.com",
    "alphastreet.com",
    "stockanalysis.com",
    "federalreserve.gov",
    "apnews.com",
]

# ---------------------------------------------------------------------------
# Finance-domain few-shot examples for query_translation
# ---------------------------------------------------------------------------
# Format mirrors query_translation defaults: each dict has
#   "original_query": str
#   "new_queries": list[str]

_FINANCE_EXPAND_EXAMPLES: list[dict] = [
    {
        "original_query": "AAPL Apple stock analysis news",
        "new_queries": [
            "Apple Inc AAPL latest earnings report and stock outlook",
            "AAPL share price drivers and analyst ratings 2024",
            "Apple stock performance and near-term catalysts",
        ],
    },
    {
        "original_query": "NVDA Nvidia stock analysis news",
        "new_queries": [
            "Nvidia NVDA earnings growth and data-center revenue trends",
            "NVDA stock valuation and competitive landscape in AI chips",
            "Nvidia share price forecast and institutional sentiment",
        ],
    },
]

_FINANCE_DECOMPOSE_EXAMPLES: list[dict] = [
    {
        "original_query": "TSLA Tesla stock analysis news",
        "new_queries": [
            "What are Tesla's latest quarterly revenue and margin figures?",
            "How is Tesla's EV market share changing relative to competitors?",
            "What is the current analyst consensus rating and price target for TSLA?",
            "What macro or regulatory risks could affect Tesla's stock?",
            "What are the key catalysts expected in the next 1–2 quarters for TSLA?",
        ],
    },
]

_FINANCE_STEP_BACK_EXAMPLES: list[dict] = [
    {
        "original_query": "MSFT Microsoft stock analysis news",
        "new_queries": [
            "Macroeconomic conditions affecting large-cap technology stocks",
            "Cloud computing industry growth trends and competitive dynamics",
            "Impact of AI investment cycle on enterprise software valuations",
        ],
    },
]

# ---------------------------------------------------------------------------
# Sector-domain few-shot examples for query_translation
# ---------------------------------------------------------------------------
# These steer the LLM toward sector-level macro content rather than
# individual company news.  One example per "style" is sufficient because
# the LLM generalises across sector names.

_SECTOR_EXPAND_EXAMPLES: list[dict] = [
    {
        "original_query": "Semiconductor sector financial health outlook",
        "new_queries": [
            "Semiconductor industry current performance and future growth prospects",
            "Chip sector revenue trends, margins, and competitive dynamics",
            "State of the global semiconductor market and near-term drivers",
        ],
    },
    {
        "original_query": "Consumer discretionary sector financial health outlook",
        "new_queries": [
            "Consumer spending trends and retail sector financial performance",
            "Consumer discretionary industry earnings and valuation outlook",
            "Retail and luxury goods sector growth drivers and risks",
        ],
    },
]

_SECTOR_DECOMPOSE_EXAMPLES: list[dict] = [
    {
        "original_query": "Healthcare sector financial health outlook",
        "new_queries": [
            "What are current revenue growth rates across healthcare sub-sectors?",
            "How are interest rates and cost pressures affecting healthcare margins?",
            "What regulatory or policy changes are reshaping the healthcare sector?",
            "Which healthcare sub-sectors (pharma, biotech, MedTech) are outperforming?",
            "What is the analyst consensus on healthcare sector valuations?",
        ],
    },
]

_SECTOR_STEP_BACK_EXAMPLES: list[dict] = [
    {
        "original_query": "Energy sector financial health outlook",
        "new_queries": [
            "Impact of global commodity cycles and geopolitics on energy markets",
            "Long-term capital allocation trends in fossil fuels versus renewables",
            "Macroeconomic conditions driving inflation and demand in energy sector",
        ],
    },
]

# ---------------------------------------------------------------------------
# TickerResearch — validated output schema (Requirement 2)
# ---------------------------------------------------------------------------


class TickerResearch(BaseModel):
    """
    Structured equity research snapshot for a single ticker.

    All fields are populated by LLM synthesis over retrieved search results.
    """

    ticker: str = Field(
        ...,
        description="Uppercase ticker symbol exactly as provided (e.g. 'NVDA').",
    )
    company_name: str = Field(
        ...,
        description="Full legal or common company name (e.g. 'Nvidia Corporation').",
    )
    sentiment: Literal["bullish", "bearish", "neutral", "mixed"] = Field(
        ...,
        description=(
            "Overall market sentiment inferred from search results. "
            "Use 'mixed' when sources conflict."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the sentiment assessment on a 0–1 scale. "
            "Lower when sources are sparse, contradictory, or stale."
        ),
    )
    key_catalyst: str = Field(
        ...,
        description=(
            "The single most important near-term catalyst identified "
            "(earnings, product launch, macro event, regulatory action, etc.)."
        ),
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="3–5 specific risk factors that could negatively impact the stock.",
    )
    analyst_consensus: str = Field(
        default="",
        description=(
            "Summary of analyst ratings and/or price targets if mentioned in sources. "
            "Empty string if no analyst data is available."
        ),
    )
    macro_context: str = Field(
        default="",
        description=(
            "Relevant macroeconomic or sector-level context from step-back queries. "
            "Empty string if not applicable."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs used in the synthesis.",
    )
    freshness_warning: bool = Field(
        default=False,
        description=(
            "Set True if the retrieved sources appear to be older than 30 days "
            "and may not reflect current market conditions."
        ),
    )
    earnings_date: Optional[str] = Field(
        default=None,
        description=(
            "Upcoming or most-recent earnings date provided by the caller "
            "(any human-readable format, e.g. '2026-05-28'). None if not supplied."
        ),
    )


# ---------------------------------------------------------------------------
# SectorResearch — validated output schema for sector-level research
# ---------------------------------------------------------------------------


class SectorResearch(BaseModel):
    """Structured financial health snapshot for a market sector.

    All fields are populated by LLM synthesis over retrieved search results.
    The ``sector`` field is always exactly the string passed by the caller so
    downstream consumers can safely group or key on it.
    """

    sector: str = Field(
        ...,
        description="Sector name as provided by the caller (e.g. 'Semiconductors').",
    )
    overall_health: Literal["strong", "weak", "stable", "deteriorating", "mixed"] = Field(
        ...,
        description="Overall financial health of the sector inferred from sources.",
    )
    key_trends: list[str] = Field(
        default_factory=list,
        description="3–5 major structural or cyclical trends shaping the sector.",
    )
    tailwinds: list[str] = Field(
        default_factory=list,
        description="2–4 positive near-term drivers (demand catalysts, policy, innovation).",
    )
    headwinds: list[str] = Field(
        default_factory=list,
        description="2–4 negative pressures or risks (costs, regulation, competition).",
    )
    valuation_note: str = Field(
        default="",
        description=(
            "Commentary on sector-level valuations (P/E, multiples, etc.); "
            "empty string if unavailable."
        ),
    )
    leading_companies: list[str] = Field(
        default_factory=list,
        description="Top 3–5 company names most prominently mentioned in the sources.",
    )
    macro_sensitivity: str = Field(
        default="",
        description=(
            "How sensitive the sector is to rates, inflation, and credit cycles; "
            "empty string if unavailable."
        ),
    )
    outlook: str = Field(
        default="",
        description="1–2 sentence forward-looking statement for the sector.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs used in the synthesis.",
    )
    freshness_warning: bool = Field(
        default=False,
        description=(
            "True if sources appear older than 30 days or dates are indeterminate."
        ),
    )


# ---------------------------------------------------------------------------
# TickerResearcher — main orchestrator
# ---------------------------------------------------------------------------


class TickerResearcher:
    """
    Orchestrates multi-query Tavily fan-out and LLM synthesis into a
    validated ``TickerResearch`` result.

    Usage::

        from langchain_openai import ChatOpenAI
        from webai import TickerResearcher

        model = ChatOpenAI(model="gpt-4o-mini")
        researcher = TickerResearcher(model=model, debug=True)
        result = researcher.research_ticker("NVDA", company_name="Nvidia")
        print(result.model_dump_json(indent=2))

    Args:
        model: Any LangChain ``BaseChatModel`` that supports
            ``with_structured_output``.
        tavily_api_key: Tavily API key.  Falls back to ``TAVILY_API_KEY`` env var.
        openai_api_key: Passed through to ``WebSearcher`` for its optional
            OpenAI initialization.
        max_results_per_query: Number of Tavily results per query (default 5).
        trusted_domains: Override the module-level ``TRUSTED_DOMAINS`` list.
        filter_by_domain: Enable/disable domain filtering (default ``True``).
        max_workers: Thread pool size for parallel Tavily searches (default 8).
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

        # Fail-fast: Tavily is required (Requirement 1).
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            raise ValueError(
                "TickerResearcher requires a Tavily API key. "
                "Pass tavily_api_key= or set the TAVILY_API_KEY environment variable."
            )

        self._model = model
        self.max_workers = max_workers
        self.filter_by_domain = filter_by_domain
        # Copy so callers can't mutate the module-level list through this instance.
        self.trusted_domains: list[str] = list(trusted_domains or TRUSTED_DOMAINS)

        # Requirement 1: always use topic="finance" and include_raw_content=True.
        self._searcher = WebSearcher(
            provider=SearchProvider.TAVILY,
            tavily_api_key=tavily_key,
            openai_api_key=openai_api_key,
            max_results=max_results_per_query,
            include_raw_content=True,
            debug=debug,
        )

        # Requirement 2: build structured-output chains once; reuse per call.
        self._synthesis_chain = model.with_structured_output(TickerResearch)
        # Sector chain uses the same pattern — one chain per output schema.
        self._sector_synthesis_chain = model.with_structured_output(SectorResearch)

        logger.debug(
            "TickerResearcher initialized | filter_by_domain=%s trusted_domains=%d",
            filter_by_domain,
            len(self.trusted_domains),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def research_ticker(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        earnings_date: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> TickerResearch:
        """
        Run the full research pipeline for a ticker symbol.

        Pipeline:
            1. Normalise symbol to uppercase.
            2. Build base query (appends earnings date when provided).
            3. Fan-out to multiple query variants (LLM-assisted; degrades gracefully).
            4. Execute all queries in parallel against Tavily ``topic="finance"``.
            5. Deduplicate by URL.
            6. Filter to trusted domains (falls back to unfiltered if all removed).
            7. Synthesise into a validated ``TickerResearch`` via structured output.
            8. Post-set ``earnings_date`` on the result when provided.

        Args:
            symbol: Ticker symbol (e.g. ``"NVDA"``).  Whitespace is stripped.
            company_name: Optional company name to anchor queries and synthesis.
            earnings_date: Optional upcoming/recent earnings date in any
                human-readable format (e.g. ``"2026-05-28"``).  When provided,
                it is appended to the base query and recorded in the result.
            days_back: Restrict Tavily results to the last *N* days.  ``None``
                applies no date restriction (default behaviour).

        Returns:
            A fully validated ``TickerResearch`` instance.

        Raises:
            ValueError: If ``symbol`` is empty.
            RuntimeError: If no search results survive the pipeline, or if LLM
                synthesis fails.
        """
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must be a non-empty string.")

        logger.debug(
            "research_ticker | symbol=%s company_name=%r earnings_date=%r days_back=%r",
            symbol, company_name, earnings_date, days_back,
        )

        base_query = self._build_base_query(symbol, company_name)
        if earnings_date:
            company_label = company_name or symbol
            base_query = (
                f"{symbol} {company_label} stock analysis news earnings {earnings_date}"
            )

        all_queries = self._fan_out_queries(
            base_query,
            expand_examples=_FINANCE_EXPAND_EXAMPLES,
            decompose_examples=_FINANCE_DECOMPOSE_EXAMPLES,
            step_back_examples=_FINANCE_STEP_BACK_EXAMPLES,
        )
        logger.debug(
            "research_ticker | %d queries after fan-out: %s", len(all_queries), all_queries
        )

        raw_results = self._run_searches_parallel(all_queries, days=days_back)
        logger.debug("research_ticker | %d raw results", len(raw_results))

        deduped = self._deduplicate(raw_results)
        logger.debug("research_ticker | %d after dedup", len(deduped))

        if self.filter_by_domain:
            filtered = self._filter_by_domain(deduped)
            if not filtered:
                logger.warning(
                    "Domain filter removed all %d results — falling back to unfiltered.",
                    len(deduped),
                )
                filtered = deduped
            logger.debug("research_ticker | %d after domain filter", len(filtered))
        else:
            filtered = deduped

        if not filtered:
            raise RuntimeError(
                f"No search results survived the pipeline for symbol={symbol!r}. "
                "Check your Tavily API key and network connectivity."
            )

        earnings_note = ""
        if earnings_date is not None:
            earnings_note = (
                f"- The caller has provided an earnings date of '{earnings_date}'. "
                f"Reference it in key_catalyst if relevant.\n"
            )

        result_obj = self._synthesize(symbol, company_name, filtered, earnings_note=earnings_note)

        if earnings_date is not None:
            result_obj = result_obj.model_copy(update={"earnings_date": earnings_date})

        return result_obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_base_query(
        self, symbol: str, company_name: Optional[str]
    ) -> str:
        """
        Build the anchor query for a ticker.

        The phrase "stock analysis news" steers Tavily's finance topic toward
        relevant equity content rather than general company information.
        """
        if company_name:
            return f"{symbol} {company_name} stock analysis news"
        return f"{symbol} stock analysis news"

    def _fan_out_queries(
        self,
        base_query: str,
        expand_examples: list[dict],
        decompose_examples: list[dict],
        step_back_examples: list[dict],
    ) -> list[str]:
        """Thin wrapper — delegates to :func:`webai.utils.fan_out_queries`."""
        from webai import utils
        return utils.fan_out_queries(
            self._model,
            base_query,
            expand_examples,
            decompose_examples,
            step_back_examples,
        )

    def _run_searches_parallel(
        self,
        queries: list[str],
        days: Optional[int] = None,
    ) -> list[SearchResult]:
        """Thin wrapper — delegates to :func:`webai.utils.run_searches_parallel`."""
        from webai import utils
        return utils.run_searches_parallel(
            self._searcher,
            queries,
            topic="finance",
            max_workers=self.max_workers,
            days=days,
        )

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Thin wrapper — delegates to :func:`webai.utils.deduplicate`."""
        from webai import utils
        return utils.deduplicate(results)

    def _filter_by_domain(self, results: list[SearchResult]) -> list[SearchResult]:
        """Thin wrapper — delegates to :func:`webai.utils.filter_by_domain`."""
        from webai import utils
        return utils.filter_by_domain(results, self.trusted_domains)

    def _synthesize(
        self,
        symbol: str,
        company_name: Optional[str],
        results: list[SearchResult],
        earnings_note: str = "",
    ) -> TickerResearch:
        """
        Call the LLM structured-output chain to synthesise a ``TickerResearch``
        from the gathered search results (Requirement 2).

        The prompt includes:
            - Ticker and optional company name.
            - Numbered source blocks (title / URL / content).
            - Instruction to set ``freshness_warning=True`` if sources appear
              older than 30 days.
            - The full list of source URLs for the ``sources`` field.

        Args:
            symbol: Normalised uppercase ticker symbol.
            company_name: Optional company name string.
            results: Filtered, deduplicated ``SearchResult`` list.

        Returns:
            Validated ``TickerResearch`` instance.

        Raises:
            RuntimeError: If the LLM chain raises or returns an invalid object.
        """
        company_label = company_name or symbol
        source_blocks: list[str] = []
        source_urls: list[str] = []

        for i, result in enumerate(results, 1):
            source_blocks.append(
                f"[{i}] {result.title}\n"
                f"    Source: {result.source}\n"
                f"    {result.content}"
            )
            if result.source:
                source_urls.append(result.source)

        sources_section = "\n\n".join(source_blocks)
        sources_list = "\n".join(f"- {url}" for url in source_urls)

        prompt = (
            f"You are a professional equity research analyst. "
            f"Using ONLY the search results below, produce a structured research "
            f"snapshot for the stock ticker {symbol} ({company_label}).\n\n"
            f"SEARCH RESULTS:\n{sources_section}\n\n"
            f"SOURCES:\n{sources_list}\n\n"
            f"Instructions:\n"
            f"- Set `ticker` to '{symbol}'.\n"
            f"- Set `company_name` to the full company name.\n"
            f"- Set `sentiment` to 'bullish', 'bearish', 'neutral', or 'mixed'.\n"
            f"- Set `confidence` (0.0–1.0) reflecting how consistent and fresh the "
            f"  evidence is (lower when sparse, contradictory, or stale).\n"
            f"- Set `key_catalyst` to the single most important near-term catalyst.\n"
            f"- Set `risk_factors` to 3–5 specific risks.\n"
            f"- Set `analyst_consensus` if analyst ratings or price targets are mentioned; "
            f"  otherwise leave empty.\n"
            f"- Set `macro_context` from any macro/sector context found; leave empty if absent.\n"
            f"- Set `sources` to the list of source URLs provided above.\n"
            f"- Set `freshness_warning` to true if the sources appear to be more than "
            f"  30 days old or if dates cannot be determined.\n"
            + earnings_note
        )

        logger.debug(
            "_synthesize | invoking chain for %s with %d sources", symbol, len(results)
        )

        try:
            result_obj = self._synthesis_chain.invoke(prompt)
        except Exception as exc:
            logger.error("_synthesize | chain invocation failed: %s", exc, exc_info=True)
            raise RuntimeError(
                f"LLM synthesis failed for {symbol}: {exc}"
            ) from exc

        if not isinstance(result_obj, TickerResearch):
            raise RuntimeError(
                f"Synthesis returned unexpected type {type(result_obj).__name__} "
                f"instead of TickerResearch for {symbol}."
            )

        logger.debug(
            "_synthesize | done: sentiment=%s confidence=%.2f",
            result_obj.sentiment,
            result_obj.confidence,
        )
        return result_obj


    # ------------------------------------------------------------------
    # Sector research — public entry point
    # ------------------------------------------------------------------

    def research_sector(self, sector: str) -> SectorResearch:
        """
        Run the full research pipeline for a market sector.

        Pipeline (mirrors ``research_ticker`` structure):
            1. Normalise sector name (strip whitespace).
            2. Build sector base query.
            3. Fan-out to multiple query variants using sector-specific few-shots.
            4. Execute all queries in parallel against Tavily ``topic="finance"``.
            5. Deduplicate by URL.
            6. Filter to trusted domains (falls back to unfiltered if all removed).
            7. Synthesise into a validated ``SectorResearch`` via structured output.

        Args:
            sector: Sector name (e.g. ``"Semiconductors"``).  Whitespace is stripped.

        Returns:
            A fully validated ``SectorResearch`` instance.

        Raises:
            ValueError: If ``sector`` is empty after stripping.
            RuntimeError: If no search results survive the pipeline, or if LLM
                synthesis fails.
        """
        sector = sector.strip()
        if not sector:
            raise ValueError("sector must be a non-empty string.")

        logger.debug("research_sector | sector=%r", sector)

        base_query = self._build_sector_base_query(sector)
        all_queries = self._fan_out_queries(
            base_query,
            expand_examples=_SECTOR_EXPAND_EXAMPLES,
            decompose_examples=_SECTOR_DECOMPOSE_EXAMPLES,
            step_back_examples=_SECTOR_STEP_BACK_EXAMPLES,
        )
        logger.debug(
            "research_sector | %d queries after fan-out: %s", len(all_queries), all_queries
        )

        raw_results = self._run_searches_parallel(all_queries)
        logger.debug("research_sector | %d raw results", len(raw_results))

        deduped = self._deduplicate(raw_results)
        logger.debug("research_sector | %d after dedup", len(deduped))

        if self.filter_by_domain:
            filtered = self._filter_by_domain(deduped)
            if not filtered:
                logger.warning(
                    "Domain filter removed all %d results — falling back to unfiltered.",
                    len(deduped),
                )
                filtered = deduped
            logger.debug("research_sector | %d after domain filter", len(filtered))
        else:
            filtered = deduped

        if not filtered:
            raise RuntimeError(
                f"No search results survived the pipeline for sector={sector!r}. "
                "Check your Tavily API key and network connectivity."
            )

        return self._synthesize_sector(sector, filtered)

    # ------------------------------------------------------------------
    # Portfolio research
    # ------------------------------------------------------------------

    def research_portfolio(
        self,
        symbols: list,
        days_back: Optional[int] = None,
    ) -> dict:
        """
        Run :meth:`research_ticker` for each symbol, isolating failures per-ticker.

        Each entry in *symbols* is either a bare ticker string (e.g. ``"NVDA"``)
        or a ``(ticker, company_name)`` tuple.  Failures for individual tickers
        are caught, logged at WARNING, and excluded from the output — the batch
        never aborts.

        Args:
            symbols: List of ticker strings or ``(ticker, company_name)`` tuples.
            days_back: Passed through to each :meth:`research_ticker` call.
                ``None`` applies no date restriction.

        Returns:
            ``dict[str, TickerResearch]`` keyed by uppercase ticker symbol,
            containing only the successfully researched tickers.
        """
        out: dict[str, TickerResearch] = {}

        for entry in symbols:
            if isinstance(entry, tuple):
                symbol, company_name = entry[0], entry[1] if len(entry) > 1 else None
            else:
                symbol, company_name = str(entry), None

            symbol = symbol.strip().upper()

            try:
                result = self.research_ticker(
                    symbol,
                    company_name=company_name,
                    days_back=days_back,
                )
                out[symbol] = result
            except Exception as exc:
                logger.warning("research_portfolio | %s failed: %s", symbol, exc)

        logger.info(
            "research_portfolio | %d/%d completed",
            len(out),
            len(symbols),
        )
        return out

    # ------------------------------------------------------------------
    # Sector research — private helpers
    # ------------------------------------------------------------------

    def _build_sector_base_query(self, sector: str) -> str:
        """
        Build the anchor query for a sector.

        The phrase "financial health outlook" steers Tavily's finance topic
        toward macro-sector content rather than individual company news.
        """
        return f"{sector} sector financial health outlook"

    def _synthesize_sector(
        self,
        sector: str,
        results: list[SearchResult],
    ) -> SectorResearch:
        """
        Call the LLM structured-output chain to synthesise a ``SectorResearch``
        from the gathered search results.

        The prompt includes:
            - Sector name.
            - Numbered source blocks (title / URL / content).
            - Per-field instructions matching the ``SectorResearch`` schema.
            - Instruction to set ``freshness_warning=True`` when sources appear
              older than 30 days or dates cannot be determined.

        Args:
            sector: Normalised sector name string.
            results: Filtered, deduplicated ``SearchResult`` list.

        Returns:
            Validated ``SectorResearch`` instance.

        Raises:
            RuntimeError: If the LLM chain raises or returns an invalid object.
        """
        source_blocks: list[str] = []
        source_urls: list[str] = []

        for i, result in enumerate(results, 1):
            source_blocks.append(
                f"[{i}] {result.title}\n"
                f"    Source: {result.source}\n"
                f"    {result.content}"
            )
            if result.source:
                source_urls.append(result.source)

        sources_section = "\n\n".join(source_blocks)
        sources_list = "\n".join(f"- {url}" for url in source_urls)

        prompt = (
            f"You are a professional sector equity analyst. "
            f"Using ONLY the search results below, produce a structured financial "
            f"health snapshot for the {sector} sector.\n\n"
            f"SEARCH RESULTS:\n{sources_section}\n\n"
            f"SOURCES:\n{sources_list}\n\n"
            f"Instructions:\n"
            f"- Set `sector` to '{sector}'.\n"
            f"- Set `overall_health` to one of: strong / weak / stable / "
            f"deteriorating / mixed.\n"
            f"- Set `key_trends` to 3–5 major structural or cyclical trends.\n"
            f"- Set `tailwinds` to 2–4 positive near-term drivers.\n"
            f"- Set `headwinds` to 2–4 negative pressures or risks.\n"
            f"- Set `valuation_note` to P/E / multiples commentary; leave empty "
            f"if no valuation data is present.\n"
            f"- Set `leading_companies` to the top 3–5 company names mentioned.\n"
            f"- Set `macro_sensitivity` to how sensitive the sector is to rates, "
            f"inflation, and credit cycles; leave empty if not discussed.\n"
            f"- Set `outlook` to a 1–2 sentence forward-looking statement.\n"
            f"- Set `sources` to the list of source URLs provided above.\n"
            f"- Set `freshness_warning` to true if sources appear to be more than "
            f"30 days old or if dates cannot be determined.\n"
        )

        logger.debug(
            "_synthesize_sector | invoking chain for %r with %d sources",
            sector,
            len(results),
        )

        try:
            result_obj = self._sector_synthesis_chain.invoke(prompt)
        except Exception as exc:
            logger.error(
                "_synthesize_sector | chain invocation failed: %s", exc, exc_info=True
            )
            raise RuntimeError(
                f"LLM synthesis failed for sector={sector!r}: {exc}"
            ) from exc

        if not isinstance(result_obj, SectorResearch):
            raise RuntimeError(
                f"Synthesis returned unexpected type {type(result_obj).__name__} "
                f"instead of SectorResearch for sector={sector!r}."
            )

        logger.debug(
            "_synthesize_sector | done: overall_health=%s freshness_warning=%s",
            result_obj.overall_health,
            result_obj.freshness_warning,
        )
        return result_obj


# ---------------------------------------------------------------------------
# Smoke-test / quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    _model = ChatOpenAI(model="gpt-4o-mini")
    _researcher = TickerResearcher(model=_model, max_results_per_query=3, debug=True)

    # --- ticker research ---
    _ticker_result = _researcher.research_ticker("NVDA", company_name="Nvidia")

    assert isinstance(_ticker_result, TickerResearch)
    assert _ticker_result.ticker == "NVDA"
    assert _ticker_result.sentiment in {"bullish", "bearish", "neutral", "mixed"}
    assert 0.0 <= _ticker_result.confidence <= 1.0
    assert isinstance(_ticker_result.risk_factors, list)

    print("=== TickerResearch ===")
    print(json.dumps(json.loads(_ticker_result.model_dump_json()), indent=2))

    # --- sector research ---
    _sector_result = _researcher.research_sector("Semiconductors")

    assert isinstance(_sector_result, SectorResearch)
    assert _sector_result.sector == "Semiconductors"
    assert _sector_result.overall_health in {
        "strong", "weak", "stable", "deteriorating", "mixed"
    }
    assert isinstance(_sector_result.key_trends, list)
    assert isinstance(_sector_result.tailwinds, list)
    assert isinstance(_sector_result.headwinds, list)

    print("\n=== SectorResearch ===")
    print(json.dumps(json.loads(_sector_result.model_dump_json()), indent=2))
