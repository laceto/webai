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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from webai.tools import SearchProvider, SearchResult, WebSearcher
from query_translation import decompose_query, expand_query, step_back_query

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
    ) -> TickerResearch:
        """
        Run the full research pipeline for a ticker symbol.

        Pipeline:
            1. Normalise symbol to uppercase.
            2. Build base query.
            3. Fan-out to multiple query variants (LLM-assisted; degrades gracefully).
            4. Execute all queries in parallel against Tavily ``topic="finance"``.
            5. Deduplicate by URL.
            6. Filter to trusted domains (falls back to unfiltered if all removed).
            7. Synthesise into a validated ``TickerResearch`` via structured output.

        Args:
            symbol: Ticker symbol (e.g. ``"NVDA"``).  Whitespace is stripped.
            company_name: Optional company name to anchor queries and synthesis.

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

        logger.debug("research_ticker | symbol=%s company_name=%r", symbol, company_name)

        base_query = self._build_base_query(symbol, company_name)
        all_queries = self._fan_out_queries(
            base_query,
            expand_examples=_FINANCE_EXPAND_EXAMPLES,
            decompose_examples=_FINANCE_DECOMPOSE_EXAMPLES,
            step_back_examples=_FINANCE_STEP_BACK_EXAMPLES,
        )
        logger.debug(
            "research_ticker | %d queries after fan-out: %s", len(all_queries), all_queries
        )

        raw_results = self._run_searches_parallel(all_queries)
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

        return self._synthesize(symbol, company_name, filtered)

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
        """
        Generate multiple query variants from ``base_query`` using LLM-assisted
        query translation.

        Strategy:
            - ``expand_query``    → paraphrases
            - ``decompose_query`` → sub-questions
            - ``step_back_query`` → macro / broadened context

        The few-shot examples are supplied by the caller so the same method
        works for both ticker and sector research (no hidden coupling to a
        specific example list).

        Each step is wrapped in a ``try/except``; failures are logged at WARNING
        and the pipeline continues.  ``base_query`` is always index 0 in the
        returned list, and duplicates are removed (order preserved).

        Args:
            base_query: The anchor query string.
            expand_examples: Few-shot examples for ``expand_query``.
            decompose_examples: Few-shot examples for ``decompose_query``.
            step_back_examples: Few-shot examples for ``step_back_query``.

        Returns:
            At least ``[base_query]``.
        """
        seen: dict[str, None] = {base_query: None}  # ordered-set via dict
        queries: list[str] = [base_query]

        # --- expand (paraphrases) ---
        try:
            expand_results = expand_query(
                self._model,
                [base_query],
                few_shot_examples=expand_examples,
            )
            for r in expand_results:
                q = r.paraphrased_query.strip()
                if q and q not in seen:
                    seen[q] = None
                    queries.append(q)
            logger.debug("_fan_out_queries | expand added %d", len(expand_results))
        except Exception as exc:
            logger.warning("expand_query failed (skipping): %s", exc)

        # --- decompose (sub-questions) ---
        try:
            decompose_results = decompose_query(
                self._model,
                [base_query],
                few_shot_examples=decompose_examples,
            )
            for r in decompose_results:
                q = r.decomposed_query.strip()
                if q and q not in seen:
                    seen[q] = None
                    queries.append(q)
            logger.debug("_fan_out_queries | decompose added %d", len(decompose_results))
        except Exception as exc:
            logger.warning("decompose_query failed (skipping): %s", exc)

        # --- step-back (macro context) ---
        try:
            step_back_results = step_back_query(
                self._model,
                [base_query],
                num_queries=3,
                few_shot_examples=step_back_examples,
            )
            for r in step_back_results:
                q = r.general_query.strip()
                if q and q not in seen:
                    seen[q] = None
                    queries.append(q)
            logger.debug("_fan_out_queries | step_back added %d", len(step_back_results))
        except Exception as exc:
            logger.warning("step_back_query failed (skipping): %s", exc)

        return queries

    def _run_searches_parallel(self, queries: list[str]) -> list[SearchResult]:
        """
        Execute Tavily searches for all queries concurrently.

        Each search uses ``topic="finance"`` (Requirement 1).  Individual query
        failures are logged at WARNING and skipped; the method never raises.

        Args:
            queries: List of query strings to search.

        Returns:
            Flat list of ``SearchResult`` objects in completion order.
        """
        if not queries:
            return []

        all_results: list[SearchResult] = []
        n_workers = min(len(queries), self.max_workers)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_query = {
                executor.submit(
                    self._searcher.search_tavily, q, "finance"
                ): q
                for q in queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results = future.result()
                    logger.debug(
                        "_run_searches_parallel | query=%r  results=%d", q, len(results)
                    )
                    all_results.extend(results)
                except Exception as exc:
                    logger.warning(
                        "_run_searches_parallel | query=%r failed: %s", q, exc
                    )

        return all_results

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove duplicate results by URL (``result.source``).

        Results with an empty source are always kept (no deduplication key).
        First occurrence wins; subsequent duplicates are dropped.

        Args:
            results: Raw, possibly-duplicate search results.

        Returns:
            Deduplicated list preserving original relative order.
        """
        seen_sources: set[str] = set()
        deduped: list[SearchResult] = []
        for result in results:
            if not result.source:
                # No URL — keep unconditionally (cannot dedup without a key).
                deduped.append(result)
            elif result.source not in seen_sources:
                seen_sources.add(result.source)
                deduped.append(result)
        return deduped

    def _filter_by_domain(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Retain only results whose URL belongs to ``self.trusted_domains``
        (Requirement 4).

        Matching rules:
            - ``netloc == domain`` (exact match, e.g. ``wsj.com``)
            - ``netloc.endswith("." + domain)`` (subdomain match, e.g.
              ``finance.yahoo.com`` when domain is ``finance.yahoo.com``)
            - ``www.`` prefix is stripped from ``netloc`` before comparison.

        Results with an empty source are excluded (no URL to evaluate).
        ``urlparse`` failures are logged at WARNING and the result is skipped.

        Args:
            results: Deduplicated search results.

        Returns:
            Filtered list; may be empty if no results match the allowlist.
        """
        filtered: list[SearchResult] = []
        for result in results:
            if not result.source:
                logger.debug("_filter_by_domain | skipping empty source")
                continue
            try:
                parsed = urlparse(result.source)
                netloc = parsed.netloc.lower()
                # Strip www. so "www.reuters.com" matches "reuters.com".
                if netloc.startswith("www."):
                    netloc = netloc[4:]
                matched = any(
                    netloc == domain or netloc.endswith("." + domain)
                    for domain in self.trusted_domains
                )
                if matched:
                    filtered.append(result)
                else:
                    logger.debug(
                        "_filter_by_domain | excluded %s (netloc=%s)", result.source, netloc
                    )
            except Exception as exc:
                logger.warning(
                    "_filter_by_domain | urlparse failed for %r: %s", result.source, exc
                )
        return filtered

    def _synthesize(
        self,
        symbol: str,
        company_name: Optional[str],
        results: list[SearchResult],
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
