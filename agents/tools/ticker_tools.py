"""
agents/tools/ticker_tools.py
-----------------------------
LangChain tool wrappers around TickerResearcher.

Single public function:
    make_ticker_tools(researcher: TickerResearcher) -> list[BaseTool]

Returns three tools:
    research_ticker      — deep dive on a single equity symbol
    research_portfolio   — parallel research across multiple symbols
    research_sector      — sector-level health and outlook analysis

Design invariants:
- Pydantic models are serialised to indented JSON so the LLM can quote
  individual fields in its response.
- Failures per tool are returned as "ERROR: …" strings; the orchestrator
  agent can relay them to the user without crashing.
- portfolio symbols are passed as a comma-separated string because LangChain
  tool arguments must be simple scalars or strings for reliable parsing.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from webai import TickerResearcher


def make_ticker_tools(researcher: TickerResearcher) -> list[BaseTool]:
    """Return tools backed by *researcher*."""

    @tool
    def research_ticker(
        symbol: str,
        company_name: Optional[str] = None,
        earnings_date: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> str:
        """Research a single stock ticker and return a structured analysis.

        Runs a multi-query Tavily fan-out and synthesises the results via an
        LLM into a TickerResearch snapshot with sentiment, confidence,
        key catalyst, risk factors, and more.

        Args:
            symbol: Uppercase equity ticker, e.g. "NVDA".
            company_name: Optional company name to improve search quality.
            earnings_date: Optional upcoming earnings date (ISO string or
                           descriptive, e.g. "2025-07-23") appended to the
                           search query and referenced in the synthesis prompt.
            days_back: Restrict searches to the last N days (optional).

        Returns:
            Indented JSON string of TickerResearch fields, or "ERROR: …".
        """
        try:
            result = researcher.research_ticker(
                symbol,
                company_name=company_name,
                earnings_date=earnings_date,
                days_back=days_back,
            )
            return result.model_dump_json(indent=2)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    @tool
    def research_portfolio(
        symbols: str,
        days_back: Optional[int] = None,
    ) -> str:
        """Research multiple tickers in parallel and return a batch analysis.

        Args:
            symbols: Comma-separated list of ticker symbols, optionally with
                     company names separated by a pipe, e.g.:
                     "NVDA|NVIDIA, AAPL|Apple, MSFT"
            days_back: Restrict searches to the last N days (optional).

        Returns:
            Indented JSON object keyed by uppercase symbol, or "ERROR: …".
        """
        try:
            parsed: list = []
            for part in symbols.split(","):
                part = part.strip()
                if "|" in part:
                    sym, name = part.split("|", 1)
                    parsed.append((sym.strip().upper(), name.strip()))
                else:
                    parsed.append(part.upper())

            results = researcher.research_portfolio(parsed, days_back=days_back)
            return json.dumps(
                {sym: res.model_dump() for sym, res in results.items()},
                indent=2,
                default=str,
            )
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    @tool
    def research_sector(
        sector: str,
        days_back: Optional[int] = None,
    ) -> str:
        """Research a market sector and return a structured health snapshot.

        Produces a SectorResearch model with overall_health, key_trends,
        tailwinds, headwinds, and outlook.

        Args:
            sector: Sector name, e.g. "Technology", "Healthcare", "Energy".
            days_back: Restrict searches to the last N days (optional).

        Returns:
            Indented JSON string of SectorResearch fields, or "ERROR: …".
        """
        try:
            result = researcher.research_sector(sector, days_back=days_back)
            return result.model_dump_json(indent=2)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    return [research_ticker, research_portfolio, research_sector]
