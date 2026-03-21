"""
agents/webai_agent.py
----------------------
Factory that builds the webai Deep Agent with three specialised sub-agents.

Public API:
    build_webai_agent(
        model,
        openai_api_key,
        tavily_api_key,
        *,
        openai_model="gpt-4o-mini",
    ) -> CompiledGraph

Architecture
------------

                        ┌─────────────────────────┐
                        │    Orchestrator Agent    │
                        │  (routes via task tool)  │
                        └──────┬──────┬───────┬───┘
                               │      │       │
               ┌───────────────┘      │       └───────────────┐
               ▼                      ▼                        ▼
   ┌────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
   │  web-search-agent  │ │   ticker-agent       │ │    news-agent       │
   │  ─ web_search      │ │  ─ research_ticker   │ │  ─ fetch_breaking   │
   │                    │ │  ─ research_portfolio│ │  ─ fetch_macro      │
   │                    │ │  ─ research_sector   │ │  ─ fetch_catalyst   │
   └────────────────────┘ └─────────────────────┘ │  ─ fetch_daily      │
                                                   └─────────────────────┘

All three sub-agents are stateless (ephemeral) — each `task` call starts a
fresh context.  The orchestrator coordinates their outputs.

Invariants
----------
- build_webai_agent never raises on valid inputs.
- Every sub-agent description is detailed enough for the orchestrator to
  select the right one without ambiguity.
- Sub-agents catch their own exceptions and return "ERROR: …" strings; the
  orchestrator surfaces those to the user.
"""

from __future__ import annotations

import os
from typing import Optional

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from webai import NewsResearcher, TickerResearcher, WebSearcher

from agents.tools.news_tools import make_news_tools
from agents.tools.ticker_tools import make_ticker_tools
from agents.tools.web_search_tools import make_web_search_tools

# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------
_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a financial research orchestrator powered by the webai library.
You coordinate three specialised sub-agents and synthesise their outputs
into clear, actionable answers.

## Sub-agents and when to use them

### web-search-agent
Use for raw web searches when you need fresh information that is not specific
to equity research or news pipelines.  Examples:
- "Search for the latest announcements about Apple"
- "What does the SEC say about short-selling rules?"

### ticker-agent
Use for structured equity and sector research backed by multi-query Tavily
fan-out + LLM synthesis.  Examples:
- "Research NVDA" → research_ticker
- "Give me a portfolio snapshot for AAPL, MSFT, GOOGL" → research_portfolio
- "How is the semiconductor sector doing?" → research_sector

### news-agent
Use for time-sensitive news pipelines.  Examples:
- "What's breaking today for TSLA and NVDA?" → fetch_breaking_news
- "Give me macro news from the last 2 days" → fetch_macro_news
- "What catalysts are driving AMD this week?" → fetch_catalyst_news
- "Build me a daily digest for my watchlist" → fetch_daily_digest

## Coordination rules
1. Plan first — use write_todos for multi-step requests.
2. Delegate via task — always pass complete, self-contained instructions.
3. Synthesise — combine sub-agent outputs into a single coherent answer.
4. Surface errors gracefully — if a sub-agent returns "ERROR: …", tell the
   user what failed and suggest an alternative if possible.
"""

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_webai_agent(
    model: str,
    openai_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    *,
    openai_model: str = "gpt-4o-mini",
) -> object:
    """Build and return the webai orchestrator agent.

    Args:
        model: LangChain model string for the orchestrator and sub-agents,
               e.g. "claude-sonnet-4-6" or "gpt-4o".
        openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        tavily_api_key: Tavily API key (falls back to TAVILY_API_KEY env var).
        openai_model: OpenAI model name used by TickerResearcher and
                      NewsResearcher for structured-output synthesis
                      (default "gpt-4o-mini").

    Returns:
        A compiled LangGraph / Deep Agents graph.  Invoke with::

            agent.invoke(
                {"messages": [{"role": "user", "content": "..."}]},
                config={"configurable": {"thread_id": "session-1"}},
            )

    Raises:
        ValueError: If neither API key argument nor the corresponding env var
                    is set for Tavily (required by NewsResearcher).
    """
    openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

    # ------------------------------------------------------------------
    # Shared webai clients (singletons — initialised once, reused across
    # all tool calls within the session)
    # ------------------------------------------------------------------
    searcher = WebSearcher(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
    )

    # TickerResearcher / NewsResearcher need a ChatOpenAI-compatible model
    from langchain_openai import ChatOpenAI

    synthesis_llm = ChatOpenAI(
        model=openai_model,
        api_key=openai_api_key,
    )

    ticker_researcher = TickerResearcher(
        model=synthesis_llm,
        tavily_api_key=tavily_api_key,
        openai_api_key=openai_api_key,
    )

    news_researcher = NewsResearcher(
        model=synthesis_llm,
        tavily_api_key=tavily_api_key,
        openai_api_key=openai_api_key,
    )

    # ------------------------------------------------------------------
    # Tool lists (one per sub-agent)
    # ------------------------------------------------------------------
    web_tools = make_web_search_tools(searcher)
    ticker_tools = make_ticker_tools(ticker_researcher)
    news_tools = make_news_tools(news_researcher)

    # ------------------------------------------------------------------
    # Sub-agent definitions
    # ------------------------------------------------------------------
    subagents = [
        {
            "name": "web-search-agent",
            "description": (
                "Performs raw web searches using Tavily or OpenAI. "
                "Use for general research, SEC filings, regulatory news, "
                "or any query that does not fit ticker or news pipelines."
            ),
            "system_prompt": (
                "You are a web research specialist. "
                "Search thoroughly and return a concise, sourced summary. "
                "Always include source URLs in your final answer."
            ),
            "tools": web_tools,
        },
        {
            "name": "ticker-agent",
            "description": (
                "Structured equity and sector research via multi-query fan-out "
                "synthesis. Use for TickerResearch snapshots, portfolio analysis, "
                "and sector health reports."
            ),
            "system_prompt": (
                "You are an equity research specialist. "
                "Call the appropriate research tool, then return a structured "
                "summary covering sentiment, confidence, key catalyst, risk "
                "factors, and sources. Quote the JSON fields verbatim where "
                "helpful."
            ),
            "tools": ticker_tools,
        },
        {
            "name": "news-agent",
            "description": (
                "Time-sensitive news pipelines for breaking news, macro events, "
                "catalyst analysis, and daily market digests. "
                "Use when the user asks for recent news, catalysts, or a digest."
            ),
            "system_prompt": (
                "You are a financial news specialist. "
                "Call the most appropriate fetch tool, then return a concise "
                "summary ordered by relevance. For digests, highlight the "
                "top stories and market theme."
            ),
            "tools": news_tools,
        },
    ]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    agent = create_deep_agent(
        name="webai-orchestrator",
        model=model,
        system_prompt=_ORCHESTRATOR_SYSTEM_PROMPT,
        subagents=subagents,
        checkpointer=MemorySaver(),
    )

    return agent
