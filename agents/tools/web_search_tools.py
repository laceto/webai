"""
agents/tools/web_search_tools.py
---------------------------------
LangChain tool wrappers around WebSearcher.

Single public function:
    make_web_search_tools(searcher: WebSearcher) -> list[BaseTool]

Returns one tool:
    web_search — run a query against Tavily or OpenAI web-search.

Design invariants:
- Serialisation: results are returned as a numbered markdown list so the LLM
  can read them without further processing.
- No tool raises; errors are surfaced as a plain error string so the agent can
  report them gracefully.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool

from webai import SearchProvider, WebSearcher


def make_web_search_tools(searcher: WebSearcher) -> list[BaseTool]:
    """Return tools backed by *searcher*.

    Using a closure keeps the singleton ``WebSearcher`` instance out of global
    state and makes the dependency explicit at construction time.
    """

    @tool
    def web_search(
        query: str,
        provider: str = "tavily",
        topic: str = "general",
        days: Optional[int] = None,
    ) -> str:
        """Search the web and return relevant results.

        Args:
            query: The search query string.
            provider: Backend to use – "tavily" (default) or "openai".
                      Tavily returns structured snippets; OpenAI returns one
                      synthesised prose answer.
            topic: Tavily topic hint – "general" (default) or "news" for
                   time-sensitive queries.
            days: Restrict results to the last N days (Tavily only, optional).

        Returns:
            Numbered markdown list of results with title, source URL and
            content snippet, or an error message prefixed with "ERROR:".
        """
        try:
            prov = SearchProvider.TAVILY if provider.lower() == "tavily" else SearchProvider.OPENAI
            results = searcher.search(
                query,
                provider=prov,
                topic=topic,
                fallback=True,
            )
            # Pass days through the lower-level method when specified
            if days is not None and prov == SearchProvider.TAVILY:
                results = searcher.search_tavily(query, topic=topic, days=days)

            if not results:
                return "No results found."

            lines: list[str] = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. **{r.title}**")
                lines.append(f"   Source: {r.source}")
                lines.append(f"   {r.content}")
                lines.append("")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    return [web_search]
