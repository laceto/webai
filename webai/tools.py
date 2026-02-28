import logging
import os
import traceback  # noqa: F401 — imported at top; used via exc_info in logger calls
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """Supported web search providers."""

    TAVILY = "tavily"
    OPENAI = "openai"


@dataclass
class SearchResult:
    """
    Standardized search result.

    Fields:
        title:    Human-readable title.  For OpenAI results this is a
                  generated label (``"Web search: <query>"``), not a real
                  page title.
        content:  Full result content — no truncation applied at parse time.
        source:   URL string.  For OpenAI results this may be an empty string
                  if the response did not include URL citations.
        raw_data: Original provider response dict for advanced use.
    """

    title: str
    content: str
    source: str
    raw_data: dict


class WebSearcher:
    """
    Unified web search interface supporting Tavily and OpenAI web search.

    Provider differences
    --------------------
    **Tavily** returns structured results with real titles, source URLs, and
    page snippets.  ``topic``, ``include_raw_content``, and ``max_results``
    all work as documented.

    **OpenAI** synthesizes a single prose answer using its built-in web-search
    tool.  ``topic``, ``include_raw_content``, and ``max_results`` have no
    effect when using the OpenAI provider — a :class:`UserWarning` is emitted
    if non-default values are passed.

    Debug logging
    -------------
    Enable per-module debug output via Python's standard logging API::

        import logging
        logging.basicConfig(level=logging.DEBUG)

    The legacy ``debug=True`` parameter is kept for backwards compatibility —
    it sets this module's logger to ``DEBUG`` level at construction time.
    """

    def __init__(
        self,
        provider: SearchProvider = SearchProvider.OPENAI,
        tavily_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        max_results: int = 5,
        include_raw_content: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the web searcher.

        Args:
            provider: Default search provider (``TAVILY`` or ``OPENAI``).
            tavily_api_key: Tavily API key.  Falls back to the
                ``TAVILY_API_KEY`` environment variable when omitted.
            openai_api_key: OpenAI API key.  Falls back to the
                ``OPENAI_API_KEY`` environment variable when omitted.
            max_results: Maximum results to return.  Has no effect for the
                OpenAI provider.
            include_raw_content: Return full page content (Tavily only).
                Has no effect for the OpenAI provider.
            debug: Backwards-compatibility shortcut.  When ``True``, sets
                this module's logger to ``DEBUG`` level and attaches a
                :class:`logging.StreamHandler` if none exist.  Prefer
                configuring logging externally instead.
        """
        self.provider = provider
        self.max_results = max_results
        self.include_raw_content = include_raw_content

        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                logger.addHandler(logging.StreamHandler())

        self.tavily_tool: Optional[TavilySearch] = None
        self.llm: Optional[ChatOpenAI] = None
        self.llm_with_search = None

        # --- Tavily ---
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                self.tavily_tool = TavilySearch(
                    api_key=tavily_key,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                )
                logger.debug("Tavily tool initialized successfully.")
            except Exception as e:
                logger.warning("Failed to initialize Tavily: %s", e, exc_info=True)
        else:
            logger.debug("No Tavily API key found — Tavily provider unavailable.")

        # --- OpenAI ---
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if openai_key:
            try:
                self.llm = ChatOpenAI(model=model_name, api_key=openai_key)
                self.llm_with_search = self.llm.bind_tools(
                    [{"type": "web_search_preview"}]
                )
                logger.debug(
                    "ChatOpenAI (%s) initialized with web search tool.", model_name
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize OpenAI: %s", e, exc_info=True
                )
        else:
            logger.debug("No OpenAI API key found — OpenAI provider unavailable.")

    # ── Provider-specific search methods ────────────────────────────────────

    def search_tavily(
        self,
        query: str,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> list[SearchResult]:
        """
        Search using the Tavily API.

        Args:
            query: Search query string.
            topic: Tavily topic category — ``"general"``, ``"news"``, or
                ``"finance"``.

        Returns:
            Up to ``max_results`` standardized search results with real titles,
            source URLs, and full page snippets.

        Raises:
            ValueError: If Tavily was not initialized (missing API key).
            RuntimeError: If the Tavily API call fails.
        """
        if not self.tavily_tool:
            raise ValueError(
                "Tavily is not initialized. Provide a tavily_api_key or set "
                "the TAVILY_API_KEY environment variable."
            )

        logger.debug("search_tavily | query=%r  topic=%s", query, topic)

        try:
            response = self.tavily_tool.invoke({"query": query, "topic": topic})
            logger.debug(
                "search_tavily | response type=%s", type(response).__name__
            )

            raw_items: list[dict] = []
            if isinstance(response, dict):
                raw_items = response.get("results", [])
            elif isinstance(response, list):
                raw_items = [r for r in response if isinstance(r, dict)]

            results = [
                SearchResult(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    source=item.get("url", ""),
                    raw_data=item,
                )
                for item in raw_items
            ]

            results = results[: self.max_results]
            logger.debug("search_tavily | returning %d results", len(results))
            return results

        except Exception as e:
            logger.error("search_tavily failed: %s", e, exc_info=True)
            raise RuntimeError(f"Tavily search failed: {e}") from e

    def search_openai(self, query: str) -> list[SearchResult]:
        """
        Search using OpenAI's native web search tool.

        Note:
            OpenAI synthesizes a **single prose answer** from its web-search
            tool regardless of ``max_results``.  The result's ``title`` is a
            generated label and ``source`` will be an empty string unless URL
            citations are present in the response.

        Args:
            query: Search query string.

        Returns:
            A list containing one synthesized ``SearchResult``.

        Raises:
            ValueError: If OpenAI was not initialized (missing API key).
            RuntimeError: If the OpenAI API call fails.
        """
        if not self.llm_with_search:
            raise ValueError(
                "OpenAI is not initialized. Provide an openai_api_key or set "
                "the OPENAI_API_KEY environment variable."
            )

        logger.debug("search_openai | query=%r", query)

        try:
            response = self.llm_with_search.invoke(
                f"Search the web for: {query}. Summarize the most relevant findings."
            )

            content = getattr(response, "content", None)
            logger.debug(
                "search_openai | response content type=%s",
                type(content).__name__,
            )

            results: list[SearchResult] = []

            if isinstance(content, str) and content:
                source = self._extract_openai_source(response)
                results.append(
                    SearchResult(
                        title=f"Web search: {query}",
                        content=content,
                        source=source,
                        raw_data={"response": str(response)},
                    )
                )
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        source = self._extract_openai_source(response)
                        results.append(
                            SearchResult(
                                title=f"Web search: {query}",
                                content=item.get("text", ""),
                                source=source,
                                raw_data=item,
                            )
                        )

            logger.debug("search_openai | returning %d results", len(results))
            return results

        except Exception as e:
            logger.error("search_openai failed: %s", e, exc_info=True)
            raise RuntimeError(f"OpenAI search failed: {e}") from e

    @staticmethod
    def _extract_openai_source(response: object) -> str:
        """
        Attempt to extract the first URL citation from an OpenAI response.

        LangChain surfaces OpenAI annotations in ``AIMessage.additional_kwargs``
        as a list of dicts with ``{"type": "url_citation", "url": "..."}``.
        Returns an empty string if no URL annotation is found.
        """
        try:
            additional = getattr(response, "additional_kwargs", {})
            for ann in additional.get("annotations", []):
                if isinstance(ann, dict) and ann.get("type") == "url_citation":
                    url = ann.get("url", "")
                    if url:
                        return url
        except Exception:
            pass
        return ""

    # ── Unified search entry point ───────────────────────────────────────────

    def search(
        self,
        query: str,
        provider: Optional[SearchProvider] = None,
        topic: Literal["general", "news", "finance"] = "general",
        fallback: bool = True,
    ) -> list[SearchResult]:
        """
        Perform a web search with optional provider override and automatic
        fallback.

        Args:
            query: Search query string.
            provider: Override the instance-level default provider for this
                call only.
            topic: Tavily topic category.  Has no effect when the resolved
                provider is OpenAI — a :class:`UserWarning` is emitted if a
                non-default value is passed.
            fallback: When ``True``, retry with the other provider if the
                primary one fails.

        Returns:
            List of search results from whichever provider succeeded.

        Raises:
            RuntimeError: If the primary provider fails and either ``fallback``
                is ``False`` or the fallback provider also fails.
        """
        selected = provider or self.provider

        # Warn about OpenAI-unsupported parameters so callers are not silently
        # misled into thinking their configuration had an effect.
        if selected == SearchProvider.OPENAI:
            if topic != "general":
                warnings.warn(
                    f"topic={topic!r} has no effect when using the OpenAI "
                    "provider.  Set provider=SearchProvider.TAVILY to use "
                    "topic filtering.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.include_raw_content:
                warnings.warn(
                    "include_raw_content=True has no effect when using the "
                    "OpenAI provider.",
                    UserWarning,
                    stacklevel=2,
                )

        logger.debug(
            "search | query=%r  provider=%s  topic=%s  fallback=%s",
            query, selected, topic, fallback,
        )

        try:
            if selected == SearchProvider.TAVILY:
                return self.search_tavily(query, topic=topic)
            elif selected == SearchProvider.OPENAI:
                return self.search_openai(query)
            else:
                raise ValueError(f"Unknown provider: {selected!r}")

        except Exception as primary_error:
            logger.debug("Primary provider (%s) failed: %s", selected, primary_error)

            if not fallback:
                raise

            fallback_provider = (
                SearchProvider.OPENAI
                if selected == SearchProvider.TAVILY
                else SearchProvider.TAVILY
            )
            logger.debug("Trying fallback provider: %s", fallback_provider)

            try:
                if fallback_provider == SearchProvider.TAVILY:
                    return self.search_tavily(query, topic=topic)
                else:
                    return self.search_openai(query)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Both providers failed.\n"
                    f"  Primary   ({selected.value}): {primary_error}\n"
                    f"  Fallback ({fallback_provider.value}): {fallback_error}"
                ) from fallback_error

    # ── Result formatting ────────────────────────────────────────────────────

    def format_results(self, results: list[SearchResult]) -> str:
        """
        Format results as a numbered list with title, source URL, and content.
        """
        if not results:
            return "No results found."
        lines = []
        for i, result in enumerate(results, 1):
            lines.append(
                f"{i}. {result.title}\n"
                f"   Source: {result.source}\n"
                f"   {result.content}\n"
            )
        return "\n".join(lines)

    def format_minimal(self, results: list[SearchResult]) -> str:
        """
        Format results with title and content only (no source URL).
        """
        if not results:
            return "No results found."
        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.title}")
            lines.append(result.content)
            lines.append("")
        return "\n".join(lines)

    def format_content_only(self, results: list[SearchResult]) -> str:
        """
        Format results with content only, separated by numbered dividers.
        """
        if not results:
            return "No results found."
        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(result.content)
            lines.append("")
        return "\n".join(lines)

    def get_content_only(self, results: list[SearchResult]) -> str:
        """
        Return all result content joined by newlines.
        """
        return "\n".join(result.content for result in results)

    def get_content_list(self, results: list[SearchResult]) -> list[str]:
        """
        Return result content as a list of strings (one per result).
        """
        return [result.content for result in results]

    def get_first_content(self, results: list[SearchResult]) -> Optional[str]:
        """
        Return the content of the first result, or ``None`` if empty.
        """
        return results[0].content if results else None

    def to_documents(self, results: list[SearchResult]) -> list[Document]:
        """
        Convert results to LangChain ``Document`` objects.

        Each ``Document`` has:
            - ``page_content``: the result content.
            - ``metadata``: ``{"title": ..., "source": ..., "raw_data": ...}``.
        """
        return [
            Document(
                page_content=result.content,
                metadata={
                    "title": result.title,
                    "source": result.source,
                    "raw_data": result.raw_data,
                },
            )
            for result in results
        ]
