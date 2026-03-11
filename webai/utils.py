"""
webai.utils — Shared pipeline utilities.

These four module-level functions are the **single source of truth** for the
fan-out / search / deduplicate / filter pipeline that is shared between
``TickerResearcher`` and ``NewsResearcher``.  Neither ``ticker.py`` nor
``news.py`` should re-implement these steps — they delegate here instead.

Import safety
-------------
``utils.py`` only imports from ``webai.tools`` and third-party packages
(``query_translation``, ``langchain_core``).  It does **not** import from
``ticker.py`` or ``news.py``, so there are no circular dependencies.

Invariants
----------
- ``fan_out_queries``      — always returns ``>= [base_query]``; never raises.
- ``run_searches_parallel`` — never raises; individual query failures are WARNING.
- ``deduplicate``           — pure function; never raises.
- ``filter_by_domain``      — never raises; callers handle the empty-return case.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from query_translation import decompose_query, expand_query, step_back_query

from webai.tools import SearchResult, WebSearcher

logger = logging.getLogger(__name__)


def fan_out_queries(
    model: BaseChatModel,
    base_query: str,
    expand_examples: list[dict],
    decompose_examples: list[dict],
    step_back_examples: list[dict],
) -> list[str]:
    """
    Expand *base_query* via LLM-assisted expand / decompose / step-back.

    Each translation step is wrapped in an independent ``try/except`` so a
    failure in one step (e.g. ``expand_query``) does not abort the others.
    Failures are logged at WARNING and skipped.  Duplicate queries are removed
    while preserving insertion order.

    The few-shot examples are supplied by the caller so the same function works
    for ticker research, sector research, and news extraction without hidden
    coupling to any particular example set.

    Args:
        model: Any LangChain ``BaseChatModel`` that supports text invocation.
        base_query: The anchor query string; always index 0 in the output.
        expand_examples: Few-shot examples for ``expand_query``
            (``{"original_query": str, "new_queries": list[str]}``).
        decompose_examples: Few-shot examples for ``decompose_query``.
        step_back_examples: Few-shot examples for ``step_back_query``.

    Returns:
        Deduplicated list of query strings.  At least ``[base_query]``.
    """
    seen: dict[str, None] = {base_query: None}  # ordered-set via dict
    queries: list[str] = [base_query]

    # --- expand (paraphrases) ---
    try:
        for r in expand_query(model, [base_query], few_shot_examples=expand_examples):
            q = r.paraphrased_query.strip()
            if q and q not in seen:
                seen[q] = None
                queries.append(q)
        logger.debug("fan_out_queries | expand done, total=%d", len(queries))
    except Exception as exc:
        logger.warning("fan_out_queries | expand_query failed (skipping): %s", exc)

    # --- decompose (sub-questions) ---
    try:
        for r in decompose_query(model, [base_query], few_shot_examples=decompose_examples):
            q = r.decomposed_query.strip()
            if q and q not in seen:
                seen[q] = None
                queries.append(q)
        logger.debug("fan_out_queries | decompose done, total=%d", len(queries))
    except Exception as exc:
        logger.warning("fan_out_queries | decompose_query failed (skipping): %s", exc)

    # --- step-back (macro context) ---
    try:
        for r in step_back_query(
            model, [base_query], num_queries=3, few_shot_examples=step_back_examples
        ):
            q = r.general_query.strip()
            if q and q not in seen:
                seen[q] = None
                queries.append(q)
        logger.debug("fan_out_queries | step_back done, total=%d", len(queries))
    except Exception as exc:
        logger.warning("fan_out_queries | step_back_query failed (skipping): %s", exc)

    return queries


def run_searches_parallel(
    searcher: WebSearcher,
    queries: list[str],
    topic: Literal["general", "news", "finance"] = "finance",
    max_workers: int = 8,
    days: Optional[int] = None,
) -> list[SearchResult]:
    """
    Execute all *queries* concurrently against the Tavily backend.

    Individual query failures are logged at WARNING and skipped; the function
    never raises.  Results are returned in completion order (non-deterministic).

    Args:
        searcher: An initialised ``WebSearcher`` with a Tavily client.
        queries: Query strings to execute.
        topic: Tavily topic category — ``"general"``, ``"news"``, or
            ``"finance"``.
        max_workers: Thread-pool size; capped at ``len(queries)``.
        days: When set, passed to ``search_tavily`` to filter results to the
            last *N* days via Tavily's ``start_date`` parameter.  ``None``
            applies no date restriction.

    Returns:
        Flat list of ``SearchResult`` objects in completion order.
    """
    if not queries:
        return []

    all_results: list[SearchResult] = []
    n_workers = min(len(queries), max_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_query = {
            executor.submit(searcher.search_tavily, q, topic, days): q
            for q in queries
        }
        for future in as_completed(future_to_query):
            q = future_to_query[future]
            try:
                results = future.result()
                logger.debug(
                    "run_searches_parallel | query=%r  results=%d", q, len(results)
                )
                all_results.extend(results)
            except Exception as exc:
                logger.warning(
                    "run_searches_parallel | query=%r failed: %s", q, exc
                )

    return all_results


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove duplicate search results by URL (``result.source``).

    Results with an empty source are always kept (no deduplication key is
    available).  First occurrence wins; subsequent duplicates are dropped.

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


def filter_by_domain(
    results: list[SearchResult],
    trusted_domains: list[str],
) -> list[SearchResult]:
    """
    Retain only results whose URL belongs to *trusted_domains*.

    Matching rules (applied after stripping a leading ``www.`` from the netloc):

    - ``netloc == domain``                    (exact, e.g. ``wsj.com``)
    - ``netloc.endswith("." + domain)``       (subdomain, e.g.
      ``finance.yahoo.com`` when domain is ``finance.yahoo.com``)

    Results with an empty source are excluded (no URL to evaluate).
    ``urlparse`` failures are logged at WARNING and the result is skipped.
    The function never raises.

    Args:
        results: Deduplicated search results.
        trusted_domains: Allowlist of domain strings.

    Returns:
        Filtered list; may be empty if no results match the allowlist.
        Callers are responsible for handling the empty-return fallback.
    """
    filtered: list[SearchResult] = []
    for result in results:
        if not result.source:
            logger.debug("filter_by_domain | skipping empty source")
            continue
        try:
            parsed = urlparse(result.source)
            netloc = parsed.netloc.lower()
            # Strip www. so "www.reuters.com" matches "reuters.com".
            if netloc.startswith("www."):
                netloc = netloc[4:]
            matched = any(
                netloc == domain or netloc.endswith("." + domain)
                for domain in trusted_domains
            )
            if matched:
                filtered.append(result)
            else:
                logger.debug(
                    "filter_by_domain | excluded %s (netloc=%s)",
                    result.source,
                    netloc,
                )
        except Exception as exc:
            logger.warning(
                "filter_by_domain | urlparse failed for %r: %s", result.source, exc
            )
    return filtered
