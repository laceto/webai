# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**webai** is a Python library (v0.1.0) with two layers:

1. **Web search** (`webai/tools.py`) — unified interface over Tavily and OpenAI web search,
   normalizing results into a `SearchResult` dataclass with optional LangChain `Document` conversion.
2. **Research pipelines** (`webai/ticker.py`) — multi-query Tavily fan-out + LLM structured-output
   synthesis for individual stock tickers (`research_ticker`) and market sectors (`research_sector`).

## Setup & Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or with Poetry
poetry install

# Load environment variables (required before running)
# Needs OPENAI_API_KEY and TAVILY_API_KEY in .env

# Run usage examples
python usage_examples.py

# Interactive notebooks
jupyter notebook usage_example.ipynb      # WebSearcher demo
jupyter notebook tools_guide.ipynb        # webai.tools user guide
jupyter notebook ticker_guide.ipynb       # webai.ticker user guide
```

There is no test suite or lint configuration yet.

## Architecture

### `webai/tools.py` — Web search

| Symbol | Role |
|---|---|
| `SearchProvider` (Enum) | Selects backend: `TAVILY` or `OPENAI` |
| `SearchResult` (dataclass) | Normalized result: `title`, `content`, `source`, `raw_data` |
| `WebSearcher` | Main class — initialize once, call `search()` repeatedly |

**Initialization:** Both provider clients are initialized at construction time when their respective
API keys are present.  Missing keys are silently skipped; a `ValueError` is raised at search time
if the missing provider is called.  `__init__` never raises.

**Search flow:**
1. `search(query, provider, topic, fallback)` dispatches to `search_tavily()` or `search_openai()`
2. On failure with `fallback=True`, it automatically retries with the other provider
3. Tavily returns structured results (real titles, URLs, snippets); OpenAI returns one synthesized
   prose answer wrapped as a single `SearchResult`
4. No content truncation is applied at parse time

**Result formatting methods on `WebSearcher`:**
- `format_results()` — numbered list with title, source URL, content
- `format_minimal()` — title + content, no source
- `format_content_only()` — content only, with numbered dividers
- `get_content_only()` — plain newline-joined string
- `get_content_list()` — `list[str]`
- `get_first_content()` — `str | None`
- `to_documents()` — converts to `list[langchain_core.documents.Document]`

**Debug mode:** Pass `debug=True` to set `webai.tools` logger to `DEBUG` and attach a
`StreamHandler`.  This is a backwards-compat shortcut; prefer configuring logging externally.

---

### `webai/ticker.py` — Research pipelines

| Symbol | Role |
|---|---|
| `TickerResearch` (Pydantic) | Validated equity snapshot: sentiment, confidence, key_catalyst, risk_factors, sources, … |
| `SectorResearch` (Pydantic) | Validated sector snapshot: overall_health, key_trends, tailwinds, headwinds, outlook, … |
| `TickerResearcher` | Orchestrator — `research_ticker(symbol)` and `research_sector(sector)` |
| `TRUSTED_DOMAINS` | Module-level domain allowlist for result filtering |

**Both pipelines share the same steps (1–5); only base-query phrase, few-shot examples, and
synthesis chain differ:**

1. Build anchor query
   - Ticker: `"{SYMBOL} {company_name} stock analysis news"`
   - Sector: `"{sector} sector financial health outlook"`
2. `_fan_out_queries(base, expand_examples, decompose_examples, step_back_examples)`
   — LLM-assisted expand + decompose + step-back; each step in try/except; always returns ≥ `[base_query]`
3. `_run_searches_parallel` — ThreadPoolExecutor, `search_tavily(q, topic="finance")`
4. `_deduplicate` — by URL; empty-source results always kept
5. `_filter_by_domain` — `TRUSTED_DOMAINS` allowlist; falls back to unfiltered if all results removed
6. Synthesis — `model.with_structured_output(TickerResearch|SectorResearch).invoke(prompt)`
   — two chains built once in `__init__`: `_synthesis_chain` and `_sector_synthesis_chain`

**Invariants:**
- `_fan_out_queries` always returns ≥ `[base_query]`
- `_run_searches_parallel` returns empty list, never raises
- `_filter_by_domain` returns empty list, never raises; falls back in `research_*`
- `_synthesize` / `_synthesize_sector` always raise on failure; never return partial/None

**Few-shot example constants (module-level):**
- `_FINANCE_EXPAND_EXAMPLES`, `_FINANCE_DECOMPOSE_EXAMPLES`, `_FINANCE_STEP_BACK_EXAMPLES` — ticker
- `_SECTOR_EXPAND_EXAMPLES`, `_SECTOR_DECOMPOSE_EXAMPLES`, `_SECTOR_STEP_BACK_EXAMPLES` — sector

---

### `webai/__init__.py` — Public API (6 exports)

```python
from webai import (
    WebSearcher, SearchProvider, SearchResult,      # tools layer
    TickerResearcher, TickerResearch, SectorResearch,  # research layer
)
```

### Empty placeholders (not yet implemented)

- `webai/utils.py` — empty
- `webai/data/` — empty directory

## Known Issues

- **`OPENAI_MODEL` env var** — `tools.py` reads `os.getenv("OPENAI_MODEL", "gpt-4o-mini")`, so the
  model is configurable via environment.  The old `model` key documented in early `.env` examples
  (`gpt-4.1-nano`) has no effect; rename it to `OPENAI_MODEL` to take effect.
- **No async support** — `aiohttp` is listed in `requirements.txt` but unused.  All Tavily searches
  run synchronously within a `ThreadPoolExecutor`.
- **`query_translation` fan-out degrades silently** — In practice, `expand_query`, `decompose_query`,
  and `step_back_query` each log a `WARNING` and the pipeline continues with just `[base_query]`
  when the library returns a bare list rather than objects with `.paraphrased_query` /
  `.decomposed_query` / `.general_query` attributes.  The graceful-degradation invariant holds, but
  the fan-out benefit is lost until this upstream issue is resolved.
