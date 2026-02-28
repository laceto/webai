# webai

Python library for web search and LLM-powered financial research. Two independent layers:

1. **`webai.tools`** — unified web search over Tavily and OpenAI, normalized into `SearchResult` objects.
2. **`webai.ticker`** — multi-query Tavily fan-out + structured LLM synthesis for stock tickers and market sectors.

---

## Installation

```bash
pip install -r requirements.txt
# or
poetry install
```

Create a `.env` file with your API keys:

```
TAVILY_API_KEY=tvly-...
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # optional; defaults to gpt-4o-mini
```

---

## Quick start

### Web search (`webai.tools`)

```python
from dotenv import load_dotenv
from webai import WebSearcher, SearchProvider

load_dotenv()

searcher = WebSearcher(provider=SearchProvider.TAVILY, max_results=5)

# Structured results with real titles and URLs
results = searcher.search("NVIDIA earnings 2025", topic="finance")

print(searcher.format_results(results))   # numbered list, title + source + content
print(searcher.get_content_only(results)) # plain text for LLM context
docs = searcher.to_documents(results)     # list[langchain_core.Document]
```

### Equity research (`webai.ticker`)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from webai import TickerResearcher

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
researcher = TickerResearcher(model=model, max_results_per_query=5)

result = researcher.research_ticker("NVDA", company_name="Nvidia")

print(result.sentiment)      # "bullish" | "bearish" | "neutral" | "mixed"
print(result.confidence)     # float 0–1
print(result.key_catalyst)
print(result.risk_factors)   # list[str]
print(result.model_dump_json(indent=2))
```

### Sector research (`webai.ticker`)

```python
result = researcher.research_sector("Semiconductors")

print(result.overall_health)    # "strong" | "weak" | "stable" | "deteriorating" | "mixed"
print(result.key_trends)        # list[str]
print(result.tailwinds)         # list[str]
print(result.headwinds)         # list[str]
print(result.outlook)
print(result.model_dump_json(indent=2))
```

---

## API reference

### `webai.tools`

#### `SearchProvider` (Enum)

| Member | Behaviour |
|---|---|
| `TAVILY` | Returns N structured results with real titles and URLs |
| `OPENAI` | Returns 1 synthesized prose answer |

#### `SearchResult` (dataclass)

| Field | Type | Notes |
|---|---|---|
| `title` | `str` | Page title; `"Web search: <query>"` for OpenAI |
| `content` | `str` | Page snippet or synthesized answer |
| `source` | `str` | Source URL; may be `""` for OpenAI |
| `raw_data` | `dict` | Original provider response |

#### `WebSearcher`

```python
WebSearcher(
    provider=SearchProvider.OPENAI,   # default provider
    tavily_api_key=None,              # falls back to TAVILY_API_KEY env var
    openai_api_key=None,              # falls back to OPENAI_API_KEY env var
    max_results=5,                    # Tavily only
    include_raw_content=False,        # Tavily only
    debug=False,                      # sets webai.tools logger to DEBUG
)
```

| Method | Returns | Notes |
|---|---|---|
| `search_tavily(query, topic)` | `list[SearchResult]` | Direct Tavily call; topic: `"general"` \| `"news"` \| `"finance"` |
| `search_openai(query)` | `list[SearchResult]` | Direct OpenAI call; always 1 result |
| `search(query, provider, topic, fallback)` | `list[SearchResult]` | Unified; retries other provider on failure when `fallback=True` |
| `format_results(results)` | `str` | Numbered list: title, source, content |
| `format_minimal(results)` | `str` | Title + content, no source |
| `format_content_only(results)` | `str` | Content with numbered dividers |
| `get_content_only(results)` | `str` | Newline-joined content string |
| `get_content_list(results)` | `list[str]` | One string per result |
| `get_first_content(results)` | `str \| None` | Top result content |
| `to_documents(results)` | `list[Document]` | LangChain `Document` objects |

---

### `webai.ticker`

#### `TickerResearcher`

```python
TickerResearcher(
    model,                    # any LangChain BaseChatModel
    tavily_api_key=None,      # falls back to TAVILY_API_KEY env var
    openai_api_key=None,
    max_results_per_query=5,
    trusted_domains=None,     # overrides TRUSTED_DOMAINS allowlist
    filter_by_domain=True,
    max_workers=8,            # thread pool size for parallel searches
    debug=False,
)
```

| Method | Returns |
|---|---|
| `research_ticker(symbol, company_name=None)` | `TickerResearch` |
| `research_sector(sector)` | `SectorResearch` |

#### `TickerResearch` fields

| Field | Type |
|---|---|
| `ticker` | `str` |
| `company_name` | `str` |
| `sentiment` | `"bullish" \| "bearish" \| "neutral" \| "mixed"` |
| `confidence` | `float` (0–1) |
| `key_catalyst` | `str` |
| `risk_factors` | `list[str]` |
| `analyst_consensus` | `str` |
| `macro_context` | `str` |
| `sources` | `list[str]` |
| `freshness_warning` | `bool` |

#### `SectorResearch` fields

| Field | Type |
|---|---|
| `sector` | `str` |
| `overall_health` | `"strong" \| "weak" \| "stable" \| "deteriorating" \| "mixed"` |
| `key_trends` | `list[str]` |
| `tailwinds` | `list[str]` |
| `headwinds` | `list[str]` |
| `valuation_note` | `str` |
| `leading_companies` | `list[str]` |
| `macro_sensitivity` | `str` |
| `outlook` | `str` |
| `sources` | `list[str]` |
| `freshness_warning` | `bool` |

---

## User guide notebooks

| Notebook | Covers |
|---|---|
| `tools_guide.ipynb` | `SearchProvider`, `SearchResult`, all 7 `WebSearcher` formatters, error handling |
| `ticker_guide.ipynb` | `TickerResearcher`, both pipelines, model schemas, portfolio end-to-end pattern |
| `usage_example.ipynb` | Original `WebSearcher` demo |

```bash
jupyter notebook tools_guide.ipynb
jupyter notebook ticker_guide.ipynb
```

---

## Pipeline overview (`webai.ticker`)

Both `research_ticker` and `research_sector` run the same 6-step pipeline:

```
1. Build anchor query
      ticker : "{SYMBOL} {company} stock analysis news"
      sector : "{sector} sector financial health outlook"

2. Fan-out via LLM query translation
      expand_query   → paraphrases
      decompose_query → sub-questions
      step_back_query → macro context
      (each step degrades gracefully on failure)

3. Run all queries in parallel against Tavily topic="finance"

4. Deduplicate results by URL

5. Filter to TRUSTED_DOMAINS allowlist
      (falls back to unfiltered if all results removed)

6. Synthesise TickerResearch | SectorResearch via structured LLM output
```

---

## Known issues

- **`query_translation` fan-out** — In practice the library logs `WARNING` and degrades to `[base_query]` only. The graceful-degradation invariant holds; the fan-out benefit is not yet realized.
- **No async support** — `aiohttp` is in `requirements.txt` but unused. Parallel searches run in a `ThreadPoolExecutor`.
- **Empty placeholders** — `webai/utils.py` and `webai/data/` are not yet implemented.

---

## License

MIT License
