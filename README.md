# webai

Python library for web search and LLM-powered financial research. Three independent layers:

1. **`webai.tools`** — unified web search over Tavily and OpenAI, normalized into `SearchResult` objects.
2. **`webai.ticker`** — multi-query Tavily fan-out + structured LLM synthesis for stock tickers and market sectors.
3. **`webai.news`** — breaking news extraction, catalyst analysis, and daily market digest synthesis.

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

results = searcher.search("NVIDIA earnings 2025", topic="finance")

print(searcher.format_results(results))   # numbered list: title + source + content
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

result = researcher.research_ticker(
    "NVDA",
    company_name="Nvidia",
    earnings_date="2026-05-28",  # optional; appended to query and recorded in result
    days_back=7,                 # optional; restrict Tavily results to last N days
)

print(result.sentiment)      # "bullish" | "bearish" | "neutral" | "mixed"
print(result.confidence)     # float 0–1
print(result.key_catalyst)
print(result.risk_factors)   # list[str]
print(result.earnings_date)  # "2026-05-28" (echoed back when provided)
print(result.model_dump_json(indent=2))
```

### Portfolio batch research (`webai.ticker`)

```python
# Each entry is a bare ticker or a (ticker, company_name) tuple.
# Failures per-ticker are logged at WARNING and excluded; the batch never aborts.
results: dict = researcher.research_portfolio(
    ["NVDA", ("AAPL", "Apple"), "MSFT"],
    days_back=7,
)
for symbol, r in results.items():
    print(symbol, r.sentiment)
```

### Sector research (`webai.ticker`)

```python
result = researcher.research_sector("Semiconductors")

print(result.overall_health)  # "strong" | "weak" | "stable" | "deteriorating" | "mixed"
print(result.key_trends)      # list[str]
print(result.tailwinds)       # list[str]
print(result.headwinds)       # list[str]
print(result.outlook)
```

### Breaking news (`webai.news`)

```python
from webai import NewsResearcher

nr = NewsResearcher(model=model)

items = nr.fetch_breaking_news(["NVDA", "AAPL"], days_back=1)
for item in items:
    print(item.headline, item.sentiment, item.relevance_score)
```

### Daily digest (`webai.news`)

```python
digest = nr.fetch_daily_digest(
    tickers=["NVDA", "AAPL"],
    sectors=["Technology", "Energy"],
    days_back=1,
)

print(digest.market_theme)
for story in digest.top_stories:
    print(story.headline, story.relevance_score)
for mover in digest.sector_movers:
    print(mover.sector, mover.note)
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
| `search_tavily(query, topic, days=None)` | `list[SearchResult]` | `days` restricts to last N days via Tavily `start_date` |
| `search_openai(query)` | `list[SearchResult]` | Always 1 result |
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
    trusted_domains=None,     # overrides TRUSTED_DOMAINS allowlist (22 entries by default)
    filter_by_domain=True,
    max_workers=8,            # thread pool size for parallel searches
    debug=False,
)
```

| Method | Returns | Notes |
|---|---|---|
| `research_ticker(symbol, company_name=None, earnings_date=None, days_back=None)` | `TickerResearch` | `earnings_date` enriches query and is echoed in result; `days_back` filters Tavily |
| `research_sector(sector)` | `SectorResearch` | |
| `research_portfolio(symbols, days_back=None)` | `dict[str, TickerResearch]` | `symbols` is `list[str \| tuple[str, str]]`; failures excluded, batch never aborts |

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
| `earnings_date` | `str \| None` |

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

### `webai.news`

#### `NewsResearcher`

```python
NewsResearcher(
    model,                    # any LangChain BaseChatModel
    tavily_api_key=None,      # falls back to TAVILY_API_KEY env var; required
    openai_api_key=None,
    max_results_per_query=5,
    trusted_domains=None,     # overrides default allowlist
    filter_by_domain=True,
    max_workers=8,
    debug=False,
)
# Raises ValueError if no Tavily API key is available.
```

| Method | Returns | Notes |
|---|---|---|
| `fetch_breaking_news(tickers, days_back=1)` | `list[NewsItem]` | Sorted by `relevance_score` desc |
| `fetch_macro_news(days_back=1)` | `list[NewsItem]` | Fed, inflation, interest rates; sorted by score |
| `fetch_catalyst_news(ticker, days_back=7)` | `list[NewsItem]` | Finance-topic search; post-filtered to ticker mentions or score ≥ 0.5 |
| `fetch_daily_digest(tickers=None, sectors=None, date=None, days_back=1)` | `NewsDigest` | Raises `RuntimeError` on synthesis failure |

#### `NewsItem` fields

| Field | Type |
|---|---|
| `headline` | `str` |
| `summary` | `str` |
| `source_url` | `str` |
| `published_date` | `str` (YYYY-MM-DD or `""`) |
| `tickers_mentioned` | `list[str]` |
| `event_type` | `"earnings" \| "guidance" \| "analyst_action" \| "macro" \| "regulatory" \| "m_and_a" \| "insider" \| "technical" \| "other"` |
| `sentiment` | `"bullish" \| "bearish" \| "neutral"` |
| `relevance_score` | `float` (0–1; 0.9+ = clearly market-moving) |

#### `NewsDigest` fields

| Field | Type |
|---|---|
| `date` | `str` (YYYY-MM-DD) |
| `top_stories` | `list[NewsItem]` (3–7 highest-relevance) |
| `market_theme` | `str` |
| `sector_movers` | `list[SectorMover]` (up to 5) |
| `macro_events` | `list[str]` |
| `watchlist_alerts` | `list[str]` |

#### `SectorMover` fields

| Field | Type |
|---|---|
| `sector` | `str` |
| `note` | `str` |

---

## User guide notebooks

| Notebook | Covers |
|---|---|
| `tools_guide.ipynb` | `SearchProvider`, `SearchResult`, all 7 `WebSearcher` formatters, error handling |
| `ticker_guide.ipynb` | `TickerResearcher`, `research_ticker`/`research_portfolio`/`research_sector`, model schemas |
| `news_guide.ipynb` | `NewsResearcher`, all four fetch methods, `NewsItem`/`NewsDigest`/`SectorMover` schemas |
| `usage_example.ipynb` | Original `WebSearcher` demo |

```bash
jupyter notebook tools_guide.ipynb
jupyter notebook ticker_guide.ipynb
jupyter notebook news_guide.ipynb
```

---

## Pipeline overview

All three research classes (`TickerResearcher`, `NewsResearcher`) share the same 5-step
pipeline implemented in `webai.utils`:

```
1. Build anchor query (class-specific phrasing)

2. Fan-out via LLM query translation (webai.utils.fan_out_queries)
      expand_query    → paraphrases
      decompose_query → sub-questions
      step_back_query → macro context
      (each step degrades gracefully on failure; always returns >= [base_query])

3. Run all queries in parallel against Tavily (webai.utils.run_searches_parallel)
      topic="finance" for ticker/catalyst research
      topic="news"    for breaking news and digest

4. Deduplicate results by URL (webai.utils.deduplicate)

5. Filter to TRUSTED_DOMAINS allowlist (webai.utils.filter_by_domain)
      (falls back to unfiltered if all results removed)

6. Synthesise via structured LLM output chain
      TickerResearch | SectorResearch | NewsDigest | list[NewsItem]
```

---

## Known issues

- **No async support** — `aiohttp` is in `requirements.txt` but unused. Parallel searches run in a `ThreadPoolExecutor`.
- **`OPENAI_MODEL` env var** — `tools.py` reads `os.getenv("OPENAI_MODEL", "gpt-4o-mini")`; an old `.env` key named `model` has no effect.

---

## License

MIT License
