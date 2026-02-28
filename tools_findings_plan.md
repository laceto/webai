# Code Review: webai/tools.py

**Review Date:** 2026-02-28
**Reviewer:** Claude Code
**Files Reviewed:** `webai/tools.py`, `usage_examples.py`, `webai/__init__.py`, `webai/utils.py`

---

## Executive Summary

The codebase is a single-file web search abstraction (~490 LOC) with a clean concept: normalize Tavily and OpenAI search results behind a shared `SearchResult` dataclass and offer bidirectional fallback. The data model and type annotations are solid, and the provider enum prevents magic-string bugs.

However, the implementation has one correctness bug (a silently overwritten method), two API contract violations that break the "unified interface" promise (OpenAI returns a fake result structure vs. Tavily's real one), and a debug default that makes the library hostile to production use. Several passed parameters are silently ignored, creating invisible coupling between what users expect and what the code does. These issues are the primary source of cognitive debt.

---

## Findings

### 🔴 Critical Issues (Count: 1)

#### Issue 1: `format_results` is defined twice — second silently overwrites the first
**Severity:** Critical
**Category:** Correctness
**Lines:** 311–328, 408–422

**Description:**
Python classes resolve method names at lookup time using the last definition in the class body. The first `format_results` (line 311) contains a debug print statement that the second (line 408) omits. The first definition is dead code — it can never be reached.

**Current Code:**
```python
# Line 311 — DEAD, never called
def format_results(self, results: list[SearchResult]) -> str:
    """Format search results as readable text."""
    if self.debug:
        print(f"[DEBUG] Formatting {len(results)} results")  # ← only here
    ...

# Line 408 — this is what actually runs
def format_results(self, results: list[SearchResult]) -> str:
    """Format search results as readable text (original method with all info)."""
    ...
```

**Impact:**
- The debug branch in the first definition is silently unreachable.
- Any future maintainer editing the first definition will see no effect and waste time debugging.
- Static analysis tools (mypy, pylint) will warn about this but the warning is easy to miss.

**Recommendation:**
Delete the first definition entirely (lines 311–328). The second definition (line 408) is the correct one and should be kept. Remove the "(original method with all info)" qualifier from its docstring since it is now the only version.

---

### 🟠 High Priority Issues (Count: 4)

#### Issue 2: `debug=True` is the default — partial API keys printed to stdout on every init
**Severity:** High
**Category:** Security / Production-readiness
**Lines:** 39, 64–65, 80–81

**Description:**
The constructor defaults to `debug=True` and unconditionally prints the first 10 characters of each API key to stdout. Since `WebSearcher` is typically created at module load, every application that imports the library will leak key prefixes to standard output — which in containerized or cloud environments is captured in logs, often stored indefinitely.

**Current Code:**
```python
def __init__(self, ..., debug: bool = True):
    ...
    if tavily_key and self.debug:
        print(f"✓ Tavily API key loaded: {tavily_key[:10]}...")

    if openai_key and self.debug:
        print(f"✓ OpenAI API key loaded: {openai_key[:10]}...")
```

**Impact:**
- API key prefixes in application logs (security risk in shared log systems).
- Stdout pollution for any caller that does not explicitly pass `debug=False`.
- All eight result-formatting methods also emit debug prints, making stdout capture in tests and pipelines noisy.

**Recommendation:**
1. Flip the default to `debug: bool = False`.
2. Remove API key prefix printing entirely — confirming a key is loaded does not require showing any characters.
3. Replace all `print()` calls with `logging.getLogger(__name__).debug(...)` so callers can control output via standard log level configuration.

---

#### Issue 3: `search_openai()` produces a structurally different result than `search_tavily()`
**Severity:** High
**Category:** API Contract / Cognitive Debt
**Lines:** 210–233

**Description:**
The library's value proposition is a unified interface. But the two providers produce fundamentally different `SearchResult` objects:

| Field | Tavily | OpenAI |
|---|---|---|
| `title` | Actual page title | User's query string |
| `source` | Real URL | Hardcoded string `"openai-search"` |
| `content` | Snippet from the page | Synthesized LLM prose |
| Result count | Up to `max_results` | Always exactly 1 |

Callers who iterate results and display `result.source` as a link will get a broken UI for OpenAI results. The type signature (`list[SearchResult]`) promises equivalence that the implementation does not deliver.

**Current Code:**
```python
if isinstance(content, str):
    results.append(
        SearchResult(
            title=query,              # ← user's query, not a title
            content=content[:500],
            source="openai-search",   # ← not a URL
            raw_data={"response": str(response)},
        )
    )
```

**Impact:**
- Callers cannot write provider-agnostic result handling.
- Fallback from OpenAI to Tavily (or vice versa) changes result structure mid-stream.
- `to_documents()` metadata will contain a useless `source` for OpenAI results.

**Recommendation:**
Parse the OpenAI response's annotations (URLs cited by the model) to populate `source` properly. If citations are not available, set `source` to `""` rather than a misleading string. Document explicitly in the docstring that OpenAI always returns at most one synthesized result, so callers are not surprised.

---

#### Issue 4: Explicitly passed `tavily_api_key` is accepted but silently ignored
**Severity:** High
**Category:** API Contract
**Lines:** 62–74

**Description:**
The constructor resolves `tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")` and uses it to decide *whether* to init Tavily, but never passes it to `TavilySearch`. The tool reads the env var again internally. So `WebSearcher(tavily_api_key="my-key")` silently falls back to the env var (or fails if it is unset).

**Current Code:**
```python
tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
...
if tavily_key:
    self.tavily_tool = TavilySearch(   # ← tavily_key never passed here
        max_results=max_results,
        include_raw_content=include_raw_content,
    )
```

**Impact:**
- The `tavily_api_key` parameter is a lie — passing it does nothing.
- Testing with a fixture key is impossible without mutating the environment.
- Same pattern applies to `openai_api_key` (line 88: `api_key=openai_key` is correct here, but `tavily_api_key` is not).

**Recommendation:**
Pass the resolved key to `TavilySearch`:
```python
self.tavily_tool = TavilySearch(
    api_key=tavily_key,
    max_results=max_results,
    include_raw_content=include_raw_content,
)
```

---

#### Issue 5: Exception chaining is broken — original stack trace lost on re-raise
**Severity:** High
**Category:** Observability / Debuggability
**Lines:** 164–169, 240–245

**Description:**
Both `search_tavily()` and `search_openai()` catch exceptions and re-raise as `RuntimeError(str(e))`. This creates a new exception object that has no `__cause__`, severing the chain. Debuggers and error reporting tools (Sentry, etc.) lose the original traceback and exception type.

**Current Code:**
```python
except Exception as e:
    print(f"✗ Tavily search error: {e}")
    if self.debug:
        import traceback
        traceback.print_exc()
    raise RuntimeError(f"Tavily search failed: {str(e)}")  # chain broken
```

**Impact:**
- `except RuntimeError` in caller code cannot distinguish network errors from API quota errors from parsing errors.
- `traceback.print_exc()` is only called in debug mode, meaning non-debug production runs lose the stack trace entirely.
- Exception wrapping without chaining is an anti-pattern in Python 3.

**Recommendation:**
```python
raise RuntimeError(f"Tavily search failed: {e}") from e
```
Move `traceback` import to the top of the file. Remove the conditional `traceback.print_exc()` — if the exception is re-raised, the caller's traceback handler will print it.

---

### 🟡 Medium Priority Issues (Count: 4)

#### Issue 6: `include_raw_content` and `topic` parameters are silently ignored for OpenAI
**Severity:** Medium
**Category:** API Contract / Cognitive Debt
**Lines:** 49–50, 254, 276–278

**Description:**
Two constructor parameters have no effect when using the OpenAI provider:
- `include_raw_content` is stored but never read in `search_openai()`.
- `topic` is accepted by `search()` but only forwarded to `search_tavily()`; for OpenAI it is dropped silently.

There is no warning or error to signal this to the caller.

**Recommendation:**
Either support these for OpenAI (inject topic into the search prompt) or emit a `warnings.warn()` when the user passes them with an OpenAI-only context. Document the provider-specific behavior explicitly in `search()` and `__init__()` docstrings.

---

#### Issue 7: `max_results` has no real effect for OpenAI searches
**Severity:** Medium
**Category:** API Contract
**Lines:** 191, 238

**Description:**
The search prompt says "Return the top N results" but the LLM produces a single synthesized string regardless. The `[:self.max_results]` slice at line 238 never trims anything because `results` always has exactly one element. The user-facing parameter appears functional but is not.

**Recommendation:**
Document this limitation clearly. Consider structuring the prompt to request a specific format (e.g., JSON with `N` items) and parsing it, or remove `max_results` from the OpenAI path and document that OpenAI always returns one synthesized answer.

---

#### Issue 8: Content truncated to 500 chars at parse time — full content is unrecoverable from `content` field
**Severity:** Medium
**Category:** Maintainability
**Lines:** 142, 155, 219, 229

**Description:**
`SearchResult.content` is always sliced to 500 characters at construction time. The full content is available in `raw_data`, but callers must know the raw Tavily dict structure to retrieve it — defeating the abstraction. Even setting `include_raw_content=True` does not affect the `content` field.

**Recommendation:**
Store full content in `SearchResult.content`. Move the truncation to the formatting methods (`format_results`, `format_content_only`, etc.) where it is a display concern, or expose a `max_content_length` parameter that defaults to `None` (no truncation).

---

#### Issue 9: `usage_examples.py` executes live API calls at top level
**Severity:** Medium
**Category:** Testability / Developer Experience
**Lines:** 58–113

**Description:**
Lines 58–113 are top-level executable code, not wrapped in `if __name__ == "__main__":`. This means:
- `import usage_examples` fires real API calls.
- Any test that touches this module incurs network I/O and API costs.
- CI pipelines that scan imports will trigger billing.

**Recommendation:**
Wrap all executable code in `if __name__ == "__main__":`. Additionally, remove the duplicate `load_dotenv()` calls (lines 7 and 26 are both present in the commented block).

---

### 🟢 Low Priority Issues (Count: 3)

#### Issue 10: `webai/__init__.py` exports nothing — public API is undefined
**Severity:** Low
**Category:** Maintainability
**Lines:** `webai/__init__.py`

**Description:**
Users must write `from webai.tools import WebSearcher, SearchProvider` rather than `from webai import WebSearcher`. There is no declared public interface.

**Recommendation:**
```python
# webai/__init__.py
from webai.tools import WebSearcher, SearchProvider, SearchResult

__all__ = ["WebSearcher", "SearchProvider", "SearchResult"]
```

---

#### Issue 11: Hardcoded model `gpt-4o-mini` — env var `model` in `.env` is never read
**Severity:** Low
**Category:** Configuration / Flexibility
**Lines:** 88

**Description:**
The `.env` file defines `model = gpt-4.1-nano`, but `tools.py` hardcodes `model="gpt-4o-mini"`. The env var is silently unused.

**Recommendation:**
```python
model = openai_api_key and os.getenv("OPENAI_MODEL", "gpt-4o-mini")
self.llm = ChatOpenAI(model=model, api_key=openai_key)
```

---

#### Issue 12: `lazily imported traceback` inside exception handler
**Severity:** Low
**Category:** Code Style
**Lines:** 102, 167, 243

**Description:**
`import traceback` appears inside exception handlers. `traceback` is a stdlib module with negligible import cost — repeated conditional imports inside hot paths are a code smell and suggest the module was added ad hoc.

**Recommendation:**
Move `import traceback` to the top of the file alongside other imports.

---

## Positive Observations

- The `SearchProvider` enum prevents magic-string bugs and makes the API self-documenting.
- The `SearchResult` dataclass with `raw_data: dict` is the right pattern — it allows structured access while preserving the original API response for advanced use.
- Bidirectional fallback logic is clear and readable (`TAVILY → OPENAI` and `OPENAI → TAVILY`).
- `to_documents()` is a clean LangChain integration point with well-chosen metadata fields.
- Type hints are present on all public methods.
- All public methods have docstrings with `Args` and `Returns` sections.

---

## Action Plan

### Phase 1: Critical & High (This Sprint)
- [ ] Delete the first (dead) `format_results` definition at lines 311–328
- [ ] Flip `debug` default to `False`; remove API key prefix printing; migrate all debug `print()` to `logging.getLogger(__name__).debug()`
- [ ] Fix `TavilySearch` initialization to pass the resolved `tavily_key` explicitly
- [ ] Add `from e` to all `raise RuntimeError(...)` calls; move `import traceback` to top-level

### Phase 2: API Contract (Next Sprint)
- [ ] Document (or fix) the structural difference between OpenAI and Tavily `SearchResult` objects; parse OpenAI response citations for proper `source` URLs
- [ ] Emit `warnings.warn()` when `include_raw_content` or `topic` are passed with an OpenAI provider
- [ ] Document that `max_results` has no effect for OpenAI; consider structured prompt output

### Phase 3: Medium Priority (Sprint 2)
- [ ] Move content truncation from parse time to display time (formatting methods)
- [ ] Wrap `usage_examples.py` executable code in `if __name__ == "__main__":`

### Phase 4: Low Priority (Backlog)
- [ ] Populate `webai/__init__.py` with public exports and `__all__`
- [ ] Read `OPENAI_MODEL` env var instead of hardcoding `gpt-4o-mini`
- [ ] Move all `import traceback` to top of file

---

## Technical Debt Estimate

- **Total Issues:** 12 (1 critical, 4 high, 4 medium, 3 low)
- **Estimated Fix Time:** 6–10 hours
- **Risk Level:** Medium (no data loss risk, but multiple silent API contract violations)
- **Recommended Refactor:** No — all fixes are incremental within the existing file structure. No architectural rewrite needed.
