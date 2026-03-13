"""
webai Streamlit demo — covers all three public layers:
  1. Web Search  (WebSearcher)
  2. Ticker & Sector Research  (TickerResearcher)
  3. News  (NewsResearcher)

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="webai demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — API keys + researcher instantiation (A2.1 / A2.2)
# ---------------------------------------------------------------------------
openai_key = os.getenv("OPENAI_API_KEY", "")
tavily_key = os.getenv("TAVILY_API_KEY", "")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

with st.sidebar:
    st.title("🔑 API Keys")
    st.markdown(f"OpenAI: {'✅ set' if openai_key else '❌ missing'}")
    st.markdown(f"Tavily: {'✅ set' if tavily_key else '❌ missing'}")
    st.caption(f"Model: `{openai_model}`")
    st.caption("Keys loaded from `.env`")
    st.divider()
    max_results = st.number_input(
        "Docs per query", min_value=1, max_value=20, value=5, step=1,
        help="Number of Tavily results fetched per search query (max_results / max_results_per_query).",
        key="max_results_input",
    )
    init_btn = st.button("Reinitialize", use_container_width=True)


def _init_researchers(oai_key: str, tav_key: str, model: str, max_results: int = 5) -> dict:
    """Instantiate all three researchers; return a status dict."""
    from langchain_openai import ChatOpenAI
    from webai import (
        NewsResearcher,
        SearchProvider,
        TickerResearcher,
        WebSearcher,
    )

    errors: list[str] = []
    result: dict = {}

    # WebSearcher — never raises; missing keys just disable that provider
    result["searcher"] = WebSearcher(
        tavily_api_key=tav_key or None,
        openai_api_key=oai_key or None,
        max_results=max_results,
    )

    # LLM — needed for TickerResearcher and NewsResearcher
    if oai_key:
        llm = ChatOpenAI(model=model, api_key=oai_key)
        result["llm"] = llm

        result["ticker"] = TickerResearcher(
            model=llm,
            tavily_api_key=tav_key or None,
            openai_api_key=oai_key or None,
            max_results_per_query=max_results,
        )
        try:
            result["news"] = NewsResearcher(
                model=llm,
                tavily_api_key=tav_key or None,
                openai_api_key=oai_key or None,
                max_results_per_query=max_results,
            )
        except ValueError as exc:
            errors.append(f"NewsResearcher: {exc}")
    else:
        errors.append("OpenAI key required for TickerResearcher and NewsResearcher.")

    result["errors"] = errors
    return result


# Initialize on first load or when the button is pressed
if "researchers" not in st.session_state or init_btn:
    with st.spinner("Initializing researchers…"):
        try:
            st.session_state["researchers"] = _init_researchers(
                openai_key, tavily_key, openai_model, int(max_results)
            )
            st.sidebar.success("Ready.")
        except Exception as exc:
            st.sidebar.error(f"Init failed: {exc}")
            st.session_state["researchers"] = {}

rs = st.session_state.get("researchers", {})

# Show any init errors in sidebar
for err in rs.get("errors", []):
    st.sidebar.warning(err)

# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _render_search_results(results, fmt: str) -> None:
    searcher = rs.get("searcher")
    if not results or not searcher:
        st.info("No results.")
        return
    searcher._results = results  # inject so formatters work
    if fmt == "Full (title + source + content)":
        st.text(searcher.format_results(results))
    elif fmt == "Minimal (title + content)":
        st.text(searcher.format_minimal(results))
    elif fmt == "Content only":
        st.text(searcher.format_content_only(results))
    elif fmt == "Content list":
        for i, c in enumerate(searcher.get_content_list(results), 1):
            st.markdown(f"**{i}.** {c}")
    else:
        for r in results:
            with st.expander(r.title or r.source or "Result"):
                st.write(r.content)
                if r.source:
                    st.caption(r.source)


def _render_news_items(items) -> None:
    if not items:
        st.info("No news items returned.")
        return
    for item in items:
        sentiment_color = {
            "bullish": "🟢", "bearish": "🔴", "neutral": "⚪"
        }.get((item.sentiment or "").lower(), "⚪")
        with st.expander(
            f"{sentiment_color} [{item.event_type or 'news'}] {item.headline}  "
            f"*(score: {item.relevance_score:.2f})*"
        ):
            st.write(item.summary)
            if item.tickers_mentioned:
                st.caption("Tickers: " + ", ".join(item.tickers_mentioned))


def _render_ticker_research(res) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", res.sentiment or "—")
        st.metric("Confidence", f"{res.confidence:.0%}" if res.confidence else "—")
        st.metric("Earnings Date", res.earnings_date or "—")
    with col2:
        st.markdown(f"**Key Catalyst:** {res.key_catalyst or '—'}")
        st.markdown(f"**Risk Factors:** {', '.join(res.risk_factors) if res.risk_factors else '—'}")
    if res.sources:
        with st.expander("Sources"):
            for s in res.sources:
                st.markdown(f"- {s}")
    with st.expander("Raw JSON"):
        st.json(res.model_dump())


def _render_sector_research(res) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Health", res.overall_health or "—")
        st.metric("Outlook", res.outlook or "—")
    with col2:
        st.markdown(f"**Key Trends:** {', '.join(res.key_trends) if res.key_trends else '—'}")
        st.markdown(f"**Tailwinds:** {', '.join(res.tailwinds) if res.tailwinds else '—'}")
        st.markdown(f"**Headwinds:** {', '.join(res.headwinds) if res.headwinds else '—'}")
    with st.expander("Raw JSON"):
        st.json(res.model_dump())


# ---------------------------------------------------------------------------
# Main tabs (A1.2)
# ---------------------------------------------------------------------------
tab_search, tab_ticker, tab_news = st.tabs(
    ["🔍 Web Search", "📈 Ticker & Sector", "📰 News"]
)

# ===========================================================================
# TAB 1 — Web Search (A3)
# ===========================================================================
with tab_search:
    st.header("Web Search")
    searcher = rs.get("searcher")

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        query = st.text_input("Query", placeholder="e.g. NVIDIA earnings Q1 2025", key="search_query")
    with c2:
        provider = st.selectbox("Provider", ["TAVILY", "OPENAI"], key="search_provider")
    with c3:
        topic = st.selectbox("Topic", ["general", "news", "finance"], key="search_topic")
    with c4:
        days_back = st.number_input("Days back", min_value=0, value=0, step=1, key="search_days",
                                    help="0 = no date filter")

    fmt = st.selectbox(
        "Result format",
        ["Full (title + source + content)", "Minimal (title + content)",
         "Content only", "Content list", "Expanded cards"],
        key="search_fmt",
    )

    if st.button("Search", key="search_btn", use_container_width=True):
        if not searcher:
            st.error("Researcher not initialized.")
        elif not query.strip():
            st.warning("Enter a query.")
        else:
            from webai import SearchProvider as SP
            prov = SP.TAVILY if provider == "TAVILY" else SP.OPENAI
            days = int(days_back) if days_back > 0 else None
            with st.spinner("Searching…"):
                try:
                    if prov.name == "TAVILY":
                        results = searcher.search_tavily(query, topic=topic, days=days)
                    else:
                        results = searcher.search(query, provider=prov, topic=topic)
                    st.success(f"{len(results)} result(s)")
                    _render_search_results(results, fmt)
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

# ===========================================================================
# TAB 2 — Ticker & Sector (A4)
# ===========================================================================
with tab_ticker:
    st.header("Ticker & Sector Research")
    ticker_researcher = rs.get("ticker")

    if not ticker_researcher:
        st.warning("Initialize with a valid OpenAI key to use this tab.")
    else:
        sub_ticker, sub_portfolio, sub_sector = st.tabs(
            ["Single Ticker", "Portfolio", "Sector"]
        )

        # --- A4.1  Single Ticker ---
        with sub_ticker:
            tc1, tc2, tc3, tc4 = st.columns([1, 2, 1, 1])
            with tc1:
                symbol = st.text_input("Symbol", placeholder="NVDA", key="ticker_symbol").upper()
            with tc2:
                company = st.text_input("Company name (optional)", placeholder="NVIDIA", key="ticker_company")
            with tc3:
                earnings = st.text_input("Earnings date (optional)", placeholder="2025-05-28", key="ticker_earnings")
            with tc4:
                t_days = st.number_input("Days back", min_value=1, value=7, key="ticker_days")

            if st.button("Research Ticker", key="ticker_btn", use_container_width=True):
                if not symbol:
                    st.warning("Enter a symbol.")
                else:
                    with st.spinner(f"Researching {symbol}…"):
                        try:
                            res = ticker_researcher.research_ticker(
                                symbol,
                                company_name=company or None,
                                earnings_date=earnings or None,
                                days_back=int(t_days),
                            )
                            st.subheader(f"{symbol} — {res.sentiment or 'n/a'}")
                            _render_ticker_research(res)
                        except Exception as exc:
                            st.error(f"Research failed: {exc}")

        # --- A4.2  Portfolio ---
        with sub_portfolio:
            st.caption(
                "Enter one symbol per line. Optionally add company name after a comma: `AAPL, Apple`"
            )
            port_input = st.text_area(
                "Symbols", placeholder="NVDA\nAAPL, Apple\nMSFT", height=120, key="portfolio_input"
            )
            p_days = st.number_input("Days back", min_value=1, value=7, key="portfolio_days")

            if st.button("Research Portfolio", key="portfolio_btn", use_container_width=True):
                lines = [l.strip() for l in port_input.splitlines() if l.strip()]
                if not lines:
                    st.warning("Enter at least one symbol.")
                else:
                    symbols: list = []
                    for line in lines:
                        parts = [p.strip() for p in line.split(",", 1)]
                        symbols.append(tuple(parts) if len(parts) == 2 else parts[0])

                    with st.spinner(f"Researching {len(symbols)} ticker(s)…"):
                        try:
                            results = ticker_researcher.research_portfolio(
                                symbols, days_back=int(p_days)
                            )
                            st.success(f"Got results for: {', '.join(results.keys())}")
                            for sym, res in results.items():
                                with st.expander(f"{sym} — {res.sentiment or 'n/a'}"):
                                    _render_ticker_research(res)
                        except Exception as exc:
                            st.error(f"Portfolio research failed: {exc}")

        # --- A4.3  Sector ---
        with sub_sector:
            sector_name = st.text_input(
                "Sector", placeholder="e.g. Semiconductors, Energy, Healthcare", key="sector_name"
            )

            if st.button("Research Sector", key="sector_btn", use_container_width=True):
                if not sector_name.strip():
                    st.warning("Enter a sector name.")
                else:
                    with st.spinner(f"Researching {sector_name}…"):
                        try:
                            res = ticker_researcher.research_sector(sector_name)
                            st.subheader(f"{sector_name} — {res.overall_health or 'n/a'}")
                            _render_sector_research(res)
                        except Exception as exc:
                            st.error(f"Sector research failed: {exc}")

# ===========================================================================
# TAB 3 — News (A5)
# ===========================================================================
with tab_news:
    st.header("News Research")
    news_researcher = rs.get("news")

    if not news_researcher:
        st.warning("Initialize with valid OpenAI + Tavily keys to use this tab.")
    else:
        sub_breaking, sub_macro, sub_catalyst, sub_digest = st.tabs(
            ["Breaking News", "Macro News", "Catalyst News", "Daily Digest"]
        )

        # --- A5.1  Breaking news ---
        with sub_breaking:
            br_tickers = st.text_input(
                "Tickers (comma-separated)", placeholder="NVDA, AAPL, MSFT", key="breaking_tickers"
            )
            br_days = st.number_input("Days back", min_value=1, value=1, key="breaking_days")

            if st.button("Fetch Breaking News", key="breaking_btn", use_container_width=True):
                tickers = [t.strip().upper() for t in br_tickers.split(",") if t.strip()]
                if not tickers:
                    st.warning("Enter at least one ticker.")
                else:
                    with st.spinner("Fetching…"):
                        try:
                            items = news_researcher.fetch_breaking_news(tickers, days_back=int(br_days))
                            st.success(f"{len(items)} item(s)")
                            _render_news_items(items)
                        except Exception as exc:
                            st.error(f"Failed: {exc}")

        # --- A5.2  Macro news ---
        with sub_macro:
            mac_days = st.number_input("Days back", min_value=1, value=1, key="macro_days")

            if st.button("Fetch Macro News", key="macro_btn", use_container_width=True):
                with st.spinner("Fetching…"):
                    try:
                        items = news_researcher.fetch_macro_news(days_back=int(mac_days))
                        st.success(f"{len(items)} item(s)")
                        _render_news_items(items)
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

        # --- A5.3  Catalyst news ---
        with sub_catalyst:
            cat_ticker = st.text_input("Ticker", placeholder="NVDA", key="catalyst_ticker").upper()
            cat_days = st.number_input("Days back", min_value=1, value=7, key="catalyst_days")

            if st.button("Fetch Catalyst News", key="catalyst_btn", use_container_width=True):
                if not cat_ticker:
                    st.warning("Enter a ticker.")
                else:
                    with st.spinner(f"Fetching catalysts for {cat_ticker}…"):
                        try:
                            items = news_researcher.fetch_catalyst_news(cat_ticker, days_back=int(cat_days))
                            st.success(f"{len(items)} item(s)")
                            _render_news_items(items)
                        except Exception as exc:
                            st.error(f"Failed: {exc}")

        # --- A5.4  Daily digest ---
        with sub_digest:
            dg_tickers = st.text_input(
                "Tickers (comma-separated)", placeholder="NVDA, AAPL", key="digest_tickers"
            )
            dg_sectors = st.text_input(
                "Sectors (comma-separated)", placeholder="Technology, Energy", key="digest_sectors"
            )
            dg_date = st.date_input("Date", value=date.today(), key="digest_date")
            dg_days = st.number_input("Days back", min_value=1, value=1, key="digest_days")

            if st.button("Fetch Daily Digest", key="digest_btn", use_container_width=True):
                tickers = [t.strip().upper() for t in dg_tickers.split(",") if t.strip()]
                sectors = [s.strip() for s in dg_sectors.split(",") if s.strip()]
                with st.spinner("Building digest…"):
                    try:
                        digest = news_researcher.fetch_daily_digest(
                            tickers=tickers,
                            sectors=sectors,
                            date=str(dg_date),
                            days_back=int(dg_days),
                        )
                        st.subheader(f"Market Theme: {digest.market_theme or '—'}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Top Stories**")
                            _render_news_items(digest.top_stories)
                        with col2:
                            st.markdown("**Macro Events**")
                            _render_news_items(digest.macro_events)

                        if digest.sector_movers:
                            st.markdown("**Sector Movers**")
                            for mover in digest.sector_movers:
                                st.markdown(
                                    f"- **{mover.sector}**: {mover.direction or ''} — {mover.note or ''}"
                                )

                        if digest.watchlist_alerts:
                            st.markdown("**Watchlist Alerts**")
                            _render_news_items(digest.watchlist_alerts)

                        with st.expander("Raw JSON"):
                            st.json(digest.model_dump())

                    except RuntimeError as exc:
                        st.error(f"Digest synthesis failed: {exc}")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
