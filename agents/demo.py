"""
agents/demo.py
--------------
Interactive CLI demo for the webai Deep Agent.

Usage
-----
    python -m agents.demo

    # or with explicit model override:
    ORCHESTRATOR_MODEL=gpt-4o python -m agents.demo

Environment variables required
-------------------------------
    OPENAI_API_KEY   — for WebSearcher (OpenAI fallback) + synthesis LLM
    TAVILY_API_KEY   — for all Tavily searches (required)

    ORCHESTRATOR_MODEL  — orchestrator model string (default: claude-sonnet-4-6)
    SYNTHESIS_MODEL     — OpenAI model for structured-output synthesis
                          (default: gpt-4o-mini)

Example prompts to try
----------------------
    Research NVDA for the last 7 days
    Give me a portfolio snapshot for AAPL, MSFT, GOOGL
    How is the semiconductor sector doing?
    What's breaking today for TSLA?
    Build a daily digest for NVDA and AAPL in Technology for today

Type "quit" or press Ctrl-C to exit.
"""

from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

# Silence LangSmith tracing noise in demo mode
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from agents.webai_agent import build_webai_agent  # noqa: E402


def _extract_text(result: dict) -> str:
    """Pull the last assistant message text from the agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if role in ("ai", "assistant") and content:
            return content if isinstance(content, str) else str(content)
    return str(result)


def main() -> None:
    orchestrator_model = os.environ.get("ORCHESTRATOR_MODEL", "claude-sonnet-4-6")
    synthesis_model = os.environ.get("SYNTHESIS_MODEL", "gpt-4o-mini")

    print("Building webai agent …")
    agent = build_webai_agent(
        model=orchestrator_model,
        openai_model=synthesis_model,
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nwebai Deep Agent ready  (thread: {thread_id})")
    print("Orchestrator model :", orchestrator_model)
    print("Synthesis model    :", synthesis_model)
    print("Type 'quit' to exit.\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            sys.exit(0)

        if not user_input:
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"\nAgent: {_extract_text(result)}\n")
            print("-" * 60)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
