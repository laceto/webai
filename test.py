import json
import logging
import os

from dotenv import load_dotenv

from webai.ticker import TRUSTED_DOMAINS, SectorResearch, TickerResearch, TickerResearcher

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,  # flip to DEBUG to see the full pipeline trace
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)

_HAS_TAVILY = bool(os.environ.get("TAVILY_API_KEY"))
_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
_HAS_KEYS   = _HAS_TAVILY and _HAS_OPENAI

print(f"TAVILY_API_KEY : {'set' if _HAS_TAVILY else 'MISSING'}")
print(f"OPENAI_API_KEY : {'set' if _HAS_OPENAI else 'MISSING'}")
print(f"Live sections  : {'enabled' if _HAS_KEYS else 'SKIPPED — set both keys in .env'}")