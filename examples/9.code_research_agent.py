# agent_code_research_eth_lc1.py
# Pattern: Autonomous tools agent (reactive planning) with terminal finalize step
# Deps: pip install -U langchain langchain-openai langchain-community langchain-experimental python-dotenv requests
# Env : export OPENAI_API_KEY=... ; export TAVILY_API_KEY=...

from __future__ import annotations
import json
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

import requests

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

# 1) LLM (tool-calling capable)
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # or "gpt-4o-mini"

# 2) Tools
# 2a) Web search (pass Tavily tool directly; needs TAVILY_API_KEY)
tavily = TavilySearchResults(max_results=4)

# 2b) Python REPL (for deterministic math/checks)
python_repl = PythonREPLTool()

# 2c) Crypto price via CoinGecko (no key required)
@tool
def crypto_price(symbol: str, vs_currency: str = "usd") -> Dict[str, Any]:
    """Return latest spot price for a crypto symbol (e.g., 'ETH') in vs_currency (e.g., 'usd')."""
    symbol_map = {"ETH": "ethereum", "BTC": "bitcoin", "SOL": "solana"}
    coin_id = symbol_map.get((symbol or "").upper())
    if not coin_id:
        return {"ok": False, "error": f"Unsupported symbol '{symbol}'. Try one of: {list(symbol_map)}"}
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={vs_currency}"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        price = float(data[coin_id][vs_currency.lower()])
        return {"ok": True, "symbol": symbol.upper(), "vs_currency": vs_currency.lower(), "price": price, "source_url": url}
    except Exception as e:
        return {"ok": False, "error": str(e), "source_url": url}

# 2d) Finalize (terminal tool) — enforce strict JSON and end the run
def _finalize_impl(final_json: str) -> str:
    """
    Input JSON MUST be:
    {
      "price_usd": 1234.56,
      "eth_for_500": 0.123456,
      "sources": [{"title": "...", "url": "..."}]
    }
    """
    try:
        data = json.loads(final_json)
        assert isinstance(data.get("price_usd"), (int, float))
        assert isinstance(data.get("eth_for_500"), (int, float))
        assert isinstance(data.get("sources"), list)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Invalid finalize payload: {e}"}, ensure_ascii=False)

finalize = Tool.from_function(
    name="finalize",
    description=(
        "Finish by returning STRICT JSON with keys: "
        "price_usd (number), eth_for_500 (number), sources (list of {title,url}). Input: JSON string."
    ),
    func=_finalize_impl,
    return_direct=True,  # <-- terminal action
)

TOOLS = [tavily, python_repl, crypto_price, finalize]

# 3) Agent (no ChatPromptTemplate; use system_prompt)
SYSTEM_PROMPT = (
    "You are a code & research assistant. Plan and execute with the available tools.\n"
    "Steps:\n"
    "1) Get ETH price in USD using 'crypto_price' FIRST. If it fails, use 'tavily-search' to find a reliable public API, "
    "   and (optionally) 'python_repl' to fetch/parse JSON from that API.\n"
    "2) Use 'python_repl' to compute how much ETH $500 buys. Round to 6 decimals.\n"
    "3) End by calling 'finalize' with STRICT JSON: "
    '{"price_usd": <number>, "eth_for_500": <number>, "sources": [{"title":"...","url":"..."}]}.\n'
    "Guidelines: Prefer authoritative sources; include CoinGecko URL in sources if used. "
    "Do NOT reveal chain-of-thought. The LAST tool call MUST be 'finalize'."
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# 4) Run
if __name__ == "__main__":
    task = (
        "What's the current USD price of Ethereum (ETH)? Then compute how much ETH I can buy for $500. "
        "Return strict JSON via 'finalize'."
    )

    state = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": 12}
    )

    # The last message content is the JSON returned by `finalize`
    print("\nFinal JSON:\n", state["messages"][-1].content)
