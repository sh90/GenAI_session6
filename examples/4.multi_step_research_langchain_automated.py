# autonomous_tweet_agent_lc1.py
# pip install -U langchain langchain-openai langchain-community tavily-python
# env: OPENAI_API_KEY=...  TAVILY_API_KEY=...

from __future__ import annotations
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# 1) Model (tool-calling capable)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2) Tools
# 2a) Tavily search as a ready-made tool object (requires TAVILY_API_KEY)
tavily = TavilySearchResults(max_results=3)

# 2b) Optional helper tool: constrain tweet under a discovered limit
@tool
def enforce_limit(text: str, limit: int = 280) -> str:
    """Trim text to fit within a character limit; prefers keeping key points + citations."""
    if len(text) <= limit:
        return text
    # naive trim with ellipsis
    trimmed = text[: max(0, limit - 1)]
    return trimmed if len(trimmed) < limit else (trimmed[:-1] + "…")

tools = [tavily, enforce_limit]

# 3) Agent creation langchain
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a content research & creation agent. "
        "Plan and execute: use web search to find 2–3 key features of gpt-4o-mini and the current X/Twitter "
        "character limit. Draft one compelling tweet that fits under that limit. "
        "Cite sources inline with brief [1], [2] markers where helpful. "
        "Use `enforce_limit` if needed. Do not reveal chain-of-thought."
    ),
)

if __name__ == "__main__":
    user_task = (
        "Find 2–3 key features of the new gpt-4o-mini and the current tweet character limit for X. "
        "Then produce a single tweet under that limit summarizing those features."
    )

    # 4) Invoke: pass message state (no AgentExecutor)
    result = agent.invoke({"messages": [{"role": "user", "content": user_task}]})

    # The agent returns a state dict; the last AI message has the final text.
    final_msg = result["messages"][-1].content
    print("\n--- Final Tweet ---\n")
    print(final_msg)
