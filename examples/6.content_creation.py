# autonomous_tweet_agent_refactored_lc1.py
from __future__ import annotations
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader

# 1) LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)  # or "gpt-4o-mini"

# 2) Tools
tavily = TavilySearchResults(max_results=4)  # needs TAVILY_API_KEY

@tool
def load_url(url: str) -> str:
    """Fetch & extract visible text from a URL. Input: url string."""
    docs = WebBaseLoader(url).load()
    return docs[0].page_content if docs else ""

@tool
def count_chars(s: str) -> int:
    """Return character count of the provided string."""
    return len(s or "")

def _finalize_impl(final_json: str) -> str:
    """
    Input JSON MUST be:
    {
      "features": ["...", "..."],
      "char_limit": 280,
      "tweet": "final tweet text",
      "sources": [{"title": "...", "url": "..."}]
    }
    """
    try:
        data = json.loads(final_json)
        assert isinstance(data.get("features"), list)
        assert isinstance(data.get("char_limit"), int)
        assert isinstance(data.get("tweet"), str)
        assert isinstance(data.get("sources"), list)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Invalid finalize payload: {e}"}, ensure_ascii=False)

finalize = Tool.from_function(
    name="finalize",
    description=(
        "Finish by returning STRICT JSON with keys: "
        "features (list[str]), char_limit (int), tweet (str), "
        "sources (list[{title,url}]). Input: JSON string."
    ),
    func=_finalize_impl,
    return_direct=True,  # <-- makes this the terminal action
)

TOOLS = [tavily, load_url, count_chars, finalize]

# 3) Agent (no prompt=, just system_prompt=)
SYSTEM_PROMPT = (
    "You are a content research & creation agent. Plan and execute your own strategy using the tools.\n"
    "Goals:\n"
    "1) Identify 2–3 concrete features of the specified GPT model from authoritative pages.\n"
    "2) Determine the CURRENT X (Twitter) tweet character limit; if multiple tiers exist, pick the standard/public limit and state it.\n"
    "3) Draft a compelling tweet under that limit incorporating the features.\n"
    "4) Verify the tweet length using the 'count_chars' tool; if over the limit, revise and re-check.\n"
    "5) End by calling 'finalize' with STRICT JSON: "
    '{"features":[...], "char_limit": <int>, "tweet": "...", "sources":[{"title":"...","url":"..."}]}.\n'
    "Constraints:\n"
    "- Prefer recent, authoritative sources. Use 'load_url' to verify claims from search results.\n"
    "- Do NOT reveal chain-of-thought. The LAST tool call MUST be 'finalize'."
)

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# 4) Run
if __name__ == "__main__":
    task = (
        "I want to post on X (Twitter) about the new `gpt-4o-mini` model. "
        "Research 2–3 key features, check the current X tweet character limit, "
        "and produce a final tweet under that limit with sources."
    )

    # Bound the loop; 'finalize' is terminal
    state = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": 12}
    )

    print("\nFinal JSON:\n", state["messages"][-1].content)
