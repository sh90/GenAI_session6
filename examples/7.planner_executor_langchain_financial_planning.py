# langchain_financial_planner_autonomous_lc1.py
# Pattern: Autonomous tools agent (reactive planning) with terminal finalize step
# Requires: pip install -U langchain langchain-openai langchain-community python-dotenv

from __future__ import annotations
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain.tools import tool

Number = Union[int, float, str]

# ---- 1) LLM (tool-calling capable) ----
llm = ChatOpenAI(model="gpt-4o", temperature=0)   # or "gpt-4o-mini"

# ---- 2) Helpers ----
def _to_int(x: Number, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

# ---- 3) Tools (robust to str/int inputs) ----
@tool
def calc_monthly_surplus(monthly_income_inr: Number, monthly_expenses_inr: Number) -> Dict[str, Any]:
    """Compute user's monthly surplus and savings rate (%) for India users."""
    mi = _to_int(monthly_income_inr)
    me = _to_int(monthly_expenses_inr)
    surplus = mi - me
    savings_rate = round((surplus / mi) * 100, 2) if mi > 0 else 0.0
    return {"monthly_surplus": surplus, "savings_rate_pct": savings_rate}

@tool
def calc_emergency_fund_target(monthly_expenses_inr: Number, dependents: Number) -> Dict[str, Any]:
    """Emergency fund target: 6× expenses if dependents>0 else 3×."""
    me = _to_int(monthly_expenses_inr)
    deps = _to_int(dependents)
    months = 6 if deps > 0 else 3
    return {"efund_target": me * months, "months_of_expense": months}

@tool
def short_term_goal_allocation(remaining_surplus: Number, goals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Allocate part of surplus to short-term goals (<=24 months).
    Each goal: {"name": str, "amount_inr": int, "time_horizon_months": int}
    """
    rs = _to_int(remaining_surplus)
    short = [g for g in goals if _to_int(g.get("time_horizon_months", 0)) <= 24]
    alloc, used = [], 0
    for g in short:
        total = _to_int(g.get("amount_inr", 0))
        months = max(_to_int(g.get("time_horizon_months", 1)), 1)
        need_pm = max(total // months, 0)
        if used + need_pm <= rs:
            alloc.append({"goal": g.get("name", "goal"), "monthly": need_pm})
            used += need_pm
    return {"short_term_allocations": alloc, "surplus_after_short_term": rs - used}

@tool
def long_term_allocation(remaining_surplus: Number, risk_profile: str) -> Dict[str, Any]:
    """
    Allocate remaining surplus into equity/debt by risk profile.
    risk_profile ∈ {"aggressive","moderate","conservative"/other}
    """
    rs = _to_int(remaining_surplus)
    if rs <= 0:
        return {"equity_monthly": 0, "debt_monthly": 0}
    rp = (risk_profile or "").lower().strip()
    if rp == "aggressive":
        e, d = 0.75, 0.25
    elif rp == "moderate":
        e, d = 0.60, 0.40
    else:
        e, d = 0.40, 0.60
    return {"equity_monthly": int(rs * e), "debt_monthly": int(rs * d)}

# ---- 4) Terminal "finalize" tool (forces strict JSON & stops the run) ----
def _finalize_impl(final_json: str) -> str:
    """
    Input JSON MUST contain:
    {
      "plan_text": "human-friendly final plan",
      "numbers": {
        "monthly_surplus": int, "savings_rate_pct": number,
        "efund_target": int, "months_of_expense": int,
        "short_term_allocations": [{"goal": str, "monthly": int}],
        "surplus_after_short_term": int,
        "equity_monthly": int, "debt_monthly": int
      },
      "tips": ["...", "..."]   # 3–5 actionable tips
    }
    """
    import json
    try:
        data = json.loads(final_json)
        assert isinstance(data.get("plan_text"), str)
        assert isinstance(data.get("numbers"), dict)
        assert isinstance(data.get("tips"), list)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Invalid finalize payload: {e}"}, ensure_ascii=False)

finalize = Tool.from_function(
    name="finalize",
    description=(
        "Finish by returning STRICT JSON with keys: plan_text (str), numbers (object), tips (list[str]). "
        "Input: JSON string as specified."
    ),
    func=_finalize_impl,
    return_direct=True,  # <-- makes this the terminal action
)

TOOLS = [calc_monthly_surplus, calc_emergency_fund_target, short_term_goal_allocation, long_term_allocation, finalize]

# ---- 5) Agent (LangChain v1) ----
SYSTEM_PROMPT = (
    "You are a financial planning agent for India. Plan and execute your own strategy using the available tools.\n"
    "Typical flow: calc_monthly_surplus → calc_emergency_fund_target → short_term_goal_allocation → long_term_allocation.\n"
    "Use tool outputs as inputs to subsequent tools; reason over when the remaining surplus becomes zero.\n"
    "If surplus is fully consumed by emergency fund or short-term goals, set long-term allocation to 0.\n"
    "At the end, call the 'finalize' tool with STRICT JSON containing: plan_text, numbers, tips.\n"
    "Do NOT reveal chain-of-thought. The LAST tool call MUST be 'finalize'."
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# ---- 6) Run it ----
if __name__ == "__main__":
    user_profile = {
        "age": 35,
        "dependents": 1,
        "monthly_income_inr": 210000,
        "monthly_expenses_inr": 115000,
        "risk_profile": "moderate",
        "goals": [
            {"name": "Car purchase", "amount_inr": 1200000, "time_horizon_months": 18},
            {"name": "Retirement", "amount_inr": 35000000, "time_horizon_months": 240},
        ],
    }

    task = (
        "Create a monthly financial plan. Use the tools to compute numbers and allocations.\n"
        "Return a human-friendly final plan plus a compact JSON of the final numbers via 'finalize'.\n\n"
        f"{user_profile}"
    )

    # Bound the loop; 'finalize' is terminal
    state = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": 12}
    )

    # The last AI message is the JSON returned by `finalize`
    print("\nFINAL JSON:\n", state["messages"][-1].content)
