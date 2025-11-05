#!/usr/bin/env python
"""
demo6_tot_root_cause_lc2025.py
Pattern: Deliberate planning — Tree-of-Thought (multiple branches) + Self-Consistency (voter)

Usage:
  export OPENAI_API_KEY=sk-...
  python demo6_tot_root_cause_lc2025.py --branches 3 --gen-model gpt-4o-mini --vote-model gpt-4o
"""
from __future__ import annotations
import os, json, time, argparse, re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ---------------------------
# OpenAI client (reads env)
# ---------------------------
client = OpenAI()  # uses OPENAI_API_KEY from env

# ---------------------------
# Shared context & complaint
# ---------------------------
CONTEXT = """
System: VistaPrice is the internal pricing engine.
- Rule changes are written to DB immediately.
- UI (pricing page) reads from a caching layer updated every 30 minutes.
- Yesterday there was a deploy to the pricing service.
- Known issue: sometimes UI keeps old cache if region was not invalidated.
- Support said: "customer is seeing old discount".
"""

CUSTOMER_COMPLAINT = """
Customer: "Since yesterday, your pricing page is showing an outdated discount for some SKUs.
Our ops team says they updated the rule in the morning, but the UI still shows the old price.
Can you check?"
"""

# ---------------------------
# Prompts
# ---------------------------
BRANCH_PROMPT = """You are an experienced L2 support engineer.

You will reason about a customer complaint by exploring plausible root causes.
Think in 3 phases:
1. Restate the complaint in your own words.
2. Explore 2–3 plausible causes grounded in the context.
3. Pick the SINGLE most likely root cause and justify it.

Return JSON ONLY with this EXACT shape:
{{
  "restatement": "string",
  "candidates": [
    {{"name": "string", "why": "string"}},
    {{"name": "string", "why": "string"}}
  ],
  "selected": {{
    "name": "string",
    "why": "string"
  }}
}}

Context:
{context}

Complaint:
{complaint}
"""

VOTE_PROMPT = """You are ranking root-cause analyses.

You are given multiple analyses (from different people). Each has:
- a list of candidate causes
- a final selected cause

Your goal: pick the SINGLE most plausible and context-consistent root cause.

Rules:
- Prefer causes that mention the caching layer or delayed UI refresh (because the context says UI reads from cache).
- Prefer causes that connect "yesterday deploy" with "from yesterday issue".
- Reject causes that ignore the context.

Return JSON ONLY:
{{
  "best_cause": "string",
  "justification": "string"
}}

Analyses:
{analyses}
"""

# ---------------------------
# JSON Schemas (for LLM hints)
# ---------------------------
BRANCH_SCHEMA = {
    "type": "object",
    "properties": {
        "restatement": {"type": "string"},
        "candidates": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why": {"type": "string"}
                },
                "required": ["name", "why"]
            }
        },
        "selected": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "why": {"type": "string"}
            },
            "required": ["name", "why"]
        }
    },
    "required": ["restatement", "candidates", "selected"]
}

VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "best_cause": {"type": "string"},
        "justification": {"type": "string"}
    },
    "required": ["best_cause", "justification"]
}

# ---------------------------
# Parsing helpers (robust)
# ---------------------------
def _brace_slice(text: str) -> Optional[str]:
    """Extract the first top-level JSON object from text (very lenient)."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        # naive repair of trailing commas in JSON5-ish output
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return candidate
    return None

def _parse_json_maybe(text: str) -> dict:
    if not text:
        raise ValueError("Empty text for JSON parse")
    try:
        return json.loads(text)
    except Exception:
        sl = _brace_slice(text)
        if sl:
            return json.loads(sl)
        raise

def _extract_text_from_responses(resp) -> str:
    # Prefer output_text (Responses API)
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # Older path: resp.output[0].content[0].text
    output = getattr(resp, "output", None)
    if output and isinstance(output, list) and getattr(output[0], "content", None):
        first_block = output[0].content[0]
        return getattr(first_block, "text", "")
    return ""

def _call_responses(prompt: str, model: str, temperature: float, schema: Optional[dict]) -> dict:
    """
    Try Responses API with json_schema; if not supported, try json_object; else plain text + parse.
    """
    # 1) Try with json_schema
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": {"name": "schema", "schema": schema}} if schema else {"type": "json_object"},
        )
        # New SDKs sometimes offer parsed JSON directly
        if getattr(resp, "output_parsed", None) is not None:
            return resp.output_parsed  # type: ignore[return-value]
        return _parse_json_maybe(_extract_text_from_responses(resp))
    except TypeError:
        # 2) Older SDK: response_format kw unsupported → try without it, then parse
        resp = client.responses.create(model=model, input=prompt, temperature=temperature)
        return _parse_json_maybe(_extract_text_from_responses(resp))

def _call_chat_completions(prompt: str, model: str, temperature: float, schema: Optional[dict]) -> dict:
    """
    Fallback to Chat Completions. If response_format not supported, rely on prompt-only JSON instruction.
    """
    # Prefer JSON mode if available
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        return _parse_json_maybe(text)
    except TypeError:
        # Very old SDK: no response_format
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt + "\n\nReturn **JSON ONLY**."}],
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return _parse_json_maybe(text)

def call_llm_json(
    prompt: str,
    model: str,
    temperature: float,
    schema: Optional[dict] = None,
    retries: int = 3,
    backoff_sec: float = 1.2,
) -> dict:
    """
    SDK-version-tolerant JSON caller:
    1) Try Responses API (+json_schema/json_object)
    2) Fallback to Chat Completions (+json_object if supported)
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return _call_responses(prompt, model, temperature, schema)
        except Exception as e1:
            last_err = e1
            try:
                return _call_chat_completions(prompt, model, temperature, schema)
            except Exception as e2:
                last_err = e2
                if attempt < retries:
                    time.sleep(backoff_sec * attempt)
                else:
                    raise last_err
    # not reached
    raise RuntimeError("Unhandled error in call_llm_json")

# ---------------------------
# Core functions
# ---------------------------
def generate_branch(
    context: str,
    complaint: str,
    model: str,
    temperature: float = 0.4
) -> dict:
    prompt = BRANCH_PROMPT.format(context=context, complaint=complaint)
    return call_llm_json(prompt, model=model, temperature=temperature, schema=BRANCH_SCHEMA)

def vote_on_branches(
    branches: List[dict],
    model: str,
    temperature: float = 0.2
) -> dict:
    analyses_txt = ""
    for i, b in enumerate(branches, start=1):
        slim = {
            "restatement": b.get("restatement", ""),
            "candidates": b.get("candidates", []),
            "selected": b.get("selected", {}),
        }
        analyses_txt += f"Analysis {i}:\n{json.dumps(slim, ensure_ascii=False)}\n\n"
    prompt = VOTE_PROMPT.format(analyses=analyses_txt)
    return call_llm_json(prompt, model=model, temperature=temperature, schema=VOTE_SCHEMA)

def pretty_print_branches(branches: List[dict]) -> None:
    print("=== BRANCHES ===")
    for i, b in enumerate(branches, start=1):
        print(f"\n--- Branch {i} ---")
        print("Restatement:", b.get("restatement", ""))
        print("Candidates:")
        for c in b.get("candidates", []):
            print("  -", c.get("name", "?"), "→", c.get("why", ""))
        sel = b.get("selected", {})
        print("Selected:", sel.get("name", "?"))

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="ToT + Self-Consistency Root Cause Demo (2025-refactor)")
    ap.add_argument("--branches", type=int, default=3, help="number of parallel reasoning branches")
    ap.add_argument("--gen-model", type=str, default="gpt-4o-mini", help="model for branch generation")
    ap.add_argument("--vote-model", type=str, default="gpt-4o-mini", help="model for voting")
    ap.add_argument("--gen-temp", type=float, default=0.4, help="temperature for branch generation")
    ap.add_argument("--vote-temp", type=float, default=0.2, help="temperature for voting")
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) Generate branches
    branches: List[dict] = []
    for _ in range(max(1, args.branches)):
        b = generate_branch(CONTEXT, CUSTOMER_COMPLAINT, model=args.gen_model, temperature=args.gen_temp)
        branches.append(b)

    pretty_print_branches(branches)

    # 2) Vote / self-consistency
    final = vote_on_branches(branches, model=args.vote_model, temperature=args.vote_temp)

    print("\n=== FINAL PICK (self-consistency) ===")
    print("Best cause:", final.get("best_cause", ""))
    print("Why:", final.get("justification", ""))

if __name__ == "__main__":
    main()
