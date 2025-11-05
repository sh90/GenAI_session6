# rss_json_plan_executor.py
from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Optional
import re

from dotenv import load_dotenv
import feedparser
import newspaper
from openai import OpenAI

# ----------------------------
# 0) Setup
# ----------------------------
load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from env

MODEL_PLAN = "gpt-4o-mini"
MODEL_TASK = "gpt-4o-mini"

MAX_ARTICLES = 5        # how many items to analyze from the feed
MAX_CONTENT_CHARS = 5000  # trim article text to keep prompts light

FEED_URL = "http://feeds.bbci.co.uk/news/rss.xml"

# ----------------------------
# 1) Utilities
# ----------------------------
def safe_get_article_text(url: str) -> str:
    """Fetch article content with newspaper3k; fall back to raw if needed."""
    try:
        art = newspaper.Article(url)
        art.download()
        art.parse()
        txt = (art.text or "").strip()
        if txt:
            return txt
    except Exception:
        pass
    # Optional: add trafilatura fallback if installed
    try:
        import requests, trafilatura
        html = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}).text
        txt = trafilatura.extract(html) or ""
        return txt.strip()
    except Exception:
        return ""

def _parse_json_from_response(resp) -> Dict[str, Any]:
    """
    Robust JSON parse for Responses API objects (new and older SDKs).
    """
    # Newer SDKs may give parsed JSON
    try:
        if getattr(resp, "output_parsed", None) is not None:
            return resp.output_parsed  # type: ignore[return-value]
    except Exception:
        pass

    # Try output_text (some SDKs expose it)
    text = getattr(resp, "output_text", None)
    if text:
        return json.loads(_strip_code_fences(text))

    # Fallback: parse first text block (older responses objects)
    blocks = getattr(resp, "output", [])
    if blocks and getattr(blocks[0], "content", []):
        text = blocks[0].content[0].text
        return json.loads(_strip_code_fences(text))

    raise ValueError("Could not parse JSON from model response (empty or unrecognized format).")


def _strip_code_fences(s: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if the model accidentally adds them.
    """
    s = s.strip()
    if s.startswith("```"):
        # remove leading ```json or ```
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        # remove trailing ```
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def call_llm_json(
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    model: str = MODEL_TASK,
    temperature: float = 0
) -> Dict[str, Any]:
    """
    Version-adaptive JSON call:
    1) Try Responses API with response_format (json_schema/json_object)
    2) If TypeError (older SDK), try Responses API WITHOUT response_format
    3) If that fails, fall back to Chat Completions with a JSON-only system prompt
    """
    # --- Attempt 1: Responses API with response_format ---
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            response_format=(
                {"type": "json_schema", "json_schema": {"name": "schema", "schema": schema}}
                if schema else {"type": "json_object"}
            ),
        )
        return _parse_json_from_response(response)
    except TypeError as e:
        # Older openai SDKs don’t accept response_format on Responses API
        if "response_format" not in str(e):
            raise

    # --- Attempt 2: Responses API WITHOUT response_format ---
    try:
        response = client.responses.create(
            model=model,
            input=(
                "Return ONLY a valid JSON object. Do not add code fences or any extra text.\n\n" + prompt
                if schema is None else
                "Return ONLY a valid JSON object that matches this schema (no extra keys, no text):\n"
                f"{json.dumps(schema)}\n\n{prompt}"
            ),
            temperature=temperature,
            # no response_format here (older SDK compatibility)
        )
        return _parse_json_from_response(response)
    except Exception:
        pass

    # --- Attempt 3: Chat Completions fallback ---
    # Note: some SDK versions support response_format here too, but we’ll keep it simple + instruction-based.
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": (
                "You must return ONLY a valid JSON object. No prose, no code fences, no explanations."
                if schema is None else
                "You must return ONLY a valid JSON object that matches this JSON schema exactly. "
                "No extra keys, no prose, no code fences. Here is the schema:\n"
                f"{json.dumps(schema)}"
             )},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    text = completion.choices[0].message.content or ""
    return json.loads(_strip_code_fences(text))


# ----------------------------
# 2) JSON Plan
# ----------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {
                        "type": "string",
                        "enum": ["summarize", "detect_topic", "sentiment", "company_ner", "final_report"]
                    },
                    "inputs": {"type": "object"}
                },
                "required": ["id", "name"]
            }
        }
    },
    "required": ["steps"]
}

def plan_for_article(title: str, content: str) -> Dict[str, Any]:
    """Ask LLM to produce a minimal plan tailored to this article."""
    trimmed = content[:MAX_CONTENT_CHARS]
    prompt = f"""
You are a news analysis planner. Choose a small set of steps for this article.

Allowed steps (choose subset and order them sensibly):
- summarize: 80–120 word neutral summary
- detect_topic: classify into topic (e.g., politics, business, markets, technology, science, sports, world, health)
- sentiment: label article sentiment (positive/negative/neutral) with 0–1 confidence
- company_ner: list company names mentioned (unique)
- final_report: produce final structured news record

Rules:
- Include 'summarize' first.
- Include 'detect_topic' early.
- Include 'company_ner' only if topic is business/markets/technology/startups.
- Include 'sentiment' for business/markets/technology/politics.
- Always end with 'final_report'.

Return a strict JSON object matching the provided schema.

Title: {title}

Article (truncated):
\"\"\"{trimmed}\"\"\"
"""
    return call_llm_json(prompt, schema=PLAN_SCHEMA, model=MODEL_PLAN)

# ----------------------------
# 3) Step Executors (structured outputs)
# ----------------------------
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"]
}

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "reason": {"type": "string"}
    },
    "required": ["topic"]
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"}
    },
    "required": ["label", "confidence"]
}

COMPANY_NER_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["companies"]
}

FINAL_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "topic": {"type": "string"},
        "sentiment": {"type": "string"},
        "companies": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "string"}
    },
    "required": ["title", "summary", "topic"]
}

def step_summarize(title: str, content: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Summarize the article in 80–120 words, neutral tone. Return JSON {{"summary": "..."}}

Title: {title}
Article:
\"\"\"{content[:MAX_CONTENT_CHARS]}\"\"\""""
    return call_llm_json(prompt, schema=SUMMARY_SCHEMA)

def step_detect_topic(title: str, content: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Classify the main topic of this article (politics, business, markets, technology, science, sports, world, health, other).
Return JSON like {{"topic":"business","reason":"..."}}.

Title: {title}
Article:
\"\"\"{content[:MAX_CONTENT_CHARS]}\"\"\""""
    return call_llm_json(prompt, schema=TOPIC_SCHEMA)

def step_sentiment(title: str, content: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Classify sentiment of the article content as positive, negative, or neutral. 
Return JSON like {{"label":"neutral","confidence":0.74}}.

Title: {title}
Article:
\"\"\"{content[:MAX_CONTENT_CHARS]}\"\"\""""
    return call_llm_json(prompt, schema=SENTIMENT_SCHEMA)

def step_company_ner(title: str, content: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Extract company names mentioned (public or private). Return unique list, JSON: {{"companies":["..."]}}.

Title: {title}
Article:
\"\"\"{content[:MAX_CONTENT_CHARS]}\"\"\""""
    res = call_llm_json(prompt, schema=COMPANY_NER_SCHEMA)
    # tiny cleanup
    uniq = sorted({c.strip() for c in res.get("companies", []) if c and isinstance(c, str)})
    return {"companies": uniq}

def step_final_report(title: str, content: str, plan: Dict[str, Any], stash: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble final JSON using previously computed fields (no LLM)."""
    return {
        "title": title,
        "summary": stash.get("summary", ""),
        "topic": stash.get("topic", "other"),
        "sentiment": stash.get("label", "unknown"),
        "companies": stash.get("companies", []),
        "priority": plan.get("priority", "medium"),
    }

STEP_IMPLS = {
    "summarize": step_summarize,
    "detect_topic": step_detect_topic,
    "sentiment": step_sentiment,
    "company_ner": step_company_ner,
    "final_report": None,  # handled separately
}

# ----------------------------
# 4) Runner
# ----------------------------
def analyze_article(item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("link") or item.get("id") or ""
    title = item.get("title", "").strip()
    published = item.get("published", "") or item.get("updated", "")
    text = safe_get_article_text(url)

    plan = plan_for_article(title, text)

    # Execute steps
    stash: Dict[str, Any] = {}
    for step in sorted(plan["steps"], key=lambda s: s["id"]):
        name = step["name"]
        if name == "final_report":
            continue
        impl = STEP_IMPLS.get(name)
        if not impl:
            continue
        res = impl(title, text, step.get("inputs", {}))
        # flatten stash (avoid collisions by convention)
        stash.update(res)

    final = step_final_report(title, text, plan, stash)
    final["url"] = url
    final["published"] = published
    final["plan"] = plan  # optional: include for debugging/teaching
    return final

# ----------------------------
# 5) Main
# ----------------------------
def fetch_feed_items(feed_url: str, limit: int = MAX_ARTICLES) -> List[Dict[str, Any]]:
    feed = feedparser.parse(feed_url)
    return list(feed.entries)[:limit]

def main():
    items = fetch_feed_items(FEED_URL, MAX_ARTICLES)
    outputs: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for it in items:
        try:
            rec = analyze_article(it)
            outputs.append(rec)
            print(f"\n--- Processed: {rec['title'][:80]}...")
            print(json.dumps({k: rec[k] for k in ['title','topic','sentiment','companies'] if k in rec}, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[WARN] Failed on item: {it.get('title','(no title)')}: {e}")
    t1 = time.perf_counter()

    # Save a JSONL for downstream use
    out_path = "news_analysis.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(outputs)} records to {out_path} in {(t1 - t0):.2f}s")

if __name__ == "__main__":
    main()
