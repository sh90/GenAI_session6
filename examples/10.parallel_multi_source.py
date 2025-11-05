#!/usr/bin/env python
"""
session6_parallel_planned_fanout_end_to_end.py

Session 6 (Planning) demo:
- Tiny explicit planner (decides which sources to fetch)
- Parallel fan-out with asyncio.TaskGroup (Py 3.11+/3.12)
- Per-task timeouts + latency + overall latency
- Optional reflexive re-plan for KB if confidence low
- Robust JSON serialization (handles dataclasses/Pydantic/custom objects)

Run:
  python session6_parallel_planned_fanout_end_to_end.py \
      --customer C001 \
      --sku VP-001 \
      --kb "common issues in pricing" \
      --timeout-customer 2.0 \
      --timeout-product 2.0 \
      --timeout-kb 2.5 \
      --kb-threshold 0.55 \
      --replan-on-low-kb \
      --customers-csv customers.csv \
      --products-json products.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import date, datetime
from dataclasses import is_dataclass, asdict
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -----------------------------
# Flexible imports for your project layout
# -----------------------------
DEFAULT_KB_QUERY_FALLBACK = "common issues in pricing cache invalidation"

try:
    # when running as a package module
    from services.config import DEFAULT_KB_QUERY  # type: ignore
    from services.fetchers import (      # type: ignore
        initialize_data,
        fetch_customer_async,
        fetch_product_async,
        fetch_kb_async,
    )
except Exception:
    # when running as a flat script
    try:
        from config import DEFAULT_KB_QUERY  # type: ignore
        from services.fetchers import (      # type: ignore
            initialize_data,
            fetch_customer_async,
            fetch_product_async,
            fetch_kb_async,
        )
    except Exception:
        # If you don't have these modules on sys.path, set DEFAULT_KB_QUERY and
        # provide your own implementations in the same directory or adjust imports.
        DEFAULT_KB_QUERY = DEFAULT_KB_QUERY_FALLBACK  # type: ignore
        raise RuntimeError(
            "Could not import project modules. Ensure 'config.py' and 'services/fetchers.py' "
            "are importable, or adjust the imports above."
        )

# -----------------------------
# JSON sanitizer (fixes “not JSON serializable”)
# -----------------------------
try:
    import numpy as np  # optional
except Exception:
    np = None  # type: ignore


def to_jsonable(x: Any) -> Any:
    """Recursively convert common Python / dataclass / Pydantic / numpy / datetime
    values into types json.dumps can handle."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, Path):
        return str(x)

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if is_dataclass(x):
        return to_jsonable(asdict(x))

    # Pydantic v1/v2
    try:
        from pydantic import BaseModel  # type: ignore
        if isinstance(x, BaseModel):
            to_dict = getattr(x, "model_dump", None) or getattr(x, "dict", None)
            if callable(to_dict):
                return to_jsonable(to_dict())
    except Exception:
        pass

    if np is not None:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()

    for attr in ("to_dict", "dict", "as_dict"):
        if hasattr(x, attr) and callable(getattr(x, attr)):
            try:
                return to_jsonable(getattr(x, attr)())
            except Exception:
                break

    if hasattr(x, "__dict__"):
        return to_jsonable(vars(x))

    return repr(x)


# -----------------------------
# Planning
# -----------------------------
def plan_sources(customer_id: str, sku: str, kb_query: Optional[str]) -> Dict[str, bool]:
    """
    Tiny explicit planner: decide which sources to fetch.
    Extend with more rules (feature flags, user intent) as needed.
    """
    plan = {
        "customer": True,
        "product": True,
        "kb": bool(kb_query),
    }
    # Example rule: for VistaPrice SKUs, always include KB lookup
    if sku.upper().startswith("VP-"):
        plan["kb"] = True
    return plan


# -----------------------------
# Executor helpers
# -----------------------------
async def with_timeout(coro, seconds: float, name: str) -> Any:
    """
    Wrap a coroutine with a timeout. Returns an error-shaped dict on failure so
    siblings keep running.
    """
    try:
        if seconds and seconds > 0:
            async with asyncio.timeout(seconds):
                return await coro
        return await coro
    except Exception as e:
        return {"error": f"{name} failed: {e!s}"}


async def timed(name: str, coro) -> Tuple[Any, int]:
    """Measure per-task latency (ms) and return (result, latency_ms)."""
    t0 = time.perf_counter()
    res = await coro
    ms = int((time.perf_counter() - t0) * 1000)
    return res, ms


def _safe_get_score(kb_obj: Any) -> float:
    """Extract numeric 'score' from KB result; 0.0 if missing/invalid."""
    try:
        if isinstance(kb_obj, dict) and "score" in kb_obj and isinstance(kb_obj["score"], (int, float)):
            return float(kb_obj["score"])
    except Exception:
        pass
    return 0.0


def _refine_kb_query(base_query: str, product: Any, sku: str) -> str:
    """
    Heuristic refinement for low-confidence KB: add product name/category + SKU.
    """
    parts = [base_query]
    if isinstance(product, dict):
        name = str(product.get("name", "") or "").strip()
        category = str(product.get("category", "") or "").strip()
        if name:
            parts.append(name)
        if category:
            parts.append(category)
    if sku:
        parts.append(f"SKU:{sku}")

    seen, uniq = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            uniq.append(p)
            seen.add(p)
    return " | ".join(uniq)


# -----------------------------
# Main flow (plan → parallel execute → integrate → optional re-plan)
# -----------------------------
async def main_flow(
    customer_id: str,
    sku: str,
    kb_query: Optional[str] = None,
    timeout_customer: float = 2.0,
    timeout_product: float = 2.0,
    timeout_kb: float = 2.5,
    kb_threshold: float = 0.55,
    replan_on_low_kb: bool = False,
) -> Dict[str, Any]:
    """
    Execute the plan in parallel, then (optionally) reflexively re-plan KB if low confidence.
    Returns a deterministic dict suitable for logging/evaluation.
    """
    base_kb_query = (kb_query or DEFAULT_KB_QUERY or DEFAULT_KB_QUERY_FALLBACK).strip()
    plan = plan_sources(customer_id, sku, base_kb_query)

    overall_t0 = time.perf_counter()

    results: Dict[str, Any] = {}
    per_task_ms: Dict[str, int] = {}
    kb_queries_used = [base_kb_query] if plan.get("kb") and base_kb_query else []

    # Fan-out according to plan
    async with asyncio.TaskGroup() as tg:
        tasks: Dict[str, asyncio.Task] = {}

        if plan.get("customer"):
            tasks["customer"] = tg.create_task(
                timed("customer", with_timeout(fetch_customer_async(customer_id), timeout_customer, "customer"))
            )
        if plan.get("product"):
            tasks["product"] = tg.create_task(
                timed("product", with_timeout(fetch_product_async(sku), timeout_product, "product"))
            )
        if plan.get("kb") and base_kb_query:
            tasks["kb"] = tg.create_task(
                timed("kb", with_timeout(fetch_kb_async(base_kb_query), timeout_kb, "kb"))
            )

    # Gather first-pass results
    for name, t in tasks.items():
        val, ms = t.result()  # safe: with_timeout guards exceptions
        results[name] = val
        per_task_ms[name] = ms

    # Optional: reflexive re-plan for KB
    replanned = False
    if plan.get("kb") and replan_on_low_kb:
        kb = results.get("kb", {})
        score = _safe_get_score(kb)
        if score < kb_threshold:
            product_obj = results.get("product", {})
            refined = _refine_kb_query(base_kb_query, product_obj, sku)
            kb_queries_used.append(refined)
            replanned = True

            kb2, ms2 = await timed(
                "kb.refined",
                with_timeout(fetch_kb_async(refined), timeout_kb, "kb.refined"),
            )
            per_task_ms["kb.refined"] = ms2

            score2 = _safe_get_score(kb2)
            if score2 >= score:
                results["kb"] = kb2

    overall_ms = int((time.perf_counter() - overall_t0) * 1000)

    out: Dict[str, Any] = {
        "customer": results.get("customer"),
        "product": results.get("product"),
        "kb": results.get("kb") if plan.get("kb") else None,
        "meta": {
            "plan": plan,
            "kb_threshold": kb_threshold,
            "replanned": replanned,
            "kb_queries": kb_queries_used,
            "per_task_ms": per_task_ms,
            "overall_ms": overall_ms,
        },
    }
    return out


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Parallel multi-source fan-out with tiny planner + optional KB re-plan"
    )
    ap.add_argument("--customer", default="C001", help="customer_id to fetch")
    ap.add_argument("--sku", default="VP-001", help="product SKU to fetch")
    ap.add_argument("--kb", default=None, help="KB query (default: config.DEFAULT_KB_QUERY)")
    ap.add_argument("--timeout-customer", type=float, default=2.0, dest="timeout_customer")
    ap.add_argument("--timeout-product", type=float, default=2.0, dest="timeout_product")
    ap.add_argument("--timeout-kb", type=float, default=2.5, dest="timeout_kb")
    ap.add_argument("--kb-threshold", type=float, default=0.55, dest="kb_threshold")
    ap.add_argument("--replan-on-low-kb", action="store_true", dest="replan_on_low_kb")
    ap.add_argument("--customers-csv", default="customers.csv", dest="customers_csv")
    ap.add_argument("--products-json", default="products.json", dest="products_json")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load datasets once (fast). Your implementation should accept these kwargs.
    initialize_data(customers_csv=args.customers_csv, products_json=args.products_json)

    out = asyncio.run(
        main_flow(
            customer_id=args.customer,
            sku=args.sku,
            kb_query=args.kb,
            timeout_customer=args.timeout_customer,
            timeout_product=args.timeout_product,
            timeout_kb=args.timeout_kb,
            kb_threshold=args.kb_threshold,
            replan_on_low_kb=args.replan_on_low_kb,
        )
    )

    # 🔧 JSON-safe print (handles custom classes)
    print(json.dumps(to_jsonable(out), indent=2, ensure_ascii=False, sort_keys=False))


if __name__ == "__main__":
    main()
