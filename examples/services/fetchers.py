from __future__ import annotations
import asyncio, csv, json
from pathlib import Path
from typing import Dict, Any, Optional, List


from examples.services.models import Customer, Product, KBResult
from examples.services.config import TIMEOUTS, RETRY

# ---- Imports & fallbacks ----------------------------------------------------
try:
    from common.data_loader import load_csv as _load_csv, load_json as _load_json
except Exception:
    _load_csv = None
    _load_json = None

try:
    from common.rag_db import rag_answer as _rag_answer
except Exception:
    _rag_answer = None

# ---- Local fallbacks (no extra deps) ----------------------------------------
def _fallback_load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def _fallback_load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _fallback_rag_answer(query: str):
    # Return a harmless default if RAG backend is unavailable
    return ("No KB available (fallback).", 0.0, [])

def load_csv(path: str) -> List[Dict[str, Any]]:
    if _load_csv:
        return _load_csv(path)
    return _fallback_load_csv(path)

def load_json(path: str) -> Any:
    if _load_json:
        return _load_json(path)
    return _fallback_load_json(path)

def rag_answer(query: str):
    if _rag_answer:
        return _rag_answer(query)
    return _fallback_rag_answer(query)

# ---- Data caches ------------------------------------------------------------
_CUSTOMERS: Dict[str, Customer] = {}
_PRODUCTS_BY_SKU: Dict[str, Product] = {}

def initialize_data(
    customers_csv: str = "customers.csv",
    products_json: str = "products.json",
) -> None:
    global _CUSTOMERS, _PRODUCTS_BY_SKU
    # Customers
    _CUSTOMERS = {
        str(r.get("customer_id", "")).strip(): Customer.from_dict(r)
        for r in load_csv(customers_csv)
        if str(r.get("customer_id", "")).strip()
    }
    # Products (index by SKU for O(1) lookup)
    products = load_json(products_json) or []
    _PRODUCTS_BY_SKU = {}
    for p in products:
        prod = Product.from_dict(p)
        if prod.sku:
            _PRODUCTS_BY_SKU[prod.sku] = prod

# ---- Retry helper (async) ---------------------------------------------------
async def _with_retry(func, *args, attempts: int = None, backoff_s: float = None, **kwargs):
    attempts = attempts or RETRY.attempts
    backoff_s = backoff_s or RETRY.backoff_s
    last_err: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                await asyncio.sleep(backoff_s * (i + 1))
    if last_err:
        raise last_err

# ---- Async fetchers with timeouts ------------------------------------------
async def fetch_customer_async(customer_id: str) -> Optional[Customer]:
    async def _inner():
        await asyncio.sleep(0)  # yield control, mimic I/O
        return _CUSTOMERS.get(customer_id)
    try:
        return await asyncio.wait_for(
            _with_retry(_inner),
            timeout=TIMEOUTS.customer_s
        )
    except Exception:
        return None

async def fetch_product_async(sku: str) -> Optional[Product]:
    async def _inner():
        await asyncio.sleep(0)
        return _PRODUCTS_BY_SKU.get(sku)
    try:
        return await asyncio.wait_for(
            _with_retry(_inner),
            timeout=TIMEOUTS.product_s
        )
    except Exception:
        return None

async def fetch_kb_async(query: str) -> Optional[KBResult]:
    async def _inner():
        # rag_answer is sync: run in a thread so we don't block the loop
        ans, score, _chunks = await asyncio.to_thread(rag_answer, query)
        return KBResult(answer=ans or "", score=float(score or 0.0))
    try:
        return await asyncio.wait_for(
            _with_retry(_inner),
            timeout=TIMEOUTS.kb_s
        )
    except Exception:
        return None
