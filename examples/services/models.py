from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class Customer:
    customer_id: str
    name: Optional[str] = None
    segment: Optional[str] = None
    email: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Customer":
        return Customer(
            customer_id=str(d.get("customer_id", "")).strip(),
            name=d.get("name"),
            segment=d.get("segment"),
            email=d.get("email"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Product:
    sku: str
    title: Optional[str] = None
    price_inr: Optional[float] = None
    category: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Product":
        price = d.get("price_inr")
        try:
            price = float(price) if price is not None else None
        except Exception:
            price = None
        return Product(
            sku=str(d.get("sku", "")).strip(),
            title=d.get("title"),
            price_inr=price,
            category=d.get("category"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class KBResult:
    answer: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "score": self.score}

@dataclass
class AggregateResult:
    customer: Optional[Customer]
    product: Optional[Product]
    kb: Optional[KBResult]
    latency_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer": self.customer.to_dict() if self.customer else None,
            "product": self.product.to_dict() if self.product else None,
            "kb": self.kb.to_dict() if self.kb else None,
            "latency_ms": self.latency_ms,
        }
