from dataclasses import dataclass

@dataclass(frozen=True)
class Timeouts:
    customer_s: float = 1.0
    product_s: float = 1.0
    kb_s: float = 2.5

@dataclass(frozen=True)
class Retry:
    attempts: int = 2
    backoff_s: float = 0.15  # seconds

# Single place to tweak behavior
TIMEOUTS = Timeouts()
RETRY = Retry()
DEFAULT_KB_QUERY = "common issues in pricing"
