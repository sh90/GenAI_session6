# session6/common/data_loader.py
from __future__ import annotations
import os, csv, json
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_csv(name: str) -> list[dict]:
    path = os.path.join(DATA_DIR, name)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def load_json(name: str) -> Any:
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
