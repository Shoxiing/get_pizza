"""Загрузка датасета few-shot примеров и построение каталога товаров."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

# Записи датасета, которые не являются товарами
NON_PRODUCTS = {
    "Продукт может быть без лука",
    "Письмо от управляющего",
    "Листовка с акцией",
}
# Явные опечатки в разметке
TYPOS = {"Домашняяl": "Домашняя"}


def normalize_name(name: str) -> str:
    name = " ".join(name.split())  # двойные/концевые пробелы
    return TYPOS.get(name, name)


@dataclass(frozen=True)
class Example:
    query: str
    products: tuple[str, ...]  # с повторами: повтор = количество

    def as_order(self) -> list[dict]:
        return [{"name": n, "quantity": q} for n, q in Counter(self.products).items()]


class Catalog:
    def __init__(self, names: list[str]):
        self.names = sorted(set(names))
        self._lower = {n.lower(): n for n in self.names}

    def __contains__(self, name: str) -> bool:
        return name in self._lower.values()

    def __len__(self) -> int:
        return len(self.names)

    def resolve(self, name: str, min_score: float = 85) -> str | None:
        """Привести название от LLM к точному названию из каталога."""
        name = normalize_name(name)
        if exact := self._lower.get(name.lower()):
            return exact
        match = process.extractOne(name, self.names, scorer=fuzz.WRatio, score_cutoff=min_score)
        return match[0] if match else None


def load_examples(path: Path) -> tuple[list[Example], Catalog]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # первая строка датасета — заголовок {"query": "Question", "answer": "Products_name"}
    rows = [r for r in raw if r["query"] != "Question"]

    # Канонизируем регистр: "Грибной Стартер" / "Грибной стартер" -> самый частый вариант
    freq = Counter(normalize_name(p) for r in rows for p in json.loads(r["answer"]))
    variants: dict[str, list[str]] = defaultdict(list)
    for name in freq:
        variants[name.lower()].append(name)
    canon = {v: max(vs, key=freq.__getitem__) for vs in variants.values() for v in vs}

    examples, seen = [], set()
    for r in rows:
        query = " ".join(r["query"].split())
        products = tuple(
            canon[normalize_name(p)]
            for p in json.loads(r["answer"])
            if normalize_name(p) not in NON_PRODUCTS
        )
        key = (query.lower(), tuple(sorted(products)))
        if not products or key in seen:  # точные дубли пар не нужны
            continue
        seen.add(key)
        examples.append(Example(query, products))

    catalog = Catalog([p for e in examples for p in e.products])
    return examples, catalog


def load_prices(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {normalize_name(k): float(v) for k, v in json.load(f).items()}
