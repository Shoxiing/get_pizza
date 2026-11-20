"""Оценка качества: retrieval (leave-one-out) и end-to-end сборка корзины.

Запуск из bot/:
    python -m scripts.evaluate retrieval
    python -m scripts.evaluate e2e --n 150 [--model Qwen/Qwen3-14B-AWQ] [--base-url http://localhost:8000/v1]

E2E: отложенная выборка запросов (их нет в индексе), LLM собирает корзину,
сравниваем с разметкой. Метрики:
  item_f1   — F1 по мультимножеству товаров (с учётом количества)
  base_f1   — то же, но без размера/объёма ("Пепперони Medium" == "Пепперони Large")
  qty_acc   — совпало общее число позиций
  exact     — корзина совпала полностью
"""

import argparse
import asyncio
import random
import re
import statistics
import time
from collections import Counter

import numpy as np

from pizza_bot.app import build_order_chain
from pizza_bot.catalog import Example, load_examples
from pizza_bot.config import settings
from pizza_bot.retriever import build_embeddings, doc_text

SIZE_RE = re.compile(r"\s+(Small|Medium|Large|Big|\d+([.,]\d+)?)$")


def base(name: str) -> str:
    return SIZE_RE.sub("", name)


def f1(gold: Counter, pred: Counter) -> float:
    if not gold and not pred:
        return 1.0
    tp = sum((gold & pred).values())
    if tp == 0:
        return 0.0
    p, r = tp / sum(pred.values()), tp / sum(gold.values())
    return 2 * p * r / (p + r)


def split(examples: list[Example], n_test: int, seed: int = 42):
    keys = sorted({e.query.lower() for e in examples})
    random.Random(seed).shuffle(keys)
    test_keys = set(keys[:n_test])
    train = [e for e in examples if e.query.lower() not in test_keys]
    test = list({e.query.lower(): e for e in examples if e.query.lower() in test_keys}.values())
    return train, test


def eval_retrieval(examples, ks=(4, 8, 12)):
    """Leave-one-out: доля товаров из правильной корзины, встречающихся в найденных примерах."""
    embeddings = build_embeddings(settings.embedding_model, settings.embedding_device)
    doc_emb = np.array(embeddings.embed_documents([doc_text(e) for e in examples]))
    q_emb = np.array(embeddings.embed_documents([e.query for e in examples]))
    sims = q_emb @ doc_emb.T
    keys = np.array([e.query.lower() for e in examples])
    for k in ks:
        recall = []
        for i, e in enumerate(examples):
            s = np.where(keys == keys[i], -np.inf, sims[i])
            got = {p for j in np.argsort(-s)[:k] for p in examples[j].products}
            recall.append(len(set(e.products) & got) / len(set(e.products)))
        print(f"k={k:2d} product_recall={np.mean(recall):.3f}")


async def eval_e2e(examples, catalog, n_test, concurrency):
    train, test = split(examples, n_test)
    # в индексе только train, меню — полное
    chain = build_order_chain(settings, examples=train, catalog=catalog)
    sem = asyncio.Semaphore(concurrency)

    async def run(e: Example):
        async with sem:
            t0 = time.perf_counter()
            try:
                cart = await chain.ainvoke(e.query)
                return e, cart, time.perf_counter() - t0, None
            except Exception as exc:  # noqa: BLE001
                return e, None, time.perf_counter() - t0, repr(exc)[:200]

    t_all = time.perf_counter()
    results = await asyncio.gather(*(run(e) for e in test))
    wall = time.perf_counter() - t_all

    m = {"item_f1": [], "base_f1": [], "qty_acc": [], "exact": []}
    lat, errors, dropped = [], [], 0
    for e, cart, dt, err in results:
        lat.append(dt)
        gold = Counter(e.products)
        if cart is None:
            errors.append((e.query, err))
            pred = Counter()
        else:
            pred = Counter(dict(cart.items))
            dropped += len(cart.dropped)
        m["item_f1"].append(f1(gold, pred))
        m["base_f1"].append(f1(Counter(base(p) for p in gold.elements()),
                               Counter(base(p) for p in pred.elements())))
        m["qty_acc"].append(sum(gold.values()) == sum(pred.values()))
        m["exact"].append(gold == pred)

    print(f"model={settings.llm_model} n={len(test)} errors={len(errors)} off_menu_items={dropped}")
    print("  " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in m.items()))
    print(f"  latency p50={statistics.median(lat):.2f}s max={max(lat):.2f}s wall={wall:.1f}s")
    for q, err in errors[:5]:
        print("  ERR", q, err)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrieval", "e2e"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--show", type=int, default=0, help="напечатать N примеров предсказаний")
    args = ap.parse_args()
    if args.model:
        settings.llm_model = args.model
    if args.base_url:
        settings.llm_base_url = args.base_url

    examples, catalog = load_examples(settings.data_path)
    if args.mode == "retrieval":
        eval_retrieval(examples)
    else:
        results = asyncio.run(eval_e2e(examples, catalog, args.n, args.concurrency))
        for e, cart, _, _ in results[: args.show]:
            print(f"\nQ: {e.query}\n  gold: {dict(Counter(e.products))}\n  pred: {cart and dict(cart.items)}"
                  f"{'  comment: ' + cart.comment if cart and cart.comment else ''}")


if __name__ == "__main__":
    main()
