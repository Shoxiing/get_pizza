"""Векторный поиск похожих примеров запрос -> корзина.

Ключевые решения:
* 1 документ = 1 пара (запрос, корзина). Никакого text splitter'а: он резал
  длинные пары, и ответ отрывался от вопроса.
* Эмбеддится "запрос: [товары]": названия товаров в тексте помогают находить
  примеры по корзине (запросы вида "Ветчина и грибы Small").
* Нормализованные эмбеддинги + cosine similarity.
* k реально передаётся в поиск (раньше `as_retriever(k=10)` молча
  игнорировался и всегда возвращалось 4 документа).
"""

import json

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pizza_bot.catalog import Example


def build_embeddings(model_name: str, device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def doc_text(e: Example) -> str:
    return f"{e.query}: {json.dumps(list(e.products), ensure_ascii=False)}"


def build_vectorstore(examples: list[Example], embeddings: HuggingFaceEmbeddings) -> FAISS:
    docs = [
        Document(page_content=doc_text(e), metadata={"query": e.query, "products": list(e.products)})
        for e in examples
    ]
    return FAISS.from_documents(
        docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )


class ExampleRetriever:
    def __init__(self, store: FAISS, k: int):
        self.store = store
        self.k = k

    def _to_examples(self, docs: list[Document]) -> list[Example]:
        # одинаковые запросы с разной разметкой не дублируем в промпте
        seen, out = set(), []
        for d in docs:
            key = (d.metadata["query"].lower(), tuple(sorted(d.metadata["products"])))
            if key in seen:
                continue
            seen.add(key)
            out.append(Example(d.metadata["query"], tuple(d.metadata["products"])))
        return out

    async def aretrieve(self, query: str) -> list[Example]:
        docs = await self.store.asimilarity_search(query, k=self.k)
        return self._to_examples(docs)

    def retrieve(self, query: str) -> list[Example]:
        return self._to_examples(self.store.similarity_search(query, k=self.k))
