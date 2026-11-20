"""RAG-цепочка: запрос -> похожие примеры -> LLM со structured output -> корзина."""

import logging
from collections import Counter

from langchain_openai import ChatOpenAI

from pizza_bot.cart import Cart
from pizza_bot.catalog import Catalog
from pizza_bot.config import Settings
from pizza_bot.llm.prompts import build_prompt, format_examples
from pizza_bot.llm.schemas import Order, make_order_schema
from pizza_bot.retriever import ExampleRetriever

log = logging.getLogger(__name__)


def build_llm(settings: Settings) -> ChatOpenAI:
    extra_body = {}
    if settings.llm_disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.request_timeout_s,
        max_retries=2,
        extra_body=extra_body or None,
    )


class OrderChain:
    def __init__(self, settings: Settings, retriever: ExampleRetriever, catalog: Catalog):
        self.retriever = retriever
        self.catalog = catalog

        schema = make_order_schema(catalog.names) if settings.llm_strict_catalog else Order
        llm = build_llm(settings).with_structured_output(schema, method=settings.llm_structured_method)
        self.chain = build_prompt(catalog.names) | llm

    async def ainvoke(self, question: str) -> Cart:
        examples = await self.retriever.aretrieve(question)
        order: Order = await self.chain.ainvoke(
            {"question": question, "examples": format_examples(examples)}
        )
        return self._to_cart(order)

    def _to_cart(self, order: Order) -> Cart:
        counts: Counter[str] = Counter()
        dropped = []
        for item in order.items:
            raw = getattr(item.name, "value", item.name)  # Enum или str
            name = self.catalog.resolve(raw)
            if name is None:
                dropped.append(raw)
                continue
            counts[name] += item.quantity
        if dropped:
            log.warning("LLM вернула товары не из меню: %s", dropped)
        return Cart(items=list(counts.items()), comment=order.comment, dropped=dropped)
