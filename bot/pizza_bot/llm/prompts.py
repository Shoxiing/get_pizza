import json

from langchain_core.prompts import ChatPromptTemplate

from pizza_bot.catalog import Example

# Системный промпт статичен (правила + меню) -> vLLM переиспользует KV-кэш
# этого префикса между запросами (--enable-prefix-caching).
SYSTEM_PROMPT = """Ты — оператор пиццерии. По сообщению клиента собери корзину заказа.

Правила:
- Используй только товары из МЕНЮ ниже, названия пиши точно как в меню (включая размер: Small / Medium / Large, объём и т.п.).
- Если размер не указан — бери Medium для пиццы, для остального — вариант из похожих примеров.
- Количество указывай в поле quantity, а не повтором позиции.
- "Разные" / "на выбор" — выбирай разные товары. Похожие примеры показывают, как интерпретировать запросы.
- Если нужного товара нет в меню — предложи ближайшую замену из меню; если замены нет или запрос не про еду — верни пустой items.
- comment — null, если заказ собран как просили. Заполняй одним коротким предложением по-русски только при замене или отказе. Не объясняй свой выбор.

МЕНЮ:
{menu}"""

USER_PROMPT = """Похожие запросы и правильные корзины:
{examples}

Запрос клиента: {question}"""


def build_prompt(menu: list[str]) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
    return prompt.partial(menu="\n".join(menu))


def format_examples(examples: list[Example]) -> str:
    return "\n".join(
        f"Запрос: {e.query}\nКорзина: {json.dumps(e.as_order(), ensure_ascii=False)}"
        for e in examples
    )
