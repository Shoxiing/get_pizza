from enum import Enum

from pydantic import BaseModel, Field, create_model


class OrderItem(BaseModel):
    name: str = Field(description="Точное название товара из меню")
    quantity: int = Field(ge=1, le=50, description="Количество")


class Order(BaseModel):
    items: list[OrderItem] = Field(
        max_length=20, description="Позиции корзины. Пусто, если ничего из меню не подходит"
    )
    comment: str | None = Field(
        default=None,
        max_length=200,
        description="Только если запрос нельзя выполнить как есть (товара нет в меню, "
        "запрос не про еду) — одно короткое предложение для клиента. Иначе null",
    )


def make_order_schema(catalog_names: list[str]) -> type[Order]:
    """Order, у которого name — enum из каталога.

    При constrained decoding (vLLM, response_format=json_schema) модель физически
    не сможет вернуть товар, которого нет в меню.
    """
    product_enum = Enum("Product", {f"p{i}": n for i, n in enumerate(catalog_names)}, type=str)
    item = create_model(
        "OrderItem",
        __base__=OrderItem,
        name=(product_enum, Field(description="Точное название товара из меню")),
    )
    return create_model("Order", __base__=Order, items=(list[item], Field(max_length=20)))
