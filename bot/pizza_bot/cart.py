"""Корзина заказа и её текстовое представление для Telegram."""

from dataclasses import dataclass, field

EMPTY_CART_TEXT = "Не нашёл в меню подходящих товаров. Уточните, пожалуйста, заказ."


@dataclass
class Cart:
    items: list[tuple[str, int]] = field(default_factory=list)  # (товар, количество)
    comment: str | None = None
    dropped: list[str] = field(default_factory=list)  # названия от LLM, которых нет в меню


def render_cart(cart: Cart, prices: dict[str, float]) -> str:
    if not cart.items:
        return cart.comment or EMPTY_CART_TEXT

    lines = ["Ваш заказ:"]
    total, all_priced = 0.0, True
    for name, qty in cart.items:
        price = prices.get(name)
        if price is None:
            all_priced = False
            lines.append(f"• {name} × {qty}")
        else:
            total += price * qty
            lines.append(f"• {name} × {qty} — {price * qty:g} ₽")
    if prices and all_priced:
        lines.append(f"Итого: {total:g} ₽")
    if cart.comment:
        lines += ["", cart.comment]
    return "\n".join(lines)
