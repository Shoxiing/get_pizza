import asyncio
import logging

from pizza_bot.app import run
from pizza_bot.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    asyncio.run(run(settings))
