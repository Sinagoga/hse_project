import asyncio
import logging
from aiogram import Bot, Dispatcher, Router
from handlers import result

TOKEN = "6988515939:AAGuc60wFwC6nccacfiC1c3MNZTxFCe7h_Y"

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN, parse_mode="HTML")
dp = Dispatcher(bot=bot)
router = Router()

async def main():
    dp.include_router(result.router)
    try:
        await dp.start_polling(bot)

    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
