import asyncio
import logging
import aiogram
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.enums.dice_emoji import DiceEmoji

TOKEN = "12345678"

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_name = message.from_user.full_name
    await message.reply(f"Привет, {user_name}!")

@dp.message(Command("place"))
async def place(message: types.Message):
    kb = [[types.KeyBoardButton(text="Онлайн")],
          [types.KeyBoardButton(text="В офисе")]
          ]
    keyboard = types.ReplyKeyBoardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder = "Выберите формат"
    )
    await message("Онлайн/оффлайн", reply_markup=keyboard)

@dp.message(Command("exp"))
async def exp(message: types.Message):
    kb = [[types.KeyBoardButton(text="Есть опыт работы")],
          [types.KeyBoardButton(text="Без опыта работы")]
          ]
    keyboard = types.ReplyKeyBoardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder = "Опыт"
    )
    await message("Опыт", reply_markup=keyboard)

@dp.message(Command("Activity"))
async def place(message: types.Message):
    kb = [[types.KeyBoardButton(text="IT")],
          [types.KeyBoardButton(text="Медицина")],
          [types.KeyBoardButton(text="Преподавание")],
          [types.KeyBoardButton(text="Дизайн")],
          [types.KeyBoardButton(text="Экономика")],
          [types.KeyBoardButton(text="Администратор")]
          [types.KeyBoardButton(text="Право")]
          ]
    keyboard = types.ReplyKeyBoardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder = "Сфера деятельности"
    )
    await message("Сфера деятельности", reply_markup=keyboard)
async def main():
    await dp.start_polling(bot)

if __name__ = "__main__":
    asyncio.run(main())
