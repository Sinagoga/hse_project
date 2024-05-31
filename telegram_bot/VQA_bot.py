import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv

from PIL import Image
from io import BytesIO

import bot_utils as model
from pathlib import Path

import os


load_dotenv()
TOKEN = os.getenv('TOKEN')
bot = Bot(TOKEN)
dp = Dispatcher(bot=bot)


start_markup = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Загрузить картинку")],
        [KeyboardButton(text="Завершить работу")]
    ],
    resize_keyboard=True
)


after_upload_markup = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Загрузить еще картинку")],
        [KeyboardButton(text="Завершить работу")]
    ],
    resize_keyboard=True
)


@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Начать?", reply_markup=start_markup)


@dp.message(lambda message: message.text == "Загрузить картинку")
async def prompt_image(message: types.Message):
    await message.answer("Пожалуйста, отправьте картинку и задайте вопрос.")


@dp.message(lambda message: message.content_type == types.ContentType.PHOTO)
async def handle_image(message: types.Message):
    photo = message.photo[-1]
    caption = message.caption
    photo_id = photo.file_id
    file_info = await bot.get_file(photo_id)
    file_path = file_info.file_path
    image = await bot.download_file(file_path)
    image = Image.open(BytesIO(image.read())).convert("RGB")
    await message.answer(f"Картинка получена.\n\n Ответ:{model.bot_pref(image, caption)}\n\n Что вы хотите сделать дальше?", reply_markup=after_upload_markup)


@dp.message(lambda message: message.text == "Загрузить еще картинку")
async def prompt_more_image(message: types.Message):
    await message.answer("Пожалуйста, отправьте еще одну картинку.")


@dp.message(lambda message: message.text == "Завершить работу")
async def finish(message: types.Message):
    await message.answer("Работа завершена. Спасибо за использование бота!", reply_markup=types.ReplyKeyboardRemove())


async def main():
    dp.message.register(start, Command("start"))
    dp.message.register(prompt_image, lambda message: message.text == "Загрузить картинку")
    dp.message.register(handle_image, lambda message: message.content_type == types.ContentType.PHOTO)
    dp.message.register(prompt_more_image, lambda message: message.text == "Загрузить еще картинку")
    dp.message.register(finish, lambda message: message.text == "Завершить работу")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
