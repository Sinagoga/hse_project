from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types.keyboard_button import KeyboardButton
import emoji
def choice() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    builder.row(
        types.KeyboardButton(text=emojize(":thumbs_up:")),
        types.KeyboardButton(text=emojize(":thumbs_down:"))
    )
    builder.row(
        types.KeyBoardButton(text="Остаться на этой вакансии")
    )
    return kb.as_markup(resize_keyboard=True)
    
