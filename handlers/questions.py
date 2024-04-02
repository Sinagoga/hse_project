from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from aiogram.types.keyboard_button import KeyboardButton
import emoji
def choice() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    builder = InlineKeyboardBuilder()
    builder.row(
        KeyboardButton(text=emoji.emojize(":thumbs_up:")),
        KeyboardButton(text=emoji.emojize(":thumbs_down:"))
    )
    builder.row(
        KeyboardButton(text="Остаться на этой вакансии")
    )
    return kb.as_markup(resize_keyboard=True)
    
