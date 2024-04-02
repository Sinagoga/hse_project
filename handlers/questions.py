from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import emoji
def choice() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    kb.button(text="Остаться на этой вакансии")
    kb.button(emoji.emojize(":thumbs_up:"))
    kb.button(emoji.emojize(":thumbs_down:"))
    kb.adjust(3)
    return kb.as_markup(resize_keyboard=True)
