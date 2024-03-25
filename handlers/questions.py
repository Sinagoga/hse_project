from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder

def choice() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    kb.button(text="Остаться на этой вакансии")
    kb.button(text="Следующая!")
    kb.adjust(2)
    return kb.as_markup(resize_keyboard=True)
