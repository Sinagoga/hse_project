from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import emoji
def choice() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    kb.add(KeyBoardButton(text="Остаться на этой вакансии"))
    kb.add(KeyboardButton(text=emojize(":thumbs_up:")))
    kb.add(KeyboardButton(text=emojize(":thumbs_down:")))
    kb.adjust(3)
    return kb.as_markup(resize_keyboard=True)
