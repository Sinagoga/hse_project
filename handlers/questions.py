from aiogram.types import ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types.keyboard_button import KeyboardButton
import emoji
def choice() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardBuilder()
    kb.add(KeyboardButton(text="Остаться на этой вакансии"))
    kb.add(KeyboardButton(text=emoji.emojize(":thumbs_up:")))
    kb.add(KeyboardButton(text=emoji.emojize(":thumbs_down:")))
    kb.adjust(3)
    return kb.as_markup(resize_keyboard=True)
