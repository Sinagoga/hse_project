import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.keyboard import ReplyKeyboardBuilder, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardBuilder
import pandas as pd
from aiogram.fsm.state import State, StatesGroup
from IPython.display import display, HTML
from aiogram.fsm.context import FSMContext
TOKEN = "6988515939:AAGuc60wFwC6nccacfiC1c3MNZTxFCe7h_Y"

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

df = pd.read_parquet("vacancies.pq")

cur_code2rus = {"KZT": "₸", "BYR": "Br.", "EUR": "€", "KGS": "som", "RUR": "₽", "USD": "$", "UZS": "so'm"}

sch2rus = {
    "fullDay": "Полный рабочий день",
    "remote": "Удаленная работа",
    "flexible": "Гибкий график",
    "shift": "Работа по сменам",
    "flyInFlyOut": "Работа на вылет"
}

empl2rus = {
    "part": "Частичная занятость",
    "probation": "Испытательный срок",
    "full": "Полная занятость",
    "project": "Проектная работа",
    "volunteer": "Волонтерство"
}

exp2rus = {
    "moreThan6": "более 6 лет",
    "between1And3": "от 1 до 3 лет",
    "between3And6": "от 3 до 6 лет",
    "noExperience": "без опыта работы"
}
i = 0
async def VacancyInfo(message: types.Message, i: int):
    parse_vac = df[i]
    name = parse_vac["name"].item()
    description = parse_vac["description"].item()
    compensation_from = parse_vac["compensation.from"].item() if not (
                parse_vac["compensation.from"].item() is None) else 0
    compensation_to = parse_vac["compensation.to"].item() if not (
                parse_vac["compensation.to"].item() is None) else compensation_from
    res_compensation = f"{compensation_from} - {compensation_to}" if compensation_from != compensation_to else compensation_from
    compensation_code = cur_code2rus[parse_vac["compensation.currencyCode"].item()] if not (
                parse_vac["compensation.currencyCode"].item() is None) else ""
    skills = ", ".join(parse_vac["keySkills.keySkill"].item().to_list()) if not (
                parse_vac["keySkills.keySkill"].item() is None) else "Ключевые навыки не требуются"
    employment = empl2rus[parse_vac["employment"].item()]
    parse_work_sch = sch2rus[parse_vac["workSchedule"].item()]
    exp = exp2rus[parse_vac["workExperience"].item()]

    vacancy_info = f"<h3>{name}</h3>\n\n"
    vacancy_info += f"<b>Описание:</b>\n{description}\n\n"
    vacancy_info += f"<b>Условия:</b>\n{parse_work_sch}\n"
    vacancy_info += f"<b>Опыт работы:</b> {exp}\n"
    vacancy_info += f"<b>Трудоустройство:</b> {employment}\n"
    vacancy_info += f"<b>Зарплата:</b> {res_compensation} {compensation_code}\n"
    vacancy_info += f"<b>Ключевые навыки:</b> {skills}\n"
    i += 1
    await message.answer(vacancy_info, parse_mode="HTML")
    

class VacancyState(StatesGroup):
    SHOW_VACANCY = State()
    WAIT_FOR_CHOICE = State()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Показать вакансии?")
    await VacancyState.SHOW_VACANCY.set()
    await show_next_vacancy(message)

async def show_next_vacancy(message: types.Message):
    await VacancyInfo(message="Новая вакансия")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton(text="Выбрать эту вакансию"),
               types.KeyboardButton(text="Следующая")
               )
    await VacancyState.WAIT_FOR_CHOICE.set()

@dp.message(state=VacancyState.WAIT_FOR_CHOICE)
async def handle_choice(message: types.Message, state: FSMContext):
    if message.text == "Выбрать эту вакансию":
        await message.answer("Success!")
        await state.clear()
    elif message.text == "Следующая":
        await show_next_vacancy(message)
        
async def main():
    dp.message.register(start)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())


