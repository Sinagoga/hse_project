from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove

from handlers.questions import choice
from handlers.vacancies import VacancyInfo, t
router = Router()  # [1]

@router.message(Command("start")) 
async def cmd_start(message: Message):
    vacancy = VacancyInfo(t)
    await message.answer(text=vacancy)
    await message.answer(
        "Как вам вакансия?",
        reply_markup=choice()
    )
t = t + 1
@router.message(F.text.lower() == "Остаться на этой вакансии")
async def answer_yes(message: Message):
    await message.answer(
        "Success!",
        reply_markup=ReplyKeyboardRemove()
    )

@router.message(F.text.lower() == "Следующая")
async def answer_no(message: Message):
    vacancy = VacancyInfo(t)
    await message.answer(
        text=vacancy,
        reply_markup=choice()
    )
