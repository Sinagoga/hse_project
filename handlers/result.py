from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove
import random

from handlers.questions import choice
from handlers.vacancies import VacancyInfo, t
router = Router()  # [1]

@router.message(Command("start")) 
async def cmd_start(message: Message):
    vacancy = VacancyInfo()
    await message.answer(text=vacancy)
    await message.answer(
        "Как вам вакансия?",
        reply_markup=choice()
    )

@router.message(F.text.lower() == "остаться на этой вакансии")
async def answer_yes(message: Message):
    await message.answer(
        text="Надеюсь, я смог вам помочь в выборе вакансии",
        reply_markup=ReplyKeyboardRemove()
    )

@router.message(F.text.lower() == "следующая!")
async def answer_no(message: Message):
    vacancy = VacancyInfo([random.randint(1, 10)])
    await message.answer(
        text=vacancy,
        reply_markup=choice()
    )
