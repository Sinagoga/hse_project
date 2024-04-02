from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove
import random
import polars as pl

from handlers.questions import choice
from handlers.vacancies import VacancyInfo, labels, vac_rate
router = Router()  # [1]

counter = 0
user = [0]*labels.shape[0]

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
    global counter
    if counter < 10:
        counter+=1
        ind = random.randint(0, labels.shape[0])
        vacancy = VacancyInfo(ind)
        user[ind] = 1*vac_rate.filter(pl.col("vacancy_id")==ind)["rate"].item()
    else: # model plug in
        vacancy = VacancyInfo(0)
    await message.answer(
        text=vacancy,
        reply_markup=choice()
    )
