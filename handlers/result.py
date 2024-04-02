from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove
import random
import emoji
from handlers.questions import choice
from handlers.vacancies import VacancyInfo, labels
router = Router()  # [1]

counter = 0

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

@router.message(F.emoji(emoji.emojize(":thumbs_down:")))
async def answer_no(message: Message):
    global counter
    if counter < 10:
        counter+=1
        vacancy = VacancyInfo(random.randint(0, labels.shape[0]))
    else: # model plug in
        vacancy = VacancyInfo(0)
    await message.answer(
        text=vacancy,
        reply_markup=choice()
    )

@router.message(F.emoji(emoji.emojize(":thumbs_up:")))
async def answer_yes(message: Message):
    #add to Favourites?
