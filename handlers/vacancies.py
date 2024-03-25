import polars as pl
import html

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

t = 0

df = pl.read_parquet("C:/Users/User/Downloads/hh_recsys_vacancies.pq")

def VacancyInfo(i = t) -> str:
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

    vacancy_info = f"<b>{name}</b>\n\n"
    vacancy_info += f"<b>Описание:</b>\n{description}\n\n"
    vacancy_info += f"<b>Условия:</b>\n{parse_work_sch}\n"
    vacancy_info += f"<b>Опыт работы:</b> {exp}\n"
    vacancy_info += f"<b>Трудоустройство:</b> {employment}\n"
    vacancy_info += f"<b>Зарплата:</b> {res_compensation} {compensation_code}\n"
    vacancy_info += f"<b>Ключевые навыки:</b> {skills}\n"

    return vacancy_info
