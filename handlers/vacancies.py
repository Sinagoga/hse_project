import polars as pl
from html.parser import HTMLParser

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

df = pl.read_parquet("C:\labs\hh_hack\hh_recsys_vacancies.pq")
labels = pl.read_parquet("C:\labs\hh_hack\hh_vac2idx_data.pq")
vac2idx = {labels["vacancy_id"][i]: int(labels["idx"][i]) for i in range(labels.shape[0])}
vac_rate = pl.read_parquet("C:\labs\hh_hack\vac_rate.pq").with_columns(pl.col("vacancy_id").cast(int))

df = df.filter(pl.col("vacancy_id").is_in(labels["vacancy_id"]))
df = df.with_columns(df["vacancy_id"].replace(vac2idx).cast(int))

s = ""
class MHP(HTMLParser):
    def handle_data(self, data):
        global s
        s+=data+"\n\n" if data != " " else ""
p = MHP()

def VacancyInfo(i = t) -> str:
    parse_vac = df.filter(pl.col("vacancy_id") == i)
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

    p.feed(description)
    global s
    vacancy_info = f'''
<b>{name}</b>

{s}
<b>Компенсация:</b> {res_compensation} {compensation_code}

<b>Ключевые навыки:</b> {skills}

<b>Трудоустройтво:</b> {employment}

<b>Расписание:</b> {parse_work_sch}

<b>Опыт работы:</b> {exp}
'''

    s=""

    return vacancy_info