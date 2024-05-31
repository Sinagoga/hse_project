# Visual Question Answering
Создание модели для Visual Question Answering и других image+text задач в рамках весеннего проекта ВШЭ ПАДИИ 2024. 
## Описание
Модель была обучена на переведенном на русский датасет VQAv2.  
VQA — это задача ответа на открытые вопросы на основе изображения.Входные данные для модели представляют собой комбинацию изображения и вопроса, а выходные данные — ответ, выраженный на естественном языке
## Структура проекта
- <code>src</code> - актуальная модель  
- <code>experiments</code> - эксперименты с моделью  
- <code>Datasets</code> - форматирование данных  
- <code>Translation</code> - перевод данных  
- <code>LLM_tune</code> - работа с моделями LLM  
    - <code>LLM_tune/Experiments</code> - эксперименты с LLM  
    - <code>LLM_tune/FRED-T5-tune</code> - акутальная LLM  
- <code>accelerate_config</code> - конфиг-файл для accelerate

## Демонстрация
Можно пощупать как работает:  
[VQA_project_bot](https://t.me/VQA_project_bot)
