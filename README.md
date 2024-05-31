# Visual Question Answering
Создание модели для Visual Question Answering и других image+text задач в рамках весеннего проекта, ПАДИИ ВШЭ СПб, весна 2024
## Описание
Модель была обучена на датасете VQAv2, который был переведен на русский язык.  
VQA — это задача, цель которой состоит в том, чтобы предоставлять ответы на открытые вопросы, основанные на анализе соответствующего изображения.
Входные данные для модели представляют собой комбинацию изображения и вопроса, а выходные данные — ответ, выраженный на естественном языке
## Структура проекта
- <code>src</code> - актуальная модель  
- <code>experiments</code> - эксперименты с моделью  
- <code>Translation</code> - перевод данных  
- <code>LLM_tune</code> - работа с моделями LLM  
    - <code>LLM_tune/Experiments</code> - эксперименты с LLM  
    - <code>LLM_tune/FRED-T5-tune</code> - акутальная LLM  
- <code>telegram_bot</code> - весь код, связанный с телеграм ботом 
## Демонстрация
Можно пощупать как это работает [тут](https://t.me/VQA_project_bot)     
Взаимодействие с ботом:
- Отправить картинку с вопросом в одном сообщении;
- Дождаться ответа и продолжить использование бота;
- Управление ботом реализовано с помощью удобных кнопок  

## Члены команды
[Купряков Дмитрий](https://t.me/qeqochec) - файнтюн LLM, соединение главной модели с ботом, перевод датасетов  
[Нам Алина](https://t.me/Nam_Alina) - написание бота, перевод датасетов  
[Хамаганов Ильдар](https://t.me/Ovllee) - написание главной модели и её обучение

