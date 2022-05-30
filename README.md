Проект: "Прогнозирование объема продаж на период по данным об объемах продаж за несколько предшествующих лет"

Целью проекта является исследование практической применимости статистического метода прогнозирования объемов продаж и получение прогнозных данных об объеме продаж на требуемый период в розничной сети производителя мягкой мебели.
Данные представлены файлом ts_days.csv и ts_days_prov.csv, признаковое пространство содержит объем продаж на дату.

Вопросы от заказчика:
Необходимо получить прогноз объемов продаж на некоторый период (в разрезе дня - на 90 дней, в разрезе недели - на 12 недель)

Описание признаков: (данные об объемах продаж за несколько лет):
1 - date: дата продаж
2 - summ: объем продаж

Исследование переменных
В рамках работы предполагается провести стандартное исследование переменных: пропущенные значения, выбросы.

Модели и их настройка
По результатам предварительной подготовки планируется провести исследование нескольких моделей машинного обучения для решения задачи прогнозирования объема продаж.

Выбор финальной модели
Выбор финальной модели планируется проводить на основе критерия "rmse" на нормализованном (0;1) наборе данных.
