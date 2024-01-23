#!/usr/bin/env python
# coding: utf-8

# __Развлекательное приложение Procrastinate Pro+__
# 
# __Цель проекта :__
# Опредлить: откуда приходят пользователи и какими устройствами они пользуются,
# сколько стоит привлечение пользователей из различных рекламных каналов;
# сколько денег приносит каждый клиент,
# когда расходы на привлечение клиента окупаются,
# какие факторы мешают привлечению клиентов.

# __Общий план выполнения проекта :__
# 1.Загрузить данные и подготовить их к анализу
# 2.Задать функции для расчёта и анализа LTV, ROI, удержания и конверсии.
# 3.Выполнить исследовательский анализ данных
# 4.Маркетинг (посчитать затраты , визуализировать основные параметры на графиках )
# 5.Оценить окупаемость рекламы
# 6.Написать общие вывод и дать рекомендации.

# Для удобства добавим описание данных
# Структура visits_info_short.csv:
# User Id — уникальный идентификатор пользователя,
# Region — страна пользователя,
# Device — тип устройства пользователя,
# Channel — идентификатор источника перехода,
# Session Start — дата и время начала сессии,
# Session End — дата и время окончания сессии.
# Структура orders_info_short.csv:
# User Id — уникальный идентификатор пользователя,
# Event Dt — дата и время покупки,
# Revenue — сумма заказа.
# Структура costs_info_short.csv:
# dt — дата проведения рекламной кампании,
# Channel — идентификатор рекламного источника,
# costs — расходы на эту кампанию.

# ### Загрузка данных и подготовка к анализу
# 

# Загрузите данные о визитах, заказах и рекламных расходах из CSV-файлов в переменные.
# 
# **Пути к файлам**
# 
# - визиты: `/datasets/visits_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/visits_info_short.csv);
# - заказы: `/datasets/orders_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/orders_info_short.csv);
# - расходы: `/datasets/costs_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/costs_info_short.csv).
# 
# Изучите данные и выполните предобработку. Есть ли в данных пропуски и дубликаты? Убедитесь, что типы данных во всех колонках соответствуют сохранённым в них значениям. Обратите внимание на столбцы с датой и временем.

# In[1]:


#Импортируем библиотеки
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


# In[2]:


#Присваиваем переменные датасетам
visits=pd.read_csv('/datasets/visits_info_short.csv') #Визиты
orders=pd.read_csv('/datasets/orders_info_short.csv') #Заказы
costs=pd.read_csv('/datasets/costs_info_short.csv') # Расходы


# In[3]:


#Ознакомимся с таблицей визитов
visits.head() 


# In[4]:


#Выведем информацию по таблице визитов
visits.info()
# Обратим внимание на Session start u session end - dtype должен быть формата datetime, object не подойдёт


# In[5]:


#Ознакомимся с таблицей orders 
orders.head()


# In[6]:


#Выведем информацию по таблице заказов 
orders.info()
#Обратим внимание на столбец Event Dt, необходимо изменить dtype на datetime


# In[7]:


#Ознакомимся с таблиецй расходов
costs.head()


# In[8]:


#Выведем информацию по таблице расходов
costs.info()
#Обратим внимание на столбец Event Dt , dtype необходимо изменить на datetime


# In[9]:


# Выведем также описательную таблицу по датасетам с помощью describe
visits.describe()


# In[10]:


orders.describe()


# In[11]:


costs.describe()


# 
# Ознакомились с общей информацией о данных, оценили масштаб, подчеркнули для себя dtype столбцов связанных с временем.
# </div>
# 

# In[1]:


#Проверим наличие пропусков в датасетах
visits.isna().sum()


# In[13]:


orders.isna().sum()


# In[14]:


costs.isna().sum()


# In[15]:


#Проверим наличие дубликатов в датасетах
visits.duplicated().sum()


# In[16]:


orders.duplicated().sum()


# In[17]:


costs.duplicated().sum()


#  
# Пропуски и дубликаты в датасетах отсутствуют , возможно мы имеем дело уже с предобработанными для анализа данными.

# In[18]:


# Изменим dtype значений времени в датасетах.
#visits:
visits['Session Start'] = pd.to_datetime(visits['Session Start'])
visits['Session End'] = pd.to_datetime(visits['Session End'])
#Проверяем 
visits.info()


# In[19]:


# Изменим dtype значений времени для orders:
orders['Event Dt'] = pd.to_datetime(orders['Event Dt'])
#Проверяем
orders.info()


# In[20]:


costs.columns = ['dt', 'channel','costs'] #переименуем столбцы в costs для "подгонки" под функцию


# In[21]:


#Изменим dtype значений времени для costs:
costs['dt'] = pd.to_datetime(costs['dt']).dt.date 
#Проверяем
costs.info()


# In[22]:


#Напишем функцию для приведения к "привычному"нижнему регистру и нижней чёрточке вместо пробела:
for i in [visits, orders, costs]:
    i.columns = i.columns.str.replace(' ', '_').str.lower()
#Проверим
visits.info()


# In[23]:


#Переименуем строки в costs для "подгонки под функцию"
costs.columns = ['dt', 'channel','costs']


# In[24]:


#Запись для себя : при вызове get_profiles все значения в столбце end_costs = 0 
visits['session_start'] = pd.to_datetime(visits['session_start'])
visits['session_end'] = pd.to_datetime(visits['session_end'])
orders['event_dt'] = pd.to_datetime(orders['event_dt'])
costs['dt'] = pd.to_datetime(costs['dt']).dt.date 


# 
# 
# 
# 
# ### Написание функции для расчёта и анализа LTV, ROI, удержания и конверсии.
# 
# Разрешается использовать функции, с которыми вы познакомились в теоретических уроках.
# 
# Это функции для вычисления значений метрик:
# 
# - `get_profiles()` — для создания профилей пользователей,
# - `get_retention()` — для подсчёта Retention Rate,
# - `get_conversion()` — для подсчёта конверсии,
# - `get_ltv()` — для подсчёта LTV.
# 
# А также функции для построения графиков:
# 
# - `filter_data()` — для сглаживания данных,
# - `plot_retention()` — для построения графика Retention Rate,
# - `plot_conversion()` — для построения графика конверсии,
# - `plot_ltv_roi` — для визуализации LTV и ROI.

# Пишем функцию get_proifles для создания профилей пользователей

# In[25]:


def get_profiles(visits, orders, costs):

    # находим параметры первых посещений
    profiles = (
        visits.sort_values(by=['user_id', 'session_start'])
        .groupby('user_id')
        .agg(
            {
                'session_start': 'first',
                'channel': 'first',
                'device': 'first',
                'region': 'first',
            }
        )
        .rename(columns={'session_start': 'first_ts'})
        .reset_index()
    )

    # для когортного анализа определяем дату первого посещения
    # и первый день месяца, в который это посещение произошло
    profiles['dt'] = profiles['first_ts'].dt.date
    profiles['month'] = profiles['first_ts'].astype('datetime64[M]')

    # добавляем признак платящих пользователей
    profiles['payer'] = profiles['user_id'].isin(orders['user_id'].unique())


    # считаем количество уникальных пользователей
    # с одинаковыми источником и датой привлечения
    new_users = (
        profiles.groupby(['dt', 'channel'])
        .agg({'user_id': 'nunique'})
        .rename(columns={'user_id': 'unique_users'})
        .reset_index()
    )

    # объединяем траты на рекламу и число привлечённых пользователей
    costs = costs.merge(new_users, on=['dt', 'channel'], how='left')

    # делим рекламные расходы на число привлечённых пользователей
    costs['cost_end'] = costs['costs'] / costs['unique_users']

    # добавляем стоимость привлечения в профили
    profiles = profiles.merge(
        costs[['dt', 'channel', 'cost_end']],
        on=['dt', 'channel'],
        how='left',
    )

    # стоимость привлечения органических пользователей равна нулю
    profiles['cost_end'] = profiles['cost_end'].fillna(0)

    return profiles


# Пишем функцию get_retention для рассчёта удержания пользователей

# In[26]:


#Пишем функцию get_retention для рассчёта удержания пользователей
def get_retention(
    profiles,
    visits,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # добавляем столбец payer в передаваемый dimensions список
    dimensions = ['payer'] + dimensions

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # собираем «сырые» данные для расчёта удержания
    result_raw = result_raw.merge(
        visits[['user_id', 'session_start']], on='user_id', how='left'
    )
    result_raw['lifetime'] = (
        result_raw['session_start'] - result_raw['first_ts']
    ).dt.days

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу удержания
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # получаем таблицу динамики удержания
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time 
    


# Пишем функцию get_conversion для подсчёта конверсии

# In[27]:


# пишем функцию get_conversion — для подсчёта конверсии
def get_conversion(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # определяем дату и время первой покупки для каждого пользователя
    first_purchases = (
        purchases.sort_values(by=['user_id', 'event_dt'])
        .groupby('user_id')
        .agg({'event_dt': 'first'})
        .reset_index()
    )

    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_dt']], on='user_id', how='left'
    )

    # рассчитываем лайфтайм для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users' 
        dimensions = dimensions + ['cohort']

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        result = result.fillna(0).cumsum(axis = 1)
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # делим каждую «ячейку» в строке на размер когорты
        # и получаем conversion rate
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу конверсии
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # для таблицы динамики конверсии убираем 'cohort' из dimensions
    if 'cohort' in dimensions: 
        dimensions = []

    # получаем таблицу динамики конверсии
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time 


# Пишем функцию get_ltv для подсчёта LTV и ROI

# In[28]:


#пишем функцию get_ltv — для подсчёта LTV и ROI
def get_ltv(
    users,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = users.query('dt <= @last_suitable_acquisition_date')
    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        purchases[['user_id', 'event_dt', 'revenue']], on='user_id', how='left'
    )
    # рассчитываем лайфтайм пользователя для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days
    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    # функция группировки по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        # строим «треугольную» таблицу выручки
        result = df.pivot_table(
            index=dims, columns='lifetime', values='revenue', aggfunc='sum'
        )
        # находим сумму выручки с накоплением
        result = result.fillna(0).cumsum(axis=1)
        # вычисляем размеры когорт
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        # объединяем размеры когорт и таблицу выручки
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # считаем LTV: делим каждую «ячейку» в строке на размер когорты
        result = result.div(result['cohort_size'], axis=0)
        # исключаем все лайфтаймы, превышающие горизонт анализа
        result = result[['cohort_size'] + list(range(horizon_days))]
        # восстанавливаем размеры когорт
        result['cohort_size'] = cohort_sizes

        # собираем датафрейм с данными пользователей и значениями CAC, 
        # добавляя параметры из dimensions
        cac = df[['user_id', 'cost_end'] + dims].drop_duplicates()

        # считаем средний CAC по параметрам из dimensions
        cac = (
            cac.groupby(dims)
            .agg({'cost_end': 'mean'})
            .rename(columns={'cost_end': 'cac'})
        )

        # считаем ROI: делим LTV на CAC
        roi = result.div(cac['cac'], axis=0)

        # удаляем строки с бесконечным ROI
        roi = roi[~roi['cohort_size'].isin([np.inf])]

        # восстанавливаем размеры когорт в таблице ROI
        roi['cohort_size'] = cohort_sizes

        # добавляем CAC в таблицу ROI
        roi['cac'] = cac['cac']

        # в финальной таблице оставляем размеры когорт, CAC
        # и ROI в лайфтаймы, не превышающие горизонт анализа
        roi = roi[['cohort_size', 'cac'] + list(range(horizon_days))]

        # возвращаем таблицы LTV и ROI
        return result, roi

    # получаем таблицы LTV и ROI
    result_grouped, roi_grouped = group_by_dimensions(
        result_raw, dimensions, horizon_days
    )

    # для таблиц динамики убираем 'cohort' из dimensions
    if 'cohort' in dimensions:
        dimensions = []

    # получаем таблицы динамики LTV и ROI
    result_in_time, roi_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return (
        result_raw,  # сырые данные
        result_grouped,  # таблица LTV
        result_in_time,  # таблица динамики LTV
        roi_grouped,  # таблица ROI
        roi_in_time,  # таблица динамики ROI
    ) 


# Пишем функции для визуализации метрик:
# filter_data, plot_retention, plot_conversion, plot_ltv_roi

# In[29]:


#пишем функцию filter_data для сглаживания графиков фрейма
def filter_data(df, window):        
    for column in df.columns.values:
        df[column] = df[column].rolling(window).mean() 
    return df


# пишем функцию plot_retention для визуализации данных по удержанию пользователей:

# In[30]:


#пишем функцию plot_retention для визуализации данных по удержанию пользователей:
def plot_retention(retention, retention_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 10))

    # исключаем размеры когорт и удержание первого дня
    retention = retention.drop(columns=['cohort_size', 0])
    # в таблице динамики оставляем только нужный лайфтайм
    retention_history = retention_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # если в индексах таблицы удержания только payer,
    # добавляем второй признак — cohort
    if retention.index.nlevels == 1:
        retention['cohort'] = 'All users'
        retention = retention.reset_index().set_index(['cohort', 'payer'])

    # в таблице графиков — два столбца и две строки, четыре ячейки
    # в первой строим кривые удержания платящих пользователей
    ax1 = plt.subplot(2, 2, 1)
    retention.query('payer == True').droplevel('payer').T.plot(
        grid=True, ax=ax1
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание платящих пользователей')

    # во второй ячейке строим кривые удержания неплатящих
    # вертикальная ось — от графика из первой ячейки
    ax2 = plt.subplot(2, 2, 2, sharey=ax1)
    retention.query('payer == False').droplevel('payer').T.plot(
        grid=True, ax=ax2
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание неплатящих пользователей')

    # в третьей ячейке — динамика удержания платящих
    ax3 = plt.subplot(2, 2, 3)
    # получаем названия столбцов для сводной таблицы
    columns = [
        name
        for name in retention_history.index.names
        if name not in ['dt', 'payer']
    ]
    # фильтруем данные и строим график
    filtered_data = retention_history.query('payer == True').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания платящих пользователей на {}-й день'.format(
            horizon
        )
    )

    # в чётвертой ячейке — динамика удержания неплатящих
    ax4 = plt.subplot(2, 2, 4, sharey=ax3)
    # фильтруем данные и строим график
    filtered_data = retention_history.query('payer == False').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax4)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания неплатящих пользователей на {}-й день'.format(
            horizon
        )
    )
    
    plt.tight_layout()
    plt.show() 


# #Пишем функцию plot_conversion для визуализации конверсии:

# In[31]:


#Пишем функцию plot_conversion для визуализации конверсии:
def plot_conversion(conversion, conversion_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 5))

    # исключаем размеры когорт
    conversion = conversion.drop(columns=['cohort_size'])
    # в таблице динамики оставляем только нужный лайфтайм
    conversion_history = conversion_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # первый график — кривые конверсии
    ax1 = plt.subplot(1, 2, 1)
    conversion.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Конверсия пользователей')

    # второй график — динамика конверсии
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    columns = [
        # столбцами сводной таблицы станут все столбцы индекса, кроме даты
        name for name in conversion_history.index.names if name not in ['dt']
    ]
    filtered_data = conversion_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика конверсии пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show()


# #Пишем функцию plot_ltv_roi для визуализации ltv и roi:

# In[32]:


#Пишем функцию plot_ltv_roi для визуализации ltv и roi:
def plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon, window=7):

    # задаём сетку отрисовки графиков
    plt.figure(figsize=(20, 10))

    # из таблицы ltv исключаем размеры когорт
    ltv = ltv.drop(columns=['cohort_size'])
    # в таблице динамики ltv оставляем только нужный лайфтайм
    ltv_history = ltv_history.drop(columns=['cohort_size'])[[horizon - 1]]

    # стоимость привлечения запишем в отдельный фрейм
    cac_history = roi_history[['cac']]

    # из таблицы roi исключаем размеры когорт и cac
    roi = roi.drop(columns=['cohort_size', 'cac'])
    # в таблице динамики roi оставляем только нужный лайфтайм
    roi_history = roi_history.drop(columns=['cohort_size', 'cac'])[
        [horizon - 1]
    ]

    # первый график — кривые ltv
    ax1 = plt.subplot(2, 3, 1)
    ltv.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('LTV')

    # второй график — динамика ltv
    ax2 = plt.subplot(2, 3, 2, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in ltv_history.index.names if name not in ['dt']]
    filtered_data = ltv_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика LTV пользователей на {}-й день'.format(horizon))

    # третий график — динамика cac
    ax3 = plt.subplot(2, 3, 3)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in cac_history.index.names if name not in ['dt']]
    filtered_data = cac_history.pivot_table(
        index='dt', columns=columns, values='cac', aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика стоимости привлечения пользователей')

    # четвёртый график — кривые roi
    ax4 = plt.subplot(2, 3, 4)
    roi.T.plot(grid=True,ax=ax4)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('ROI')

    # пятый график — динамика roi
    ax5 = plt.subplot(2, 3, 5)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in roi_history.index.names if name not in ['dt']]
    filtered_data = roi_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax5)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.xlabel('Дата привлечения')
    plt.title('Динамика ROI пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show() 


# ### Исследовательский анализ данных
# 
# - Составьте профили пользователей. Определите минимальную и максимальную даты привлечения пользователей.
# - Выясните, из каких стран пользователи приходят в приложение и на какую страну приходится больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих из каждой страны.
# - Узнайте, какими устройствами пользуются клиенты и какие устройства предпочитают платящие пользователи. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого устройства.
# - Изучите рекламные источники привлечения и определите каналы, из которых пришло больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения.
# 
# После каждого пункта сформулируйте выводы.

# Составляем профили пользователей. Определяем минимальную и максимальную даты привлечения пользователей.

# In[33]:


users = get_profiles(visits, orders, costs)
users


# In[34]:


min_date = users['dt'].min()
max_date = users['dt'].max()

min_date, max_date


# Составили профили пользователей с помощью ранее написанной функции get profiles , определили минимальную и максимальную дату привлечения клиентов.
# Минимальная дата : 1 мая 2019 года
# Максимальная дата 27 октября 2019 
# Что совпадает со "спеком" задания.

# Выясним из каких стран пользователи приходят в приложение и в каких странах пользователи более "Платящие":

# In[35]:


users['region'].unique() #Определим уникальные значения столбца регион


# Пользователи приложения из : США,Франции,Германии,Великобритании.

# In[36]:


users_paying = users.query('payer==True').groupby(
    'region').agg({'payer':'count'}).sort_values('payer',ascending=False) #группируем по региону, считаем платящих, сортируем по платящим
users_paying
users_paying['fraction']=users_paying['payer']*100/users_paying['payer'].sum() # Для наглядности посчитаем долю в процентах
users_paying


# In[37]:


# Группировка данных по столбцу "region" и подсчет числа пользователей и платящих пользователей
users_data = users.groupby('region').agg({'user_id': 'count', 'payer': 'sum'})

# Расчет доли платящих пользователей для каждой страны
users_data['paying_ratio'] = (users_data['payer'] / users_data['user_id']) * 100
users_data


# In[38]:


#Добавим число пользователей по каждой стране устройству и каналу
user_count = users.groupby(['region', 'device', 'channel']).size().reset_index(name='user_count')
user_count


# Наибольшее количество платящих пользователей из США ( почти 78 процентов)
# Второе место Великобритания (почти 8 процентов)
# Третье место Франция (Чуть больше 7 процентов)
# Четвёртое место Германия (Чуть меньше 7 процентов)
# Стоит отметить что разрыв между лидером по платящим пользователям(США) и другими странами весьма внушителен, а между остальными странами разница составляет пару процентов .

# Узнаем, какими устройствами пользуются клиенты и какие устройства предпочитают платящие пользователи. Построим таблицу, отражающую количество пользователей и долю платящих для каждого устройства.

# In[39]:


users['device'].unique() #Определим уникальные значения столбца девайс


# Клиенты пользуются : Мак,Айфон,ПК,Андройд

# In[40]:


#Также извлечем платящих пользователей из датасета users и сгруппируем по девйсу.


# In[41]:


users_device = users.query('payer==True').groupby(
    'device').agg({'payer':'count'}).sort_values('payer',ascending=False) #Извлекаем платящих, группируем по девайс, сортируем по платящим по убыванию.
users_device
users_device['fraction'] = users_device['payer']*100/users_device['payer'].sum() #считаем в процентах
users_device


# Большинство платящих пользователей пользуются приложением с устройства iPhone ( 38 процентов)
# Платящие пользователи на андройде составляют 23 процента от платящих
# Пользователи на маке составляют 21 процент платящих пользователей
# Пользователи на PC платят реже остальных и являются 17ю процентами от платящих пользователей.
# Также отмечаем, что разрыв между платящими пользователями айфон и андройд порядка 15 процентов,
# в то время как разницы между андройд,мак,пк от 2 до 5 процентов.

# Рассмотрим рекламные источники привлечения и определим каналы, из которых пришло больше всего платящих пользователей. Построим таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения.

# In[42]:


#Определим уникальные значения столбца channel
users['channel'].unique()


# In[43]:


#Извлечем платящих пользователей , сгруппируем по channel , посчитаем кол-во и отсортируем.
users_channel=users.query('payer==True').groupby(
'channel').agg({'payer':'count'}).sort_values('payer',ascending=False)
users_channel
users_channel['fraction']=users_channel['payer']*100/users_channel['payer'].sum()
users_channel


# Из канала FaceBoom пришло 40 процентов платящих пользователей
# Почти в два раза меньше (20 процентов) платящих пользователей из канала TipTop
# Органические пользователи заняли составляют 13 процентов платящих пользователей
# Остальные каналы имеют от 2 до 5 процентов платящих пользователей
# 
# Тоже отметим "высокую ступень " между лидером и вторым местом каналов по привлечению платящих пользователей.
# Ещё , отметим высокий процент платящих пользователей для органических пользователей, т.к стоимость привлечения таких пользователей равна нулю.

# ### Маркетинг
# 
# - Посчитайте общую сумму расходов на маркетинг.
# - Выясните, как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник.
# - Постройте график с визуализацией динамики изменения расходов во времени по неделям по каждому источнику. Затем на другом графике визуализируйте динамику изменения расходов во времени по месяцам по каждому источнику.
# - Узнайте, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника. Используйте профили пользователей.
# 
# Напишите промежуточные выводы.

# Посчитаем общую сумму расходов на маркетинг, для этого сложим все значения из столбца cost_end

# In[44]:


users['cost_end'].sum()


# Посчитаем сколько денег потратили на каждый источник ( тоесть сгруппируем по столбцу channel)

# In[45]:


users.groupby('channel').agg({'cost_end':'sum'}).sort_values('cost_end',ascending=False)


# Больше всего денежных средств потратили на канал привлечения TipTop 
# Почти в два раза меньше потратили на канал привлечения FaceBoom 
# При этом стоит отметить , что количество платящих пользователей с канала FaceBoom почти в два раза превышает количество 
# платящих пользователей с канала TipTop , выводы делать рано, но в голове будем держать "качество" привлекаемых пользователей.

# Построим график с визуализацией динамики изменения расходов во времени по неделям и по месяцам по каждому источнику.
# 

# При построении графика из профилей пользователя ( датасета users) все линии графика находятся в одной области, не выделяется
# никакой канал, кроме organic, подскажите пожалуйста, почему так?

# In[46]:


costs['dt']=pd.to_datetime(costs['dt'], errors='coerce') # Выходила ошибка по dtype dt, нашёл такое решение проблемы.
costs.info()


# In[47]:


advertising_expense=costs
advertising_expense['week'] = costs['dt'].dt.isocalendar().week
advertising_expense['month'] = costs['dt'].dt.month


# In[48]:


#Изменение расходов во времени по неделям
advertising_expense_table = pd.pivot_table(advertising_expense,index=['week'],columns=['channel'])
advertising_expense_table.plot(figsize=(15,10),title='Изменения расходов во времени по неделям')
xlabel="Недели"
ylabel="Расходы"


# In[49]:


#Изменения расходов во времени по месяцам
advertising_expense_table = pd.pivot_table(advertising_expense,index=['month'],columns=['channel'])
advertising_expense_table.plot(figsize=(15,10),title='Изменения расходов во времени по месяцам')
xlabel="Месяцы"
ylabel="Расходы"


# Расходы на рекламу в каналах TipTop и FaceBoom постоянно растёт, это видно как на графике с недельным, так и месячным "срезом"

# Узнаем, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника. Используя профили пользователей.

# In[50]:


#Сгруппируем данные из функции для вызова профилей пользователей по channel , посчитаем среднее затрат на рекламу , сортировка
## по уменьшению затрат
users.groupby('channel').agg({'cost_end':'mean'}).sort_values('cost_end',ascending=False)


# Привлечение  пользователя из канала TipTop , в среднем, выходит более чем в два раза дороже чем привлечение пользователя из канала FaceBoom.

# Подытожим:
# Из канала FaceBoom пришло 40 процентов платящих пользователей
# Почти в два раза меньше (20 процентов) платящих пользователей из канала TipTop
# Больше всего денежных средств потратили на канал привлечения TipTop почти в два раза меньше потратили 
# на канал привлечения FaceBoom 
# Канал TipTop : 20 % платящих пользователей, большие затраты на привлечение
# Канал FaceBoom :40 % платящих пользователей, сравнительно низкие затраты на привлечение.

# ### Оценка окупаемости рекламы
# 
# Используя графики LTV, ROI и CAC, проанализируйте окупаемость рекламы. Считайте, что на календаре 1 ноября 2019 года, а в бизнес-плане заложено, что пользователи должны окупаться не позднее чем через две недели после привлечения. Необходимость включения в анализ органических пользователей определите самостоятельно.
# 
# - Проанализируйте окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проверьте конверсию пользователей и динамику её изменения. То же самое сделайте с удержанием пользователей. Постройте и изучите графики конверсии и удержания.
# - Проанализируйте окупаемость рекламы с разбивкой по устройствам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по странам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Ответьте на такие вопросы:
#     - Окупается ли реклама, направленная на привлечение пользователей в целом?
#     - Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
#     - Чем могут быть вызваны проблемы окупаемости?
# 
# Напишите вывод, опишите возможные причины обнаруженных проблем и промежуточные рекомендации для рекламного отдела.

# Проанализируем окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.

# Установим момент и горизонт анализа данных

# In[51]:


observation_date = datetime(2019, 11, 1).date()  # момент анализа, сегодняшний день
horizon_days = 14  # зададим двухнедельный горизонт анализа


# In[52]:


#Включать в анализ органических пользователей не будем, т.к. стоимость привлечения таких пользовтаелей 0.
users = users.query('channel !="organic"')


# In[53]:


tv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    users, orders, observation_date, horizon_days
)
plot_ltv_roi(ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days)


# По графикам можно сделать такие выводы:
# Реклама не окупается. ROI к концу второй недели преодолела отметку в 80 %.
# Однако в динамике, мы видим что с начала до конца мая ROI был выше единицы и реклама окупалась.
# Показатель CAC не стабилен,поэтому можно предположить что дело в изменении рекламного бюджета, в момент падения ROI наблюдается рост САС.

# Проверим конверсию пользователей и динамику её изменения. То же самое сделаем с удержанием пользователей. Построим графики конверсии и удержания.

# In[54]:


#Конверсия
conversion_raw, conversion_grouped, conversion_history = get_conversion(
    users, orders, observation_date, horizon_days
)

plot_conversion(conversion_grouped, conversion_history, horizon_days)


# In[55]:


#Удержание 
retention_raw, retention_grouped, retention_history = get_retention(
    users, visits, observation_date, horizon_days
)

plot_retention(retention_grouped, retention_history, horizon_days) 


# Конверсия и удержание пользователей выглядят стабильно, закономерно что удержание платящих пользователей выше чем неплатящих,т.к
# пользователям"есть что терять ".

# Проанализируем окупаемость рекламы с разбивкой по устройствам. Построим графики LTV и ROI, а также графики динамики LTV, CAC и ROI.

# In[56]:


dimensions = ['device']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    users, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days, window=14
) 


# Анализ графиков :
# LTV: Показатель LTV на всех девайсах идут "плечо к плечу" все стабильно растут, выделяется лишь немного отстающий от всех PC
# cac: на графике видно что стоимость привлечения пользователей с девайсами mac и iphone  выросло с мая больше остальных
# продолжая лидировать весь период.
# динамика стоимости привлечения пользователей PC стабильно ниже всех
# ROI:На графике видно что окупаются только пользователи PC, лидеры предыдущих графиков mac и iphone здесь аутсайдеры
# ПРи это на графике ROI в динамике мы видим что в период исследования пользователи всех устройств выходили за еденицу, тоесть
# окупались .

# In[57]:


#Конверсия по девайсам
dimensions = ['device']
conversion_raw, conversion_grouped, conversion_history = get_conversion(
    users, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, horizon_days)

#Удержание по девайсам
retention_raw, retention_grouped, retention_history = get_retention(
    users, visits, observation_date, horizon_days, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, horizon_days) 


# 

# 

# При разбивке по девайсам конверсия iphone и mac немного выше чем конверсия android и значительно выше PC
# Лидером по удержанию является PC, немного ниже удержание android и показатели удержания iphone и mac имеют похожие показатели
# удержания. (Хотел указать на зеркальность конверсии к удержанию)

# Проанализирум окупаемость рекламы с разбивкой по странам. Построим графики LTV и ROI, а также графики динамики LTV, CAC и ROI.

# In[58]:


dimensions = ['region']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    users, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days, window=14
) 


# Разбивка по региону показывает что пользовательская статистика из США выглядит аномально:
# Параметр LTV пользователей из США ярко выделяется в сравнении с другими странами.
# Затраты на привлечение пользователей резко растут в то время как в других странах сначала стабильно движутся вниз , затем
# идут "по ровной".
# ROI пользователей из всех стран отлично окупаются, кроме США.
# Подытжив : Пользователи из США имеют более высокий LTV, но и многократно выше затраты на привлечение таких клиентов, 
# как итог отрицатльный показатель ROI.
# 

# Построим графики конверсии и удержания по регионам.

# In[59]:


#Конверсия по регионам
dimensions = ['region']
conversion_raw, conversion_grouped, conversion_history = get_conversion(
    users, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, horizon_days)

#Удержание по регионам
retention_raw, retention_grouped, retention_history = get_retention(
    users, visits, observation_date, horizon_days, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, horizon_days) 


# Опять видим зеркальность, более высокая конверсия пользователей означает более низкое удержание пользователей.
# Пользователи из США выделяются из 4х стран , имея более высокую конверсию и более низкое удержание пользователей.

# Проанализируем окупаемость рекламы с разбивкой по рекламным каналам. Построим графики LTV и ROI, а также графики динамики LTV, CAC и ROI.

# In[60]:


dimensions = ['channel']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    users, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, horizon_days, window=14
) 


# Анализ графиков :
# LTV: Выделяются два канала с наибольшим LTV : LambdaMediaAds и TipTop - тоесть пользователи с этих каналов приносят наибольшее
# количествоо денежных средств "за жизненный цикл"
# сac : выделяется канал TipTop , стоимость привлечения пользователей с этого канала кратно растёт,стоимость привлеения пользоватлей остальных каналов движется стабильно почти "по ровной" линии.
# ROI:Из ранее выделенных каналов стоит отметить , что лишь lambdamediaads является хорошо окупаемым , тоесть огромные затраты
# на канал TipTop не окупаются.

# In[61]:


#Конверсия по каналама
dimensions = ['channel']
conversion_raw, conversion_grouped, conversion_history = get_conversion(
    users, orders, observation_date, horizon_days, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, horizon_days)

#Удержание по регионам
retention_raw, retention_grouped, retention_history = get_retention(
    users, visits, observation_date, horizon_days, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, horizon_days) 


# График конверсии выглядит стабильно 
# На графике удержания выделяются каналы AdNonSense и FaceBoom , как имеющие низкие показатели удержания.

# Ответьте на вопросы
# Окупается ли реклама, направленная на привлечение пользователей в целом? В целом, реклама направленная на привлечение пользователей не окупается.
# Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
# Девайсы , оказывающие негативное влияние : Mac,Iphone
# Страны, оказывающие негативное влияние : США
# Каналы, оказывающие негативное влияние : Выделю самый ярко - выражающийся - TipTop
# Чем могут быть вызваны проблемы окупаемости? Проблемы окупаемости на мой взгляд вызваны большим количеством денежных средств на привлечение пользователей (рекламу). Тоесть пользователи не приносят столько денег, сколько потрачено на их привлечение.
# Также могут быть локальные системные ошибки, например технические ошибки на каком-то этапе у пользователей из США.
# Также может быть что приложение плохо оптимизировано под Iphone и Mac.

# ### Выводы
# 
# - Выделите причины неэффективности привлечения пользователей.
# - Сформулируйте рекомендации для отдела маркетинга.

# Причинами неэффективности привлечения пользователей является :
# 1.высокая стоимость привлечения клиентов
# 2.вложение большого количества денежных средств в рекламу на одном канале
# 3.проблемы в конкретном регионе , с конкретными устройствами
# 
# 
# Рекомендации:
# Необходимо более равномерно распределять денежные средства на рекламу в разных каналах.
# Необходимо разобраться с пользователями из США , определить почему такая высокая стоимость привлечения, если дело в рекламе, то расммотреть вариант перенаправления денежных потоков на рекламу в США в другие страны.
# Необходимо провести анализ ошибок и багов для пользователей Mac,Iphone.
# Следует обращать внимание на каналы с низкой стоимостью привлечения, и высоким LTV, таким как lambdamediaads например.
# "допиливать" читай - оптимизировать приложение под пользователей с самым высоким количеством платящих пользователей (iphone)
