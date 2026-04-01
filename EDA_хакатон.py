import pandas as pd
"""
Тоня! Подставишь сюда новый файл - чтобы органично смотрелось?
По идее со старым будет корреляция и все равно колонок 40 улетит псоле фильтрации на data.isna!
По сути после такой очистки я и отдал тебе dataset
"""
PATH_FILE = 'MIPT_hackathon_dataset.csv' # (пример) !!!файл!!!
df = pd.read_csv(PATH_FILE, sep=',')
data = df.copy()
df['lead_Скидка'].isna().sum() # подумать про скидку!
data.isna().sum()
data.describe()
data.info()

import re
def extract_articles_fast(text_list):
    """
    Один проход regex для артикулов
    """
    full_text = '\n'.join(str(text_list).strip().split('\n'))
    
    return re.findall(r'Артикул:\s*(\d+)', full_text)

article_data = data['lead_Состав заказа'].apply(extract_articles_fast)

# проверим все уникальные артикли по товарам
from collections import Counter
ex = (sum(article_data, []))
articles = pd.DataFrame(Counter(ex).most_common(), columns=['article', 'count']) # 150 уникальных
data['articles'] = article_data

def extract_cost_fast(text_list):
    """
    Один проход regex для стоимости
    """
    full_text = '\n'.join(str(text_list).strip().split('\n'))
    
    cost_list = re.findall(r'Розничная цена:\s*(\d+)', full_text)
    cost_list = map(int, cost_list)
    
    return sum(cost_list)
    
cost_data = data['lead_Состав заказа'].apply(extract_cost_fast)
data['cost'] = cost_data


def extract_delivery_cost(text):
    """
    Извлекает стоимость доставки из текста заказа
    """
    if pd.isna(text):
        return 0
    
    text = str(text)
    
    # Ищем блок с "Доставка" и забираем цену после "Розничная цена:"
    # Паттерн ищет: Доставка, затем любой текст до "Розничная цена:", затем число
    pattern = r'Доставка.*?Розничная цена:\s*(\d+)'
    
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    else:
        return 0

# Применяем
delivery_data = data['lead_Состав заказа'].apply(extract_delivery_cost)
data['delivery_cost'] = delivery_data

# почистим исходный датасет - слишком много пропусков и нет смысла кормить модель этим
info_data = data.isna().sum().sort_values(ascending=False)
LIMIT = data.shape[0] * 0.60
cols_to_delete = info_data[info_data > LIMIT].index.to_list()
data.drop(cols_to_delete, axis=1, inplace=True)

data['buyout_flag'] = data['buyout_flag'].map(lambda x: 1 if x else 0)

# чистим еще одну грязную колонку
data.drop('outcome_unknown', axis=1, inplace=True)

def find_datetime_columns(df):
    """
    Находит все потенциально временные колонки по названию
    """
    
    patterns = {
        'timestamp': r'_ts$|_at$|time',           # _ts, _at, time
        'date': r'date|день|дата|дн|год|месяц',   # date, день, дата
        'russian_time': r'создан|обновлен|закрыт|переход|получен|выдан|доставк|возврат|сборк|продаж',
        'unix_like': r'^\d{10,13}$'               # похоже на unix timestamp - иначе проверить
    }
    
    datetime_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Проверяем каждый паттерн
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, col_lower):
                # Дополнительная проверка: значения похожи на даты или нет?
                sample = df[col].dropna()
                if len(sample) > 0:
                    sample_val = sample.iloc[0]
                    
                    # Проверка на unix timestamp (число)
                    if df[col].dtype in ['int64', 'float64']:
                        if 1e8 < sample_val < 2e9 or 1e11 < sample_val < 2e12:
                            datetime_cols.append(col)
                            break
                    
                    # Проверка на строковую дату
                    elif df[col].dtype == 'object':
                        if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', str(sample_val)):
                            datetime_cols.append(col)
                            break
                        if re.search(r'\d{2}[./]\d{2}[./]\d{4}', str(sample_val)):
                            datetime_cols.append(col)
                            break
    
    return datetime_cols

# Находим все временные колонки
all_datetime_cols = find_datetime_columns(data)


def convert_to_date_simple(data, cols):
    """
    Простое преобразование в даты (без времени)
    """
    for col in cols:
        if col in data.columns:
            # Преобразуем числа в даты
            # Делим на 86400 (секунд в дне) чтобы получить дни от 1970 - мб тут есть дата в формате строки
            try:
                data[col] = pd.to_datetime(data[col], unit='s').dt.date
            except ValueError:
                pass
    return data

# Преобразуем
data = convert_to_date_simple(data, all_datetime_cols)

# сортируем датасет, но чуть позже - так как не конкатили еще таблицу
# data.sort_values(by='sale_date', ascending=True, inplace=True)
data.drop('lifecycle_incomplete', axis=1, inplace=True)

# засунуть в частотное кодирование
data['lead_responsible_user_id'].value_counts() # проверить на трейне и тесте

data['lead_Модель телефона'].value_counts() # можно и удалить - вклад крайне мал и фичей много
"""
lead_Модель телефона
Смартфон             2771
Не удалось узнать    2520
Кнопочный              73
Name: count, dtype: int64
"""
data['lead_будущие покупки'].value_counts()
# кодируем так: есть покупка или нет покупки - слишком много категорий с низкой дисперсией
data['lead_будущие покупки'][data['lead_будущие покупки'] != 'не известно'].value_counts().sum()


"""
Тоже перезапись файла!
data.to_csv('MIPT_hackathon_dataset_MOD.csv', sep=',', index=False)
"""
# проверить данные колонки - можно преобразовать в фичу относительного объема или НЕ ДАНО
data.columns

check_for_LLM = data.columns.to_list()
size_cols = ['lead_Ширина', 'lead_Высота', 'lead_Длина', 'lead_Вес (грамм)*']
data[size_cols]
lst_check = []
for col in size_cols:
    isna_count = data[col].isna().sum()
    summary = col, isna_count
    lst_check.append(summary)
print(lst_check)


"""
Сохранение в отдельный файл артикулов
data[['lead_Состав заказа', 'lead_id']].to_csv('NLP_data.csv', sep=',')
"""
data.drop('sale_ts', axis=1, inplace=True)

data['lead_responsible_user_id'] # частотное кодирование - можно не трогать
data['lead_responsible_user_id'].isna().sum() # заполнен идеально

data.drop('lead_is_deleted', axis=1, inplace=True) # константный - шум
data['lead_group_id'].value_counts() # сделать ohe или частотное кодирование

data['lead_status_id'] # утечка - отражение купил/продал! 
data['current_status_id'] # утечка - отражение купил/продал!
#data['current_status_id'] и data['lead_status_id'] одинаковые!
data.drop('lead_status_id', axis=1, inplace=True)
data['issued_or_pvz_ts'] # выдача заказа на пвз - утечка !
data['received_ts'] # выкуп заказа - утечка!
print(data['days_sale_to_handed'].describe()) # отрицательных чисел нет - это хорошо!
data['days_handed_to_issued_pvz'] # утечка!
data['days_to_outcome'] # утечка!
data['closed_ts'] # утечка!

#data['closed_ts'] и data['received_ts'] одинаковые !
data.drop('received_ts', axis=1, inplace=True)
data['lead_pipeline_id'].value_counts() # удалить - шум
data.drop('lead_pipeline_id', axis=1, inplace=True)

#data['lead_created_at'] и data['lead_Дата создания сделки'] одинаковые и отличаются в 1 день!
data.drop('lead_Дата создания сделки', axis=1, inplace=True)

# пора удалить - вернусь к этому позже, если метрика у модели будет плохой
data.drop('lead_Состав заказа', axis=1, inplace=True)

lead_qual = data['lead_Квалификация лида'].unique()

"""
sorted_data = data.sort_values(by='sale_date', ascending=True).reset_index(drop='first')

lead_id   sale_date  ...   cost delivery_cost
0     LEAD_0217  2025-11-01  ...   5257           335

"""
sorted_data = data.sort_values(by='sale_date', ascending=True).reset_index(drop='first')
test_data = sorted_data[sorted_data['sale_date'] > '2026-01-30']

lead_names_list = data['lead_Квалификация лида'].value_counts().index.to_list()
lead_names_count = list(data['lead_Квалификация лида'].value_counts().values)

#data['lead_Квалификация лида'][~data['lead_Квалификация лида'].isin(lead_names_list[:3])] = lead_names_list[2]

data['lead_Квалификация лида'].value_counts() # можно OHE

lead_category_list = data['lead_Категория и варианты выбора'].value_counts().index.to_list()
lead_category_count = list(data['lead_Категория и варианты выбора'].value_counts().values)

"""
Проверка популярных значений!
"""
#data['lead_Категория и варианты выбора'][~data['lead_Категория и варианты выбора'].isin(lead_category_list[:4])] = \
#    lead_category_list[3]

data['lead_Категория и варианты выбора'].value_counts()

"""
Не носить изменения в исходный dataset до разделения на тест и трейн!
Ниже приведены мысли и выкладки!
"""

data['lead_будущие покупки'].value_counts() # чистокровный бинарный признак 
data['lead_будущие покупки'] = data['lead_будущие покупки'].map(lambda x: 0 if x == 'не известно' else 1)

data['lead_Модель телефона'].value_counts() # тоже нечто бинарное (мб там не телефон, а компьютер)
data['lead_Модель телефона'] = data['lead_Модель телефона'].map(lambda x: 1 if x == 'Смартфон' else 0)

data['lead_Проблема'].value_counts() # проверить на итоговом датасете - тут частотное кодирование

data['buyout_flag'].value_counts()

data['delivery_cost'][data['buyout_flag']==0].sum()


data['lead_Проблема'].value_counts() # frequency encoder
data['lead_Тариф Доставки'].value_counts() # frequency encoder
data['lead_Служба доставки'].value_counts() # frequency encoder
data['lead_Вид оплаты'].value_counts() # frequency encoder

data['lead_Дата перехода в Сборку'].value_counts() # утечка!
data['lead_Дата получения денег на Р/С'] # утечка!
data['contact_id'].value_counts() # удалить - нет нормальной дисперсии 

data.columns
data['contact_Адрес ПВЗ'].value_counts() # удалить - дисперсия ужасна!
data['contact_Код ПВЗ'].value_counts() # удалить - не несет информации и долго вытаскивать - нет смысла!
data['contact_Город'].value_counts().head(10) # можно OHE до топ-5 и в категорию another!
data['lead_utm_source'].value_counts() # взять частотное кодирование (топ-5 и остальное в топку)

data['lead_utm_term'].value_counts() # удалить - просто дублирует utm_source
data['lead_utm_campaign'].value_counts() # удалить!
data['lead_utm_content'] # удалить!

data['lead_utm_group'].value_counts() # OHE или частотное кодировние - проверить
data['lead_utm_sky'].value_counts() # удалить и не смотреть даже!
data['lead_FORMNAME'].value_counts() # можно сохранить - но не факт, что нужно!
data['lead_utm_referrer'].value_counts() # удалить сразу!
data['lead_Дата перехода Передан в доставку'] # удалить сразу!

