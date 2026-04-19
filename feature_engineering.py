import pandas as pd
import numpy as np
from transformers import transform_weight_to_category, make_feature_binary, transform_lead_qualification
from preprocessors import parse_tags_column

def apply_all_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет все преобразования к датафрейму"""
    df = df.copy()
    
    # Удаление строк с пропущенным target
    df = df[~df['buyout_flag'].isna()]
    df['buyout_flag'] = df['buyout_flag'].astype('float')
    
    # Парсинг тегов
    df['lead_tags'] = df['lead_tags'].map(parse_tags_column)
    
    # Трансформации отдельных колонок
    df['lead_Вес (грамм)*'] = df['lead_Вес (грамм)*'].map(transform_weight_to_category)
    df['lead_Модель телефона'] = df['lead_Модель телефона'].map(make_feature_binary)
    df['lead_будущие покупки'] = df['lead_будущие покупки'].map(make_feature_binary)
    df['lead_Квалификация лида'] = df['lead_Квалификация лида'].map(transform_lead_qualification)
    
    # Бинарные признаки
    df['lead_FORMID'] = np.where(df['lead_FORMID'].isna(), 0, 1)
    df['lead_REFERER'] = np.where(df['lead_REFERER'].isna(), 0, 1)
    df['lead_FORMNAME'] = np.where(df['lead_FORMNAME'].isna(), 0, 1)
    
    # Заполнение пропусков
    df['lead_utm_source'] = df['lead_utm_source'].fillna('Неизвестно')
    df['contact_Число сделок'] = df['contact_Число сделок'].fillna(1)
    df['lead_Категория и варианты выбора'] = df['lead_Категория и варианты выбора'].fillna('Нет категории')
    df['lead_utm_group'] = df['lead_utm_group'].fillna('unknown_status')
    
    # Размерные признаки
    max_dim = df[['lead_Длина', 'lead_Ширина', 'lead_Высота']].max(axis=1)
    min_dim = df[['lead_Длина', 'lead_Ширина', 'lead_Высота']].min(axis=1)
    df['size_ratio'] = np.where(min_dim > 0, max_dim / min_dim, 0)
    df.drop(['lead_Длина', 'lead_Ширина', 'lead_Высота'], inplace=True, axis=1)
    
    # Скидки и подарки
    df.loc[df['delivery_cost'] == 0, 'discount'] = 1
    df['discount'] = df['discount'].fillna(0)
    df.loc[df['delivery_cost'] == 1, 'is_gift'] = 1
    df['is_gift'] = df['is_gift'].fillna(0)
    
    # Обработка артикулов
    df['articles'] = df['articles'].map(lambda x: ([item for item in x if item != '1']))
    df['unique_articles'] = df['articles'].map(lambda x: len(x))
    
    # Стоимость без доставки
    df['cost_without_delivery'] = df['cost_with_delivery'] - df['delivery_cost']
    df.drop('cost_with_delivery', axis=1, inplace=True)
    
    # Время жизни клиента
    sale_date = pd.to_datetime(df['sale_date'])
    contact_created = pd.to_datetime(df['contact_created_at'])
    df['customer_tenure_days_frac'] = (sale_date - contact_created).dt.total_seconds() / (24 * 3600)
    df.loc[df['customer_tenure_days_frac'] < 0, 'customer_tenure_days_frac'] = 0
    df['is_same_day'] = (df['sale_date'] == df['contact_created_at']).astype(int)
    
    # Год продажи и сортировка
    df['sale_date'] = pd.to_datetime(df['sale_date']).dt.year
    df = df.sort_values(by='sale_date', ascending=True)
    
    # Биннинг числа сделок
    df['contact_deals_bin'] = pd.cut(df['contact_Число сделок'], 
                                     bins=[0, 1, 3, 6, 10, np.inf], 
                                     labels=['0', '1-2', '3-5', '6-10', '10+']).astype('str')
    
    # Комбинированные признаки
    df['deals_x_weight'] = df['contact_Число сделок'] * df['lead_Вес (грамм)*'].map({'light': 1, 'medium': 2, 'heavy': 3})
    df['deals_x_delivery'] = df['contact_Число сделок'].astype(str) + '_' + df['lead_Служба доставки']
    
    threshold_price = df['cost_without_delivery'].quantile(0.75)
    df['has_expensive_item'] = (df['cost_without_delivery'] > threshold_price).astype(int)
    df['delivery_cost_ratio'] = df['delivery_cost'] / (df['cost_without_delivery'] + 1)
    
    # Временные признаки
    df['sale_dayofweek'] = pd.to_datetime(df['sale_date']).dt.dayofweek
    df['sale_month'] = pd.to_datetime(df['sale_date']).dt.month
    df['sale_is_weekend'] = (df['sale_dayofweek'] >= 5).astype(int)
    
    # Логарифмические преобразования
    df['deals_log'] = np.log1p(df['contact_Число сделок'])
    df['deals_x_city'] = df['contact_Число сделок'].astype(str) + '_' + df['contact_Город'].astype(str)
    
    # Период зарплаты
    df['is_payday_period'] = pd.to_datetime(df['sale_date']).dt.day.between(1, 5) | \
                             pd.to_datetime(df['sale_date']).dt.day.between(25, 31)
    df['is_payday_period'] = df['is_payday_period'].astype(int)
    
    # Логирование дней доставки
    df['log_days_handed_to_issued_pvz'] = np.log1p(df['days_handed_to_issued_pvz'].map(lambda x: abs(x)))
    df.drop('days_handed_to_issued_pvz', axis=1, inplace=True)
    
    return df