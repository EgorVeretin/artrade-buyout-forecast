import re
from collections import Counter
import pandas as pd

DANGER_COLUMNS = [
    'contact_created_at', 'contact_updated_at', 'contact_Телефон', 'lead_Комментарий',
    'contact_name', 'contact_first_name', 'handed_to_delivery_ts', 'lead_COOKIES',
     'lead_Тариф Доставки', 'lead_Компания Отправитель', 'lead_Вид оплаты','lead_name',
    'lead_TRANID', 'lead_account_id', 'lead_closed_at', 'lead_updated_at',
    'lead_created_at', 'lead_loss_reason_id', 'rejected_ts', 'returned_ts',
    'outcome_unknown', 'sale_ts', 'lead_is_deleted', 'lead_status_id',
    'current_status_id', 'issued_or_pvz_ts', 'received_ts', 'days_handed_to_issued_pvz', 
    'days_to_outcome', 'closed_ts', 'lead_pipeline_id', 'lead_Дата создания сделки',
    'lead_Дата перехода в Сборку', 'lead_Дата получения денег на Р/С', 'contact_id', 'contact_Адрес ПВЗ',
    'contact_Код ПВЗ', 'lead_utm_term', 'lead_utm_campaign', 'lead_utm_sky', 'lead_utm_referrer', 
    'lead_Дата перехода Передан в доставку', 'lifecycle_incomplete'
]

NULL_THRESHOLD = 0.60  # 60% пропусков

def read_data(path: str, sep: str = ',') -> pd.DataFrame:
    """Загружает данные из CSV"""
    return pd.read_csv(path, sep=sep)

def extract_articles_from_row(text: str) -> list:
    """Извлекает артикулы из одной строки"""
    if pd.isna(text):
        return []
    return re.findall(r'Артикул:\s*(\d+)', text)

def extract_cost_from_row(text: str) -> float:
    """Извлекает общую стоимость из одной строки"""
    if pd.isna(text):
        return 0.0
    prices = re.findall(r'Розничная цена:\s*(\d+)', text)
    return sum(int(p) for p in prices)

def extract_delivery_cost_from_row(text) -> float:
    """Извлекает стоимость доставки из текста заказа"""
    if pd.isna(text):
        return 0.0
    
    text = str(text)
    pattern = r'Доставка.*?Розничная цена:\s*(\d+)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    return int(match.group(1)) if match else 0.0

def delete_null_and_danger_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет колонки с >60% пропусков и опасные колонки"""
    df = df.copy()
    
    # Колонки с >60% пропусков
    null_counts = df.isna().sum()
    threshold = len(df) * NULL_THRESHOLD
    high_null_cols = null_counts[null_counts > threshold].index.tolist()
    
    print(f"Удалено {len(high_null_cols)} колонок с >60% пропусков")
    
    # Объединяем с опасными колонками
    cols_to_delete = set(high_null_cols) | set(DANGER_COLUMNS)
    cols_to_delete = [col for col in cols_to_delete if col in df.columns]
    
    print(f"Всего удалено колонок: {len(cols_to_delete)}")
    
    return df.drop(columns=cols_to_delete)

def find_and_convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Находит и преобразует временные колонки"""
    df = df.copy()
    datetime_cols = []
    
    # важный момент 
    exclude_keywords = ['доставк', 'тариф', 'служба', 'вид оплаты', 'оплаты']
    # Паттерны для поиска
    time_patterns = r'_ts$|_at$|time|date|день|дата|создан|обновлен|закрыт|переход|получен|выдан|доставк|возврат|сборк|продаж'
    
    for col in df.columns:
        col_lower = col.lower()
        
        # спасибо за проверку - будь внимательнее!
        # проверка на содержание в коде информации по доставке не в формате timestamp / date 
        if any(kw in col_lower for kw in exclude_keywords):
            continue
        
        if re.search(time_patterns, col.lower()):
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            
            # Unix timestamp
            if df[col].dtype in ['int64', 'float64']:
                sample_val = sample.iloc[0]
                if 1e8 < sample_val < 2e9:  # секунды
                    df[col] = pd.to_datetime(df[col], unit='s')
                    datetime_cols.append(col)
                elif 1e11 < sample_val < 2e12:  # миллисекунды
                    df[col] = pd.to_datetime(df[col], unit='ms')
                    datetime_cols.append(col)
            
            # Строковые даты
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                    if df[col].notna().any():
                        datetime_cols.append(col)
                except:
                    pass
    
    print(f"Преобразовано {len(datetime_cols)} колонок: {datetime_cols}")
    return df

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонки с артикулами и стоимостью"""
    df = df.copy()
    df['articles'] = df['lead_Состав заказа'].apply(extract_articles_from_row)
    df['cost_with_delivery'] = df['lead_Состав заказа'].apply(extract_cost_from_row)
    df['delivery_cost'] = df['lead_Состав заказа'].apply(extract_delivery_cost_from_row)
    df = df.drop('lead_Состав заказа', axis=1)
    return df

def get_top_articles(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Возвращает топ-N самых частых артикулов"""
    all_articles = [art for sublist in df['articles'] for art in sublist]
    counter = Counter(all_articles)
    return pd.DataFrame(counter.most_common(top_n), columns=['article', 'count'])

def preprocess_data(path: str) -> pd.DataFrame:
    """Надстройка для основного пайплайна"""
    print("1. Загрузка данных...")
    df = read_data(path)
    print(f"   Загружено {df.shape[0]} строк, {df.shape[1]} колонок")
    
    print("2. Удаление лишних колонок...")
    df = delete_null_and_danger_columns(df)
    print(f"   Осталось {df.shape[1]} колонок")
    
    print("3. Преобразование дат...")
    df = find_and_convert_datetime_columns(df)
    
    print("4. Обогащение данных (артикулы, стоимость)...")
    df = enrich_data(df)
    
    df = df[~df['buyout_flag'].isna()]
    
    df['buyout_flag'] = df['buyout_flag'].astype('float')
    
    return df

if __name__ == "__main__":
    PATH_TO_FILE = "dataset_2025-03-01_2026-03-29_external.csv"
    
    data = preprocess_data(PATH_TO_FILE)
    top_articles = get_top_articles(data, top_n=10)
    print("\nТоп-10 артикулов:")
    print(top_articles)


# сортировка признаков и их кодировка!
