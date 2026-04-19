import re
import pandas as pd
from config import DANGER_COLUMNS, NULL_THRESHOLD

def delete_null_and_danger_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет колонки с >60% пропусков и опасные колонки"""
    df = df.copy()
    
    null_counts = df.isna().sum()
    threshold = len(df) * NULL_THRESHOLD
    high_null_cols = null_counts[null_counts > threshold].index.tolist()
    
    print(f"Удалено {len(high_null_cols)} колонок с >60% пропусков")
    
    cols_to_delete = set(high_null_cols) | set(DANGER_COLUMNS)
    cols_to_delete = [col for col in cols_to_delete if col in df.columns]
    
    print(f"Всего удалено колонок: {len(cols_to_delete)}")
    
    return df.drop(columns=cols_to_delete)

def find_and_convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Находит и преобразует временные колонки"""
    df = df.copy()
    datetime_cols = []
    
    exclude_keywords = ['доставк', 'тариф', 'служба', 'вид оплаты', 'оплаты']
    time_patterns = r'_ts$|_at$|time|date|день|дата|создан|обновлен|закрыт|переход|получен|выдан|доставк|возврат|сборк|продаж'
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(kw in col_lower for kw in exclude_keywords):
            continue
        
        if re.search(time_patterns, col.lower()):
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                sample_val = sample.iloc[0]
                if 1e8 < sample_val < 2e9:
                    df[col] = pd.to_datetime(df[col], unit='s')
                    datetime_cols.append(col)
                elif 1e11 < sample_val < 2e12:
                    df[col] = pd.to_datetime(df[col], unit='ms')
                    datetime_cols.append(col)
            
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
    from feature_extractors import extract_articles_from_row, extract_cost_from_row, extract_delivery_cost_from_row
    
    df = df.copy()
    df['articles'] = df['lead_Состав заказа'].apply(extract_articles_from_row)
    df['cost_with_delivery'] = df['lead_Состав заказа'].apply(extract_cost_from_row)
    df['delivery_cost'] = df['lead_Состав заказа'].apply(extract_delivery_cost_from_row)
    df = df.drop('lead_Состав заказа', axis=1)
    return df

def parse_tags_column(text: str) -> list:
    """Парсит строку с тегами в список"""
    text = str(text)
    return text.strip().split(',')