import re
import pandas as pd

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