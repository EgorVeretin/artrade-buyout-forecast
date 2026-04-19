import pandas as pd

def read_data(path: str, sep: str = ',') -> pd.DataFrame:
    """Загружает данные из CSV"""
    return pd.read_csv(path, sep=sep)