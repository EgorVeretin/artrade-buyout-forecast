"""
Предсказания на новых данных - с правильной обработкой категориальных признаков
"""

import pickle
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import preprocess_data
from list_features import create_top_features_from_lists
from utils import fill_missing_values

# Загружаем модель
with open('simple_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

model = pipeline['model']
optimal_threshold = pipeline['optimal_threshold']
feature_names = pipeline['feature_names']

print("Модель загружена")
print(f"Оптимальный порог: {optimal_threshold:.3f}")

# Путь к новым данным
DATA_PATH = "dataset_2025-03-01_2026-03-29_external.csv" # НОВЫЕ ДАННЫЕ

# Предобрабатываем данные
print("\nПредобработка данных...")
df = preprocess_data(DATA_PATH)

print(f"   После предобработки: {df.shape}")

# Убираем target если есть
if 'buyout_flag' in df.columns:
    df = df.drop('buyout_flag', axis=1)

# Создаем признаки из списков (как при обучении)
special_cols = ['articles', 'lead_tags']

# Нужно разделить на train/test для create_top_features_from_lists
# Для предсказаний используем все данные как test
df_temp = df.copy()
df_temp['_dummy_col'] = 0

# Применяем ту же логику, что и в main.py
_, df = create_top_features_from_lists(df_temp, df_temp, 'articles', top_n=10)
_, df = create_top_features_from_lists(df_temp, df_temp, 'lead_tags', top_n=10)

# Определяем категориальные колонки
categorical_cols = df.select_dtypes(include='object').columns.to_list()
categorical_cols = list(set(categorical_cols) - set(special_cols))
numeric_cols = df.select_dtypes(exclude='object').columns.to_list()

# Заполняем пропуски (ВАЖНО!)
df, _ = fill_missing_values(df, df, categorical_cols, numeric_cols)

# Убеждаемся, что колонки совпадают с моделью
missing_cols = set(feature_names) - set(df.columns)
for col in missing_cols:
    df[col] = 0

extra_cols = set(df.columns) - set(feature_names)
df = df.drop(columns=extra_cols, errors='ignore')

X = df[feature_names]

print(f"Данные готовы: {X.shape}")
print(f"Категориальных признаков: {len([c for c in X.columns if X[c].dtype == 'object'])}")
print(f"Пропусков в данных: {X.isna().sum().sum()}")

# Проверяем наличие NaN в категориальных колонках
cat_cols_in_X = X.select_dtypes(include='object').columns
for col in cat_cols_in_X:
    if X[col].isna().sum() > 0:
        print(f"В колонке {col} есть NaN, заполняем 'unknown'")
        X[col] = X[col].fillna('unknown')

# Предсказания
print("\nПредсказания...")
y_proba = model.predict_proba(X)[:, 1]
y_class = (y_proba >= optimal_threshold).astype(int)

# Сохраняем
results = pd.DataFrame({
    'prediction_proba': y_proba,
    'prediction_class': y_class
})

if 'lead_id' in df.columns:
    results.insert(0, 'lead_id', df['lead_id'])

results.to_csv('new_predictions.csv', index=False)

print("\nСохранено: new_predictions.csv")
print(f"Средняя вероятность: {y_proba.mean():.3f}")
print(f"Доля класса 1: {y_class.mean():.3f}")
print("\nПервые 5:")
print(results.head())