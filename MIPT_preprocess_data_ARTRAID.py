import re
from collections import Counter
import pandas as pd
import numpy as np

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
    'lead_Дата перехода Передан в доставку', 'lifecycle_incomplete', 'lead_utm_medium', 'lead_utm_content',
    'lead_roistat', 'lead__ym_uid', 'lead_yclid', 'lead_Трек-номер СДЭК', 'contact_LTV'
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
    time_patterns = \
        r'_ts$|_at$|time|date|день|дата|создан|обновлен|закрыт|переход|получен|выдан|доставк|возврат|сборк|продаж'
    
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

def transform_weight_to_category(weight: float | None) -> str:
    if pd.isna(weight):
        return 'unknown'
    
    elif weight >= 5000.0:
        return 'heavy'
    
    elif weight >= 2000.0:
        return 'medium'
    
    else:
        return 'light'

def make_feature_binary(text: str) -> int:
    if pd.isna(text):
        return 0
    
    else:
        return 1

def transform_lead_qualification(text: str) -> str:
    if pd.isna(text) or text not in ['D - лид', 'Е - лид', 'А - лид', 'В - лид', 'С - лид']:
        return 'Неквал лид'
    
    if text in ['D - лид', 'Е - лид']:
        return 'С - лид'
    
    else:
        return text


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

def parcing_data_from_tags_column(text: str) -> list:
    text = str(text)
    list_with_tags = text.strip().split(',')
    return list_with_tags

def get_top_tags(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Возвращает топ-N самых частых тэгов"""
    all_tags = [tag for tags in df['lead_tags'] for tag in tags]
    counter = Counter(all_tags)
    return pd.DataFrame(counter.most_common(top_n), columns=['tag', 'count'])

def preprocess_data(path: str) -> pd.DataFrame:
    """Надстройка для основного пайплайна"""
    #print("1. Загрузка данных...")
    df = read_data(path)
    #print(f"   Загружено {df.shape[0]} строк, {df.shape[1]} колонок")
    
    #print("2. Удаление лишних колонок...")
    df = delete_null_and_danger_columns(df)
    #print(f"   Осталось {df.shape[1]} колонок")
    
    #print("3. Преобразование дат...")
    df = find_and_convert_datetime_columns(df)
    
    #print("4. Обогащение данных (артикулы, стоимость)...")
    df = enrich_data(df)
    
    #print("5. Преобразование целевой переменной (buyout_flag)...")
    df = df[~df['buyout_flag'].isna()]
    
    df['buyout_flag'] = df['buyout_flag'].astype('float')
    
    #print("6. Извлечение информации из тэгов...")
    df['lead_tags'] = df['lead_tags'].map(parcing_data_from_tags_column)
    
    #print("7. Преобразование веса (граммы) в категориальные переменные...")
    df['lead_Вес (грамм)*'] = df['lead_Вес (грамм)*'].map(transform_weight_to_category)
    
    #print("8. Преобразуем 'lead_Модель телефона' в бинарный признак...")
    df['lead_Модель телефона'] = df['lead_Модель телефона'].map(make_feature_binary)
    
    #print("9. Преобразуем 'lead_будущие покупки' в бинарный признак...")
    df['lead_будущие покупки'] = df['lead_будущие покупки'].map(make_feature_binary)
    
    #print("10. Преобразуем колонку 'lead_Квалификация лида'...")
    df['lead_Квалификация лида'] = df['lead_Квалификация лида'].map(transform_lead_qualification)
    
    #print("11. Бинаризация признака lead_FORMID...")
    df['lead_FORMID'] = np.where(df['lead_FORMID'].isna(), 0, 1)
    
    #print("12. Бинаризация признака lead_REFERER...")    
    df['lead_REFERER'] = np.where(df['lead_REFERER'].isna(), 0, 1)
    
    #print("13. Обработка признака lead_utm_source...")
    df['lead_utm_source'] = df['lead_utm_source'].fillna('Неизвестно')
    
    #print("14. Обработка признака contact_Число сделок...")
    df['contact_Число сделок'] = df['contact_Число сделок'].fillna(1)
    
    #print("16. Обработка признака lead_Категория и варианты выбора...")
    df['lead_Категория и варианты выбора'] = df['lead_Категория и варианты выбора'].fillna('Нет категории')
    
    #print("17. Обработаем колонку размеров - введем относительный размер...")
    max_dim = df[['lead_Длина', 'lead_Ширина', 'lead_Высота']].max(axis=1)
    min_dim = df[['lead_Длина', 'lead_Ширина', 'lead_Высота']].min(axis=1)
    df['size_ratio'] = np.where(min_dim > 0, max_dim / min_dim, 0)
    df.drop(['lead_Длина', 'lead_Ширина', 'lead_Высота'], inplace=True, axis=1)
    
    #print("Обработаем пропуски в данной графе lead_utm_group...")
    df['lead_utm_group'] = df['lead_utm_group'].fillna('unknown_status')
    
    #print("Обработаем пропуски в колонке lead_FORMNAME...")
    df['lead_FORMNAME'] = np.where(df['lead_FORMNAME'].isna(), 0, 1)
    
    df.loc[df['delivery_cost']==0, 'discount'] = 1
    df['discount'] = df['discount'].fillna(0)
    
    df.loc[df['delivery_cost']==1,'is_gift'] = 1
    df['is_gift'] = df['is_gift'].fillna(0)
    
    df['articles'] = df['articles'].map(lambda x: ([item for item in x if item != '1']))
    df['unique_articles'] = df['articles'].map(lambda x: len(x))
    
    df['cost_without_delivery'] = df['cost_with_delivery'] - df['delivery_cost']
    df.drop('cost_with_delivery', axis=1, inplace=True)
    
    
    df['sale_date'] = pd.to_datetime(df['sale_date']).dt.year
    df = df.sort_values(by='sale_date', ascending=True)
    return df

if __name__ == "__main__":
    PATH_TO_FILE = "dataset_2025-03-01_2026-03-29_external.csv"
    
    data = preprocess_data(PATH_TO_FILE)
    
    top_articles = get_top_articles(data, top_n=10)
    print(top_articles)
    
    top_tags = get_top_tags(data, top_n=10)
    print(top_tags)

data = data.set_index('lead_id', drop='first')
data.to_csv('clear_data_for_train_test_split.csv', sep=',') # для аналитика!

categorical_cols = data.select_dtypes(include='object').columns.to_list()
numeric_cols = data.select_dtypes(exclude='object').columns.to_list()
special_cols = ['articles', 'lead_tags']
categorical_cols = list(set(categorical_cols) - set(special_cols))


"""
print(categorical_cols)
['lead_Проблема', 'contact_responsible_user_id', 'lead_Вес (грамм)*', 'contact_Город', 
 'lead_Категория и варианты выбора', 'lead_Квалификация лида', 'lead_utm_group', 
 'lead_Служба доставки', 'lead_responsible_user_id', 'lead_utm_source']
"""

data['lead_Проблема'].value_counts() # OHE топ-5
data['contact_responsible_user_id'].value_counts() # OHE топ-5
data['lead_Вес (грамм)*'].value_counts() # сразу в OHE
data['contact_Город'].value_counts() # OHE топ-10
data['lead_Категория и варианты выбора'].value_counts() # сразу OHE 
data['lead_Квалификация лида'].value_counts() # сразу в OHE 
data['lead_utm_group'].value_counts() # OHE топ-5
data['lead_Служба доставки'].value_counts() # OHE топ-5
data['lead_responsible_user_id'].value_counts() # OHE топ-5
data['lead_utm_source'].value_counts() # OHE топ-5

features_categorical_data = {
    'features_OHE_top_5' : ['lead_Проблема', 'contact_responsible_user_id', 'lead_utm_group',
                          'lead_Служба доставки', 'lead_responsible_user_id', 'lead_utm_source'],
    'features_OHE_top_10' : ['contact_Город'],
    'features_OHE' : ['lead_Вес (грамм)*', 'lead_Категория и варианты выбора', 'lead_Квалификация лида']
    }

# разбиение колонок в итоге!
print(numeric_cols)
print(special_cols)
print(features_categorical_data)

# начинаем загрузку в модель
data_2025 = data.loc[data['sale_date'] == 2025,:]
data_2026 = data.loc[data['sale_date']==2026,:]

X_train = data_2025.drop('buyout_flag', axis=1)
X_test = data_2026.drop('buyout_flag', axis=1)
y_train = data_2025['buyout_flag']
y_test = data_2026['buyout_flag']


def create_top_features_from_lists(train_df, test_df, column_name, top_n=10):
    """
    Создает признаки на основе топ-N значений в списках.
    Обучается на train, применяется к train и test.
    """
    
    #Собираем все значения из списков в train
    all_values = []
    for lst in train_df[column_name]:
        if isinstance(lst, list):
            all_values.extend(lst)
    
    #Считаем топ-N на train
    counter = Counter(all_values)
    top_values = [v for v, _ in counter.most_common(top_n)]
    
    #Функция для создания признаков
    def make_features(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return {
                f'has_top1_{column_name}': 0,
                f'has_any_top{top_n}_{column_name}': 0,
                f'top_share_{column_name}': 0,
                f'unique_count_{column_name}': 0,
                f'avg_freq_{column_name}': 0
            }
        
        #Попадание в топ-1
        has_top1 = 1 if top_values and top_values[0] in lst else 0
        
        #Попадание в любой из топ-N
        has_any_top = 1 if any(v in top_values for v in lst) else 0
        
        #Доля топ-значений
        top_share = sum(1 for v in lst if v in top_values) / len(lst)
        
        #Количество уникальных
        unique_count = len(lst)
        
        #Средняя частота (популярность) значений в заказе
        freqs = [counter.get(v, 0) for v in lst]
        avg_freq = sum(freqs) / len(freqs) if freqs else 0
        
        return {
            f'has_top1_{column_name}': has_top1,
            f'has_any_top{top_n}_{column_name}': has_any_top,
            f'top_share_{column_name}': top_share,
            f'unique_count_{column_name}': unique_count,
            f'avg_freq_{column_name}': avg_freq
        }
    
    #Применяем к train и test
    train_features = train_df[column_name].apply(make_features).apply(pd.Series)
    test_features = test_df[column_name].apply(make_features).apply(pd.Series)
    
    #Добавляем в датафреймы и удаляем исходную колонку
    train_df = pd.concat([train_df.drop(columns=[column_name]), train_features], axis=1)
    test_df = pd.concat([test_df.drop(columns=[column_name]), test_features], axis=1)
    
    return train_df, test_df

X_train, X_test = create_top_features_from_lists(X_train, X_test, 'articles', top_n=10)
X_train, X_test = create_top_features_from_lists(X_train, X_test, 'lead_tags', top_n=10)

# Заполняем пропуски
X_train.loc[:, 'days_sale_to_handed'] = X_train['days_sale_to_handed'].fillna(X_train['days_sale_to_handed'].median())
X_test.loc[:, 'days_sale_to_handed'] = X_test['days_sale_to_handed'].fillna(X_train['days_sale_to_handed'].median())

X_train.loc[:, 'lead_Проблема'] = X_train['lead_Проблема'].fillna(X_train['lead_Проблема'].mode()[0])
X_test.loc[:, 'lead_Проблема'] = X_test['lead_Проблема'].fillna(X_train['lead_Проблема'].mode()[0])

X_train.loc[:, 'contact_Город'] = X_train['contact_Город'].fillna(X_train['contact_Город'].mode()[0])
X_test.loc[:, 'contact_Город'] = X_test['contact_Город'].fillna(X_train['contact_Город'].mode()[0])

X_train.loc[:, 'lead_Служба доставки'] = X_train['lead_Служба доставки'].\
    fillna(X_train['lead_Служба доставки'].mode()[0])
X_train.loc[:, 'contact_responsible_user_id'] = X_train['contact_responsible_user_id'].\
    fillna(X_train['contact_responsible_user_id'].mode()[0])

# сохранение винальной версии датасета! (на 05.04.2026)
data.to_csv('dataset_for_model_version_1', sep=',')

from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    cat_features=categorical_cols,  # просто передаём список категориальных колонок
    logging_level='Silent',
    random_seed=42,
    eval_metric='AUC',
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)]
)

from sklearn.metrics import average_precision_score, roc_auc_score
y_pred = model.predict_proba(X_test)[:, 1]
y_pred_from_model = model.predict(X_test)
pr_auc = average_precision_score(y_test, y_pred)
print(f"\nPR-AUC на тесте: {pr_auc:.3f}")
auc = roc_auc_score(y_test, y_pred)
print(f"\nROC-AUC на тесте: {auc:.3f}")

import pandas as pd
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop='first')

print("\nТоп-10 важных признаков:")
print(importance.head(10).to_string(index=False))


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='CatBoost')
plt.plot([0, 1], [0, 1], '--', label='Perfect')
plt.xlabel('Predicted probability')
plt.ylabel('Actual probability')
plt.title('Calibration plot')
plt.legend()
plt.show()

print(f"\nПредсказания: min={y_pred.min():.3f}, max={y_pred.max():.3f}, mean={y_pred.mean():.3f}")
print(f"Реальное среднее: {y_test.mean():.3f}")

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



from sklearn.metrics import f1_score, precision_score, recall_score

thresholds = np.arange(0.1, 0.9, 0.02)
best_th = 0.5
best_f1 = 0

for th in thresholds:
    y_pred_th = (y_pred >= th).astype(int)
    f1 = f1_score(y_test, y_pred_th)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"Оптимальный порог: {best_th:.2f}")
print(f"F1 при этом пороге: {best_f1:.4f}")

# Посчитай все метрики при лучшем пороге
y_pred_optimal = (y_pred >= best_th).astype(int)
print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_optimal):.4f}")
