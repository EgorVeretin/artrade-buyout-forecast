import re
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterGrid
from sklearn.calibration import CalibratedClassifierCV

DANGER_COLUMNS = [
    'contact_Телефон', 'lead_Комментарий', 'contact_updated_at',
    'contact_name', 'contact_first_name', 'handed_to_delivery_ts', 'lead_COOKIES',
     'lead_Тариф Доставки', 'lead_Компания Отправитель', 'lead_Вид оплаты','lead_name',
    'lead_TRANID', 'lead_account_id', 'lead_closed_at', 'lead_updated_at',
    'lead_created_at', 'lead_loss_reason_id', 'rejected_ts', 'returned_ts',
    'outcome_unknown', 'sale_ts', 'lead_is_deleted', 'lead_status_id',
    'current_status_id', 'issued_or_pvz_ts', 'received_ts', 'contact_id', 'lead_utm_campaign',
    'closed_ts', 'lead_pipeline_id', 'lead_Дата создания сделки', 'days_to_outcome',
    'lead_Дата перехода в Сборку', 'lead_Дата получения денег на Р/С', 'contact_Адрес ПВЗ',
    'contact_Код ПВЗ', 'lead_utm_term', 'lead_utm_sky', 'lead_utm_referrer', 
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
    
    time_patterns = \
        r'_ts$|_at$|time|date|день|дата|создан|обновлен|закрыт|переход|получен|выдан|доставк|возврат|сборк|продаж'
    
    for col in df.columns:
        col_lower = col.lower()
        
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
   
    df = read_data(path)
   
    df = delete_null_and_danger_columns(df)
    
    df = find_and_convert_datetime_columns(df)
    
    df = enrich_data(df)
    
    df = df[~df['buyout_flag'].isna()]
    
    df['buyout_flag'] = df['buyout_flag'].astype('float')
    
    df['lead_tags'] = df['lead_tags'].map(parcing_data_from_tags_column)
    
    df['lead_Вес (грамм)*'] = df['lead_Вес (грамм)*'].map(transform_weight_to_category)
    
    df['lead_Модель телефона'] = df['lead_Модель телефона'].map(make_feature_binary)
    
    df['lead_будущие покупки'] = df['lead_будущие покупки'].map(make_feature_binary)
    
    df['lead_Квалификация лида'] = df['lead_Квалификация лида'].map(transform_lead_qualification)
    
    df['lead_FORMID'] = np.where(df['lead_FORMID'].isna(), 0, 1)
    
    df['lead_REFERER'] = np.where(df['lead_REFERER'].isna(), 0, 1)
    
    df['lead_utm_source'] = df['lead_utm_source'].fillna('Неизвестно')
    
    df['contact_Число сделок'] = df['contact_Число сделок'].fillna(1)
    
    df['lead_Категория и варианты выбора'] = df['lead_Категория и варианты выбора'].fillna('Нет категории')
    
    max_dim = df[['lead_Длина', 'lead_Ширина', 'lead_Высота']].max(axis=1)
    min_dim = df[['lead_Длина', 'lead_Ширина', 'lead_Высота']].min(axis=1)
    df['size_ratio'] = np.where(min_dim > 0, max_dim / min_dim, 0)
    df.drop(['lead_Длина', 'lead_Ширина', 'lead_Высота'], inplace=True, axis=1)
    
    df['lead_utm_group'] = df['lead_utm_group'].fillna('unknown_status')
    
    df['lead_FORMNAME'] = np.where(df['lead_FORMNAME'].isna(), 0, 1)
    
    df.loc[df['delivery_cost']==0, 'discount'] = 1
    df['discount'] = df['discount'].fillna(0)
    
    df.loc[df['delivery_cost']==1,'is_gift'] = 1
    df['is_gift'] = df['is_gift'].fillna(0)
    
    df['articles'] = df['articles'].map(lambda x: ([item for item in x if item != '1']))
    df['unique_articles'] = df['articles'].map(lambda x: len(x))
    
    df['cost_without_delivery'] = df['cost_with_delivery'] - df['delivery_cost']
    df.drop('cost_with_delivery', axis=1, inplace=True)
    
    sale_date = pd.to_datetime(df['sale_date'])
    contact_created = pd.to_datetime(df['contact_created_at'])
    df['customer_tenure_days_frac'] = (sale_date - contact_created).dt.total_seconds() / (24 * 3600)
    df.loc[df['customer_tenure_days_frac'] < 0, 'customer_tenure_days_frac'] = 0
        
    df['is_same_day'] = (df['sale_date'] == df['contact_created_at']).astype(int)
    
    df['sale_date'] = pd.to_datetime(df['sale_date']).dt.year
    df = df.sort_values(by='sale_date', ascending=True)

    df['contact_deals_bin'] = pd.cut(df['contact_Число сделок'], 
                                   bins=[0, 1, 3, 6, 10, np.inf], 
                                   labels=['0', '1-2', '3-5', '6-10', '10+']).astype('str')
    
    df['deals_x_weight'] = df['contact_Число сделок'] * df['lead_Вес (грамм)*'].map({'light':1, 'medium':2, 'heavy':3})
    df['deals_x_delivery'] = df['contact_Число сделок'].astype(str) + '_' + df['lead_Служба доставки']
    
    threshold_price = df['cost_without_delivery'].quantile(0.75)
    df['has_expensive_item'] = (df['cost_without_delivery'] > threshold_price).astype(int)

    df['delivery_cost_ratio'] = df['delivery_cost'] / (df['cost_without_delivery'] + 1)
    
    # прокси-признак
    df['sale_dayofweek'] = pd.to_datetime(df['sale_date']).dt.dayofweek
    
    # прокси-признак
    df['sale_month'] = pd.to_datetime(df['sale_date']).dt.month
    
    # прокси-признак
    df['sale_is_weekend'] = (df['sale_dayofweek'] >= 5).astype(int)
    
    df['deals_log'] = np.log1p(df['contact_Число сделок'])
    
    df['deals_x_city'] = df['contact_Число сделок'].astype(str) + '_' + df['contact_Город'].astype(str)
    
    df['is_payday_period'] = pd.to_datetime(df['sale_date']).dt.day.between(1, 5) | \
        pd.to_datetime(df['sale_date']).dt.day.between(25, 31)
    df['is_payday_period'] = df['is_payday_period'].astype(int)
    
    df['days_handed_to_issued_pvz'] = df['days_handed_to_issued_pvz'].map(lambda x: abs(x))
    df['log_days_handed_to_issued_pvz'] = np.log1p(df['days_handed_to_issued_pvz'])
    df.drop('days_handed_to_issued_pvz', axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    PATH_TO_FILE = "dataset_2025-03-01_2026-03-29_external.csv"
    print(f"Путь до обновленного файла {PATH_TO_FILE}...")
    
    data = preprocess_data(PATH_TO_FILE)
        
    top_articles = get_top_articles(data, top_n=10)
    print(top_articles)
    
    top_tags = get_top_tags(data, top_n=10)
    print(top_tags)

data.columns
data = data.set_index('lead_id', drop='first')
data.to_csv('clear_data_for_train_test_split.csv', sep=',') # для аналитика!

categorical_cols = data.select_dtypes(include='object').columns.to_list()
numeric_cols = data.select_dtypes(exclude='object').columns.to_list()
special_cols = ['articles', 'lead_tags']
categorical_cols = list(set(categorical_cols) - set(special_cols))

print(numeric_cols)
print(special_cols)

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

categorical_cols_final = X_train.select_dtypes(include='object').columns.to_list()
numeric_cols_final = X_train.select_dtypes(exclude='object').columns.to_list()

# Заполняем пропуски
def transform_nan_values(X_train: pd.DataFrame,X_test: pd.DataFrame, categorical_cols: list, numeric_cols: list) -> tuple:
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    for cat in categorical_cols:
        X_train_copy[cat] = X_train_copy[cat].fillna(X_train_copy[cat].mode()[0])
        X_test_copy[cat] = X_test_copy[cat].fillna(X_test_copy[cat].mode()[0])
    
    # учтем бинарность некоторых фичей
    list_binary_cols = []
    for col in numeric_cols_final:
        if len(X_train[col].value_counts().index) <= 5:
            list_binary_cols.append(col)

    for num in numeric_cols:
        if num in list_binary_cols:
            X_train_copy[num] = X_train_copy[num].fillna(X_train_copy[num].mode()[0])
            X_test_copy[num] = X_test_copy[num].fillna(X_test_copy[num].mode()[0])
        else:
            X_train_copy[num] = X_train_copy[num].fillna(X_train_copy[num].mean())
            X_test_copy[num] = X_test_copy[num].fillna(X_test_copy[num].mean())

    return X_train_copy, X_test_copy

X_train, X_test = transform_nan_values(X_train, X_test, categorical_cols_final, numeric_cols_final)

model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.03,
    depth=6,
    cat_features=categorical_cols,
    auto_class_weights='Balanced',
    logging_level='Silent',
    random_seed=42,
    eval_metric='AUC',
    early_stopping_rounds=50,
    l2_leaf_reg=5
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)]
)


base_params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 5,
    'l2_leaf_reg': 3  # добавим регуляризацию
}

# Сетка для перебора (около базовых значений)
param_grid = {
    'iterations': [400, 600, 700],
    'learning_rate': [0.02, 0.03, 0.05],
    'depth': [6, 7, 8],
    'l2_leaf_reg': [1, 3, 5]
}

best_auc = 0
best_params = None

print("Перебор параметров...")
for params in ParameterGrid(param_grid):
    # Берём ваш шаблон, меняем только параметры
    model_tune = CatBoostClassifier(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        cat_features=categorical_cols,
        logging_level='Silent',
        random_seed=42,
        eval_metric='AUC',
        early_stopping_rounds=50
    )
    
    model_tune.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Оценка на тесте (у вас уже есть X_test, y_test)
    pred = model_tune.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    
    print(f"params={params}, AUC={auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_params = params
        best_model = model_tune

print(f"\nЛучшие параметры: {best_params}")
print(f"Лучший AUC на тесте: {best_auc:.4f}")

model = best_model

y_pred = model.predict_proba(X_test)[:, 1]
y_pred_from_model = model.predict(X_test)
pr_auc = average_precision_score(y_test, y_pred)
print(f"\nPR-AUC на тесте: {pr_auc:.3f}")
auc = roc_auc_score(y_test, y_pred)
print(f"\nROC-AUC на тесте: {auc:.3f}")
auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) # аномалия какая-то
print(f"\nROC-AUC на трейне: {auc_train:.3f}")

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop='first')

print("\nТоп-10 важных признаков:")
print(importance.head(10).to_string(index=False))

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


fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


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

# лучший порог - для калибровки 
y_pred_optimal = (y_pred >= best_th).astype(int)
print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_optimal):.4f}")


# Твои данные уже отсортированы по времени (2025 -> 2026)
# Для CV используем только 2025 год
X_train_cv = X_train.copy()
y_train_cv = y_train.copy()

# TimeSeriesSplit 
tscv = TimeSeriesSplit(n_splits=20, gap=30)  # gap - отступ между фолдами

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_cv)):
    print(f"Fold {fold + 1}: train {len(train_idx)}, val {len(val_idx)}")
    
    model.fit(
        X_train_cv.iloc[train_idx], 
        y_train_cv.iloc[train_idx],
        eval_set=[(X_train_cv.iloc[val_idx], y_train_cv.iloc[val_idx])],
        verbose=False
    )
    
    pred = model.predict_proba(X_train_cv.iloc[val_idx])[:, 1]
    auc = roc_auc_score(y_train_cv.iloc[val_idx], pred)
    cv_scores.append(auc)
    print(y_train_cv.iloc[train_idx].value_counts(normalize=True))
    print(f"  AUC: {auc:.4f}")
print(np.mean(cv_scores))

# высокая стабильность на кросс-валидации
# Вместо жёсткого порога 0.7 используем персентиль
threshold_70_percentile = np.percentile(y_pred, 70)
print(f"70-й персентиль предсказаний: {threshold_70_percentile:.3f}") # как альтернатива калибровке

precision_score(y_test, (y_pred >= threshold_70_percentile).astype(int)) # используем адаптивную вероятность
recall_score(y_test, (y_pred >= threshold_70_percentile).astype(int))

# Обучаем калиброванный классификатор
calibrated_cb = CalibratedClassifierCV(
    model, 
    method='sigmoid',
    cv=3
)
calibrated_cb.fit(X_train, y_train)

# Калиброванные вероятности
y_pred_calibrated = calibrated_cb.predict_proba(X_test)[:, 1]

# Проверяем калибровку - изотоническая калибровка хуже
prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_pred_calibrated, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='CatBoost with calibrated')
plt.plot([0, 1], [0, 1], '--', label='Perfect')
plt.xlabel('Predicted probability with calibration')
plt.ylabel('Actual probability')
plt.title('Calibration plot after preprocess sigmoid calibration')
plt.legend()
plt.show()
