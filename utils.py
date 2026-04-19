import pandas as pd

def get_top_articles(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Возвращает топ-N самых частых артикулов"""
    from collections import Counter
    all_articles = [art for sublist in df['articles'] for art in sublist]
    counter = Counter(all_articles)
    return pd.DataFrame(counter.most_common(top_n), columns=['article', 'count'])

def get_top_tags(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Возвращает топ-N самых частых тэгов"""
    from collections import Counter
    all_tags = [tag for tags in df['lead_tags'] for tag in tags]
    counter = Counter(all_tags)
    return pd.DataFrame(counter.most_common(top_n), columns=['tag', 'count'])

def fill_missing_values(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         categorical_cols: list, numeric_cols: list) -> tuple:
    """Заполняет пропуски в train и test"""
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    # Определяем бинарные колонки
    list_binary_cols = []
    for col in numeric_cols:
        if len(X_train[col].value_counts().index) <= 5:
            list_binary_cols.append(col)
    
    for cat in categorical_cols:
        X_train_copy[cat] = X_train_copy[cat].fillna(X_train_copy[cat].mode()[0])
        X_test_copy[cat] = X_test_copy[cat].fillna(X_test_copy[cat].mode()[0])
    
    for num in numeric_cols:
        if num in list_binary_cols:
            X_train_copy[num] = X_train_copy[num].fillna(X_train_copy[num].mode()[0])
            X_test_copy[num] = X_test_copy[num].fillna(X_test_copy[num].mode()[0])
        else:
            X_train_copy[num] = X_train_copy[num].fillna(X_train_copy[num].mean())
            X_test_copy[num] = X_test_copy[num].fillna(X_test_copy[num].mean())
    
    return X_train_copy, X_test_copy