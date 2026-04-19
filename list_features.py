import pandas as pd
from collections import Counter

def create_top_features_from_lists(train_df, test_df, column_name, top_n=10):
    """
    Создает признаки на основе топ-N значений в списках.
    Обучается на train, применяется к train и test.
    """
    # Собираем все значения из списков в train
    all_values = []
    for lst in train_df[column_name]:
        if isinstance(lst, list):
            all_values.extend(lst)
    
    # Считаем топ-N на train
    counter = Counter(all_values)
    top_values = [v for v, _ in counter.most_common(top_n)]
    
    def make_features(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return {
                f'has_top1_{column_name}': 0,
                f'has_any_top{top_n}_{column_name}': 0,
                f'top_share_{column_name}': 0,
                f'unique_count_{column_name}': 0,
                f'avg_freq_{column_name}': 0
            }
        
        has_top1 = 1 if top_values and top_values[0] in lst else 0
        has_any_top = 1 if any(v in top_values for v in lst) else 0
        top_share = sum(1 for v in lst if v in top_values) / len(lst)
        unique_count = len(lst)
        freqs = [counter.get(v, 0) for v in lst]
        avg_freq = sum(freqs) / len(freqs) if freqs else 0
        
        return {
            f'has_top1_{column_name}': has_top1,
            f'has_any_top{top_n}_{column_name}': has_any_top,
            f'top_share_{column_name}': top_share,
            f'unique_count_{column_name}': unique_count,
            f'avg_freq_{column_name}': avg_freq
        }
    
    train_features = train_df[column_name].apply(make_features).apply(pd.Series)
    test_features = test_df[column_name].apply(make_features).apply(pd.Series)
    
    train_df = pd.concat([train_df.drop(columns=[column_name]), train_features], axis=1)
    test_df = pd.concat([test_df.drop(columns=[column_name]), test_features], axis=1)
    
    return train_df, test_df