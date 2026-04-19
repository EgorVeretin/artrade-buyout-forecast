import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from data_loader import read_data
from preprocessors import delete_null_and_danger_columns, find_and_convert_datetime_columns, enrich_data
from feature_engineering import apply_all_transformations
from list_features import create_top_features_from_lists
from utils import fill_missing_values, get_top_articles, get_top_tags
from train_evaluate import (train_model, evaluate_model, calibrate_and_plot, 
                            find_optimal_threshold, plot_roc_curve, timeseries_cross_validation)
from save_pipeline import save_full_pipeline
from config import DANGER_COLUMNS, NULL_THRESHOLD

def preprocess_data(path: str) -> pd.DataFrame:
    """Основной пайплайн предобработки"""
    df = read_data(path)
    df = delete_null_and_danger_columns(df)
    df = find_and_convert_datetime_columns(df)
    df = enrich_data(df)
    df = apply_all_transformations(df)
    return df

def main():
    PATH_TO_FILE = "dataset_2025-03-01_2026-03-29_external.csv"
    print(f"Загрузка файла {PATH_TO_FILE}...")
    
    data = preprocess_data(PATH_TO_FILE)
    
    top_articles = get_top_articles(data, top_n=10)
    print("Топ-10 артикулов:")
    print(top_articles)
    
    top_tags = get_top_tags(data, top_n=10)
    print("\nТоп-10 тегов:")
    print(top_tags)
    
    numeric_cols = data.select_dtypes(exclude='object').columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(data[numeric_cols].corr(),
                yticklabels=data[numeric_cols].columns,
                xticklabels=data[numeric_cols].columns,
                vmax=1, vmin=-1, cmap='BrBG')
    plt.title("Correlation Heatmap")
    plt.show()
    
    data = data.set_index('lead_id', drop='first')
    data.to_csv('clear_data_for_train_test_split.csv', sep=',')
    
    data_2025 = data.loc[data['sale_date'] == 2025, :]
    data_2026 = data.loc[data['sale_date'] == 2026, :]
    
    X_train = data_2025.drop('buyout_flag', axis=1)
    X_test = data_2026.drop('buyout_flag', axis=1)
    y_train = data_2025['buyout_flag']
    y_test = data_2026['buyout_flag']
    
    special_cols = ['articles', 'lead_tags']
    categorical_cols = X_train.select_dtypes(include='object').columns.to_list()
    categorical_cols = list(set(categorical_cols) - set(special_cols))
    
    X_train, X_test = create_top_features_from_lists(X_train, X_test, 'articles', top_n=10)
    X_train, X_test = create_top_features_from_lists(X_train, X_test, 'lead_tags', top_n=10)
    
    categorical_cols_final = X_train.select_dtypes(include='object').columns.to_list()
    numeric_cols_final = X_train.select_dtypes(exclude='object').columns.to_list()
    
    X_train, X_test = fill_missing_values(X_train, X_test, categorical_cols_final, numeric_cols_final)
    
    model = train_model(X_train, y_train, X_test, y_test, categorical_cols, tune=True)
    
    y_pred_raw = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    auc_raw = roc_auc_score(y_test, y_pred_raw)
    plot_roc_curve(y_test, y_pred_raw, auc_raw)
    
    y_pred_calibrated = calibrate_and_plot(model, X_train, X_test, y_train, y_test)
    
    auc_calibrated = roc_auc_score(y_test, y_pred_calibrated)
    print(f"\nROC-AUC после калибровки: {auc_calibrated:.3f}")
    
    print("\n=== Поиск оптимального порога для откалиброванных вероятностей ===")
    best_th = find_optimal_threshold(y_test, y_pred_calibrated)
    
    print("\nTimeSeries кросс-валидация:")
    timeseries_cross_validation(model, X_train, y_train, categorical_cols)
    
    threshold_70_percentile = np.percentile(y_pred_calibrated, 70)
    print(f"\n70-й персентиль откалиброванных предсказаний: {threshold_70_percentile:.3f}")
    
    print("\n=== Сравнение подходов ===")
    y_pred_optimal = (y_pred_calibrated >= best_th).astype(int)
    y_pred_percentile = (y_pred_calibrated >= threshold_70_percentile).astype(int)
    
    print(f"При оптимальном пороге {best_th:.2f}:")
    print(f"Precision: {precision_score(y_test, y_pred_optimal):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_optimal):.3f}")
    
    print(f"\nПри 70-м персентиле {threshold_70_percentile:.3f}:")
    print(f"Precision: {precision_score(y_test, y_pred_percentile):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred_percentile):.3f}")
    
    predictions_df = pd.DataFrame({
        'lead_id': X_test.index,
        'y_true': y_test.values,
        'y_pred_raw': y_pred_raw,
        'y_pred_calibrated': y_pred_calibrated,
        'y_pred_optimal_threshold': y_pred_optimal,
        'y_pred_70_percentile': y_pred_percentile
    })
    
    predictions_df.to_csv('predictions.csv', index=False)
    print("\nПредсказания сохранены в predictions.csv")
    
    # СОХРАНЯЕМ ПОЛНЫЙ ПАЙПЛАЙН ДЛЯ ИСПОЛЬЗОВАНИЯ НА НОВЫХ ДАННЫХ
    from feature_extractors import extract_articles_from_row, extract_cost_from_row, extract_delivery_cost_from_row
    from transformers import transform_weight_to_category, make_feature_binary, transform_lead_qualification
    
    feature_extractors = {
        'extract_articles_from_row': extract_articles_from_row,
        'extract_cost_from_row': extract_cost_from_row,
        'extract_delivery_cost_from_row': extract_delivery_cost_from_row
    }
    
    transformers = {
        'transform_weight_to_category': transform_weight_to_category,
        'make_feature_binary': make_feature_binary,
        'transform_lead_qualification': transform_lead_qualification
    }
    
    config = {
        'DANGER_COLUMNS': DANGER_COLUMNS,
        'NULL_THRESHOLD': NULL_THRESHOLD,
        'optimal_threshold': best_th,
        'percentile_70': threshold_70_percentile
    }
    
    save_full_pipeline(
        model=model,
        preprocess_func=preprocess_data,
        feature_extractors=feature_extractors,
        transformers=transformers,
        config=config
    )
    
    print("\nPipeline сохранен! Теперь можно использовать на новых данных.")

if __name__ == "__main__":
    main()