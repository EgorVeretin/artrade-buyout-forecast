import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             average_precision_score, roc_auc_score, roc_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from config import DEFAULT_CATBOOST_PARAMS, PARAM_GRID, DANGER_COLUMNS, NULL_THRESHOLD

def train_model(X_train, y_train, X_test, y_test, categorical_cols, tune=True):
    """Обучает модель с опциональным Grid Search"""
    
    if tune:
        best_auc = 0
        best_params = None
        best_model = None
        
        print("Перебор параметров...")
        for params in ParameterGrid(PARAM_GRID):
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
    else:
        model = CatBoostClassifier(**DEFAULT_CATBOOST_PARAMS, cat_features=categorical_cols)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Оценивает модель и возвращает предсказания"""
    
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict_proba(X_train)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    auc_train = roc_auc_score(y_train, y_pred_train)
    
    print(f"\nPR-AUC на тесте: {pr_auc:.3f}")
    print(f"ROC-AUC на тесте: {auc:.3f}")
    print(f"ROC-AUC на трейне: {auc_train:.3f}")
    
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\nТоп-10 важных признаков:")
    print(importance.head(10).to_string(index=False))
    
    return y_pred

def calibrate_and_plot(model, X_train, X_test, y_train, y_test):
    """Калибровка модели и возврат откалиброванных вероятностей"""
    
    calibrated_cb = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated_cb.fit(X_train, y_train)
    
    y_pred_calibrated = calibrated_cb.predict_proba(X_test)[:, 1]
    
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test, y_pred_calibrated, n_bins=10)
    
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='CatBoost calibrated')
    plt.plot([0, 1], [0, 1], '--', label='Perfect')
    plt.xlabel('Predicted probability with calibration')
    plt.ylabel('Actual probability')
    plt.title('Calibration plot after sigmoid calibration')
    plt.legend()
    plt.show()
    
    return y_pred_calibrated

def find_optimal_threshold(y_test, y_pred):
    """Находит оптимальный порог по F1"""
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
    
    y_pred_optimal = (y_pred >= best_th).astype(int)
    print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_optimal):.4f}")
    
    return best_th

def plot_roc_curve(y_test, y_pred, auc):
    """Строит ROC-кривую"""
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def timeseries_cross_validation(model, X_train, y_train, categorical_cols, n_splits=20, gap=30):
    """TimeSeries кросс-валидация"""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"Fold {fold + 1}: train {len(train_idx)}, val {len(val_idx)}")
        
        model.fit(
            X_train.iloc[train_idx], 
            y_train.iloc[train_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            verbose=False
        )
        
        pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_train.iloc[val_idx], pred)
        cv_scores.append(auc)
        print(f"AUC: {auc:.3f}")
    
    print(f"Средний AUC по CV: {np.mean(cv_scores):.3f}")
    return cv_scores