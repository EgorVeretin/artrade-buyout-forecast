import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

def save_full_pipeline(model, preprocess_func, feature_extractors, transformers, config, filepath=None):
    """
    Сохраняет модель и все необходимые компоненты для предобработки
    """
    
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"saved_pipeline_{timestamp}.pkl"
    
    pipeline = {
        'model': model,
        'model_params': model.get_params(),
        'feature_names': list(model.feature_names_),  # CatBoost использует feature_names_
        'feature_importances': dict(zip(model.feature_names_, model.feature_importances_)),
        
        'preprocess_func': preprocess_func,
        'feature_extractors': feature_extractors,
        'transformers': transformers,
        
        'config': config,
        
        'training_metadata': {
            'save_date': datetime.now().isoformat(),
            'n_features': len(model.feature_names_),  # вместо feature_count_
            'cat_features_indices': model.get_cat_feature_indices(),
            'model_type': 'CatBoostClassifier',
            'n_classes': len(model.classes_) if hasattr(model, 'classes_') else 2
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Pipeline сохранен в: {filepath}")
    print(f"Размер: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath

def load_full_pipeline(filepath):
    """Загружает полный пайплайн"""
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    
    print(f"Pipeline загружен из: {filepath}")
    print(f"Модель: {pipeline['training_metadata']['model_type']}")
    print(f"Кол-во признаков: {pipeline['training_metadata']['n_features']}")
    
    return pipeline