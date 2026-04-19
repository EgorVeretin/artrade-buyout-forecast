"""
Конвертирует saved_pipeline в простой формат без функций
"""

import pickle
import glob

# Находим pipeline
pipeline_files = glob.glob("saved_pipeline_*.pkl")
if not pipeline_files:
    print("❌ Pipeline не найден!")
    exit()

latest_pipeline = max(pipeline_files)
print(f"Загрузка: {latest_pipeline}")

# Загружаем с ignore
import sys

# Создаем заглушку для функции
def dummy_preprocess_data(*args, **kwargs):
    return None

# Подменяем в модуле
sys.modules['__main__'].preprocess_data = dummy_preprocess_data

try:
    with open(latest_pipeline, 'rb') as f:
        pipeline = pickle.load(f)
    print("✅ Pipeline загружен")
except Exception as e:
    print(f"Ошибка: {e}")
    
    # Альтернативный способ - загружаем через dill
    try:
        import dill
        with open(latest_pipeline, 'rb') as f:
            pipeline = dill.load(f)
        print("✅ Pipeline загружен через dill")
    except:
        print("Не удалось загрузить, пересоздаем модель...")
        # Если не загружается, выходим
        exit()

# Сохраняем в простом формате
simple_pipeline = {
    'model': pipeline['model'],
    'optimal_threshold': pipeline['config']['optimal_threshold'],
    'feature_names': pipeline['feature_names'],
    'feature_importances': pipeline.get('feature_importances', {}),
    'model_params': pipeline['model'].get_params()
}

with open('simple_model.pkl', 'wb') as f:
    pickle.dump(simple_pipeline, f)

print("\nСохранено в 'simple_model.pkl'")
print(f"Признаков: {len(simple_pipeline['feature_names'])}")
print(f"Оптимальный порог: {simple_pipeline['optimal_threshold']:.3f}")