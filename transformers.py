import pandas as pd

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