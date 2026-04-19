# Константы и конфигурация

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

NULL_THRESHOLD = 0.60

# Параметры модели по умолчанию
DEFAULT_CATBOOST_PARAMS = {
    'iterations': 400,
    'learning_rate': 0.03,
    'depth': 6,
    'auto_class_weights': 'Balanced',
    'logging_level': 'Silent',
    'random_seed': 42,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 50,
    'l2_leaf_reg': 5
}

# Сетка для Grid Search
PARAM_GRID = {
    'iterations': [400, 600, 700],
    'learning_rate': [0.02, 0.03, 0.05],
    'depth': [6, 7, 8],
    'l2_leaf_reg': [1, 3, 5]
}