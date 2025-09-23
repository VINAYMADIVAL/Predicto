# PERSON 2 (Navneethan)
"""
Person 2 - production-ready model functions.
Each function accepts 'features' dict and returns:
    predictions (list[{"date": "YYYY-MM-DD", "forecast": float}]),
    model_info (dict),
    metrics (dict)
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List

def _make_future_dates(start_date_str: str, days: int = 7) -> List[str]:
    try:
        start = datetime.fromisoformat(start_date_str).date()
    except Exception:
        start = datetime.utcnow().date()
    return [(start + timedelta(days=i+1)).isoformat() for i in range(days)]

def xgboost_model(features: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Simulated XGBoost model prediction (production).
    Replace inside with actual model loading/predict code.
    """
    # Simple deterministic formula to simulate output
    base = features["feature_1"] * 0.5 + features["feature_2"] * 0.3
    dates = _make_future_dates(features["date"], days=7)
    predictions = [{"date": d, "forecast": round(base + idx * 1.5, 2)} for idx, d in enumerate(dates)]
    model_info = {"model_name": "xgboost_model", "version": "v1.0"}
    metrics = {"rmse": 1.8, "mae": 1.2, "r2": 0.85}
    print("[Person2] xgboost_model -> base:", base)
    return predictions, model_info, metrics

def naive_model(features: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    A naive baseline model: repeat last known value or apply small drift.
    """
    base = features["feature_1"] * 0.4 + features["feature_2"] * 0.25
    dates = _make_future_dates(features["date"], days=7)
    predictions = [{"date": d, "forecast": round(base + (idx * 0.8), 2)} for idx, d in enumerate(dates)]
    model_info = {"model_name": "naive_model", "version": "v1.0"}
    metrics = {"rmse": 3.2, "mae": 2.6, "r2": 0.62}
    print("[Person2] naive_model -> base:", base)
    return predictions, model_info, metrics

def prophet_model(features: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Simulated Prophet-like model.
    """
    base = features["feature_1"] * 0.55 + features["feature_2"] * 0.2
    dates = _make_future_dates(features["date"], days=7)
    predictions = [{"date": d, "forecast": round(base + (idx * 2.0), 2)} for idx, d in enumerate(dates)]
    model_info = {"model_name": "prophet_model", "version": "v1.0"}
    metrics = {"rmse": 2.1, "mae": 1.5, "r2": 0.78}
    print("[Person2] prophet_model -> base:", base)
    return predictions, model_info, metrics

# Add new production model functions here (e.g., def some_new_model(...))
