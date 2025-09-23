# PERSON 1 (-)
from datetime import date
from typing import Dict, Any

def generate_features(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate feature generation for production.
    raw_input is expected to contain: {"sku": "...", "date": "YYYY-MM-DD"} OR date object
    Returns a dict of features (serializable).
    """
    # Ensure date as ISO string
    dt = raw_input.get("date")
    if hasattr(dt, "isoformat"):
        date_str = dt.isoformat()
    else:
        date_str = str(dt)

    # Example derived features (replace with real feature code from Person 1)
    features = {
        "sku": raw_input.get("sku"),
        "date": date_str,
        "feature_1": 100.0,  # example numeric feature
        "feature_2": 200.0
    }
    # Logging for your visibility (Person 4 will see prints in console)
    print("[Person1] generate_features ->", features)
    return features
