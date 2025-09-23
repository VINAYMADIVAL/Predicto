# PERSON 6 (Kushal)
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any
from datetime import datetime

def _dates_to_datetimes(dates):
    dt_list = []
    for d in dates:
        try:
            dt_list.append(datetime.fromisoformat(d))
        except Exception:
            dt_list.append(datetime.utcnow())
    return dt_list

def create_plot(predictions: List[Dict[str, Any]]) -> str:
    """
    Create a line plot from predictions and return base64 PNG (utf-8 string).
    predictions: list of {"date": "YYYY-MM-DD", "forecast": float}
    """
    dates = [p["date"] for p in predictions]
    values = [p["forecast"] for p in predictions]
    x = _dates_to_datetimes(dates)

    plt.figure(figsize=(8, 4))
    plt.plot(x, values, marker='o')
    plt.title("Forecast Predictions")
    plt.xlabel("Date")
    plt.ylabel("Forecast Value")
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    print("[Person6] plot created")
    return img_base64
