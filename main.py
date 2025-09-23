# PERSON 4 (Vinay)
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Dict, Any
import importlib

from data_ingest import generate_features
import train_model  # contains model functions
from plot import create_plot

# ---------- Configuration ----------
# Choose which model function name to call from train_model module.
# Set the environment variable BEST_MODEL to one of the function names in train_model.py,
# e.g., BEST_MODEL=xgboost_model
BEST_MODEL = os.environ.get("BEST_MODEL", "xgboost_model")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("person4_orchestrator")

# FastAPI app and templates
app = FastAPI(title="Person 4 - Production Hub")
templates = Jinja2Templates(directory="templates")

# ---------- Request schemas ----------
class SKUInput(BaseModel):
    sku: str = Field(..., example="SKU123")
    date: str = Field(..., example="2025-09-21")  # YYYY-MM-DD

# ---------- Utilities ----------
def get_model_callable(model_name: str):
    """
    Retrieve the model function from train_model by name.
    This protects you from calling arbitrary functions.
    """
    allowed = {name: getattr(train_model, name) for name in dir(train_model) if callable(getattr(train_model, name))}
    if model_name not in allowed:
        raise KeyError(f"Model '{model_name}' not available. Allowed: {list(allowed.keys())}")
    return allowed[model_name]

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})               #fetching data from front-end (Sadhwi Part)

@app.post("/api/forecast")
def api_forecast(payload: SKUInput):
    """
    Primary production API endpoint used by frontend (JSON).
    Orchestrates: Person1 -> Person2 (selected model) -> Person6 -> aggregate response.
    """
    logger.info("Received API forecast request: %s", payload.dict())
    try:
        # Step 1: Person 1 -> features
        features = generate_features(payload.dict())                                    #Sending data to PERSON 1        

        # Step 2: choose and call Person 2's chosen model function
        model_callable = get_model_callable(BEST_MODEL)                                  #Calling Pre-determined and feeding data to model (Navneethan Part)        
        predictions, model_info, metrics = model_callable(features)
        logger.info("Model %s returned %d predictions", model_info.get("model_name"), len(predictions))

        # Step 3: Person 6 -> plot
        plot_base64 = create_plot(predictions)                                           #Sending processed data for plotting (Kushal Part)

        # Step 4: aggregate response
        response = {
            "features": features,
            "model_info": model_info,
            "metrics": metrics,
            "predictions": predictions,
            "plot_base64": plot_base64
        }
        return JSONResponse(response)
    except KeyError as e:
        logger.error("Configuration error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during orchestration")
        raise HTTPException(status_code=500, detail="Internal server error")

#------ A separate POST route for form submissions (the <form> in index.html) --------
# @app.post("/forecast", response_class=HTMLResponse)
# async def form_forecast(request: Request):
#     """
#     Optional: handles form POSTs from the index.html <form action="/forecast" method="post">.
#     Reads form data and behaves like /api/forecast but returns the raw JSON for simplicity.
#     """
#     form = await request.form()
#     sku = form.get("sku")
#     date = form.get("date")
#     if not sku or not date:
#         raise HTTPException(status_code=400, detail="sku and date are required")
#     payload = SKUInput(sku=sku, date=date)
#     return await api_forecast(payload)
