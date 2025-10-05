# ================================================================
# SMART INDIA HACKATHON - DEMAND FORECASTING BACKEND (PRODUCTION)
# JWT Auth + Company Registration + Forecast System
# ================================================================

import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional

import psycopg2
import jwt
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Depends, status, Response, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from dotenv import load_dotenv
from pathlib import Path
from typing import Generator
from sqlalchemy.orm import Session
# --- ML Integration ---
from model import PowerGridForecastSystem

# ================================================================
# ENVIRONMENT CONFIGURATION
# ================================================================
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "1234")
POSTGRES_DB = os.getenv("POSTGRES_DB", "predictodb")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

JWT_SECRET = os.getenv("JWT_SECRET", "supplypredict_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "60"))

# ================================================================
# LOGGING CONFIGURATION
# ================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast_backend")

# ================================================================
# SECURITY CONFIGURATION
# ================================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ================================================================
# DATABASE SETUP
# ================================================================
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    gstin = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


Base.metadata.create_all(bind=engine)

# ================================================================
# FASTAPI INITIALIZATION
# ================================================================
app = FastAPI(title="Smart India Hackathon - Demand Forecasting API")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================================================================
# FORECAST SYSTEM INITIALIZATION
# ================================================================
forecast_system = PowerGridForecastSystem()


@app.on_event("startup")
def initialize_forecast_model():
    """Train or load model at startup"""
    try:
        df = pd.read_csv("training_data.csv")
        forecast_system.train(df)
        logger.info("✅ Forecast system initialized successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Forecast system initialization failed: {e}")


# ================================================================
# SCHEMAS
# ================================================================
class SKUInput(BaseModel):
    Budget_Cr: float
    Tower_Count: int
    Substations_Count: int
    Voltage: float
    Tower_Type: str
    Circuit: str
    Line_Length_CKM: float
    Transformation_Capacity_MVA: float
    Terrain_Difficulty: str
    Substation_Type: str
    Tax_Rate: float
    Geographic_Region: str
    Location: str
    Start_Year: int
    Completion_Year: int
    Start_Month: int


class TokenData(BaseModel):
    company_id: int
    company_name: str
    exp: Optional[int]


# ================================================================
# DATABASE UTILITIES
# ================================================================

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_company(db: Session, company_name: str, email: str, gstin: str, password: str) -> int:
    MAX_BCRYPT_BYTES = 72
    MIN_PASSWORD_LENGTH = 8

    password_bytes = password.encode("utf-8")
    if len(password_bytes) > MAX_BCRYPT_BYTES:
        raise HTTPException(status_code=400, detail=f"Password too long (max {MAX_BCRYPT_BYTES} bytes).")
    if len(password_bytes) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password too short (min {MIN_PASSWORD_LENGTH} chars).")

    hashed = pwd_context.hash(password)

    new_company = Company(
        company_name=company_name.strip(),
        email=email.lower().strip(),
        gstin=gstin.upper().strip(),
        password_hash=hashed,
        created_at=datetime.utcnow().isoformat()
    )

    db.add(new_company)
    try:
        db.commit()
        db.refresh(new_company)
        logger.info(f"✅ Company created: {new_company.company_name} (id={new_company.id})")
        return new_company.id
    except Exception as e:
        db.rollback()
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="Email or GSTIN already exists.")
        logger.exception("Database error creating company.")
        raise HTTPException(status_code=500, detail="Internal server error.")


def find_company_by_email(db: Session, email: str) -> Optional[Company]:
    return db.query(Company).filter(Company.email == email.lower()).first()


def find_company_by_id(db: Session, company_id: int) -> Optional[Company]:
    return db.query(Company).filter(Company.id == company_id).first()


# ================================================================
# JWT AUTHENTICATION UTILITIES
# ================================================================
def create_access_token(data: Dict[str, Any], expires_minutes: int = JWT_EXP_MINUTES) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    payload.update({"exp": expire})
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(
            company_id=payload.get("company_id"),
            company_name=payload.get("company_name"),
            exp=payload.get("exp"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def get_current_company(request: Request, db: Session = Depends(get_db)):
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    else:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token.")

    token_data = decode_access_token(token)
    company = find_company_by_id(db, token_data.company_id)
    if not company:
        raise HTTPException(status_code=401, detail="Company not found.")
    return company


# ================================================================
# ROUTES
# ================================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, company: Company = Depends(get_current_company)):
    return templates.TemplateResponse("dashboard.html", {"request": request, "company": company})


@app.post("/register")
async def register_form(
    request: Request,
    company_name: str = Form(...),
    email: str = Form(...),
    gstin: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    company_id = create_company(db, company_name, email, gstin, password)
    token = create_access_token({"company_id": company_id, "company_name": company_name})
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax", secure=False)
    return response


@app.post("/login")
async def login_form(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    company = find_company_by_email(db, email)
    if not company or not pwd_context.verify(password, company.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token({"company_id": company.id, "company_name": company.company_name})
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax", secure=False)
    return response


@app.post("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("access_token")
    return response


# ================================================================
# FORECASTING ENDPOINTS
# ================================================================

@app.post("/api/forecast")
def api_forecast(payload: SKUInput, company: Company = Depends(get_current_company)):
    """Run demand forecasting for provided project details"""
    logger.info("Forecast request received from %s: %s", company.company_name, payload.dict())
    upcoming_projects_df = pd.DataFrame([payload.dict()])
    results = forecast_system.forecast_quarterly_demand(upcoming_projects_df, quarters=4)

    return JSONResponse({
        "forecast_summary": results['executive_summary'],
        "quarterly_forecast": results['quarterly_forecast'].to_dict(orient='records'),
        "procurement_plan": results['procurement_schedule'].to_dict(orient='records'),
        "confidence_intervals": results['confidence_intervals'],
        "company": {
            "id": company.id,
            "company_name": company.company_name,
            "gstin": company.gstin
        }
    })


@app.post("/api/export_forecast")
def export_forecast(payload: SKUInput, company: Company = Depends(get_current_company)):
    """Export forecast to Excel"""
    upcoming_projects_df = pd.DataFrame([payload.dict()])
    results = forecast_system.forecast_quarterly_demand(upcoming_projects_df, quarters=4)
    filename = f"{company.company_name}_Forecast.xlsx"
    forecast_system.export_to_excel(results, filename=filename)
    if os.path.exists(filename):
        return FileResponse(filename, filename=filename)
    return {"message": "Forecast generated but Excel export failed."}


# ================================================================
# END OF FILE
# ================================================================
