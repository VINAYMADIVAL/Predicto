# PowerGrid Material Demand Forecasting System
*Smart India Hackathon 2025 – Predicto Team*

## Overview

A *production-ready ML system* for forecasting material demand in power grid projects (Steel, Conductor, Insulators, Transformers).

* *Hybrid Model:* Random Forest + Exponential Smoothing
* *Temporal Validation:* Train ≤2019, Test >2019
* *Procurement Planning:* Lead times + safety stock
* *Confidence Intervals:* 90% & 50%
* *Stack:* Python (FastAPI, scikit-learn, statsmodels), PostgreSQL, JS/HTML/CSS, Excel

<details>
<summary><b>Features & Highlights</b></summary>

* Forecast quarterly material demand
* Excel export: forecast, procurement, confidence intervals
* Web dashboard with login/registration
* JWT authentication for security

</details>

<details>
<summary><b>Project Structure</b></summary>

*Backend:*

* `model.py` → ML pipeline, feature engineering, forecasting, safety stock
* `main.py` → FastAPI backend, endpoints, JWT auth
* `boosted_powergrid_material_demand.csv` → 5775 historical projects

*Frontend (optional for judging):*

* `index.html` → Main dashboard
* `src/` → JS/TS logic + templates
* `public/` → Static assets
* `package.json` → Dependency manager

*Other files:* TypeScript, Tailwind, Vite, ESLint configs, lock files

</details>

<details>
<summary><b>Setup Instructions</b></summary>

*Prerequisites:* Python 3.8+, Node.js, PostgreSQL, Git

*Backend:*

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
uvicorn main:app --reload
```

* Configure `.env` with PostgreSQL & JWT
* Ensure CSV dataset is in project root

*Frontend (optional):*

```bash
cd <frontend-folder>
npm install    # or bun install
npm run dev    # or bun run dev
```

*Access:*

* API → [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Frontend → [http://localhost:5173](http://localhost:5173)

</details>

<details>
<summary><b>Usage</b></summary>

*Endpoints:*

| Method | Endpoint                            | Purpose                |
| ------ | ----------------------------------- | ---------------------- |
| GET    | `/`                                 | Home page              |
| GET    | `/register`, `/login`, `/dashboard` | Auth & dashboard       |
| POST   | `/api/forecast`                     | Forecast demand (JSON) |
| POST   | `/api/export_forecast`              | Export Excel file      |

*Example Input:*

```json
{
  "Budget_Cr": 2000, "Tower_Count": 2000, "Voltage": 400,
  "Substations_Count": 10, "Tower_Type": "Lattice",
  "Circuit": "Single", "Line_Length_CKM": 500,
  "Transformation_Capacity_MVA": 1000, "Terrain_Difficulty": "Moderate",
  "Substation_Type": "AIS", "Tax_Rate": 18, "Geographic_Region": "North",
  "Location": "Delhi", "Start_Year": 2025, "Completion_Year": 2026, "Start_Month": 1
}
```

*Output:* JSON + Excel (forecast, procurement, confidence intervals, summary)

</details>

<details>
<summary><b>Validation & Performance</b></summary>

| Material    | R²    | Notes                               |
| ----------- | ----- | ----------------------------------- |
| Steel       | 0.936 | Train ≤2019: 4432, Test >2019: 1343 |
| Conductor   | 0.987 | -                                   |
| Insulator   | 0.920 | -                                   |
| Transformer | 0.825 | -                                   |

*Overall Confidence:* ~92% average R²

</details>

<details>
<summary><b>Future Improvements</b></summary>

* Real-time data updates
* Interactive forecast charts
* Scalable API for multiple users

</details>

<details>
<summary><b>License & Contact</b></summary>

SIH 2025 participation
Contact: Predicto Team via rnavaneethn.dsce@gmail.com

</details>


