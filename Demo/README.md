# Power Grid Material Demand Forecast - Demo

> **Status:** ✅ Fully functional production demo

A machine learning-powered system for predicting material requirements in power transmission projects using XGBoost models.

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train models (first time only):**
```bash
python train_model.py
```
*Takes ~30 seconds. Creates model files in `models/` directory.*

3. **Launch application:**
```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

## What It Does

Predicts material requirements for power grid projects:
- **Steel** (tons)
- **Conductor** (kilometers) 
- **Insulators** (units)
- **Transformers** (count)

**Accuracy:** ~87% R² score  
**Speed:** <2 seconds per prediction

## How to Use

1. **Configure project** in the sidebar (11 parameters: location, budget, towers, terrain, etc.)
2. **Click "Predict Material Demand"**
3. **View results** with cost breakdown and visualizations
4. **Print/save** report as needed

## Key Features

- Real-time ML predictions using XGBoost
- Interactive Streamlit UI with visualizations
- Automatic cost estimation
- Confidence scoring
- Trained on 500+ historical projects

## Troubleshooting

**Models not found?**  
→ Run `python train_model.py`

**Import errors?**  
→ Run `pip install -r requirements.txt`

**Unrealistic predictions?**  
→ Verify input parameters are within reasonable ranges

## Limitations

- Predictions based on historical patterns only
- Best accuracy for projects similar to training data
- Use as planning tool, not replacement for engineering assessment

## Tech Stack

- **ML Framework:** XGBoost 2.0+
- **UI:** Streamlit 1.28+
- **Visualization:** Plotly 5.17+
- **Data:** Pandas, NumPy, Scikit-learn

---

For the full experimental version with advanced features, see the `src/` directory (under development).

**Developed for Smart India Hackathon 2025**