import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

print("🚀 Starting Model Training...")
print("="*50)

os.makedirs('models', exist_ok=True)

print("\n📂 Loading dataset...")
df = pd.read_csv("training_data.csv")
print(f"✅ Loaded {len(df)} projects")

print("\n🔄 Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
encoders = {}

for col in categorical_cols:
    if col != "Project_ID":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   ✓ Encoded: {col}")

print("\n🔧 Engineering features...")
df['Project_Duration'] = df['Completion_Year'] - df['Start_Year']
df['MVA_per_Substation'] = df['Transformation_Capacity_MVA'] / (df['Substations_Count'] + 1)
df['Budget_per_MVA'] = df['Budget_Cr'] / (df['Transformation_Capacity_MVA'] + 1)
df['Budget_per_Line'] = df['Budget_Cr'] / (df['Line_Length_CKM'] + 1)
print("✅ Feature engineering complete")

# Features
input_features = [
    'Location', 'Geographic_Region', 'Tower_Count', 'Substations_Count',
    'Tower_Type', 'Substation_Type', 'Tax_Rate', 'Line_Length_CKM',
    'Transformation_Capacity_MVA', 'Budget_Cr',
    'Project_Duration', 'MVA_per_Substation', 'Budget_per_MVA', 'Budget_per_Line'
]

regression_targets = [
    'Steel_Demand_Tons', 'Conductor_Demand_Km', 'Insulator_Demand_Nos'
]
classification_target = 'Transformer_Demand_Units'

print("\n📊 Splitting data (80% train, 20% test)...")
X = df[input_features]
y_reg = df[regression_targets]
y_cls = df[classification_target]

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)
print(f"✅ Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train regression models
print("\n" + "="*50)
print("🤖 Training Regression Models...")
print("="*50)

models = {}
for i, col in enumerate(regression_targets):
    print(f"\n🎯 Training: {col}")
    model = XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_reg_train.iloc[:, i])
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_reg_test.iloc[:, i], y_pred)
    r2 = r2_score(y_reg_test.iloc[:, i], y_pred)
    
    print(f"   📈 MSE: {mse:.2f}")
    print(f"   📈 R²: {r2:.4f}")
    
    model_name = col.lower().replace('_demand', '').replace('_', '_')
    models[col] = model
    pickle.dump(model, open(f'models/{model_name}_model.pkl', 'wb'))
    print(f"   💾 Saved: models/{model_name}_model.pkl")

print(f"\n🎯 Training: Transformer_Demand_Units")
transformer_model = XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6)
transformer_model.fit(X_train, y_cls_train)

y_cls_pred = transformer_model.predict(X_test)
mse = mean_squared_error(y_cls_test, y_cls_pred)
r2 = r2_score(y_cls_test, y_cls_pred)

print(f"   📈 MSE: {mse:.2f}")
print(f"   📈 R²: {r2:.4f}")

pickle.dump(transformer_model, open('models/transformer_model.pkl', 'wb'))
print(f"   💾 Saved: models/transformer_model.pkl")

pickle.dump(encoders, open('models/encoders.pkl', 'wb'))
print(f"\n💾 Saved encoders")

print("\n" + "="*50)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*50)
print("\n📁 Models saved in 'models/' folder:")
print("   - steel_demand_tons_model.pkl")
print("   - conductor_demand_km_model.pkl")
print("   - insulator_demand_nos_model.pkl")
print("   - transformer_model.pkl")
print("   - encoders.pkl")
print("\n🎉 You can now run the Streamlit app with real predictions!")