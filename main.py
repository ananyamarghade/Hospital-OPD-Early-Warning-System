from data_loader import load_and_prepare_data
from feature_engineering import build_daily_table
from labeling import create_labels
from model import train_model, FEATURES
from evaluation import evaluate_model

DATA_PATH = "ED_admission.csv"

df = load_and_prepare_data(DATA_PATH)

daily = build_daily_table(df)

daily = create_labels(daily)

model, X_train, X_test, y_train, y_test = train_model(daily)

evaluate_model(model, X_test, y_test, FEATURES)

daily["risk_3d_ahead"] = model.predict_proba(daily[FEATURES].fillna(0))[:, 1]
daily.to_csv("daily_3day_ahead_risk.csv", index=False)

print("\nSaved: daily_3day_ahead_risk.csv")
print("\nDONE. Early warning system trained successfully.")
