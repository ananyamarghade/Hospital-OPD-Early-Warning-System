from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "daily_arrivals",
    "critical_ratio",
    "avg_stay",
    "lab_load",
    "instrument_load",
    "action_load",
    "graphy_load",
    "arrivals_roll_7d",
    "stay_roll_7d",
    "critical_roll_7d"
]

def train_model(daily):
    X = daily[FEATURES].fillna(0)
    y = daily["overload_in_3d"]

    split_idx = int(len(daily) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("\nTraining early-warning model (predicting 3 days ahead)...")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
