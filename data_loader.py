import pandas as pd

def load_and_prepare_data(DATA_PATH):
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Rows:", len(df))

    # Build date column
    df["date"] = pd.to_datetime(
        dict(
            year=df["ResidentDate_year"],
            month=df["ResidentDate_month"],
            day=df["ResidentDate_day"]
        ),
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    # Define critical cases
    df["is_critical"] = df["triage_code"].isin([1, 2]).astype(int)

    # Clean numeric columns
    numeric_cols = [
        "age",
        "ResidentDay",
        "service_count_lab",
        "service_count_instrument",
        "service_count_action",
        "service_count_graphy"
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df
