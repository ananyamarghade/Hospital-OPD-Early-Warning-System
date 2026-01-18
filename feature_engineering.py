def build_daily_table(df):
    daily = df.groupby("date").agg(
        daily_arrivals=("PatientCode", "count"),
        critical_cases=("is_critical", "sum"),
        avg_age=("age", "mean"),
        avg_stay=("ResidentDay", "mean"),
        lab_load=("service_count_lab", "sum"),
        instrument_load=("service_count_instrument", "sum"),
        action_load=("service_count_action", "sum"),
        graphy_load=("service_count_graphy", "sum"),
    ).reset_index()

    daily["critical_ratio"] = daily["critical_cases"] / (daily["daily_arrivals"] + 1e-9)
    daily = daily.sort_values("date").reset_index(drop=True)

    # Rolling features
    for w in [3, 7, 14]:
        daily[f"arrivals_roll_{w}d"] = daily["daily_arrivals"].rolling(w, min_periods=1).mean()
        daily[f"stay_roll_{w}d"] = daily["avg_stay"].rolling(w, min_periods=1).mean()
        daily[f"critical_roll_{w}d"] = daily["critical_ratio"].rolling(w, min_periods=1).mean()

    return daily
