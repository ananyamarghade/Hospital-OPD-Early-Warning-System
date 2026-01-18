def create_labels(daily):
    arr_thr = daily["daily_arrivals"].quantile(0.90)
    crit_thr = daily["critical_ratio"].quantile(0.90)
    stay_thr = daily["avg_stay"].quantile(0.90)

    daily["overload_today"] = (
        (daily["daily_arrivals"] > arr_thr) |
        (daily["critical_ratio"] > crit_thr) |
        (daily["avg_stay"] > stay_thr)
    ).astype(int)

    # 3 days ahead label
    daily["overload_in_3d"] = daily["overload_today"].shift(-3)

    daily = daily.dropna(subset=["overload_in_3d"])
    daily["overload_in_3d"] = daily["overload_in_3d"].astype(int)

    print("\nFuture label distribution:")
    print(daily["overload_in_3d"].value_counts())

    return daily
