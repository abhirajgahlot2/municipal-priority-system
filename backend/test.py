import pandas as pd

df = pd.read_csv(
    "data/processed/processed_complaints.csv"
)

print(
    df["safety_risk"]
    .value_counts()
    .head(20)
)

print("\nUnique safety-risk values:")
print(df["safety_risk"].nunique())

print("\nRisk distribution:")
print(
    pd.cut(
        df["safety_risk"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True
    ).value_counts()
    .sort_index()
)


print(
    df[
        [
            "severity",
            "traffic_impact",
            "public_impact",
            "weather_risk",
            "days_pending",
            "complaint_frequency",
            "safety_risk"
        ]
    ].describe()
)