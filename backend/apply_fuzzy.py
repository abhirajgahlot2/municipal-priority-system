import pandas as pd
from fuzzy_engine import calculate_safety_risk

INPUT_FILE = "data/processed/processed_complaints.csv"
OUTPUT_FILE = "data/processed/processed_complaints.csv"


def main():

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "severity",
        "traffic_impact",
        "public_impact",
        "weather_risk",
        "days_pending",
        "complaint_frequency"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    safety_risks = []

    for _, row in df.iterrows():

        complaint = {
            "severity": row["severity"],
            "traffic_impact": row["traffic_impact"],
            "public_impact": row["public_impact"],
            "weather_risk": row["weather_risk"],
            "days_pending": row["days_pending"],
            "complaint_frequency": row["complaint_frequency"]
        }

        try:
            risk = calculate_safety_risk(complaint)

        except Exception as e:
            print(
                f"Error processing complaint "
                f"{row.get('unique_key', 'unknown')}: {e}"
            )
            risk = 0.0

        safety_risks.append(risk)

    df["safety_risk"] = safety_risks

    # Keep risk safely within 0–1
    df["safety_risk"] = (
        df["safety_risk"]
        .clip(0.0, 1.0)
        .round(4)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n================================")
    print("FUZZY PROCESSING COMPLETED")
    print("================================")

    print(f"Total complaints: {len(df)}")

    print("\nSafety risk statistics:")
    print(df["safety_risk"].describe())

    print("\nSample results:")
    print(
        df[
            [
                "complaint_type",
                "severity",
                "traffic_impact",
                "public_impact",
                "weather_risk",
                "days_pending",
                "complaint_frequency",
                "safety_risk"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()