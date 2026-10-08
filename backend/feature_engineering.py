import numpy as np
import pandas as pd

# ============================================================
# 1. TEXT PREPROCESSING
# ============================================================

def combine_text(row):
    """
    Combine complaint type and descriptor into one searchable text.
    """

    complaint_type = str(row.get("complaint_type", ""))
    descriptor = str(row.get("descriptor", ""))

    return f"{complaint_type} {descriptor}".lower()


# ============================================================
# 2. HELPER: KEYWORD MATCHING
# ============================================================

def contains_any(text, keywords):
    """
    Returns True if any keyword appears in the text.
    """

    return any(keyword in text for keyword in keywords)


# ============================================================
# 3. SEVERITY
# ============================================================

def derive_severity(row):
    """
    Estimate severity on a 0-10 scale.

    This is a domain-informed estimate based on the type
    and description of the complaint.
    """

    text = combine_text(row)

    # Very serious safety-related problems
    if contains_any(text, [
        "gas leak",
        "fire",
        "explosion",
        "collapsed",
        "major flooding",
        "water main break",
        "traffic signal",
        "traffic light",
        "dangerous",
        "hazard",
        "road obstruction"
    ]):
        score = 9

    # High severity
    elif contains_any(text, [
        "pothole",
        "flooding",
        "flood",
        "sewer",
        "drainage",
        "water leak",
        "street condition",
        "broken hydrant"
    ]):
        score = 8

    # Medium-high severity
    elif contains_any(text, [
        "streetlight",
        "street light",
        "electrical",
        "heat",
        "hot water",
        "no water"
    ]):
        score = 7

    # Medium severity
    elif contains_any(text, [
        "garbage",
        "sanitation",
        "waste",
        "sidewalk",
        "graffiti",
        "illegal parking"
    ]):
        score = 5

    # Lower severity
    elif contains_any(text, [
        "noise",
        "parking",
        "information",
        "inquiry"
    ]):
        score = 3

    else:
        score = 4

    # Descriptor-based adjustment
    if contains_any(text, [
        "dangerous",
        "major",
        "severe",
        "broken",
        "blocked",
        "emergency"
    ]):
        score += 1

    if contains_any(text, [
        "minor",
        "small"
    ]):
        score -= 1

    return float(np.clip(score, 0, 10))


# ============================================================
# 4. TRAFFIC IMPACT
# ============================================================

def derive_traffic_impact(row):
    """
    Estimate how strongly a complaint can affect traffic
    or movement through public roads.

    IMPORTANT:
    This is NOT actual traffic volume.

    Scale:
        0-2 : negligible
        3-4 : low
        5-6 : moderate
        7-8 : high
        9-10: very high
    """

    text = combine_text(row)

    # Direct traffic-control problems
    if contains_any(text, [
        "traffic signal",
        "traffic light",
        "signal malfunction",
        "signal failure"
    ]):
        score = 10

    # Major road hazards
    elif contains_any(text, [
        "pothole",
        "road obstruction",
        "road hazard",
        "street flooding",
        "street flood",
        "water main break"
    ]):
        score = 8

    # Road/street problems
    elif contains_any(text, [
        "street condition",
        "road condition",
        "road damage",
        "blocked road",
        "construction"
    ]):
        score = 7

    # Parking/vehicle movement
    elif contains_any(text, [
        "blocked driveway",
        "illegal parking",
        "parking"
    ]):
        score = 5

    # Sidewalk problems have limited direct traffic impact
    elif contains_any(text, [
        "sidewalk",
        "curb"
    ]):
        score = 4

    # Streetlights can indirectly affect road safety
    elif contains_any(text, [
        "streetlight",
        "street light"
    ]):
        score = 5

    # Garbage generally has relatively low direct traffic impact
    elif contains_any(text, [
        "garbage",
        "litter",
        "waste",
        "sanitation"
    ]):
        score = 2

    # Noise has little direct traffic impact
    elif contains_any(text, [
        "noise"
    ]):
        score = 1

    else:
        score = 3

    # Additional obstruction indicator
    if contains_any(text, [
        "blocked",
        "obstruction",
        "closed"
    ]):
        score += 1

    return float(np.clip(score, 0, 10))


# ============================================================
# 5. PUBLIC IMPACT
# ============================================================

def derive_public_impact(row):
    """
    Estimate how broadly the complaint can affect members
    of the public.

    IMPORTANT:
    This is NOT the actual number of people affected.

    Scale:
        0-2 : very limited
        3-4 : low
        5-6 : moderate
        7-8 : high
        9-10: very high
    """

    text = combine_text(row)

    # Problems potentially affecting large populations
    if contains_any(text, [
        "water main break",
        "water supply",
        "no water",
        "water outage",
        "major flooding",
        "flooding",
        "sewer failure"
    ]):
        score = 9

    # Road/traffic safety problems affect many road users
    elif contains_any(text, [
        "traffic signal",
        "traffic light",
        "pothole",
        "road hazard",
        "road obstruction",
        "street flooding"
    ]):
        score = 8

    # Drainage / infrastructure problems
    elif contains_any(text, [
        "drainage",
        "sewer",
        "street condition",
        "road condition"
    ]):
        score = 7

    # Infrastructure affecting a smaller public area
    elif contains_any(text, [
        "streetlight",
        "street light",
        "sidewalk",
        "curb"
    ]):
        score = 6

    # Sanitation
    elif contains_any(text, [
        "garbage",
        "litter",
        "waste",
        "sanitation"
    ]):
        score = 5

    # Parking/vehicle-specific complaints
    elif contains_any(text, [
        "illegal parking",
        "blocked driveway",
        "parking"
    ]):
        score = 4

    # Noise affects surrounding residents
    elif contains_any(text, [
        "noise"
    ]):
        score = 3

    # Administrative / informational issues
    elif contains_any(text, [
        "information",
        "inquiry"
    ]):
        score = 1

    else:
        score = 4

    return float(np.clip(score, 0, 10))


# ============================================================
# 6. WEATHER RISK
# ============================================================

def derive_weather_risk(row):
    """
    Estimate how sensitive the complaint is to adverse
    weather conditions.

    IMPORTANT:
    This is weather susceptibility, NOT the actual weather
    at the time of the complaint.
    """

    text = combine_text(row)

    # Extremely weather-sensitive
    if contains_any(text, [
        "flooding",
        "flood",
        "drainage",
        "storm",
        "sewer overflow"
    ]):
        score = 9

    # Strongly affected by rain/weather
    elif contains_any(text, [
        "water leak",
        "water main break",
        "pothole",
        "road condition",
        "street condition"
    ]):
        score = 7

    # Moderate weather sensitivity
    elif contains_any(text, [
        "streetlight",
        "street light",
        "electrical"
    ]):
        score = 5

    elif contains_any(text, [
        "garbage",
        "waste",
        "sanitation"
    ]):
        score = 4

    elif contains_any(text, [
        "sidewalk",
        "curb"
    ]):
        score = 4

    else:
        score = 2

    return float(np.clip(score, 0, 10))


# ============================================================
# 7. DAYS PENDING
# ============================================================

def derive_days_pending(row):
    """
    Calculate number of days between complaint creation
    and resolution.

    If complaint is still open, calculate until today.
    """

    created = pd.to_datetime(
        row.get("created_date"),
        errors="coerce"
    )

    closed = pd.to_datetime(
        row.get("closed_date"),
        errors="coerce"
    )

    if pd.isna(created):
        return 0.0

    if pd.isna(closed):
        end_date = pd.Timestamp.today()
    else:
        end_date = closed

    days = (end_date - created).total_seconds() / 86400

    return max(float(days), 0.0)


# ============================================================
# 8. COMPLAINT FREQUENCY
# ============================================================

def derive_complaint_frequency(df):
    """
    Estimate recurrence of similar complaints.

    Similar complaint =
        same complaint_type
        + same incident_zip
        + within previous 30 days

    The raw count is converted to an interpretable 0-10 score:
        0 -> 0
        1 -> 3
        2 -> 6
        3 -> 8
        4+ -> 10
    """

    data = df.copy()

    data["created_date"] = pd.to_datetime(
        data["created_date"],
        errors="coerce"
    )

    data["incident_zip"] = (
        data["incident_zip"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    data["complaint_type"] = (
        data["complaint_type"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Sort chronologically
    data = data.sort_values("created_date").reset_index(drop=True)

    frequency_counts = []

    for i, row in data.iterrows():

        current_date = row["created_date"]

        if pd.isna(current_date):
            frequency_counts.append(0)
            continue

        start_date = current_date - pd.Timedelta(days=30)

        previous = data.iloc[:i]

        matching = previous[
            (previous["complaint_type"] == row["complaint_type"]) &
            (previous["incident_zip"] == row["incident_zip"]) &
            (previous["created_date"] >= start_date)
        ]

        frequency_counts.append(len(matching))

    data["frequency_count"] = frequency_counts

    # Convert the actual recurrence count to an interpretable
    # 0-10 fuzzy score.
    #
    # 0 previous similar complaints -> 0
    # 1 previous similar complaint  -> 3
    # 2 previous similar complaints -> 6
    # 3 previous similar complaints -> 8
    # 4+ previous similar complaints -> 10
    #
    # This avoids percentile ranking, which can give every
    # zero-frequency complaint the same non-zero score.

    def frequency_to_score(count):
        if count == 0:
            return 0.0
        elif count == 1:
            return 3.0
        elif count == 2:
            return 6.0
        elif count == 3:
            return 8.0
        else:
            return 10.0

    data["complaint_frequency"] = (
        data["frequency_count"]
        .apply(frequency_to_score)
    )

    return data


# ============================================================
# 9. COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df):
    """
    Generate all features required by the fuzzy engine
    and ANN.

    Input:
        Raw NYC 311 dataset

    Output:
        Processed dataframe
    """

    data = df.copy()

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    data["complaint_type"] = (
        data["complaint_type"]
        .fillna("Unknown")
        .astype(str)
    )

    data["descriptor"] = (
        data["descriptor"]
        .fillna("Unknown")
        .astype(str)
    )

    # --------------------------------------------------------
    # Derive fuzzy-system features
    # --------------------------------------------------------

    data["severity"] = data.apply(
        derive_severity,
        axis=1
    )

    data["traffic_impact"] = data.apply(
        derive_traffic_impact,
        axis=1
    )

    data["public_impact"] = data.apply(
        derive_public_impact,
        axis=1
    )

    data["weather_risk"] = data.apply(
        derive_weather_risk,
        axis=1
    )

    data["days_pending"] = data.apply(
        derive_days_pending,
        axis=1
    )

    # --------------------------------------------------------
    # Complaint frequency
    # --------------------------------------------------------

    data = derive_complaint_frequency(data)

    return data


# ============================================================
# 10. TEST / EXAMPLE
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/municipal_complaints.csv"
    )

    processed_df = engineer_features(df)

    print("\nGenerated features:")
    print(
        processed_df[
            [
                "complaint_type",
                "descriptor",
                "severity",
                "traffic_impact",
                "public_impact",
                "weather_risk",
                "days_pending",
                "complaint_frequency"
            ]
        ].head(10)
    )

    processed_df.to_csv(
        "data/processed/processed_complaints.csv",
        index=False
    )

    print("\nFeature engineering completed.")
    print(
        "\nFeature ranges:"
    )

    print(
        processed_df[
            [
                "severity",
                "traffic_impact",
                "public_impact",
                "weather_risk",
                "days_pending",
                "complaint_frequency"
            ]
        ].describe()
    )