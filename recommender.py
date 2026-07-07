"""
Maternal Recommendation Engine
------------------------------
Reads recommendation dataset and returns
diet, exercise and medicine recommendation.
"""

import pandas as pd


# ===============================
# Load Recommendation Dataset
# ===============================

RULES = pd.read_csv("dataset/recommendation_rules.csv")

# Remove unwanted spaces from headers
RULES.columns = RULES.columns.str.strip()

# Remove unwanted spaces from all string values
for column in RULES.columns:
    if RULES[column].dtype == object:
        RULES[column] = RULES[column].astype(str).str.strip()


# ===============================
# Detect Patient Condition
# ===============================

def detect_condition(sbp, dbp, bs, temp, hr):

    conditions = []

    if sbp >= 140 or dbp >= 90:
        conditions.append("BP")

    if bs >= 7.0:
        conditions.append("Sugar")

    if temp >= 99:
        conditions.append("Temperature")

    if hr >= 100:
        conditions.append("HeartRate")

    if len(conditions) >= 2:
        return "Combined"

    if len(conditions) == 0:
        return "General"

    return conditions[0]


# ===============================
# Recommendation Function
# ===============================

def recommend(
        risk,
        sbp,
        dbp,
        bs,
        temp,
        hr,
        trimester):

    condition = detect_condition(
        sbp,
        dbp,
        bs,
        temp,
        hr
    )

    # ------------------------------
    # Exact Match
    # ------------------------------

    recommendation = RULES[
        (RULES["RiskLevel"].str.lower() == risk.lower())
        &
        (RULES["Trimester"] == trimester)
        &
        (RULES["ConditionType"] == condition)
    ]

    # ------------------------------
    # Fallback 1
    # ------------------------------

    if recommendation.empty:

        recommendation = RULES[
            (RULES["RiskLevel"].str.lower() == risk.lower())
            &
            (RULES["Trimester"] == trimester)
        ]

    # ------------------------------
    # Fallback 2
    # ------------------------------

    if recommendation.empty:

        recommendation = RULES[
            RULES["RiskLevel"].str.lower() == risk.lower()
        ]

    # ------------------------------
    # Last Resort
    # ------------------------------

    if recommendation.empty:

        recommendation = RULES.iloc[[0]]

    row = recommendation.sample(1).iloc[0]

    return {

        "breakfast": row["Breakfast"],

        "lunch": row["Lunch"],

        "dinner": row["Dinner"],

        "exercise": row["ExerciseRecommendation"],

        "medicine": row["MedicineName"],

        "condition": condition

    }