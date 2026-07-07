"""
Utility Functions
"""

import joblib
import pandas as pd


MODEL_PATH = "models/risk_model.pkl"
ENCODER_PATH = "models/encoder.pkl"


def load_model():

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


def prepare_input(
    age,
    sbp,
    dbp,
    bs,
    temp,
    hr
):

    return pd.DataFrame(
        [[age, sbp, dbp, bs, temp, hr]],
        columns=[
            "Age",
            "SystolicBP",
            "DiastolicBP",
            "BS",
            "BodyTemp",
            "HeartRate"
        ]
    )