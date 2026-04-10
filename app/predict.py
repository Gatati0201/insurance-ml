"""
Chargement du modèle et logique de prédiction.
"""

import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "model/encoders.pkl")

# Chargé une seule fois au démarrage de l'API
_model = None
_encoders = None


def load_model():
    global _model, _encoders
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        _encoders = pickle.load(f)
    print(f"✅ Modèle chargé depuis {MODEL_PATH}")


def encode_input(age: int, sex: str, bmi: float,
                 children: int, smoker: str, region: str) -> pd.DataFrame:
    """
    Encode les entrées de la même façon que pendant l'entraînement.
    """
    cat_map = {
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }
    encoded = {}
    for col, val in cat_map.items():
        le = _encoders[col]
        if val not in le.classes_:
            valid = list(le.classes_)
            raise ValueError(f"Valeur '{val}' invalide pour '{col}'. Valeurs acceptées : {valid}")
        encoded[col] = int(le.transform([val])[0])

    df = pd.DataFrame([{
        "age": age,
        "sex": encoded["sex"],
        "bmi": bmi,
        "children": children,
        "smoker": encoded["smoker"],
        "region": encoded["region"],
    }])
    return df


def predict(age: int, sex: str, bmi: float,
            children: int, smoker: str, region: str) -> float:
    """
    Retourne la prédiction du coût d'assurance en USD.
    """
    if _model is None:
        load_model()

    features = encode_input(age, sex, bmi, children, smoker, region)
    prediction = float(_model.predict(features)[0])
    return round(prediction, 2)
