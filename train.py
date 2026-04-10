"""
Entraînement du modèle de prédiction du coût d'assurance médicale.
Dataset : https://www.kaggle.com/datasets/mirichoi0218/insurance
Colonnes : age, sex, bmi, children, smoker, region, charges
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── Chargement des données ──────────────────────────────────────────────────
DATA_PATH = "data/insurance.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Chargement des données...")
df = pd.read_csv(DATA_PATH)
print(f"   {len(df)} lignes, colonnes : {list(df.columns)}")

# ── Encodage des variables catégorielles ───────────────────────────────────
cat_cols = ["sex", "smoker", "region"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"   Encodage '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Split train / test ─────────────────────────────────────────────────────
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train : {len(X_train)} | Test : {len(X_test)}")

# ── Entraînement ───────────────────────────────────────────────────────────
print("\n🚀 Entraînement du modèle (GradientBoosting)...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# ── Évaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Résultats :")
print(f"   MAE  : {mae:,.2f} $")
print(f"   R²   : {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"   R² CV (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Sauvegarde ─────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODERS_PATH, "wb") as f:
    pickle.dump(encoders, f)

print(f"\n💾 Modèle sauvegardé → {MODEL_PATH}")
print(f"💾 Encodeurs sauvegardés → {ENCODERS_PATH}")
print("\n✨ Entraînement terminé !")
