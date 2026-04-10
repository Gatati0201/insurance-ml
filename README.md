# 🏥 Insurance Cost Predictor — ML + FastAPI + Docker

Prédiction du coût d'assurance médicale avec un modèle GradientBoosting, servi via une API REST et conteneurisé avec Docker.

---

## 📁 Structure

```
insurance-ml/
├── data/
│   └── insurance.csv       ← télécharger sur Kaggle
├── model/                  ← généré par train.py
│   ├── model.pkl
│   └── encoders.pkl
├── app/
│   ├── main.py             ← API FastAPI
│   └── predict.py          ← logique de prédiction
├── train.py                ← entraînement
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Démarrage rapide

### 1. Récupérer le dataset

Télécharger `insurance.csv` depuis Kaggle :
https://www.kaggle.com/datasets/mirichoi0218/insurance

Le placer dans le dossier `data/`.

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Entraîner le modèle

```bash
python train.py
```

→ Génère `model/model.pkl` et `model/encoders.pkl`

### 4. Lancer l'API en local (sans Docker)

```bash
uvicorn app.main:app --reload
```

→ API disponible sur http://localhost:8000
→ Documentation Swagger : http://localhost:8000/docs

---

## 🐳 Lancer avec Docker

```bash
# Construire et lancer
docker-compose up --build

# En arrière-plan
docker-compose up -d --build

# Arrêter
docker-compose down
```

---

## 🧪 Tester l'API

### Via curl

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "sex": "male",
    "bmi": 27.5,
    "children": 1,
    "smoker": "no",
    "region": "northeast"
  }'
```

### Via Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 32,
        "sex": "male",
        "bmi": 27.5,
        "children": 1,
        "smoker": "no",
        "region": "northeast"
    }
)
print(response.json())
# → {"predicted_cost_usd": 5241.83, "message": "Coût estimé : 5 241.83 USD / an"}
```

---

## 📊 Paramètres d'entrée

| Paramètre  | Type  | Valeurs acceptées                                      |
|------------|-------|--------------------------------------------------------|
| `age`      | int   | 18 – 100                                               |
| `sex`      | str   | `male`, `female`                                       |
| `bmi`      | float | 10.0 – 60.0                                            |
| `children` | int   | 0 – 10                                                 |
| `smoker`   | str   | `yes`, `no`                                            |
| `region`   | str   | `northeast`, `northwest`, `southeast`, `southwest`     |

---

## 📈 Performances du modèle

| Métrique       | Valeur typique |
|----------------|----------------|
| MAE            | ~2 500 $       |
| R² (test set)  | ~0.87          |
| R² (CV 5-fold) | ~0.86          |
