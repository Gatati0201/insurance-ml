"""
API FastAPI – Prédiction du coût d'assurance médicale.

Endpoints :
  GET  /health         → santé de l'API
  POST /predict        → prédire le coût
  GET  /docs           → documentation Swagger (auto)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from app.predict import load_model, predict


# ── Schémas Pydantic ────────────────────────────────────────────────────────

class InsuranceInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Âge de l'assuré (18-100)")
    sex: str = Field(..., description="Sexe : 'male' ou 'female'")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Indice de masse corporelle")
    children: int = Field(..., ge=0, le=10, description="Nombre d'enfants à charge")
    smoker: str = Field(..., description="Fumeur : 'yes' ou 'no'")
    region: str = Field(..., description="Région : 'northeast', 'northwest', 'southeast', 'southwest'")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        if v not in ("male", "female"):
            raise ValueError("sex doit être 'male' ou 'female'")
        return v

    @field_validator("smoker")
    @classmethod
    def validate_smoker(cls, v):
        if v not in ("yes", "no"):
            raise ValueError("smoker doit être 'yes' ou 'no'")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        valid = ("northeast", "northwest", "southeast", "southwest")
        if v not in valid:
            raise ValueError(f"region doit être parmi {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 32,
                "sex": "male",
                "bmi": 27.5,
                "children": 1,
                "smoker": "no",
                "region": "northeast"
            }
        }
    }


class PredictionOutput(BaseModel):
    predicted_cost_usd: float
    message: str


# ── Application ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage."""
    load_model()
    yield


app = FastAPI(
    title="Insurance Cost Predictor",
    description="API de prédiction du coût d'assurance médicale via un modèle ML.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["monitoring"])
def health_check():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "model": "GradientBoostingRegressor"}


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict_cost(data: InsuranceInput):
    """
    Prédit le coût annuel d'assurance médicale en USD.

    - **age** : âge de l'assuré
    - **sex** : male / female
    - **bmi** : indice de masse corporelle
    - **children** : nombre d'enfants à charge
    - **smoker** : yes / no
    - **region** : northeast / northwest / southeast / southwest
    """
    try:
        cost = predict(
            age=data.age,
            sex=data.sex,
            bmi=data.bmi,
            children=data.children,
            smoker=data.smoker,
            region=data.region,
        )
        return PredictionOutput(
            predicted_cost_usd=cost,
            message=f"Coût estimé : {cost:,.2f} USD / an"
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
