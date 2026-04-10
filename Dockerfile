# ── Image de base légère ───────────────────────────────────────────────────
FROM python:3.11-slim

# Évite les fichiers .pyc et bufferise stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dépendances ────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ────────────────────────────────────────────────────────────
COPY app/ ./app/
COPY model/ ./model/

# ── Lancement ──────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
