"""
TruthLens ML Service
====================
FastAPI service that serves a pre-trained TF-IDF + Logistic Regression model
for six-class statement classification.

IMPORTANT: Outputs are statistical estimates produced by a text classifier.
They are NOT verified facts and NOT a factual credibility score.

This file only LOADS the saved artifacts. It never trains or modifies them.

Run from inside the ml-service folder:
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("truthlens")

# ---------------------------------------------------------------------------
# Paths: resolve the project root from THIS file's location, not from the
# current working directory, so it works no matter where uvicorn is launched.
#   main.py -> ml-service/ -> zero-rag-project/ (root)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Optional label mapping.
# If your model was trained on string labels ("true", "false", ...), leave this
# EMPTY: the labels are read directly from model.classes_.
# If it was trained on integer labels (0-5), fill this in to match EXACTLY
# the encoding used in train_models.py, e.g. {0: "false", 1: "half-true", ...}
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[Any, str] = {}

MAX_STATEMENT_LENGTH = 5000

DISCLAIMER = (
    "These results are model estimates based on text patterns, not verified facts. "
    "The probabilities are not a factual credibility score."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_artifact(path: Path) -> Any:
    """Load a .pkl file saved with joblib or plain pickle (read-only)."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def label_name(raw_label: Any) -> str:
    """Convert a raw class label from the model into a readable string."""
    key = raw_label.item() if hasattr(raw_label, "item") else raw_label
    return LABEL_MAP.get(key, str(key))


def preprocess_text(text: str) -> str:
    """
    Hook for text preprocessing.
    If train_models.py cleaned the text BEFORE fitting the vectorizer
    (e.g. removed punctuation, stemming), replicate EXACTLY that here.
    Otherwise inputs at prediction time will differ from training data.
    Default: whitespace trimming only.
    """
    return text.strip()


# ---------------------------------------------------------------------------
# Startup: load model + vectorizer once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.vectorizer = None
    app.state.load_error = None
    try:
        model = load_artifact(MODEL_PATH)
        vectorizer = load_artifact(VECTORIZER_PATH)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Loaded model does not support predict_proba().")

        # Sanity check: the model and vectorizer must come from the same training run.
        n_model = getattr(model, "n_features_in_", None)
        n_vec = len(getattr(vectorizer, "vocabulary_", {}))
        if n_model is not None and n_vec and n_model != n_vec:
            raise RuntimeError(
                f"Feature mismatch: model expects {n_model} features but the "
                f"vectorizer produces {n_vec}. These files are not from the same training run."
            )

        n_classes = len(model.classes_)
        if n_classes != 6:
            logger.warning("Expected 6 classes but model has %d: %s", n_classes, list(model.classes_))

        app.state.model = model
        app.state.vectorizer = vectorizer
        logger.info("Loaded model and vectorizer from %s", MODELS_DIR)
        logger.info("Classes: %s", [label_name(c) for c in model.classes_])
    except Exception as exc:  # keep the server up so /health can report the problem
        app.state.load_error = str(exc)
        logger.exception("Failed to load model artifacts")
    yield


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TruthLens ML Service",
    description="Six-class statement classifier (TF-IDF + Logistic Regression). "
    "Outputs are model estimates, not verified facts.",
    version="1.0.0",
    lifespan=lifespan,
)

# Local development origins only. Add your own ports if needed.
# Note: opening index.html by double-click (file://) sends Origin "null";
# serve it with VS Code Live Server instead, or add "null" below for dev only.
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # Node/Express backend
    "http://127.0.0.1:3000",
    "http://localhost:5500",   # VS Code Live Server
    "http://127.0.0.1:5500",
    "http://localhost:5173",   # Vite
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    statement: str

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Statement must not be empty.")
        if len(value) > MAX_STATEMENT_LENGTH:
            raise ValueError(f"Statement must be at most {MAX_STATEMENT_LENGTH} characters.")
        return value


class PredictResponse(BaseModel):
    statement: str
    prediction: str
    probabilities: Dict[str, float]
    disclaimer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    """Report whether the API is running and whether the model is loaded."""
    loaded = app.state.model is not None and app.state.vectorizer is not None
    body: Dict[str, Any] = {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
    }
    if loaded:
        body["classes"] = [label_name(c) for c in app.state.model.classes_]
    else:
        body["error"] = app.state.load_error
    return body


# Plain `def` (not async): scikit-learn is CPU-bound, and FastAPI runs sync
# endpoints in a threadpool so the event loop is not blocked.
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Classify a statement into one of the model's six categories."""
    model = app.state.model
    vectorizer = app.state.vectorizer
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health for details.",
        )

    try:
        # 1. Convert text to TF-IDF features using the SAVED vectorizer (transform only, never fit).
        features = vectorizer.transform([preprocess_text(payload.statement)])

        # 2. Predicted category and class probabilities.
        raw_prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # 3. Pair each class label with its probability (order follows model.classes_).
        probs_by_label = {
            label_name(cls): round(float(p), 4)
            for cls, p in zip(model.classes_, probabilities)
        }

        return PredictResponse(
            statement=payload.statement,
            prediction=label_name(raw_prediction),
            probabilities=probs_by_label,
            disclaimer=DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        )


# Catch-all so unexpected errors return JSON instead of an HTML page.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# Allows `python main.py` (and VS Code's Run button) to start the server.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)