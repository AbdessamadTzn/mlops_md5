from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Chargement du modèle au démarrage
MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/model_columns.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
else:
    model = None
    model_columns = None

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    support_calls: int
    usage_frequency: float

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Churn Prediction API",
        "health_check": "/health",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Conversion du JSON en DataFrame avec le bon ordre de colonnes
        df = pd.DataFrame([data.model_dump()])
        df = df[model_columns]
        
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        
        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
