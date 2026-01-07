"""
FastAPI application for plan name prediction using sentence transformers.

This API accepts payer name and dirty plan name as input and returns:
- Clean plan name
- Plan type
- Line of business (LOB)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Initialize FastAPI app
app = FastAPI(
    title="Plan Name Prediction API",
    description="Predict clean plan names, plan types, and LOB from dirty plan names",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
sentence_model = None
le_plan_name = None
le_plan_type = None
le_lob = None
model_plan_name = None
model_plan_type = None
model_lob = None


def clean_text(text: str) -> str:
    """Standardize text data for matching."""
    if not text or text.strip() == "":
        return ""
    text = str(text).lower()
    text = re.sub(r'[\W_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.on_event("startup")
async def load_models():
    """Load all models on application startup."""
    global sentence_model, le_plan_name, le_plan_type, le_lob
    global model_plan_name, model_plan_type, model_lob
    
    try:
        logger.info("Loading models...")
        
        # Load model bundle
        logger.info("Loading model bundle...")
        bundle_path = MODELS_DIR / 'plan_prediction_bundle.joblib'
        
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}")
            
        model_bundle = joblib.load(bundle_path)
        
        # Extract models from bundle
        logger.info("Extracting models...")
        le_plan_name = model_bundle['le_plan_name']
        le_plan_type = model_bundle['le_plan_type']
        le_lob = model_bundle['le_lob']
        
        model_plan_name = model_bundle['model_plan_name']
        model_plan_type = model_bundle['model_plan_type']
        model_lob = model_bundle['model_lob']
        
        # Load sentence transformer
        model_name = model_bundle.get('sentence_transformer_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading sentence transformer: {model_name}")
        sentence_model = SentenceTransformer(model_name)
        
        logger.info("✓ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for plan prediction."""
class PredictionRequest(BaseModel):
    """Request model for plan prediction."""
    payer_name: str = Field(..., min_length=1, max_length=200, description="Name of the insurance payer")
    dirty_plan_name: str = Field(..., min_length=1, max_length=500, description="Dirty/raw plan name to be cleaned")


class PredictionResponse(BaseModel):
    """Response model for plan prediction."""
    clean_plan_name: str
    clean_plan_name_score: float
    plan_type: str
    plan_type_score: float
    line_of_business: str
    line_of_business_score: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "healthy",
        "message": "Plan Name Prediction API is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if sentence_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "message": "All models loaded and ready"
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_plan(request: PredictionRequest):
    """Predict clean plan name, plan type, and LOB from dirty plan name."""
    try:
        if sentence_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        if not request.payer_name.strip() or not request.dirty_plan_name.strip():
            raise HTTPException(
                status_code=400, 
                detail="payer_name and dirty_plan_name cannot be empty"
            )
        
        # Clean and combine input
        payer_cleaned = clean_text(request.payer_name)
        plan_cleaned = clean_text(request.dirty_plan_name)
        combined_name = f"{payer_cleaned} {plan_cleaned}"
        
        # Generate embedding
        embedding = sentence_model.encode([combined_name], convert_to_numpy=True)
        
        # Make predictions & get probabilities
        # Plan Name
        pred_plan_name_probs = model_plan_name.predict_proba(embedding)
        pred_plan_name_idx = pred_plan_name_probs.argmax()
        pred_plan_name_score = float(pred_plan_name_probs[0][pred_plan_name_idx])
        pred_plan_name = le_plan_name.inverse_transform([pred_plan_name_idx])[0]

        # Plan Type
        pred_plan_type_probs = model_plan_type.predict_proba(embedding)
        pred_plan_type_idx = pred_plan_type_probs.argmax()
        pred_plan_type_score = float(pred_plan_type_probs[0][pred_plan_type_idx])
        pred_plan_type = le_plan_type.inverse_transform([pred_plan_type_idx])[0]

        # LOB
        pred_lob_probs = model_lob.predict_proba(embedding)
        pred_lob_idx = pred_lob_probs.argmax()
        pred_lob_score = float(pred_lob_probs[0][pred_lob_idx])
        pred_lob = le_lob.inverse_transform([pred_lob_idx])[0]
        
        # Apply Confidence Threshold
        CONFIDENCE_THRESHOLD = 0.4
        
        if pred_plan_name_score < CONFIDENCE_THRESHOLD:
            pred_plan_name = "Not Matched"
            # Note: We keep the score to show why it wasn't matched
            
        if pred_plan_type_score < CONFIDENCE_THRESHOLD:
            pred_plan_type = "Unknown"
            
        if pred_lob_score < CONFIDENCE_THRESHOLD:
            pred_lob = "Unknown"
        
        return PredictionResponse(
            clean_plan_name=pred_plan_name,
            clean_plan_name_score=round(pred_plan_name_score, 4),
            plan_type=pred_plan_type,
            plan_type_score=round(pred_plan_type_score, 4),
            line_of_business=pred_lob,
            line_of_business_score=round(pred_lob_score, 4)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
