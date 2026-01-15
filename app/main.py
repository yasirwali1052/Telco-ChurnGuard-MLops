import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.prediction_pipeline import PredictionPipeline
from app.schemas import CustomerData, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="Telco ChurnGuard API",
    description="MLOps API for Telco Customer Churn Prediction",
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

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize prediction pipeline
try:
    prediction_pipeline = PredictionPipeline()
    logger.info("Prediction pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing prediction pipeline: {e}")
    prediction_pipeline = None


@app.get("/")
async def root():
    """Serve the main UI"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": "Welcome to Telco ChurnGuard API",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if prediction_pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not initialized")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerData):
    """
    Predict churn for a single customer
    
    Args:
        customer_data: Customer data
        
    Returns:
        Prediction response with churn probability
    """
    if prediction_pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not initialized")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data.dict()])
        
        # Make prediction
        results = prediction_pipeline.predict(df)
        
        # Format response
        response = PredictionResponse(
            prediction=results["predictions"][0],
            prediction_label=results["prediction_labels"][0],
            churn_probability=results["churn_probability"][0],
            no_churn_probability=results["probabilities"][0][0]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers
    
    Args:
        request: Batch prediction request with list of customers
        
    Returns:
        Batch prediction response
    """
    if prediction_pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not initialized")
    
    try:
        # Convert to DataFrame
        customers_dict = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(customers_dict)
        
        # Make predictions
        results = prediction_pipeline.predict(df)
        
        # Format response
        predictions = []
        for i in range(len(results["predictions"])):
            predictions.append(
                PredictionResponse(
                    prediction=results["predictions"][i],
                    prediction_label=results["prediction_labels"][i],
                    churn_probability=results["churn_probability"][i],
                    no_churn_probability=results["probabilities"][i][0]
                )
            )
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from src.utils import read_yaml
    
    config = read_yaml("config.yaml")
    uvicorn.run(
        "app.main:app",
        host=config.get("api_host", "0.0.0.0"),
        port=config.get("api_port", 8000),
        reload=True
    )

