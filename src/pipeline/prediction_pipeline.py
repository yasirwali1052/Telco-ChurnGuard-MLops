import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import read_yaml, load_object


class PredictionPipeline:
    """
    Prediction Pipeline for making predictions on new data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize PredictionPipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.model_path = self.config["model_path"]
        self.preprocessor_path = self.config["preprocessor_path"]
        self.target_column = self.config["target_column"]
        
        # Load model and preprocessor
        try:
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
            
            # Try to load target encoder
            target_encoder_path = self.preprocessor_path.replace(".pkl", "_target_encoder.pkl")
            if os.path.exists(target_encoder_path):
                self.target_encoder = load_object(target_encoder_path)
            else:
                self.target_encoder = None
                
            logger.info("Model and preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model/preprocessor: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed array
        """
        try:
            # Create a copy
            df = data.copy()
            
            # Handle TotalCharges conversion
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            
            # Drop customerID if present
            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"], axis=1)
            
            # Apply preprocessing (handles both numerical and categorical automatically)
            processed_data = self.preprocessor.transform(df)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> dict:
        """
        Make predictions on input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            prediction_proba = self.model.predict_proba(processed_data)
            
            # Convert predictions to labels if encoder available
            if self.target_encoder:
                predictions_labels = self.target_encoder.inverse_transform(predictions)
            else:
                predictions_labels = ["Churn" if p == 1 else "No Churn" for p in predictions]
            
            results = {
                "predictions": predictions.tolist(),
                "prediction_labels": predictions_labels,
                "probabilities": prediction_proba.tolist(),
                "churn_probability": prediction_proba[:, 1].tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

