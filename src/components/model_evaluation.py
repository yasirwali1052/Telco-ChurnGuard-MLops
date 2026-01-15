import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import read_yaml, load_object


class ModelEvaluation:
    """
    Model Evaluation class for evaluating model and logging to MLflow
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelEvaluation
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.model_path = self.config["model_path"]
        self.preprocessor_path = self.config["preprocessor_path"]
        self.mlflow_uri = self.config["mlflow_tracking_uri"]
        self.experiment_name = self.config["experiment_name"]
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> dict:
        """
        Evaluate model and return metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics["roc_auc"] = None
        
        return metrics
    
    def log_into_mlflow(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metrics: dict,
        params: dict = None
    ) -> None:
        """
        Log model and metrics to MLflow
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            metrics: Dictionary of metrics
            params: Dictionary of model parameters
        """
        try:
            with mlflow.start_run():
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        mlflow.log_metric(metric_name, metric_value)
                
                # Infer signature
                predictions = model.predict(X_test[:5])
                signature = infer_signature(X_test[:5], predictions)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name="ChurnPredictionModel"
                )
                
                logger.info("Model logged to MLflow successfully")
                
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise
    
    def initiate_model_evaluation(self, test_array: np.ndarray) -> None:
        """
        Evaluate model on test data and log to MLflow
        
        Args:
            test_array: Test data array
        """
        try:
            logger.info("Starting model evaluation")
            
            # Load model
            model = load_object(self.model_path)
            
            # Split features and target
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Print metrics
            logger.info("Model Evaluation Metrics:")
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # Classification report
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification Report:\n{report}")
            
            # Log to MLflow
            params = {
                "model_type": self.config.get("model_type", "xgboost"),
                "n_estimators": self.config.get("n_estimators", 100),
                "max_depth": self.config.get("max_depth", 6),
                "learning_rate": self.config.get("learning_rate", 0.1)
            }
            
            self.log_into_mlflow(model, X_test, y_test, metrics, params)
            
            logger.info("Model evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise

