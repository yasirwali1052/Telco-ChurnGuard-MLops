import os
import sys
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import read_yaml, save_object


class ModelTrainer:
    """
    Model Trainer class for training and selecting best model
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelTrainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.model_path = self.config["model_path"]
        self.model_type = self.config.get("model_type", "xgboost")
        self.random_state = self.config["random_state"]
        
    def get_model(self):
        """
        Get model based on configuration
        
        Returns:
            Model object
        """
        model_params = {
            "random_state": self.random_state,
            "n_estimators": self.config.get("n_estimators", 100),
            "max_depth": self.config.get("max_depth", 6),
        }
        
        if self.model_type.lower() == "xgboost":
            model = XGBClassifier(
                learning_rate=self.config.get("learning_rate", 0.1),
                **model_params
            )
        elif self.model_type.lower() == "lightgbm":
            model = LGBMClassifier(
                learning_rate=self.config.get("learning_rate", 0.1),
                **model_params
            )
        elif self.model_type.lower() == "random_forest":
            model = RandomForestClassifier(**model_params)
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using XGBoost.")
            model = XGBClassifier(
                learning_rate=self.config.get("learning_rate", 0.1),
                **model_params
            )
        
        logger.info(f"Model type: {self.model_type}")
        return model
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted")
        }
        
        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics["roc_auc"] = None
        
        return metrics
    
    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> str:
        """
        Train model and save it
        
        Args:
            train_array: Training data array
            test_array: Test data array
            
        Returns:
            Path to saved model
        """
        try:
            logger.info("Starting model training")
            
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logger.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
            
            # Get model
            model = self.get_model()
            
            # Train model
            logger.info("Training model...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)
            
            # Evaluate
            train_metrics = self.evaluate_model(y_train, y_train_pred)
            test_metrics = self.evaluate_model(y_test, y_test_pred, y_test_pred_proba)
            
            logger.info("Training Metrics:")
            for metric, value in train_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("Test Metrics:")
            for metric, value in test_metrics.items():
                if value is not None:
                    logger.info(f"  {metric}: {value:.4f}")
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            save_object(self.model_path, model)
            logger.info(f"Model saved to {self.model_path}")
            
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

