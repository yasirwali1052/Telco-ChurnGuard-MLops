import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils import read_yaml, create_directories


def setup_logging(config_path: str = "config.yaml"):
    """
    Setup logging configuration
    
    Args:
        config_path: Path to configuration file
    """
    config = read_yaml(config_path)
    log_dir = config.get("log_dir", "logs")
    log_file = config.get("log_file", "logs/training.log")
    
    create_directories([log_dir])
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="10 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


def main():
    """
    Main function to run the training pipeline
    """
    try:
        # Setup logging
        setup_logging()
        logger.info("=" * 50)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        logger.info("Step 2: Data Transformation")
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        # Step 3: Model Training
        logger.info("Step 3: Model Training")
        model_trainer = ModelTrainer()
        model_path = model_trainer.initiate_model_trainer(train_array, test_array)
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Model Evaluation")
        model_evaluation = ModelEvaluation()
        model_evaluation.initiate_model_evaluation(test_array)
        
        logger.info("=" * 50)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

