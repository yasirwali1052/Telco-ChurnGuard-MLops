import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import read_yaml, create_directories, save_object


class DataIngestion:
    """
    Data Ingestion class for loading and splitting data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataIngestion
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.raw_data_path = self.config["raw_data_path"]
        self.train_data_path = self.config["train_data_path"]
        self.test_data_path = self.config["test_data_path"]
        self.test_size = self.config["test_size"]
        self.random_state = self.config["random_state"]
        
    def initiate_data_ingestion(self) -> tuple:
        """
        Load data and split into train and test sets
        
        Returns:
            Tuple of (train_data_path, test_data_path)
        """
        logger.info("Starting data ingestion")
        
        try:
            # Read the dataset
            df = pd.read_csv(self.config["data_path"])
            logger.info(f"Dataset loaded. Shape: {df.shape}")
            
            # Create directories for artifacts
            create_directories([
                os.path.dirname(self.raw_data_path),
                os.path.dirname(self.train_data_path),
                os.path.dirname(self.test_data_path)
            ])
            
            # Save raw data
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Raw data saved to {self.raw_data_path}")
            
            # Split data into train and test
            train_set, test_set = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df[self.config["target_column"]] if self.config["target_column"] in df.columns else None
            )
            
            # Save train and test sets
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)
            
            logger.info(f"Train data shape: {train_set.shape}")
            logger.info(f"Test data shape: {test_set.shape}")
            logger.info("Data ingestion completed successfully")
            
            return self.train_data_path, self.test_data_path
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise

