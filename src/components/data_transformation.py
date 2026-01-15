import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import read_yaml, save_object, load_object


class DataTransformation:
    """
    Data Transformation class for preprocessing data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataTransformation
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.preprocessor_path = self.config["preprocessor_path"]
        self.target_column = self.config["target_column"]
        
    def get_data_transformer_object(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline
        
        Args:
            df: DataFrame to identify column types
            
        Returns:
            ColumnTransformer object
        """
        try:
            # Identify numerical and categorical columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
            
            # Remove target column and customerID if present
            if self.target_column in numerical_columns:
                numerical_columns.remove(self.target_column)
            if self.target_column in categorical_columns:
                categorical_columns.remove(self.target_column)
            if "customerID" in categorical_columns:
                categorical_columns.remove("customerID")
            
            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical pipeline (using OneHotEncoder for ColumnTransformer)
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
                ]
            )
            
            # Create transformers list
            transformers = [
                ("num_pipeline", numerical_pipeline, numerical_columns)
            ]
            
            # Add categorical pipeline if there are categorical columns
            if categorical_columns:
                transformers.append(
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                )
            
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop"
            )
            
            logger.info("Preprocessing pipeline created")
            return preprocessor, numerical_columns, categorical_columns
            
        except Exception as e:
            logger.error(f"Error creating transformer: {e}")
            raise
    
    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Apply transformation to train and test data
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Tuple of (train_array, test_array, preprocessor_path)
        """
        try:
            logger.info("Starting data transformation")
            
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Handle TotalCharges conversion (from EDA, we know it needs conversion)
            for df in [train_df, test_df]:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            
            # Get preprocessing object
            preprocessor, numerical_columns, categorical_columns = self.get_data_transformer_object(train_df)
            
            # Separate target variable
            target_feature_train = train_df[self.target_column]
            target_feature_test = test_df[self.target_column]
            
            # Encode target variable
            label_encoder = LabelEncoder()
            target_feature_train = label_encoder.fit_transform(target_feature_train)
            target_feature_test = label_encoder.transform(target_feature_test)
            
            # Handle categorical columns manually
            input_feature_train_df = train_df.drop(columns=[self.target_column], axis=1)
            input_feature_test_df = test_df.drop(columns=[self.target_column], axis=1)
            
            # Drop customerID if present
            if "customerID" in input_feature_train_df.columns:
                input_feature_train_df = input_feature_train_df.drop(columns=["customerID"], axis=1)
                input_feature_test_df = input_feature_test_df.drop(columns=["customerID"], axis=1)
            
            # Apply preprocessing (handles both numerical and categorical)
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Combine with target
            train_arr = np.c_[input_feature_train_arr, target_feature_train]
            test_arr = np.c_[input_feature_test_arr, target_feature_test]
            
            # Save preprocessor
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
            save_object(self.preprocessor_path, preprocessor)
            
            # Save label encoder for target
            target_encoder_path = self.preprocessor_path.replace(".pkl", "_target_encoder.pkl")
            save_object(target_encoder_path, label_encoder)
            
            logger.info("Data transformation completed successfully")
            logger.info(f"Train array shape: {train_arr.shape}")
            logger.info(f"Test array shape: {test_arr.shape}")
            
            return train_arr, test_arr, self.preprocessor_path
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise

