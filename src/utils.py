import os
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict
from loguru import logger


def read_yaml(path_to_yaml: str) -> Dict[str, Any]:
    """
    Read yaml file and return dictionary
    
    Args:
        path_to_yaml: Path to yaml file
        
    Returns:
        Dictionary containing yaml content
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        logger.error(f"Error reading yaml file: {e}")
        raise


def create_directories(path_to_directories: list) -> None:
    """
    Create directories if they don't exist
    
    Args:
        path_to_directories: List of directory paths to create
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory at: {path}")


def save_object(file_path: str, obj: Any) -> None:
    """
    Save object to pickle file
    
    Args:
        file_path: Path to save the object
        obj: Object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object: {e}")
        raise


def load_object(file_path: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object: {e}")
        raise

