# Telco ChurnGuard MLOps

A complete MLOps pipeline for predicting Telco Customer Churn using machine learning.

## Project Structure

```
Telco-ChurnGuard-MLops/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   └── schemas.py         # Pydantic schemas
├── artifacts/             # Saved models and preprocessors
├── data/                  # Dataset
│   └── Telco-Customer-Churn.csv
├── logs/                  # Training logs
├── mlflow/                # MLflow tracking
├── notebooks/             # Jupyter notebooks
│   └── eda.ipynb         # Exploratory Data Analysis
├── src/
│   ├── components/        # ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/          # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils.py          # Utility functions
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
├── Dockerfile            # Docker configuration
└── README.md            # This file
```

## Features

- **Data Ingestion**: Automated data loading and train-test splitting
- **Data Transformation**: Preprocessing pipeline with feature engineering
- **Model Training**: Support for XGBoost, LightGBM, and Random Forest
- **Model Evaluation**: Comprehensive metrics and MLflow tracking
- **REST API**: FastAPI-based prediction service
- **Docker Support**: Containerized deployment

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Telco-ChurnGuard-MLops
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Model parameters
- MLflow settings
- API settings

## Usage

### Training the Model

Run the training pipeline:
```bash
python src/pipeline/training_pipeline.py
```

This will:
1. Load and split the data
2. Preprocess features
3. Train the model
4. Evaluate and log to MLflow

### Running the API

Start the FastAPI server:
```bash
python app/main.py
```

Or using uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check
- `POST /predict`: Predict churn for a single customer
- `POST /batch_predict`: Predict churn for multiple customers
- `GET /docs`: Interactive API documentation (Swagger UI)

### Example API Request

```python
import requests

# Single prediction
data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 53.85,
    "TotalCharges": "646.2"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Docker Deployment

Build the Docker image:
```bash
docker build -t telco-churnguard .
```

Run the container:
```bash
docker run -p 8000:8000 telco-churnguard
```

## MLflow Tracking

Model experiments are tracked using MLflow. View the MLflow UI:
```bash
mlflow ui --backend-store-uri file:./mlflow
```

Then open `http://localhost:5000` in your browser.

## Model Performance

The model provides the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## Technologies Used

- **Python 3.10+**
- **Scikit-learn**: Machine learning algorithms
- **XGBoost/LightGBM**: Gradient boosting models
- **FastAPI**: REST API framework
- **MLflow**: Experiment tracking
- **Pandas/NumPy**: Data processing
- **Docker**: Containerization

