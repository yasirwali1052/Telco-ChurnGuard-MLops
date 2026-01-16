

# Telco ChurnGuard – End-to-End MLOps Pipeline

Telco ChurnGuard is a **production-grade MLOps project** that predicts customer churn for a telecom company using machine learning.
The project demonstrates **industry-standard ML system design**, covering **data ingestion, training pipelines, experiment tracking, API serving, Dockerization, and AWS EC2 deployment**.

This repository is designed to showcase **real-world MLOps practices**, not just model training.

---

## Project Architecture Overview

The project follows a **modular, scalable, and production-oriented architecture**:

* Data is ingested and validated
* Features are transformed using reusable pipelines
* Models are trained and evaluated with experiment tracking
* Artifacts are versioned and stored
* Predictions are served via a FastAPI REST service
* The entire system is containerized and deployed on AWS EC2

---

## Project Structure

```
Telco-ChurnGuard-MLops/
├── app/                    # FastAPI application
│   ├── main.py             # API entry point
│   └── schemas.py          # Request/response validation
├── artifacts/              # Trained models and preprocessors
├── data/                   # Raw dataset
│   └── Telco-Customer-Churn.csv
├── logs/                   # Pipeline execution logs
├── mlflow/                 # MLflow experiment tracking
├── notebooks/              # EDA and experimentation
│   └── eda.ipynb
├── src/
│   ├── components/         # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/           # End-to-end pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils.py            # Common utilities
├── config.yaml             # Centralized configuration
├── requirements.txt        # Dependencies
├── setup.py                # Local package installation
├── Dockerfile              # Docker build file
└── README.md               # Documentation
```

---

## Key Features

### Data & ML Pipeline

* Automated data ingestion and train-test splitting
* Feature engineering and preprocessing pipelines
* Support for multiple models:

  * Random Forest
  * XGBoost
  * LightGBM
* Model evaluation with standard classification metrics

### Experiment Tracking

* MLflow integration for:

  * Parameters
  * Metrics
  * Artifacts
  * Model versions

### API & Serving

* FastAPI-based REST service
* Input validation using Pydantic
* Single and batch prediction support
* Interactive Swagger documentation

### Deployment

* Fully Dockerized application
* Deployed on AWS EC2
* Production-ready server configuration using Uvicorn

---

## Installation (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/yasirwali1052/Telco-ChurnGuard-MLops.git
cd Telco-ChurnGuard-MLops
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Project as Package

```bash
pip install -e .
```

---

## Configuration

All configurations are centralized in `config.yaml`, including:

* Data paths
* Model hyperparameters
* MLflow tracking URI
* API host and port

This allows **easy environment switching** without code changes.

---

## Running the Training Pipeline

```bash
python src/pipeline/training_pipeline.py
```

This executes the complete ML lifecycle:

1. Data ingestion from CSV
2. Feature preprocessing
3. Model training
4. Model evaluation
5. Artifact saving
6. MLflow logging

Trained models and preprocessors are stored in the `artifacts/` directory.

---

## Running the FastAPI Application (Local)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access:

* API Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`

---

## API Endpoints

| Method | Endpoint         | Description                |
| ------ | ---------------- | -------------------------- |
| GET    | `/`              | API information            |
| GET    | `/health`        | Service health check       |
| POST   | `/predict`       | Single customer prediction |
| POST   | `/batch_predict` | Batch predictions          |
| GET    | `/docs`          | Swagger UI                 |

---

## Sample Prediction Request

```python
import requests

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

---

## Dockerization

### Build Docker Image

```bash
docker build -t telco-churnguard-api.
```

### Run Docker Container

```bash
docker run -p 8000:8000 telco-churnguard-api
```

---

## MLflow Experiment Tracking

Start MLflow UI:

```bash
mlflow ui --backend-store-uri file:./mlflow
```

Open:

```
http://localhost:5000
```

Track:

* Experiments
* Metrics
* Models
* Artifacts

---

## AWS EC2 Deployment (Production)

### Step 1: Launch EC2 Instance

* Instance type: `t2.micro`
* OS: Ubuntu 22.04
* Open inbound ports:

  * 22 (SSH)
  * 8000 (API)

### Step 2: Connect to EC2

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Step 3: Install Docker

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Logout and reconnect.

---

### Step 4: Pull Image from Docker Hub

```bash
docker pull yasirkhan1052>/telco-churnguard-api
```

---

### Step 5: Run Container on EC2

```bash
docker run -d -p 8000:8000 yasirkhan1052/telco-churnguard-api
```

---

### Step 6: Access API

```
http://<EC2_PUBLIC_IP>:8000/docs
```

---

## Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

---

## Technologies Used

* Python 3.10+
* Scikit-learn
* XGBoost / LightGBM
* FastAPI
* MLflow
* Pandas & NumPy
* Docker
* AWS EC2

