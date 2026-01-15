from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerData(BaseModel):
    """Schema for customer data input"""
    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "1234-ABCD",
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
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: int = Field(..., description="Prediction (0: No Churn, 1: Churn)")
    prediction_label: str = Field(..., description="Prediction label")
    churn_probability: float = Field(..., description="Probability of churn")
    no_churn_probability: float = Field(..., description="Probability of no churn")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction"""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]

