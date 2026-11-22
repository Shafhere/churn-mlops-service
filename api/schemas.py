
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal[
    "Yes", "No", "No internet service"
]


class CustomerInput(BaseModel):
    """The 19 input features expected by our trained model."""

    # Reject unexpected columns such as customerID or Churn.
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)

    PhoneService: YesNo
    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]
    InternetService: Literal["DSL", "Fiber optic", "No"]

    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0, le=1000)

    # Null is allowed: our fitted pipeline can impute it.
    TotalCharges: float | None = Field(
        ..., ge=0, le=100000
    )


class BatchInput(BaseModel):
    """Allow between 1 and 100 customers per request."""

    model_config = ConfigDict(extra="forbid")

    records: list[CustomerInput] = Field(
        min_length=1,
        max_length=100,
    )


class Factor(BaseModel):
    feature: str
    probability_impact: float
    direction: Literal["increases risk", "decreases risk"]

class PredictionOutput(BaseModel):
    top_factors: list[Factor] = Field(default_factory=list)
    churn_probability: float = Field(ge=0, le=1)
    prediction: Literal["Yes", "No"]
    threshold: float
    model_version: str


class BatchOutput(BaseModel):
    count: int
    predictions: list[PredictionOutput]
