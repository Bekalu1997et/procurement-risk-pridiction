"""Prediction endpoint returning risk scores, SHAP insights, and narratives."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src import explainability, model_pipeline, recommendation


class SupplierPayload(BaseModel):
    """Schema describing the minimal features required for inference."""

    supplier_id: Optional[int] = Field(default=None, description="Optional supplier identifier")
    region: str
    industry: str
    contract_criticality: str
    annual_spend: float
    credit_score: int
    late_ratio: float
    dispute_rate: float
    avg_delay: float
    clause_risk_score: float


class PredictionResponse(BaseModel):
    """Response payload containing predictions and explanations."""

    supplier_id: Optional[int]
    model: str
    prediction: str
    probabilities: Dict[str, float]
    top_features: List[List[float | str]]
    shap_values: List[float]
    feature_names: List[str]
    narrative: str
    recommendations: List[str]


router = APIRouter()


@router.post("/", response_model=PredictionResponse)
async def predict_supplier(
    payload: SupplierPayload,
    model_name: str = Query(default="random_forest", description="Which trained model to use"),
) -> PredictionResponse:
    """Score a supplier record and return explainability insights."""

    try:
        result = model_pipeline.predict_single(model_name=model_name, features=payload.dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    explanation_obj = explainability.build_explanation(
        risk_level=result["prediction"],
        probabilities=result["probabilities"],
        shap_values=result["shap_values"],
        feature_names=result["feature_names"],
    )

    reco = recommendation.build_recommendations(
        risk_level=explanation_obj.risk_level,
        top_features=explanation_obj.top_features,
    )

    return PredictionResponse(
        supplier_id=payload.supplier_id,
        model=model_name,
        prediction=explanation_obj.risk_level,
        probabilities=result["probabilities"],
        top_features=[[name, value] for name, value in explanation_obj.top_features],
        shap_values=explanation_obj.shap_values,
        feature_names=explanation_obj.feature_names,
        narrative=explanation_obj.narrative,
        recommendations=reco,
    )

