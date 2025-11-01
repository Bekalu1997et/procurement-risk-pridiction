"""Explainability-focused API route for surfacing SHAP insights."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src import explainability, model_pipeline


class ExplainabilityPayload(BaseModel):
    """Mirror of the feature payload used for predictions."""

    region: str
    industry: str
    contract_criticality: str
    annual_spend: float
    credit_score: int
    late_ratio: float
    dispute_rate: float
    avg_delay: float
    clause_risk_score: float


class ExplainabilityResponse(BaseModel):
    """Return value summarising SHAP contributions and LLM narrative."""

    risk_level: str
    confidence: float
    top_features: List[List[float | str]]
    shap_values: List[float]
    feature_names: List[str]
    narrative: str


router = APIRouter()


@router.post("/", response_model=ExplainabilityResponse)
async def explain_prediction(
    payload: ExplainabilityPayload,
    model_name: str = Query(default="random_forest"),
) -> ExplainabilityResponse:
    """Return the top SHAP features and the Mistral-generated narrative."""

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

    return ExplainabilityResponse(
        risk_level=explanation_obj.risk_level,
        confidence=explanation_obj.confidence,
        top_features=[[name, value] for name, value in explanation_obj.top_features],
        shap_values=explanation_obj.shap_values,
        feature_names=explanation_obj.feature_names,
        narrative=explanation_obj.narrative,
    )

