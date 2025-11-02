"""Prediction endpoint returning risk scores, SHAP insights, and narratives."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src import explainability, model_pipeline, recommendation, visualization


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
    feature_trends: Optional[List[str]] = None


router = APIRouter()


@router.post("/", response_model=PredictionResponse)
async def predict_supplier(
    payload: SupplierPayload,
    model_name: str = Query(default="random_forest", description="Which trained model to use"),
    show_feature_trends: bool = Query(default=False, description="If true, return base64 PNGs of top-feature historical trends"),
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

    # Optionally produce historical trend images for top features
    feature_trends: Optional[List[str]] = None
    # Only generate plots when explicitly requested by the client
    if show_feature_trends:
        try:
            top_names = [name for name, _ in explanation_obj.top_features]
            feature_trends = visualization.plot_top_feature_trends_from_history(
                payload.dict(), top_feature_names=top_names, supplier_id=payload.supplier_id
            )
        except Exception:
            # plotting should not break prediction; in case of error, return no images
            feature_trends = None

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
        feature_trends=feature_trends,
    )

