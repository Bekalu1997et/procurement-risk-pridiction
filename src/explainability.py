"""Explainability utilities blending SHAP analytics with Ollama narratives."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
import yaml
import ollama

from . import auditing
from . import nlp_layer


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.yaml")


@dataclass
class Explanation:
    """Container describing model explanations for presentation layers."""

    risk_level: str
    confidence: float
    top_features: List[Tuple[str, float]]
    shap_values: List[float]
    feature_names: List[str]
    narrative: str


def summarize_shap_values(
    shap_values: Sequence[float],
    feature_names: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top contributing SHAP features sorted by absolute impact."""
    ranked = sorted(
        zip(feature_names, shap_values), key=lambda tpl: abs(tpl[1]), reverse=True
    )
    return ranked[:top_k]


def _build_business_prompt(risk_level: str, confidence: float, top_features: List[Tuple[str, float]]) -> str:
    """Build conversational business-friendly prompt for LLM."""
    feature_explanations = []
    for feat, val in top_features[:5]:
        clean_name = feat.replace('numeric__', '').replace('categorical__', '').replace('_', ' ').title()
        impact = "increases" if val > 0 else "decreases"
        feature_explanations.append(f"- {clean_name}: {abs(val):.3f} ({impact} risk)")
    
    features_text = "\n".join(feature_explanations)
    
    return f"""You are a business risk analyst explaining supplier risk predictions to procurement managers.

Risk Assessment: {risk_level.upper()} risk with {confidence:.1f}% confidence

Key Contributing Factors:
{features_text}

Provide a 2-3 sentence conversational explanation in simple business language. 
Translate technical metrics into actionable insights. 
Avoid jargon. Be direct and professional."""


def _generate_shap_summary(risk_level: str, confidence: float, top_features: List[Tuple[str, float]]) -> str:
    """Generate SHAP-based summary as fallback when Ollama unavailable."""
    # Build human-readable summary from SHAP values
    risk_desc = {
        "high": "elevated",
        "medium": "moderate",
        "low": "minimal"
    }.get(risk_level.lower(), "moderate")
    
    # Identify top risk drivers
    increasing = []
    decreasing = []
    
    for feat, val in top_features[:3]:
        clean_name = feat.replace('numeric__', '').replace('categorical__', '').replace('_', ' ').lower()
        if val > 0:
            increasing.append(f"{clean_name} (+{abs(val):.3f})")
        else:
            decreasing.append(f"{clean_name} (-{abs(val):.3f})")
    
    summary = f"This supplier presents {risk_desc} risk with {confidence:.1f}% confidence. "
    
    if increasing:
        summary += f"Primary concerns: {', '.join(increasing)}. "
    
    if decreasing:
        summary += f"Positive factors: {', '.join(decreasing)}. "
    
    # Add recommendation
    if risk_level.lower() == "high":
        summary += "Recommendation: Implement enhanced monitoring, request additional documentation, and consider alternative suppliers."
    elif risk_level.lower() == "medium":
        summary += "Recommendation: Monitor performance closely and establish clear KPIs with regular reviews."
    else:
        summary += "Recommendation: Maintain standard monitoring procedures with periodic reviews."
    
    return summary


def _call_ollama(prompt: str, risk_level: str, confidence: float, top_features: List[Tuple[str, float]]) -> str:
    """Call Ollama tinyllama for conversational business explanations with SHAP fallback."""
    try:
        response = ollama.generate(
            model="tinyllama",
            prompt=prompt,
            stream=False,
            options={"temperature": 0.7}
        )
        result = response["response"].strip()
        
        # If response is too short or generic, use SHAP summary
        if len(result) < 50:
            return _generate_shap_summary(risk_level, confidence, top_features)
        
        return result
        
    except Exception as e:
        # Fallback to SHAP-based summary
        return _generate_shap_summary(risk_level, confidence, top_features)


def build_explanation(
    risk_level: str,
    probabilities: Dict[str, float],
    shap_values: Sequence[float],
    feature_names: Sequence[str],
) -> Explanation:
    """Create an Explanation object combining SHAP and LLM narratives."""
    top_features = nlp_layer.summarize_shap_values(shap_values, feature_names)
    confidence = round(probabilities.get(risk_level, 0.0) * 100, 2)

    prompt = _build_business_prompt(risk_level, confidence, top_features)
    narrative = _call_ollama(prompt, risk_level, confidence, top_features)

    auditing.persist_audit_log(
        event_type="explainability_narrative",
        payload={
            "risk_level": risk_level,
            "confidence": confidence,
            "top_features": top_features,
            "prompt_length": len(prompt),
        },
    )

    return Explanation(
        risk_level=risk_level,
        confidence=confidence,
        top_features=top_features,
        shap_values=list(shap_values),
        feature_names=list(feature_names),
        narrative=narrative,
    )
