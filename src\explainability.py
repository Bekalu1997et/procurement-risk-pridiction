"""Explainability utilities blending SHAP analytics with Mistral narratives."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
import yaml

from . import auditing


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


def _load_mistral_prompt_template() -> str:
    """Read prompt template from config or fall back to default string."""

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
            template = config.get("mistral_prompt_template")
            if template:
                return template
    return (
        "You are an AI explainability assistant.\n"
        "Prediction: {risk_level} ({confidence}%).\n"
        "Top features: {top_features}.\n"
        "Explain in business language why this risk was predicted."
    )


def _call_mistral(prompt: str) -> str:
    """Call the Mistral API if credentials are present, else mock the output."""

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return (
            "[Mocked Mistral Response] Based on the highlighted drivers, the supplier "
            "shows elevated risk due to persistent payment delays and clause risk."
        )

    import requests  # Imported lazily to keep startup light-weight.

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "mistral-medium",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that writes concise business narratives.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def build_explanation(
    risk_level: str,
    probabilities: Dict[str, float],
    shap_values: Sequence[float],
    feature_names: Sequence[str],
) -> Explanation:
    """Create an `Explanation` object combining SHAP and LLM narratives."""

    top_features = summarize_shap_values(shap_values, feature_names)
    confidence = round(probabilities.get(risk_level, 0.0) * 100, 2)

    prompt = _load_mistral_prompt_template().format(
        risk_level=risk_level,
        confidence=confidence,
        top_features=top_features,
    )
    narrative = _call_mistral(prompt)

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

