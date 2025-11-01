"""Generate actionable recommendations based on risk predictions and SHAP."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def build_recommendations(risk_level: str, top_features: Iterable[Tuple[str, float]]) -> List[str]:
    """Create human-readable recommendations tailored to the risk profile."""

    suggestions: List[str] = []
    if risk_level == "high":
        suggestions.append("Escalate supplier to compliance review board within 48 hours.")
        suggestions.append("Initiate payment term renegotiation and request collateral.")
    elif risk_level == "medium":
        suggestions.append("Schedule quarterly business review to monitor remediation progress.")
    else:
        suggestions.append("Maintain standard monitoring cadence.")

    for feature, value in top_features:
        if "late" in feature and value > 0:
            suggestions.append("Improve invoice approval SLAs to reduce late payment exposure.")
        if "dispute" in feature and value > 0:
            suggestions.append("Launch dispute root-cause workshop with procurement operations.")
        if "clause_risk_score" in feature and value > 0:
            suggestions.append("Engage legal to tighten high-risk contract clauses.")

    return suggestions[:5]

