"""Natural language processing helpers for contract analytics and Q&A."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from transformers import pipeline

from . import auditing


def summarize_shap_values(shap_values, feature_names, top_k: int = 5):
    """Compatibility helper: return top-k features by absolute SHAP impact.

    Some modules historically called this helper from the NLP layer; providing
    a small wrapper keeps older call sites working when notebooks reload in
    interactive sessions.
    """
    ranked = sorted(zip(feature_names, shap_values), key=lambda tpl: abs(tpl[1]), reverse=True)
    return ranked[:top_k]


@dataclass
class NLPSummary:
    """Structured summary of contract insights for dashboards.
    The NLPSummary class is a dataclass that captures the structured summary of contract insights for dashboards.
    It uses the List[str] object to store the key risks.
    It uses the List[str] object to store the mitigation actions.
    It uses the str object to store the raw summary.
    It also logs the contract summary using the auditing module.
    """

    key_risks: List[str]
    mitigation_actions: List[str]
    raw_summary: str


def summarize_contract(contract_text: str) -> NLPSummary:
    """Use a transformer summarizer to condense contract risk signals.
    The function uses a transformer summarizer to condense the contract text.
    It uses the pipeline function from the transformers library to summarise the contract text.
    It also logs the contract summary using the auditing module.
    """

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(
            contract_text, max_length=80, min_length=30, do_sample=False
        )[0]["summary_text"]
    except Exception:  # pragma: no cover - external model dependency.
        summary = (
            "Summary unavailable due to offline model cache. Flagging key compliance and "
            "payment clauses for manual review."
        )

    key_risks = [clause.strip() for clause in summary.split(".") if clause]
    mitigation_actions = [f"Review clause: {risk}" for risk in key_risks[:3]]

    auditing.persist_audit_log(
        event_type="contract_summary",
        payload={
            "risk_count": len(key_risks),
            "text_length": len(contract_text),
        },
    )

    return NLPSummary(key_risks=key_risks, mitigation_actions=mitigation_actions, raw_summary=summary)


def contract_qna(question: str, context: str) -> str:
    """Run a question-answering model to extract answers from contract text.
    The function runs a question-answering model to extract answers from the contract text.
    It uses the pipeline function from the transformers library to answer the question.
    It also logs the contract question and answer using the auditing module.
    """

    try:
        qna_pipeline = pipeline(
            "question-answering", model="distilbert-base-cased-distilled-squad"
        )
        result = qna_pipeline(question=question, context=context)
        answer = result["answer"]
        score = float(result.get("score", 0.0))
    except Exception:  # pragma: no cover - handles offline scenarios.
        answer = "Unable to process question with offline models."
        score = 0.0

    auditing.persist_audit_log(
        event_type="contract_qna",
        payload={
            "question": question,
            "answer": answer,
            "score": score,
        },
    )
    return answer

