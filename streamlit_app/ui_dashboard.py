"""Streamlit dashboard for the supplier risk demo platform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import auditing, data_pipeline, explainability, model_pipeline, recommendation, visualization


BASE_DIR = Path(__file__).resolve().parents[1]


def load_latest_weekly_report() -> pd.DataFrame:
    """Load the most recent weekly predictions CSV."""

    reports_dir = BASE_DIR / "reports" / "weekly_reports"
    csv_files = sorted(reports_dir.glob("weekly_predictions_*.csv"))
    if not csv_files:
        st.info("Run the MLOps loop to generate weekly reports.")
        return pd.DataFrame()
    latest = csv_files[-1]
    st.success(f"Loaded report: {latest.name}")
    return pd.read_csv(latest)


def render_instant_prediction_tab() -> None:
    """UI elements for immediate supplier scoring."""

    st.header("Instant Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        region = col1.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "LATAM"])
        industry = col1.selectbox("Industry", ["Manufacturing", "Logistics", "IT Services", "Facilities"])
        contract_criticality = col1.selectbox("Contract Criticality", ["High", "Medium", "Low"])
        annual_spend = col2.number_input("Annual Spend", min_value=1000.0, value=50000.0, step=1000.0)
        credit_score = col2.number_input("Credit Score", min_value=300, max_value=900, value=700)
        late_ratio = col1.slider("Late Transaction Ratio", 0.0, 1.0, 0.2, 0.01)
        dispute_rate = col2.slider("Dispute Rate", 0.0, 1.0, 0.05, 0.01)
        avg_delay = col1.number_input("Average Delay (days)", min_value=0.0, value=5.0)
        clause_risk_score = col2.slider("Clause Risk Score", 0.0, 100.0, 35.0, 1.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "region": region,
            "industry": industry,
            "contract_criticality": contract_criticality,
            "annual_spend": annual_spend,
            "credit_score": credit_score,
            "late_ratio": late_ratio,
            "dispute_rate": dispute_rate,
            "avg_delay": avg_delay,
            "clause_risk_score": clause_risk_score,
        }
        result = model_pipeline.predict_single("random_forest", payload)
        explanation_obj = explainability.build_explanation(
            risk_level=result["prediction"],
            probabilities=result["probabilities"],
            shap_values=result["shap_values"],
            feature_names=result["feature_names"],
        )
        st.metric("Predicted Risk", explanation_obj.risk_level, delta=f"Confidence {explanation_obj.confidence}%")
        st.write(explanation_obj.narrative)

        visualization_path = visualization.plot_shap_summary(
            explanation_obj.shap_values,
            explanation_obj.feature_names,
            output_name="instant_prediction",
        )
        st.image(str(visualization_path), caption="SHAP Feature Impact")

        recos = recommendation.build_recommendations(
            risk_level=explanation_obj.risk_level,
            top_features=explanation_obj.top_features,
        )
        st.subheader("Recommended Actions")
        for reco in recos:
            st.write(f"- {reco}")


def render_weekly_reports_tab() -> None:
    """Visualise batch scoring outputs and audit metrics."""

    st.header("Weekly Reports")
    report_df = load_latest_weekly_report()
    if report_df.empty:
        return

    st.dataframe(report_df.head())
    pairplot_path = visualization.create_pairplot(report_df, output_name="weekly_report")
    st.image(str(pairplot_path))


def render_explainability_chat_tab() -> None:
    """Provide a lightweight LLM interface for supplier-related questions."""

    st.header("Explainability Chat")
    question = st.text_input("Ask about a supplier's risk profile")
    context = st.text_area("Context", value="Payment delays averaged 15 days with 12% dispute rate last quarter.")
    if st.button("Ask"):
        prompt = (
            "You are an AI explainability assistant. Given the context below, answer the question in a "
            "business-friendly tone.\nContext: {context}\nQuestion: {question}"
        ).format(context=context, question=question)
        answer = explainability._call_mistral(prompt)  # type: ignore[attr-defined]
        st.write(answer)


def render_audit_monitor_tab() -> None:
    """Display recent audit trail events."""

    st.header("Audit Monitor")
    events = auditing.db_connector.fetch_audit_trail(limit=100)
    st.dataframe(events)


def main() -> None:
    """Entrypoint for the Streamlit dashboard."""

    st.set_page_config(page_title="Supplier Risk Demo", layout="wide")
    tabs = st.tabs(["Instant Prediction", "Weekly Reports", "Explainability Chat", "Audit Monitor"])
    with tabs[0]:
        render_instant_prediction_tab()
    with tabs[1]:
        render_weekly_reports_tab()
    with tabs[2]:
        render_explainability_chat_tab()
    with tabs[3]:
        render_audit_monitor_tab()


if __name__ == "__main__":
    main()

