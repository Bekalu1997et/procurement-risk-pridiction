"""Streamlit dashboard for the supplier risk demo platform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src import auditing, data_pipeline, explainability, model_pipeline, recommendation, visualization
from backend import visualization_engine, explainability_viz


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

        col1, col2 = st.columns(2)
        with col1:
            viz_path = explainability_viz.plot_feature_importance(
                explanation_obj.feature_names,
                explanation_obj.shap_values,
                output_name="instant_feature_importance",
            )
            st.image(str(viz_path), caption="Feature Importance")
        with col2:
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
    
    col1, col2 = st.columns(2)
    with col1:
        pairplot_path = visualization.create_pairplot(report_df, output_name="weekly_report")
        st.image(str(pairplot_path), caption="Pairplot")
    
    with col2:
        if "prediction_date" in report_df.columns and "risk_score" in report_df.columns:
            risk_timeline_path = explainability_viz.plot_risk_score_timeline(
                report_df,
                date_col="prediction_date",
                risk_col="risk_score",
                output_name="weekly_risk_timeline",
            )
            st.image(str(risk_timeline_path), caption="Risk Score Timeline")


def render_visualization_tab() -> None:
    """Interactive visualization dashboard with multiple chart types."""

    st.header("Advanced Visualizations")
    report_df = load_latest_weekly_report()
    if report_df.empty:
        return

    numeric_cols = report_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = report_df.select_dtypes(include=["object"]).columns.tolist()
    all_cols = report_df.columns.tolist()

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Pairplot", "Scatter Plot", "Heatmap", "Histogram", "Bar Chart", "Line Plot", "Box Plot", "Violin Plot", "Count Plot"],
    )

    if chart_type == "Pairplot":
        hue = st.selectbox("Hue (optional)", [None] + categorical_cols)
        if st.button("Generate Pairplot"):
            path = visualization_engine.pairplot(report_df, hue=hue, output_name="custom_pairplot")
            st.image(str(path))

    elif chart_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        x = col1.selectbox("X-axis", numeric_cols)
        y = col2.selectbox("Y-axis", numeric_cols)
        hue = st.selectbox("Hue (optional)", [None] + categorical_cols)
        if st.button("Generate Scatter"):
            path = visualization_engine.scatter(report_df, x=x, y=y, hue=hue, output_name="custom_scatter")
            st.image(str(path))

    elif chart_type == "Heatmap":
        if st.button("Generate Heatmap"):
            path = visualization_engine.heatmap(report_df, output_name="custom_heatmap")
            st.image(str(path))

    elif chart_type == "Histogram":
        column = st.selectbox("Column", numeric_cols)
        bins = st.slider("Bins", 10, 100, 30)
        if st.button("Generate Histogram"):
            path = visualization_engine.histogram(report_df, column=column, bins=bins, output_name="custom_histogram")
            st.image(str(path))

    elif chart_type == "Bar Chart":
        col1, col2 = st.columns(2)
        x = col1.selectbox("X-axis (categorical)", categorical_cols if categorical_cols else all_cols)
        y = col2.selectbox("Y-axis (numeric)", numeric_cols)
        if st.button("Generate Bar Chart"):
            path = visualization_engine.bar_chart(report_df, x=x, y=y, output_name="custom_bar")
            st.image(str(path))

    elif chart_type == "Line Plot":
        col1, col2 = st.columns(2)
        x = col1.selectbox("X-axis", all_cols)
        y = col2.selectbox("Y-axis", numeric_cols)
        hue = st.selectbox("Hue (optional)", [None] + categorical_cols)
        if st.button("Generate Line Plot"):
            path = visualization_engine.line_plot(report_df, x=x, y=y, hue=hue, output_name="custom_line")
            st.image(str(path))

    elif chart_type == "Box Plot":
        col1, col2 = st.columns(2)
        x = col1.selectbox("X-axis (categorical)", categorical_cols if categorical_cols else all_cols)
        y = col2.selectbox("Y-axis (numeric)", numeric_cols)
        if st.button("Generate Box Plot"):
            path = visualization_engine.box_plot(report_df, x=x, y=y, output_name="custom_box")
            st.image(str(path))

    elif chart_type == "Violin Plot":
        col1, col2 = st.columns(2)
        x = col1.selectbox("X-axis (categorical)", categorical_cols if categorical_cols else all_cols)
        y = col2.selectbox("Y-axis (numeric)", numeric_cols)
        if st.button("Generate Violin Plot"):
            path = visualization_engine.violin_plot(report_df, x=x, y=y, output_name="custom_violin")
            st.image(str(path))

    elif chart_type == "Count Plot":
        column = st.selectbox("Column", categorical_cols if categorical_cols else all_cols)
        if st.button("Generate Count Plot"):
            path = visualization_engine.count_plot(report_df, column=column, output_name="custom_count")
            st.image(str(path))


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
    tabs = st.tabs(["Instant Prediction", "Weekly Reports", "Advanced Visualizations", "Explainability Chat", "Audit Monitor"])
    with tabs[0]:
        render_instant_prediction_tab()
    with tabs[1]:
        render_weekly_reports_tab()
    with tabs[2]:
        render_visualization_tab()
    with tabs[3]:
        render_explainability_chat_tab()
    with tabs[4]:
        render_audit_monitor_tab()


if __name__ == "__main__":
    main()

