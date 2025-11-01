# Risk Prediction System Production Demo

This repository showcases a production-inspired, explainable AI platform for
supplier risk prediction. It combines synthetic data generation, classical
machine learning models (Random Forest and XGBoost), SHAP-based
explainability, Mistral narrative generation, and a pandas-first auditing
pipeline to create an end-to-end educational demo.

## Key Features

- **Synthetic Data Factory**: `data/raw/demo_pipeline.py` creates suppliers,
  transactions, and contract clauses, then seeds baseline models.
- **Modular ML Stack**: `src/model_pipeline.py` trains, persists, and serves
  Random Forest and XGBoost pipelines with configuration-driven grids.
- **Explainability Bridge**: `src/explainability.py` merges SHAP insights with
  Mistral LLM narratives for human-friendly risk storytelling.
- **Auditing & Lineage**: `src/auditing.py` centralises pandas data-quality
  checks, lineage capture, and SQLite-backed event logging.
- **Production Simulation**: `src/mlops_loop.py` and `backend/scheduler.py`
  mimic a weekly scoring cycle with APScheduler, alerting, and reporting.
- **Interactive Interfaces**: FastAPI (`api/`) and Streamlit
  (`streamlit_app/ui_dashboard.py`) power API and UI demos.
- **Notebooks**: Step-by-step walkthroughs for preprocessing, modelling,
  explainability, NLP, MLOps simulation, and auditing.

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate  # PowerShell
pip install -r requirements.txt

# Generate synthetic data and baseline models
python data/raw/demo_pipeline.py

# Run the weekly MLOps simulation once
python src/mlops_loop.py

# Launch FastAPI
uvicorn api.app:app --reload

# Launch Streamlit dashboard
streamlit run streamlit_app/ui_dashboard.py
```

## Project Structure

```
├── api/                  # FastAPI app and routes
├── backend/              # Scheduler and alerting simulation
├── config/               # YAML configuration files
├── data/                 # Raw scripts and processed datasets
├── notebooks/            # Educational notebooks (preprocessing → auditing)
├── reports/              # Generated plots, weekly reports, audit logs
├── src/                  # Core data, feature, model, explainability code
├── streamlit_app/        # Streamlit dashboard entry point
├── tests/                # Pytest suite covering key components
└── requirements.txt
```

## Execution Flow

1. **Data Generation** – `python data/raw/demo_pipeline.py`
2. **Notebook Walkthrough** – `notebooks/01_data_preprocessing.ipynb` →
   `06_auditing_pipeline.ipynb`
3. **Explainability** – `03_explainability_shap.ipynb` and Streamlit tab
4. **Weekly MLOps Simulation** – `python src/mlops_loop.py`
5. **Dashboards** – `streamlit run streamlit_app/ui_dashboard.py`
6. **API Demo** – `uvicorn api.app:app --reload`

## Testing

Run the unit tests to validate core data pipelines and auditing logic:

```bash
pytest
```

## Configuration Highlights

- `config/model_config.yaml` – Hyperparameters and prompt template.
- `config/auditing_config.yaml` – Data-quality thresholds and auditing rules.
- `config/data_refresh.yaml` – Weekly schedule definition.

## License

This demo is intended for educational and hackathon purposes. Adapt as needed
for your organisation's internal workshops and showcases.

