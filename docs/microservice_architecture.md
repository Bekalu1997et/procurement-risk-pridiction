# Microservice Architecture Blueprint

This note captures how the project can be evolved from the current monolith into a
microservice-aligned deployment. The guiding principle is to isolate the
model-serving footprint (“AI service”) from the business workflow / reporting
layer (“backend system”), while keeping the CLI utilities available for quick
experimentation.

## Services

### 1. `ml-service`
- **Responsibility:** Real-time inference, SHAP explanations, LLM narratives
  (mocked until a Mistral key is provided).
- **Runtime:** FastAPI or Litestar packaged with the trained models built under
  `src/model_pipeline.py` and `data/raw/demo_pipeline.py`.
- **Endpoints:**
  - `POST /predict`: Accepts supplier payload, returns risk score + top
    features. Internally calls `model_pipeline.predict_single`.
  - `POST /explain`: Optional heavier route that streams SHAP base values or
    narratives when LLM is enabled.
- **Scaling:** Containerised separately; horizontal pods can reload the latest
  `models/` artefacts from shared storage (S3/Azure Blob/etc.). CronJobs or the
  MLOps scheduler publish new model versions by writing to that storage.

### 2. `ops-service`
- **Responsibility:** Business workflows, alert routing, report APIs, UI façade
  for Streamlit or a React dashboard.
- **Runtime:** FastAPI, Django, or Node—decoupled from the model code. Only
  communicates with `ml-service` over HTTP/gRPC.
- **Endpoints:** Alerts feed (`GET /alerts`), audit queries (`GET /audit`),
  weekly batch status.
- **Data persistence:** Uses `src/db_connector.py` for SQLite prototype; swap in
  Postgres with SQLAlchemy when productionising.

### 3. `scheduler-service`
- **Responsibility:** Executes `src/mlops_loop.py` on a cadence (weekly cron).
- **Runtime:** APScheduler worker or a serverless job (AWS Lambda / Azure Func).
- **Workflow:**
  1. Loads new weekly data from `data/processed/new_data_weekly.csv` (or data
     lake equivalent).
  2. Calls `ml-service` batch endpoint or runs `model_pipeline.predict_batch`
     locally.
  3. Pushes summary + alerts to `ops-service` (or directly to messaging buses).

## Data Contracts
- **Feature schema:** Centralised in `config/model_config.yaml` &
  `config/feature_mapping.yaml`. Both services import the same config package to
  avoid drift.
- **Inference payload:** JSON object with numeric features and `contract_text`.
  Documented under `api/routes/predict.py` via Pydantic models.
- **Audit events:** Emitted through `src/auditing.py` and stored in SQLite/
  CSV—can be redirected to Kafka or Cloud Logging without touching the model
  code.

## Command-line Workflow
For offline validation (without running any HTTP services):

```bash
python data/raw/demo_pipeline.py --suppliers 300 --transactions 600 --seed 123
python - <<'PY'
from data.raw import demo_pipeline
payload = {"annual_revenue": 100000, "avg_payment_delay_days": 75, ...}
print(demo_pipeline.predict_supplier_risk(payload))
PY
```

This mirrors what the `ml-service` will do when containerised.

## Version Management
- **Pinned dependencies:** `requirements.txt` now locks scikit-learn to `1.6.1`
  (matching the serialised models) and pins all major libraries to exact
  versions to keep Docker builds reproducible.
- **Build images:** Maintain two Dockerfiles—`Dockerfile.ml` installs only the
  inference stack, `Dockerfile.ops` installs API + scheduler deps. Both COPY the
  pinned `requirements.txt` to guarantee consistent wheels.
- **Model registry:** When retraining, increment a model version (e.g.
  `models/v1/random_forest.joblib`) and update the `feature_bundle.json` so that
  `ml-service` can hot-swap models without downtime.

## Next Steps
1. Extract the existing FastAPI routes in `api/` into a standalone `ml-service`
   package with its own `pyproject.toml`.
2. Introduce an `ops-service` module that imports only the REST contracts from
   the ML service, not the training code.
3. Add docker-compose (or Kubernetes manifests) with service-to-service
   networking and shared model volume mount.
4. Wire Mistral by setting `MISTRAL_API_KEY` in `ml-service` deployment
   environment; the fallback narrative remains the default until the key is in
   place.
