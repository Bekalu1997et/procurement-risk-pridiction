"""Model training, persistence, and inference utilities for the demo platform."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    shap = None  # type: ignore
    SHAP_AVAILABLE = False
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:  # pragma: no cover - depends on optional dependency.
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime.
    XGBOOST_AVAILABLE = False
    XGBClassifier = None  # type: ignore

from . import auditing


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"


@dataclass
class ModelArtifacts:
    """Captures model objects and metadata after training."""

    model_name: str
    pipeline: Pipeline
    report: Dict[str, Any]


def _load_config() -> Dict[str, Any]:
    """Load hyperparameter configuration from YAML, falling back to defaults."""

    default_config = {
        "random_forest": {
            "n_estimators": [200, 300],
            "max_depth": [None, 10],
        },
        "xgboost": {
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 400],
        },
        "categorical_features": ["region", "industry", "contract_criticality"],
        "numeric_features": [
            "annual_spend",
            "credit_score",
            "late_ratio",
            "dispute_rate",
            "avg_delay",
            "clause_risk_score",
        ],
    }

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
            user_config = yaml.safe_load(stream) or {}
        return {**default_config, **user_config}
    return default_config


def _build_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """Create a preprocessing pipeline composed of numeric scaling and OHE."""

    categorical = config["categorical_features"]
    numeric = config["numeric_features"]
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )


def _grid_search(
    estimator: Pipeline,
    params: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Run GridSearchCV returning the best estimator for the pipeline."""

    grid = GridSearchCV(
        estimator,
        param_grid={f"model__{k}": v for k, v in params.items()},
        n_jobs=-1,
        cv=3,
        scoring="f1_macro",
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def _train_single_model(
    model_name: str,
    estimator: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelArtifacts:
    """Helper to train a single model and collect evaluation metadata."""

    best_estimator = _grid_search(estimator, param_grid, X_train, y_train)
    y_pred = best_estimator.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, MODELS_DIR / f"{model_name}.joblib")

    auditing.persist_audit_log(
        event_type="model_trained",
        payload={
            "model": model_name,
            "best_params": best_estimator.named_steps["model"].get_params(),
            "macro_f1": report["macro avg"]["f1-score"],
        },
    )

    return ModelArtifacts(model_name=model_name, pipeline=best_estimator, report=report)


def train_models(X: pd.DataFrame, y: pd.Series) -> List[ModelArtifacts]:
    """Train RandomForest and XGBoost models using configuration-driven grids."""

    config = _load_config()
    preprocessor = _build_preprocessor(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )
    artifacts = [
        _train_single_model(
            model_name="random_forest",
            estimator=rf_pipeline,
            param_grid=config["random_forest"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    ]

    if XGBOOST_AVAILABLE:
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        use_label_encoder=False,
                    ),
                ),
            ]
        )

        artifacts.append(
            _train_single_model(
                model_name="xgboost",
                estimator=xgb_pipeline,
                param_grid=config["xgboost"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )


    metadata_path = MODELS_DIR / "latest_training_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                artifact.model_name: artifact.report for artifact in artifacts
            },
            stream,
            indent=2,
        )

    return artifacts


def load_model(model_name: str = "random_forest") -> Pipeline:
    """Load a persisted model pipeline from disk."""

    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model {model_name} missing. Please run training notebooks or `demo_pipeline`."
        )
    return joblib.load(path)


def _compute_shap_values(pipeline: Pipeline, data: pd.DataFrame) -> Tuple[np.ndarray | List[np.ndarray], List[str]]:
    """Compute SHAP values using the tree explainer on the trained model."""

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out().tolist()

    if not SHAP_AVAILABLE:
        classes = getattr(model, "classes_", ["output"])
        zeros = [np.zeros((transformed.shape[0], len(feature_names))) for _ in classes]
        return zeros, feature_names

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    return shap_values, feature_names


def predict_single(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict risk for a single supplier payload with SHAP explanations."""

    pipeline = load_model(model_name)
    df = pd.DataFrame([features])
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    required_columns: List[str] = []
    for _, _, cols in preprocessor.transformers:
        if isinstance(cols, list):
            required_columns.extend(cols)
    df = df[required_columns]
    prediction = pipeline.predict(df)[0]
    probabilities = pipeline.predict_proba(df)[0]
    class_indices = {label: idx for idx, label in enumerate(pipeline.classes_)}

    shap_values, feature_names = _compute_shap_values(pipeline, df)
    shap_index = class_indices.get(prediction, 0)
    shap_for_prediction = (
        shap_values[shap_index][0] if isinstance(shap_values, list) else shap_values[0]
    )

    top_features = sorted(
        zip(feature_names, shap_for_prediction), key=lambda tpl: abs(tpl[1]), reverse=True
    )[:5]

    result = {
        "prediction": str(prediction),
        "probabilities": dict(zip(pipeline.classes_, probabilities)),
        "top_features": top_features,
        "shap_values": shap_for_prediction.tolist(),
        "feature_names": feature_names,
    }

    auditing.persist_audit_log(
        event_type="predict_single",
        payload={
            "model": model_name,
            "prediction": result["prediction"],
            "prob_high": result["probabilities"].get("high"),
        },
    )
    return result


def predict_batch(model_name: str, data: pd.DataFrame) -> pd.DataFrame:
    """Run batch inference returning predictions with SHAP contribution sums."""

    pipeline = load_model(model_name)
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    required_columns: List[str] = []
    for _, _, cols in preprocessor.transformers:
        if isinstance(cols, list):
            required_columns.extend(cols)

    data_subset = data[required_columns]

    predictions = pipeline.predict(data_subset)
    probabilities = pipeline.predict_proba(data_subset)
    class_indices = {label: idx for idx, label in enumerate(pipeline.classes_)}
    shap_values, _ = _compute_shap_values(pipeline, data_subset)

    if isinstance(shap_values, list):
        shap_contrib = np.array(shap_values).sum(axis=0)
    else:
        shap_contrib = shap_values

    result = data_subset.copy()
    result["prediction"] = predictions
    result["risk_label"] = predictions
    prob_index = class_indices.get("high")
    if prob_index is not None:
        result["prob_high"] = probabilities[:, prob_index]
    else:
        result["prob_high"] = probabilities.max(axis=1)
    result["shap_contribution_sum"] = shap_contrib.sum(axis=1)

    auditing.log_batch_predictions(
        predictions=result,
        context={"model": model_name, "rows": int(result.shape[0])},
    )
    return result

