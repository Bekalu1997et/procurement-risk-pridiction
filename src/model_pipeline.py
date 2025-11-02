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
else:
    # Defensive runtime check: some shap internals lazily import transformers which
    # can raise environment-specific errors (keras/tf mismatches). If importing
    # transformers fails here, disable SHAP usage so prediction remains robust.
    try:
        import importlib
        importlib.import_module("transformers")
    except Exception:
        SHAP_AVAILABLE = False
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import auditing


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"


@dataclass
class ModelArtifacts:
    """Captures model objects and metadata after training.
    The ModelArtifacts class is a dataclass that captures the model objects and metadata after training.
    It uses the Pipeline object to store the model.
    It uses the Dict[str, Any] object to store the report.
    It also logs the training summary using the auditing module.
    """

    model_name: str
    pipeline: Pipeline
    report: Dict[str, Any]


def _load_config() -> Dict[str, Any]:
    """Load hyperparameter configuration for models from a YAML file if present, otherwise use defaults.
    The function loads the hyperparameter configuration for models from a YAML file if present, otherwise use defaults.
    It uses the yaml library to load the configuration.
    It also logs the hyperparameter configuration using the auditing module.
    """

    default_config = {
        "random_forest": {
            "n_estimators": [200, 300],
            "max_depth": [None, 10],
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
    """
    Build a ColumnTransformer for numeric scaling and one-hot encoding (OHE) of categorical features.

    Parameters
    ----------
    config : Dict[str, Any]
        The model configuration dictionary including feature lists.

    Returns
    -------
    ColumnTransformer
        A fitted/unfitted sklearn ColumnTransformer preprocessing pipeline.
    """
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
    """
    Perform grid search to identify optimal model hyperparameters.

    Parameters
    ----------
    estimator : Pipeline
        The sklearn Pipeline to optimize.
    params : Dict[str, List[Any]]
        Dictionary of parameter grids to search over.
    X_train : pd.DataFrame
        Training data features.
    y_train : pd.Series
        Training data targets.

    Returns
    -------
    Pipeline
        Best estimator/pipeline identified by grid search.
    """
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
    """
    Train a single model pipeline with hyperparameter optimization and generate evaluation report.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g., 'random_forest').
    estimator : Pipeline
        The sklearn Pipeline to be trained and optimized.
    param_grid : Dict[str, Any]
        Hyperparameter grid for search.
    X_train : pd.DataFrame
        Training data features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test data features.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    ModelArtifacts
        Object containing the trained pipeline, evaluation report, and identifier.
    """
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
    """
    Train and save both RandomForest and optionally XGBoost models on supplied data.

    Models and configuration-driven hyperparameter grids are used. Collects ModelArtifacts output
    (containing pipeline and evaluation report) for each trained model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix for training.
    y : pd.Series
        Target labels.

    Returns
    -------
    List[ModelArtifacts]
        List of ModelArtifacts for each trained model.
    """

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = MODELS_DIR / "latest_training_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump({artifact.model_name: artifact.report for artifact in artifacts}, stream, indent=2)

    return artifacts

def load_model(model_name: str = "random_forest") -> Pipeline:
    """
    Load a trained model pipeline from disk by model name.

    Parameters
    ----------
    model_name : str, default="random_forest"
        The name of the model pipeline to load.

    Returns
    -------
    Pipeline
        Sklearn Pipeline object that was persisted to disk.

    Raises
    ------
    FileNotFoundError
        If the saved model .joblib file does not exist.
    """
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model {model_name} missing. Please run training notebooks or `demo_pipeline`."
        )
    return joblib.load(path)


def _compute_shap_values(pipeline: Pipeline, data: pd.DataFrame) -> Tuple[np.ndarray | List[np.ndarray], List[str]]:
    """
    Compute SHAP values for the given pipeline and data using TreeExplainer.

    Parameters
    ----------
    pipeline : Pipeline
        The trained pipeline (must end with a tree-based classifier).
    data : pd.DataFrame
        The data for which to compute SHAP values (raw input features).

    Returns
    -------
    Tuple[np.ndarray | List[np.ndarray], List[str]]
        Tuple of (SHAP values per class or sample, feature names as used internally).
        If SHAP is not available, returns arrays of zeros.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out().tolist()

    if not SHAP_AVAILABLE:
        classes = getattr(model, "classes_", ["output"])
        zeros = [np.zeros((transformed.shape[0], len(feature_names))) for _ in classes]
        return zeros, feature_names
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        return shap_values, feature_names
    except Exception:
        # If SHAP fails at runtime (optional deps, TF/Keras version issues, etc.),
        # fall back to zero arrays so inference still works.
        classes = getattr(model, "classes_", ["output"])
        zeros = [np.zeros((transformed.shape[0], len(feature_names))) for _ in classes]
        return zeros, feature_names


def predict_single(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a single prediction with explanation for one supplier instance.

    Loads the specified model, preprocesses the features, predicts the risk label,
    outputs probability vector and the SHAP feature importance for the predicted class.

    Parameters
    ----------
    model_name : str
        Name of the trained model to use for prediction.
    features : Dict[str, Any]
        Dictionary of features for a single sample (supplier).

    Returns
    -------
    Dict[str, Any]
        Dictionary with prediction, probabilities, SHAP explanations, and top features.
    """
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

    # Normalize SHAP output into a 1-D array of length == number of features.
    # SHAP can return either a list (one array per class) or an ndarray with
    # different axis orders depending on the model and SHAP version. Handle
    # common layouts robustly so downstream code can assume a 1-D vector.
    if isinstance(shap_values, list):
        # list[class][sample, feature]
        shap_for_prediction = np.asarray(shap_values[shap_index][0])
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            s0, s1, s2 = shap_values.shape
            # common: (n_samples, n_features, n_classes)
            if s0 == 1 and s2 > 1:
                shap_for_prediction = shap_values[0, :, shap_index]
            # alternative: (n_classes, n_samples, n_features)
            elif s0 == len(getattr(pipeline, "classes_", [])):
                shap_for_prediction = shap_values[shap_index, 0, :]
            else:
                # best-effort fallback
                shap_for_prediction = np.ravel(shap_values)[: len(feature_names)]
        else:
            # (n_samples, n_features) or similar
            shap_for_prediction = shap_values[0]
    else:
        shap_for_prediction = np.asarray(shap_values)

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
    """
    Perform batch prediction and explanation on a DataFrame of supplier features.

    Parameters
    ----------
    model_name : str
        Name of the trained model to use for prediction.
    data : pd.DataFrame
        DataFrame containing features for multiple suppliers.

    Returns
    -------
    pd.DataFrame
        Input data plus columns for prediction, risk label, probability of "high" risk,
        and SHAP contribution sum. Also logs predictions.
    """
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

