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

from . import auditing


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config" / "model_config.yaml"


@dataclass
class ModelArtifacts:
    """Container for trained model artifacts and evaluation metrics.
    
    This dataclass encapsulates everything needed to persist and evaluate
    a trained machine learning model in production.
    
    Attributes
    ----------
    model_name : str
        Identifier for the model (e.g., 'random_forest', 'xgboost')
    pipeline : Pipeline
        Complete sklearn Pipeline including preprocessing and model
    report : Dict[str, Any]
        Classification report with precision, recall, F1 scores per class
    
    Example
    -------
    >>> artifacts = ModelArtifacts(
    ...     model_name="random_forest",
    ...     pipeline=trained_pipeline,
    ...     report={"accuracy": 0.85, "macro avg": {"f1-score": 0.83}}
    ... )
    """

    model_name: str
    pipeline: Pipeline
    report: Dict[str, Any]


def _load_config() -> Dict[str, Any]:
    """Load model hyperparameter configuration from YAML or use defaults.
    
    Reads config/model_config.yaml for hyperparameter grids and feature lists.
    Falls back to sensible defaults if file doesn't exist.
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with keys:
        - 'random_forest': Hyperparameter grid for RandomForest
        - 'categorical_features': List of categorical column names
        - 'numeric_features': List of numeric column names
    
    Example
    -------
    >>> config = _load_config()
    >>> config['random_forest']
    {'n_estimators': [200, 300], 'max_depth': [None, 10]}
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
    """Build preprocessing pipeline for categorical and numeric features.
    
    Creates a ColumnTransformer that:
    - One-hot encodes categorical features (region, industry, criticality)
    - Standardizes numeric features (credit_score, late_ratio, etc.)
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration with 'categorical_features' and 'numeric_features' keys

    Returns
    -------
    ColumnTransformer
        Sklearn preprocessing pipeline ready for fitting
    
    Example
    -------
    >>> config = {'categorical_features': ['region'], 'numeric_features': ['credit_score']}
    >>> preprocessor = _build_preprocessor(config)
    >>> preprocessor.fit_transform(df)
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
    """Optimize model hyperparameters using grid search with cross-validation.
    
    Performs 3-fold cross-validation grid search to find best hyperparameters.
    Uses macro F1 score as optimization metric (balanced for multi-class).
    Runs in parallel using all available CPU cores.

    Parameters
    ----------
    estimator : Pipeline
        Sklearn Pipeline with 'model' step to optimize
    params : Dict[str, List[Any]]
        Hyperparameter grid, e.g., {'n_estimators': [100, 200]}
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels (risk_label: low/medium/high)

    Returns
    -------
    Pipeline
        Best pipeline found by grid search
    
    Example
    -------
    >>> params = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    >>> best_model = _grid_search(pipeline, params, X_train, y_train)
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
    """Train machine learning models with hyperparameter optimization.
    
    Complete training workflow:
    1. Load configuration (hyperparameters, features)
    2. Build preprocessing pipeline (scaling, encoding)
    3. Split data (80/20 train/test, stratified)
    4. Train Random Forest with grid search
    5. Evaluate on test set
    6. Save model to disk (models/random_forest.joblib)
    7. Log training metrics to audit trail
    8. Save training summary JSON

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with columns: region, industry, contract_criticality,
        annual_spend, credit_score, late_ratio, dispute_rate, avg_delay,
        clause_risk_score
    y : pd.Series
        Target labels: 'low', 'medium', or 'high' risk

    Returns
    -------
    List[ModelArtifacts]
        List containing trained model artifacts with evaluation metrics
    
    Example
    -------
    >>> X, y = data_pipeline.prepare_training_data()
    >>> artifacts = train_models(X, y)
    >>> for artifact in artifacts:
    ...     print(f"{artifact.model_name}: {artifact.report['accuracy']:.3f}")
    random_forest: 0.847
    
    Notes
    -----
    - Uses stratified split to maintain class balance
    - Optimizes for macro F1 score (handles class imbalance)
    - Saves models to models/ directory
    - Logs training events to audit trail
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
    """Load a trained model pipeline from disk.
    
    Loads a previously trained and saved model for inference.
    Model must have been trained using train_models() first.

    Parameters
    ----------
    model_name : str, default="random_forest"
        Name of model to load (matches filename without .joblib extension)

    Returns
    -------
    Pipeline
        Complete sklearn Pipeline with preprocessing and trained model

    Raises
    ------
    FileNotFoundError
        If models/{model_name}.joblib doesn't exist
    
    Example
    -------
    >>> pipeline = load_model("random_forest")
    >>> prediction = pipeline.predict(X_new)
    
    Notes
    -----
    Models are saved to models/ directory as .joblib files
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
    """Predict risk for a single supplier with SHAP explanations.
    
    Complete prediction workflow:
    1. Load trained model from disk
    2. Preprocess features (scaling, encoding)
    3. Generate risk prediction (low/medium/high)
    4. Calculate class probabilities
    5. Compute SHAP values for explainability
    6. Identify top 5 contributing features
    7. Log prediction to audit trail

    Parameters
    ----------
    model_name : str
        Model to use: 'random_forest' or 'xgboost'
    features : Dict[str, Any]
        Supplier features with keys:
        - region: str ("North America", "Europe", "Asia-Pacific", "LATAM")
        - industry: str ("Manufacturing", "Logistics", "IT Services", etc.)
        - contract_criticality: str ("High", "Medium", "Low")
        - annual_spend: float
        - credit_score: int (300-900)
        - late_ratio: float (0-1)
        - dispute_rate: float (0-1)
        - avg_delay: float (days)
        - clause_risk_score: float (0-100)

    Returns
    -------
    Dict[str, Any]
        Prediction results with keys:
        - 'prediction': str - Risk level ('low', 'medium', 'high')
        - 'probabilities': Dict[str, float] - Class probabilities
        - 'top_features': List[Tuple[str, float]] - Top 5 SHAP features
        - 'shap_values': List[float] - SHAP values for all features
        - 'feature_names': List[str] - Feature names after preprocessing
    
    Example
    -------
    >>> features = {
    ...     "region": "North America",
    ...     "industry": "Manufacturing",
    ...     "contract_criticality": "High",
    ...     "annual_spend": 75000.0,
    ...     "credit_score": 650,
    ...     "late_ratio": 0.15,
    ...     "dispute_rate": 0.08,
    ...     "avg_delay": 8.0,
    ...     "clause_risk_score": 45.0
    ... }
    >>> result = predict_single("random_forest", features)
    >>> print(f"Risk: {result['prediction']}")
    >>> print(f"Confidence: {result['probabilities'][result['prediction']]*100:.1f}%")
    Risk: high
    Confidence: 54.0%
    
    Notes
    -----
    - SHAP values explain which features drove the prediction
    - Positive SHAP = increases risk, Negative = decreases risk
    - All predictions logged to audit trail for compliance
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
    """Predict risk for multiple suppliers in batch mode.
    
    Efficient batch prediction for scoring many suppliers at once.
    Used for weekly risk assessments and portfolio analysis.

    Parameters
    ----------
    model_name : str
        Model to use: 'random_forest' or 'xgboost'
    data : pd.DataFrame
        DataFrame with required feature columns (see predict_single)

    Returns
    -------
    pd.DataFrame
        Original data plus new columns:
        - 'prediction': Risk level for each supplier
        - 'risk_label': Same as prediction
        - 'prob_high': Probability of high risk (0-1)
        - 'shap_contribution_sum': Total SHAP contribution
    
    Example
    -------
    >>> suppliers_df = pd.read_csv("suppliers.csv")
    >>> results = predict_batch("random_forest", suppliers_df)
    >>> high_risk = results[results['prediction'] == 'high']
    >>> print(f"High risk suppliers: {len(high_risk)}")
    
    Notes
    -----
    - Much faster than calling predict_single() in a loop
    - Logs batch prediction event to audit trail
    - Use for weekly scoring runs and portfolio monitoring
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

