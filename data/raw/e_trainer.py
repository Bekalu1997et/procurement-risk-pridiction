"""Model training module for supplier risk prediction system.

This module handles the training of machine learning models using
numeric and text features for risk prediction.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def train_random_forest_model(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    random_state: int,
) -> Tuple[RandomForestClassifier, TfidfVectorizer, Dict[str, object]]:
    """Train a RandomForest classifier using numeric + TF-IDF text features."""

    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    text_matrix = vectorizer.fit_transform(data["contract_text"].astype(str))
    numeric_matrix = data.loc[:, numeric_features].fillna(0).to_numpy()
    feature_matrix = hstack([numeric_matrix, text_matrix])
    labels = data["risk_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )

    model = RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_labels = model.classes_.tolist()
    y_test_bin = label_binarize(y_test, classes=class_labels)
    auc_macro = float(roc_auc_score(y_test_bin, y_proba, multi_class="ovr"))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    feature_names = numeric_features + vectorizer.get_feature_names_out().tolist()
    importances = model.feature_importances_
    top_features = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:15]
    ]

    summary: Dict[str, object] = {
        "auc_macro": auc_macro,
        "classification_report": report,
        "top_importances": top_features,
        "class_labels": class_labels,
    }
    return model, vectorizer, summary
