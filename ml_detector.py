"""ml_detector.py - Random Forest Machine Learning Intrusion Detector.

Handles training, serialization, feature alignment verification, and inference
using a scikit-learn RandomForestClassifier.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from dataset_generator import FEATURE_NAMES, extract_features

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = MODELS_DIR / "rf_model.joblib"
META_FILE = MODELS_DIR / "feature_meta.json"


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a balanced Random Forest classifier on tabular protocol features.

    Args:
        X_train: Feature matrix matching FEATURE_NAMES.
        y_train: Target labels (0: normal, 1: anomalous).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_state: Reproducibility seed.

    Returns:
        Trained RandomForestClassifier.
    """
    # Enforce strict column order
    X_train = X_train[FEATURE_NAMES]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a trained model on a held-out test set with real computed metrics.

    Args:
        clf: Trained classifier.
        X_test: Test features.
        y_test: Ground truth labels.

    Returns:
        Dictionary of real evaluation metrics.
    """
    X_test = X_test[FEATURE_NAMES]
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_model(
    clf: RandomForestClassifier,
    model_path: Path = MODEL_FILE,
    meta_path: Path = META_FILE,
) -> None:
    """Persist the trained model and canonical feature metadata to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": clf.n_estimators,
        "max_depth": clf.max_depth,
        "class_weight": "balanced",
        "random_state": clf.random_state,
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model(
    model_path: Path = MODEL_FILE,
    meta_path: Path = META_FILE,
) -> Tuple[RandomForestClassifier, List[str]]:
    """Load the trained model and verify feature ordering consistency.

    Returns:
        Tuple of (loaded_classifier, feature_names_list).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    clf: RandomForestClassifier = joblib.load(model_path)

    feature_names = FEATURE_NAMES
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            feature_names = metadata.get("feature_names", FEATURE_NAMES)

    return clf, feature_names


def predict_sequence(
    clf: RandomForestClassifier,
    events: List[str],
) -> Dict[str, Any]:
    """Perform ML inference on a single sequence of protocol events.

    Args:
        clf: Trained model.
        events: List of event strings.

    Returns:
        Dictionary with features, prediction (0 or 1), probability, and verdict.
    """
    raw_features = extract_features(events)
    # Build single-row DataFrame enforcing strict feature ordering
    df_features = pd.DataFrame([raw_features])[FEATURE_NAMES]

    pred = int(clf.predict(df_features)[0])
    probas = clf.predict_proba(df_features)[0]
    prob_normal = float(probas[0])
    prob_anomalous = float(probas[1])

    return {
        "prediction": pred,
        "is_anomalous": bool(pred == 1),
        "confidence": prob_anomalous if pred == 1 else prob_normal,
        "probability_anomalous": prob_anomalous,
        "probability_normal": prob_normal,
        "features": raw_features,
    }


def get_feature_importances(
    clf: RandomForestClassifier,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return feature importances sorted descending."""
    names = feature_names or FEATURE_NAMES
    importances = clf.feature_importances_

    df_imp = pd.DataFrame({
        "feature": names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp

