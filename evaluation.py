"""evaluation.py - Comparative Evaluation of FSM, ML, and Hybrid Detectors.

Compares three intrusion detection strategies on the identical held-out test set:
    1. Rule-Based FSM (Deterministic state-transition validity)
    2. ML-Based Random Forest (Statistical pattern classification)
    3. Hybrid Detector (OR-logic combining FSM and ML)
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from dataset_generator import FEATURE_NAMES, extract_features
from fsm_detector import validate_protocol_sequence
from ml_detector import predict_sequence


def compute_detector_metrics(y_true: List[int], y_pred: List[int], detector_name: str) -> Dict[str, Any]:
    """Compute standard classification metrics from ground truth and predictions.

    Args:
        y_true: Ground truth binary labels (0: normal, 1: anomalous).
        y_pred: Predicted binary labels (0: normal, 1: anomalous).
        detector_name: Human-readable name of the detector.

    Returns:
        Dictionary of computed performance metrics.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    # Handle standard 2x2 confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case handling if only one class exists
        tn = int(cm[0, 0]) if len(cm) > 0 else 0
        fp = 0
        fn = 0
        tp = 0

    return {
        "detector": detector_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def compare_all_detectors(
    clf: Any,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Evaluate FSM, ML, and Hybrid detectors on the identical held-out test set.

    Args:
        clf: Trained Random Forest model.
        df_test: Held-out test DataFrame containing 'events' and 'is_anomalous'.

    Returns:
        Tuple of (comparison_df, detailed_metrics_dict, results_df).
    """
    y_true = df_test["is_anomalous"].tolist()

    # 1. Approach A: Rule-Based FSM
    fsm_preds = []
    for events in df_test["events"]:
        fsm_res = validate_protocol_sequence(events)
        fsm_preds.append(1 if fsm_res["is_anomalous"] else 0)

    # 2. Approach B: ML Random Forest
    X_test = df_test[FEATURE_NAMES]
    ml_preds = clf.predict(X_test).tolist()

    # 3. Approach C: Hybrid Detector (FSM OR ML)
    # Hybrid Decision Rule:
    # A session is flagged as anomalous if either the FSM detects an invalid transition
    # OR the Random Forest model detects an anomaly from statistical feature patterns.
    hybrid_preds = [1 if (f == 1 or m == 1) else 0 for f, m in zip(fsm_preds, ml_preds)]

    # Compute metrics for all three detectors
    metrics_fsm = compute_detector_metrics(y_true, fsm_preds, "Rule-Based FSM")
    metrics_ml = compute_detector_metrics(y_true, ml_preds, "ML (Random Forest)")
    metrics_hybrid = compute_detector_metrics(y_true, hybrid_preds, "Hybrid (FSM OR ML)")

    summary_rows = [
        {
            "Detector": metrics_fsm["detector"],
            "Accuracy": round(metrics_fsm["accuracy"], 4),
            "Precision": round(metrics_fsm["precision"], 4),
            "Recall": round(metrics_fsm["recall"], 4),
            "F1-Score": round(metrics_fsm["f1_score"], 4),
            "TP": metrics_fsm["true_positives"],
            "FP": metrics_fsm["false_positives"],
            "TN": metrics_fsm["true_negatives"],
            "FN": metrics_fsm["false_negatives"],
        },
        {
            "Detector": metrics_ml["detector"],
            "Accuracy": round(metrics_ml["accuracy"], 4),
            "Precision": round(metrics_ml["precision"], 4),
            "Recall": round(metrics_ml["recall"], 4),
            "F1-Score": round(metrics_ml["f1_score"], 4),
            "TP": metrics_ml["true_positives"],
            "FP": metrics_ml["false_positives"],
            "TN": metrics_ml["true_negatives"],
            "FN": metrics_ml["false_negatives"],
        },
        {
            "Detector": metrics_hybrid["detector"],
            "Accuracy": round(metrics_hybrid["accuracy"], 4),
            "Precision": round(metrics_hybrid["precision"], 4),
            "Recall": round(metrics_hybrid["recall"], 4),
            "F1-Score": round(metrics_hybrid["f1_score"], 4),
            "TP": metrics_hybrid["true_positives"],
            "FP": metrics_hybrid["false_positives"],
            "TN": metrics_hybrid["true_negatives"],
            "FN": metrics_hybrid["false_negatives"],
        },
    ]

    comparison_df = pd.DataFrame(summary_rows)

    detailed_metrics = {
        "fsm": metrics_fsm,
        "ml": metrics_ml,
        "hybrid": metrics_hybrid,
    }

    # Annotate test DataFrame with predictions
    results_df = df_test.copy()
    results_df["pred_fsm"] = fsm_preds
    results_df["pred_ml"] = ml_preds
    results_df["pred_hybrid"] = hybrid_preds

    return comparison_df, detailed_metrics, results_df


def evaluate_single_sequence(events: List[str], clf: Any) -> Dict[str, Any]:
    """Run all three detection strategies on a single sequence."""
    # FSM check
    fsm_res = validate_protocol_sequence(events)
    fsm_flag = bool(fsm_res["is_anomalous"])

    # ML check
    ml_res = predict_sequence(clf, events)
    ml_flag = bool(ml_res["is_anomalous"])

    # Hybrid check (OR logic)
    hybrid_flag = bool(fsm_flag or ml_flag)

    decision_reason = []
    if fsm_flag and ml_flag:
        decision_reason.append("Both FSM detected protocol violations and ML classified features as anomalous.")
    elif fsm_flag:
        decision_reason.append("FSM identified an invalid state transition (ML model did not flag).")
    elif ml_flag:
        decision_reason.append("ML model identified statistical feature anomalies (transitions appeared syntactically valid).")
    else:
        decision_reason.append("Both detectors agreed: valid protocol sequence with normal statistical features.")

    return {
        "events": events,
        "fsm": {
            "flag": fsm_flag,
            "verdict": "ANOMALY" if fsm_flag else "NORMAL",
            "num_invalid": fsm_res["num_invalid_transitions"],
            "trace": fsm_res["trace"],
            "explanation": fsm_res["explanation"],
            "final_state": fsm_res["final_state"],
            "is_accepted": fsm_res["is_accepted"],
        },
        "ml": {
            "flag": ml_flag,
            "verdict": "ANOMALY" if ml_flag else "NORMAL",
            "confidence": ml_res["confidence"],
            "probability_anomalous": ml_res["probability_anomalous"],
            "features": ml_res["features"],
        },
        "hybrid": {
            "flag": hybrid_flag,
            "verdict": "INTRUSION ALERT" if hybrid_flag else "BENIGN SESSION",
            "decision_rule": "FSM_ALERT OR ML_ALERT",
            "reason": " ".join(decision_reason),
        },
    }

