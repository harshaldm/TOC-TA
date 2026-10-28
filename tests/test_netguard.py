"""test_netguard.py - Unit test suite for NetGuard.

Tests:
    1. FSM valid transitions and accepted regular language strings
    2. FSM invalid transitions, unexpected symbols, and detailed violation reasons
    3. Empty and boundary sequence handling
    4. Dataset generation, class distributions, and reproducibility
    5. Feature extraction consistency and zero target leakage
    6. ML Random Forest training, persistence, and feature order invariance
    7. Hybrid detector decision logic (FSM ∨ ML)
    8. Metric calculation integrity and confusion matrix
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsm_detector import ProtocolFSM, validate_protocol_sequence, START_STATE, ACCEPT_STATES, ALPHABET
from dataset_generator import (
    FEATURE_NAMES,
    extract_features,
    generate_dataset,
    prepare_train_test_splits,
)
from ml_detector import (
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_sequence,
    get_feature_importances,
)
from evaluation import compute_detector_metrics, compare_all_detectors, evaluate_single_sequence


# ---------------------------------------------------------------------------
# 1. FSM Detector Tests
# ---------------------------------------------------------------------------
def test_fsm_valid_standard_sequence():
    """Test that a standard valid protocol sequence is fully accepted with 0 violations."""
    events = ["connect", "auth_success", "data_send", "data_receive", "disconnect", "reset"]
    res = validate_protocol_sequence(events)

    assert res["is_valid"] is True
    assert res["num_invalid_transitions"] == 0
    assert res["is_anomalous"] is False
    assert res["final_state"] in ACCEPT_STATES
    assert len(res["invalid_transitions"]) == 0
    assert len(res["trace"]) == len(events)


def test_fsm_valid_auth_retry():
    """Test valid auth retries (auth_fail staying in CONNECTING then succeeding)."""
    events = ["connect", "auth_fail", "auth_fail", "auth_success", "data_send", "disconnect"]
    res = validate_protocol_sequence(events)

    assert res["is_valid"] is True
    assert res["num_invalid_transitions"] == 0
    assert res["is_anomalous"] is False


def test_fsm_invalid_skip_authentication():
    """Test that attempting to transfer data before authentication is flagged."""
    events = ["connect", "data_send", "disconnect"]
    res = validate_protocol_sequence(events)

    assert res["is_valid"] is False
    assert res["num_invalid_transitions"] >= 1
    assert res["is_anomalous"] is True
    assert any("data_send" in str(item["symbol"]) for item in res["invalid_transitions"])


def test_fsm_invalid_out_of_order():
    """Test that out-of-order events from start state DISCONNECTED are rejected."""
    events = ["data_send", "connect", "disconnect"]
    res = validate_protocol_sequence(events)

    assert res["is_valid"] is False
    assert res["num_invalid_transitions"] >= 1
    assert res["is_anomalous"] is True
    # The first step should be an error
    assert res["trace"][0]["is_valid"] is False


def test_fsm_unknown_symbol():
    """Test that symbols outside the alphabet Σ are detected as invalid."""
    events = ["connect", "exploit_payload", "disconnect"]
    res = validate_protocol_sequence(events)

    assert res["is_valid"] is False
    assert res["num_invalid_transitions"] >= 1
    assert any("Unknown symbol" in str(item["error"]) for item in res["invalid_transitions"])


def test_fsm_empty_sequence():
    """Test that an empty sequence is handled safely without raising exceptions."""
    res = validate_protocol_sequence([])

    assert res["num_invalid_transitions"] == 0
    assert res["final_state"] == START_STATE
    assert len(res["trace"]) == 0


def test_fsm_type_safety():
    """Test that invalid types raise appropriate TypeError."""
    fsm = ProtocolFSM()
    with pytest.raises(TypeError):
        fsm.process_sequence(12345)  # type: ignore


# ---------------------------------------------------------------------------
# 2. Dataset Generation & Feature Extraction Tests
# ---------------------------------------------------------------------------
def test_dataset_generation_shape_and_distribution():
    """Test that dataset generates requested sample count and stratifies properly."""
    n_samples = 200
    df = generate_dataset(n_samples=n_samples, normal_ratio=0.7, random_state=42)

    assert len(df) == n_samples
    assert "is_anomalous" in df.columns
    assert "anomaly_type" in df.columns
    assert "events" in df.columns

    # 70% normal -> 140, 30% anomalous -> 60
    assert df["is_anomalous"].sum() == 60
    assert (df["is_anomalous"] == 0).sum() == 140


def test_dataset_reproducibility():
    """Test that the same random seed produces identical datasets."""
    df1 = generate_dataset(n_samples=100, random_state=42)
    df2 = generate_dataset(n_samples=100, random_state=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_feature_extraction_contract():
    """Verify exact 10 numerical features, correct keys, and no target leakage."""
    events = ["connect", "auth_fail", "auth_success", "data_send", "disconnect"]
    features = extract_features(events)

    # 1. Must contain all 10 canonical features
    assert list(features.keys()) == FEATURE_NAMES
    assert len(features) == 10

    # 2. Must not contain target label (prevent target leakage)
    assert "is_anomalous" not in features
    assert "label" not in features
    assert "anomaly_type" not in features

    # 3. All values must be numerical floats or ints
    for k, v in features.items():
        assert isinstance(v, (int, float, np.number))

    # 4. Values make logical sense
    assert features["sequence_length"] == 5.0
    assert features["num_auth_failures"] == 1.0
    assert features["auth_attempts"] == 2.0
    assert features["num_data_events"] == 1.0
    assert features["has_proper_disconnect"] == 1.0


def test_train_test_split_stratification():
    """Verify stratified splitting preserves class balance."""
    df = generate_dataset(n_samples=200, normal_ratio=0.7, random_state=42)
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_splits(df, test_size=0.25, random_state=42)

    assert len(X_train) == 150
    assert len(X_test) == 50
    assert list(X_train.columns) == FEATURE_NAMES
    assert list(X_test.columns) == FEATURE_NAMES

    # Check stratification ratio (30% anomalous)
    train_anom_ratio = y_train.mean()
    test_anom_ratio = y_test.mean()
    assert pytest.approx(train_anom_ratio, abs=0.02) == 0.30
    assert pytest.approx(test_anom_ratio, abs=0.02) == 0.30


# ---------------------------------------------------------------------------
# 3. ML Detector Tests
# ---------------------------------------------------------------------------
def test_ml_training_evaluation_and_persistence(tmp_path):
    """Test model training, saving, loading, and predicting with feature alignment."""
    df = generate_dataset(n_samples=200, random_state=42)
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_splits(df, test_size=0.25, random_state=42)

    clf = train_model(X_train, y_train, n_estimators=20, max_depth=6, random_state=42)
    eval_res = evaluate_model(clf, X_test, y_test)

    assert 0.0 <= eval_res["accuracy"] <= 1.0
    assert 0.0 <= eval_res["f1"] <= 1.0
    assert len(eval_res["confusion_matrix"]) == 2

    # Test persistence
    model_path = tmp_path / "test_rf.joblib"
    meta_path = tmp_path / "test_meta.json"
    save_model(clf, model_path=model_path, meta_path=meta_path)

    loaded_clf, meta_features = load_model(model_path=model_path, meta_path=meta_path)
    assert meta_features == FEATURE_NAMES

    # Test single sequence inference
    test_seq = ["connect", "auth_success", "data_send", "disconnect"]
    pred_res = predict_sequence(loaded_clf, test_seq)

    assert pred_res["prediction"] in (0, 1)
    assert 0.0 <= pred_res["probability_anomalous"] <= 1.0
    assert 0.0 <= pred_res["probability_normal"] <= 1.0
    assert pytest.approx(pred_res["probability_anomalous"] + pred_res["probability_normal"], abs=1e-5) == 1.0


def test_feature_importance_ranking():
    """Verify feature importances are computed and sum to approximately 1."""
    df = generate_dataset(n_samples=100, random_state=42)
    X_train, _, y_train, _, _, _ = prepare_train_test_splits(df, test_size=0.25, random_state=42)
    clf = train_model(X_train, y_train, n_estimators=10, max_depth=4, random_state=42)

    df_imp = get_feature_importances(clf, FEATURE_NAMES)
    assert len(df_imp) == len(FEATURE_NAMES)
    assert pytest.approx(df_imp["importance"].sum(), abs=1e-4) == 1.0


# ---------------------------------------------------------------------------
# 4. Hybrid Detector & Evaluation Tests
# ---------------------------------------------------------------------------
def test_hybrid_decision_logic():
    """Verify Hybrid OR-logic: anomalous if FSM flags OR ML flags."""
    df = generate_dataset(n_samples=100, random_state=42)
    X_train, _, y_train, _, _, _ = prepare_train_test_splits(df, test_size=0.25, random_state=42)
    clf = train_model(X_train, y_train, n_estimators=10, random_state=42)

    # Sequence with clear FSM violation
    violating_seq = ["connect", "data_send"]
    res_viol = evaluate_single_sequence(violating_seq, clf)
    assert res_viol["fsm"]["flag"] is True
    assert res_viol["hybrid"]["flag"] is True

    # Sequence with standard normal behavior
    normal_seq = ["connect", "auth_success", "data_send", "disconnect", "reset"]
    res_norm = evaluate_single_sequence(normal_seq, clf)
    assert res_norm["fsm"]["flag"] is False


def test_compute_detector_metrics_integrity():
    """Test metric computation with known TP, FP, TN, FN values."""
    # 2 TP, 1 FP, 6 TN, 1 FN -> total 10
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

    metrics = compute_detector_metrics(y_true, y_pred, "Test Detector")

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 6
    assert metrics["false_negatives"] == 1
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3

