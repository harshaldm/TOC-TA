"""dataset_generator.py - Synthetic Network Protocol Sequence Generator.

Generates reproducible synthetic protocol sessions and extracts 10 numerical
features for machine learning classification, ensuring zero target leakage.
"""

import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from fsm_detector import validate_protocol_sequence

# Canonical feature list and fixed ordering for ML training and inference
FEATURE_NAMES: List[str] = [
    "sequence_length",
    "num_auth_failures",
    "auth_attempts",
    "num_data_events",
    "num_invalid_transitions",
    "max_repeated_consecutive",
    "unique_events",
    "has_proper_disconnect",
    "connection_duration",
    "data_volume",
]

ANOMALY_STRATEGIES = [
    "excessive_auth",
    "skip_auth",
    "out_of_order",
    "long_sequence",
    "repeated_requests",
]


def extract_features(events: List[str]) -> Dict[str, float]:
    """Extract 10 numerical features from a protocol event sequence.

    Target label is NEVER included here to prevent target leakage.

    Args:
        events: List of event strings.

    Returns:
        Dictionary mapping feature names to numerical values.
    """
    seq_len = len(events)
    if seq_len == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    # 1. sequence_length
    sequence_length = float(seq_len)

    # 2. num_auth_failures
    num_auth_failures = float(events.count("auth_fail"))

    # 3. auth_attempts
    auth_attempts = float(num_auth_failures + events.count("auth_success"))

    # 4. num_data_events
    num_data_send = events.count("data_send")
    num_data_receive = events.count("data_receive")
    num_data_events = float(num_data_send + num_data_receive)

    # 5. num_invalid_transitions (Key TOC feature)
    fsm_result = validate_protocol_sequence(events)
    num_invalid_transitions = float(fsm_result["num_invalid_transitions"])

    # 6. max_repeated_consecutive
    max_rep = 1
    current_rep = 1
    for i in range(1, seq_len):
        if events[i] == events[i - 1]:
            current_rep += 1
            if current_rep > max_rep:
                max_rep = current_rep
        else:
            current_rep = 1
    max_repeated_consecutive = float(max_rep)

    # 7. unique_events
    unique_events = float(len(set(events)))

    # 8. has_proper_disconnect
    last_event = events[-1]
    has_proper_disconnect = 1.0 if last_event in ("disconnect", "reset", "timeout") else 0.0

    # 9. connection_duration (Derived proxy in seconds based on protocol event delays)
    # connect/auth take ~0.4s, data transfers take ~0.2s, disconnect takes ~0.1s
    base_duration = seq_len * 0.25
    auth_delay = num_auth_failures * 0.8
    data_delay = num_data_events * 0.15
    connection_duration = round(base_duration + auth_delay + data_delay, 2)

    # 10. data_volume (Weighted estimation of data units transferred)
    data_volume = round(float(num_data_send * 1.5 + num_data_receive * 2.0), 2)

    return {
        "sequence_length": sequence_length,
        "num_auth_failures": num_auth_failures,
        "auth_attempts": auth_attempts,
        "num_data_events": num_data_events,
        "num_invalid_transitions": num_invalid_transitions,
        "max_repeated_consecutive": max_repeated_consecutive,
        "unique_events": unique_events,
        "has_proper_disconnect": has_proper_disconnect,
        "connection_duration": connection_duration,
        "data_volume": data_volume,
    }


def generate_normal_sequence(rng: random.Random) -> List[str]:
    """Generate a valid protocol sequence following the FSM specification."""
    seq = ["connect"]

    # 0 to 2 auth failures before successful authentication
    num_fails = rng.choices([0, 1, 2], weights=[0.8, 0.15, 0.05])[0]
    for _ in range(num_fails):
        seq.append("auth_fail")
    seq.append("auth_success")

    # 1 to 8 data transfer events
    num_data = rng.randint(1, 8)
    for _ in range(num_data):
        seq.append(rng.choice(["data_send", "data_receive"]))

    # Proper session termination
    if rng.random() < 0.9:
        seq.append("disconnect")
        if rng.random() < 0.6:
            seq.append(rng.choice(["reset", "timeout"]))
    else:
        seq.append("timeout")

    return seq


def generate_anomalous_sequence(rng: random.Random) -> Tuple[List[str], str]:
    """Generate an anomalous sequence using one of 5 distinct attack strategies.

    Returns:
        Tuple of (event sequence list, anomaly strategy name).
    """
    strategy = rng.choice(ANOMALY_STRATEGIES)
    seq: List[str] = []

    if strategy == "excessive_auth":
        # Strategy 1: Brute force simulation (3 to 8 consecutive auth failures)
        seq.append("connect")
        num_fails = rng.randint(3, 8)
        for _ in range(num_fails):
            seq.append("auth_fail")
        # May or may not eventually authenticate
        if rng.random() < 0.3:
            seq.append("auth_success")
            seq.append("disconnect")
        else:
            seq.append("timeout")

    elif strategy == "skip_auth":
        # Strategy 2: Directly transferring data without authenticating
        # This causes direct DFA transition violations
        seq.append("connect")
        num_data = rng.randint(2, 6)
        for _ in range(num_data):
            seq.append(rng.choice(["data_send", "data_receive"]))
        seq.append("disconnect")

    elif strategy == "out_of_order":
        # Strategy 3: Scrambled protocol event ordering
        base_events = ["data_send", "connect", "disconnect", "auth_success", "data_receive"]
        rng.shuffle(base_events)
        seq = base_events[: rng.randint(3, 5)]

    elif strategy == "long_sequence":
        # Strategy 4: Persistently long session simulating slow data exfiltration
        seq.append("connect")
        seq.append("auth_success")
        num_data = rng.randint(22, 38)
        for _ in range(num_data):
            seq.append(rng.choice(["data_send", "data_receive"]))
        seq.append("disconnect")

    elif strategy == "repeated_requests":
        # Strategy 5: Flooding / replay attack (6 to 15 identical consecutive events)
        repeated_event = rng.choice(["connect", "data_send", "auth_fail"])
        repeat_count = rng.randint(6, 15)
        seq = [repeated_event] * repeat_count
        if rng.random() < 0.5:
            seq.append("reset")

    return seq, strategy


def generate_dataset(
    n_samples: int = 2000,
    normal_ratio: float = 0.7,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset of protocol-event sequences.

    Args:
        n_samples: Total number of sessions to generate (default 2000).
        normal_ratio: Fraction of normal sessions (default 0.7).
        random_state: Seed for reproducibility.

    Returns:
        pandas DataFrame containing sequence strings, extracted features,
        anomaly strategy, and target label `is_anomalous`.
    """
    rng = random.Random(random_state)
    n_normal = int(n_samples * normal_ratio)
    n_anomalous = n_samples - n_normal

    records = []

    # Generate normal sessions
    for i in range(n_normal):
        events = generate_normal_sequence(rng)
        features = extract_features(events)
        record = {
            "session_id": f"NORM_{i+1:04d}",
            "events": events,
            "events_str": " -> ".join(events),
            "anomaly_type": "normal",
            "is_anomalous": 0,
            **features,
        }
        records.append(record)

    # Generate anomalous sessions
    for j in range(n_anomalous):
        events, strategy = generate_anomalous_sequence(rng)
        features = extract_features(events)
        record = {
            "session_id": f"ANOM_{j+1:04d}",
            "events": events,
            "events_str": " -> ".join(events),
            "anomaly_type": strategy,
            "is_anomalous": 1,
            **features,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Shuffle the dataset deterministically
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def prepare_train_test_splits(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split dataset into stratified train and test sets.

    Ensures feature matrix X contains strictly FEATURE_NAMES without target leakage.

    Returns:
        (X_train, X_test, y_train, y_test, df_train, df_test)
    """
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_NAMES]
    y = df["is_anomalous"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
    )

    df_train = df.loc[idx_train].reset_index(drop=True)
    df_test = df.loc[idx_test].reset_index(drop=True)

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        df_train,
        df_test,
    )

