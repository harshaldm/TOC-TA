"""app.py - NetGuard Streamlit Interactive Web Dashboard.

Demonstrates Theory of Computation (FSM) combined with Machine Learning (Random Forest)
for Network Intrusion Detection on synthetic protocol sequences.
"""

import os
import sys
from pathlib import Path

# Set Matplotlib cache dir inside project to prevent Windows temp path permission issues
mpl_dir = Path(__file__).resolve().parent / ".mpl_cache"
mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Add current folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_generator import FEATURE_NAMES, generate_dataset, prepare_train_test_splits
from evaluation import compare_all_detectors, evaluate_single_sequence
from fsm_detector import (
    ACCEPT_STATES,
    ALPHABET,
    START_STATE,
    STATES,
    TRANSITION_TABLE,
    validate_protocol_sequence,
)
from ml_detector import (
    get_feature_importances,
    load_model,
    predict_sequence,
    save_model,
    train_model,
)

# Page Configuration
st.set_page_config(
    page_title="NetGuard — Network Intrusion Detection (TOC + ML)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Data & Model Pipeline with Caching
@st.cache_resource(show_spinner=True)
def get_or_create_pipeline(n_samples: int = 2000, random_state: int = 42):
    """Generate dataset, train Random Forest, and compute comparative evaluation."""
    df = generate_dataset(n_samples=n_samples, random_state=random_state)
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_splits(
        df, test_size=0.25, random_state=random_state
    )

    # Train model
    clf = train_model(X_train, y_train, random_state=random_state)
    save_model(clf)

    # Comparative evaluation
    comparison_df, metrics, results_df = compare_all_detectors(clf, df_test)
    feature_imp = get_feature_importances(clf, FEATURE_NAMES)

    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "clf": clf,
        "comparison_df": comparison_df,
        "metrics": metrics,
        "results_df": results_df,
        "feature_imp": feature_imp,
    }


def main():
    # Academic Disclaimer & Header
    st.markdown(
        """
        <div style="background-color: #f0f4f8; padding: 16px 20px; border-radius: 8px; border-left: 5px solid #1e88e5; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #0d47a1;">🛡️ NetGuard — Network Intrusion Detection</h2>
            <p style="margin: 5px 0 0 0; color: #37474f; font-size: 15px;">
                <strong>Theory of Computation Academic Project</strong>: Integrating <em>Deterministic Finite Automata (DFA)</em> with 
                <em>Machine Learning (Random Forest)</em> for hybrid protocol verification and anomaly detection.
            </p>
            <p style="margin: 4px 0 0 0; color: #d32f2f; font-size: 13px;">
                ⚠️ <strong>Note:</strong> All network protocol data is <em>synthetically generated</em> for academic evaluation and demonstration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Experiment Controls")
        n_samples = st.selectbox("Dataset Size", options=[1000, 2000, 3000], index=1)
        seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

        if st.button("🔄 Retrain / Refresh Dataset", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Regenerating dataset and retraining model...")

        st.markdown("---")
        st.markdown("### 📚 TOC Project Details")
        st.markdown("**Subject:** Theory of Computation")
        st.markdown("**Model:** Deterministic Finite Automaton (DFA) + Random Forest")
        st.markdown("**Class Ratio:** 70% Normal / 30% Anomalous")
        st.markdown("**Split:** 75% Train / 25% Test (Stratified)")

    # Load Pipeline Data
    data = get_or_create_pipeline(n_samples=n_samples, random_state=int(seed))
    clf = data["clf"]
    comparison_df = data["comparison_df"]
    metrics = data["metrics"]
    df_test = data["df_test"]
    results_df = data["results_df"]
    feature_imp = data["feature_imp"]

    # Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Sequence Inspector",
        "📊 Comparative Evaluation",
        "📐 Formal TOC / FSM Theory",
        "📁 Dataset & Suspicious Traffic",
    ])

    # -------------------------------------------------------------
    # TAB 1: LIVE SEQUENCE INSPECTOR
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Interactive Protocol Sequence Tester")
        st.markdown(
            "Select a pre-configured protocol scenario or enter custom events to inspect "
            "step-by-step FSM transitions, ML predictions, and the combined Hybrid verdict."
        )

        col_sample, col_custom = st.columns([1, 1])

        preset_options = {
            "Normal Standard Session": [
                "connect", "auth_success", "data_send", "data_receive", "disconnect", "reset"
            ],
            "Normal Session with Retried Auth": [
                "connect", "auth_fail", "auth_success", "data_send", "data_send", "disconnect"
            ],
            "Attack: Skip Authentication (Direct Data Transfer)": [
                "connect", "data_send", "data_receive", "disconnect"
            ],
            "Attack: Brute Force Authentication (Excessive Failures)": [
                "connect", "auth_fail", "auth_fail", "auth_fail", "auth_fail", "auth_fail", "timeout"
            ],
            "Attack: Out-of-Order Protocol Events": [
                "data_send", "connect", "auth_success", "disconnect"
            ],
            "Attack: Flooding / Repeated Requests": [
                "connect", "auth_success", "data_send", "data_send", "data_send",
                "data_send", "data_send", "data_send", "data_send", "data_send", "disconnect"
            ],
        }

        with col_sample:
            selected_preset = st.selectbox(
                "Choose a Preset Scenario:",
                options=list(preset_options.keys()),
            )
            default_events = preset_options[selected_preset]

        with col_custom:
            custom_input = st.text_input(
                "Or enter comma-separated protocol events:",
                value=", ".join(default_events),
                help=f"Alphabet Σ: {', '.join(sorted(ALPHABET))}",
            )

        events_to_test = [e.strip() for e in custom_input.split(",") if e.strip()]

        if st.button("🚀 Analyze Protocol Sequence", type="primary"):
            if not events_to_test:
                st.warning("Please enter at least one protocol event.")
            else:
                eval_res = evaluate_single_sequence(events_to_test, clf)
                fsm_info = eval_res["fsm"]
                ml_info = eval_res["ml"]
                hybrid_info = eval_res["hybrid"]

                st.markdown("---")
                st.subheader("Detection Results & Decision Logic")

                # Verdict Cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    is_fsm_anom = fsm_info["flag"]
                    st.metric(
                        label="Rule-Based FSM Verdict",
                        value="ANOMALY DETECTED" if is_fsm_anom else "VALID / ACCEPTED",
                        delta=f"{fsm_info['num_invalid']} invalid transition(s)",
                        delta_color="inverse" if is_fsm_anom else "normal",
                    )
                    st.caption(f"Final State: `{fsm_info['final_state']}`")

                with c2:
                    is_ml_anom = ml_info["flag"]
                    prob = ml_info["probability_anomalous"]
                    st.metric(
                        label="ML Random Forest Verdict",
                        value="ANOMALY DETECTED" if is_ml_anom else "NORMAL PATTERN",
                        delta=f"Anomaly Prob: {prob*100:.1f}%",
                        delta_color="inverse" if is_ml_anom else "normal",
                    )
                    st.caption(f"Confidence: {ml_info['confidence']*100:.1f}%")

                with c3:
                    is_hyb_anom = hybrid_info["flag"]
                    st.metric(
                        label="Hybrid Detector Verdict",
                        value="INTRUSION ALERT 🚨" if is_hyb_anom else "BENIGN SESSION ✅",
                        delta="FSM ∨ ML Triggered" if is_hyb_anom else "Passed Both Detectors",
                        delta_color="inverse" if is_hyb_anom else "normal",
                    )
                    st.caption(f"Rule: {hybrid_info['decision_rule']}")

                # Explanation Box
                st.info(f"**Decision Explanation:** {hybrid_info['reason']}")

                # FSM Step-by-Step Transition Trace Table
                st.markdown("#### 📜 FSM State Transition Trace")
                trace_rows = []
                for step in fsm_info["trace"]:
                    status_emoji = "✅ Valid" if step["is_valid"] else "❌ VIOLATION"
                    trace_rows.append({
                        "Step": step["step"],
                        "Current State (q)": step["from_state"],
                        "Event (σ ∈ Σ)": step["symbol"],
                        "Next State (δ(q, σ))": step["to_state"],
                        "Validity": status_emoji,
                        "Violation Reason": step["error"] or "Follows defined protocol transition",
                    })

                trace_df = pd.DataFrame(trace_rows)
                st.dataframe(trace_df, use_container_width=True)

                # Feature Extraction Breakdown
                with st.expander("🔬 View 10 Extracted Features for ML Model"):
                    feat_dict = ml_info["features"]
                    f_df = pd.DataFrame([feat_dict]).T.reset_index()
                    f_df.columns = ["Feature Name", "Extracted Value"]
                    st.table(f_df)

    # -------------------------------------------------------------
    # TAB 2: COMPARATIVE EVALUATION
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Comparative Performance on Held-Out Test Data")
        st.markdown(
            "All three detectors are evaluated on the **exact same held-out test split** (25% of data). "
            "No metrics are hardcoded; all results are dynamically computed."
        )

        # Summary Metric Cards
        col_fsm, col_ml, col_hyb = st.columns(3)
        with col_fsm:
            st.markdown("### 🏛️ Rule-Based FSM")
            st.metric("Accuracy", f"{metrics['fsm']['accuracy']*100:.2f}%")
            st.metric("Recall (Sensitivity)", f"{metrics['fsm']['recall']*100:.2f}%")
            st.metric("Precision", f"{metrics['fsm']['precision']*100:.2f}%")
            st.metric("F1-Score", f"{metrics['fsm']['f1_score']:.4f}")

        with col_ml:
            st.markdown("### 🌲 ML (Random Forest)")
            st.metric("Accuracy", f"{metrics['ml']['accuracy']*100:.2f}%")
            st.metric("Recall (Sensitivity)", f"{metrics['ml']['recall']*100:.2f}%")
            st.metric("Precision", f"{metrics['ml']['precision']*100:.2f}%")
            st.metric("F1-Score", f"{metrics['ml']['f1_score']:.4f}")

        with col_hyb:
            st.markdown("### ⚡ Hybrid (FSM ∨ ML)")
            st.metric("Accuracy", f"{metrics['hybrid']['accuracy']*100:.2f}%")
            st.metric("Recall (Sensitivity)", f"{metrics['hybrid']['recall']*100:.2f}%")
            st.metric("Precision", f"{metrics['hybrid']['precision']*100:.2f}%")
            st.metric("F1-Score", f"{metrics['hybrid']['f1_score']:.4f}")

        st.markdown("---")
        st.markdown("#### 📊 Metric Comparison Table")
        st.dataframe(comparison_df.set_index("Detector"), use_container_width=True)

        # Charts: Bar Comparison and Confusion Matrices
        c_chart1, c_chart2 = st.columns([1, 1])

        with c_chart1:
            st.markdown("#### Metric Comparison Chart")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_df = comparison_df.melt(
                id_vars=["Detector"],
                value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                var_name="Metric",
                value_name="Score",
            )
            sns.barplot(data=plot_df, x="Metric", y="Score", hue="Detector", palette="Set2", ax=ax)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("Score (0.0 to 1.0)")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

        with c_chart2:
            st.markdown("#### Confusion Matrix (Hybrid Detector)")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            cm_hyb = np.array(metrics["hybrid"]["confusion_matrix"])
            sns.heatmap(
                cm_hyb,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Anomalous"],
                yticklabels=["Normal", "Anomalous"],
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Ground Truth")
            st.pyplot(fig)
            plt.close(fig)

        # Theoretical Trade-off Analysis Box
        st.markdown(
            """
            <div style="background-color: #f7fafc; padding: 14px; border-radius: 6px; border: 1px solid #cbd5e0; margin-top: 15px;">
                <h4 style="margin-top: 0; color: #2d3748;">💡 Academic Insight: Why Use a Hybrid Detector?</h4>
                <p style="margin-bottom: 6px; font-size: 14px;">
                    1. <strong>Rule-based FSM</strong> provides deterministic guarantees. If an invalid transition occurs (e.g. data transmission before authentication), it flags with 100% precision and zero ambiguity. However, it cannot catch attacks that follow valid protocol syntax (such as DoS flooding or slow scans).
                </p>
                <p style="margin-bottom: 6px; font-size: 14px;">
                    2. <strong>Machine Learning (Random Forest)</strong> excels at statistical pattern detection across multiple features (burst length, volume, rate). However, it is an approximate statistical classifier and lacks protocol semantics.
                </p>
                <p style="margin-bottom: 0; font-size: 14px;">
                    3. <strong>Hybrid (OR-logic: FSM ∨ ML)</strong> maximizes <strong>Recall</strong>. In cybersecurity intrusion detection, missing an attack (False Negative) is far more dangerous than inspecting an extra alert (False Positive).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # TAB 3: FORMAL TOC / FSM THEORY
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Theory of Computation: Deterministic Finite Automaton (DFA)")
        st.markdown(
            r"""
            In formal language theory, a **Deterministic Finite Automaton (DFA)** is defined by the 5-tuple:
            $$\mathbf{M = (Q, \Sigma, \delta, q_0, F)}$$
            """
        )

        col_def1, col_def2 = st.columns(2)
        with col_def1:
            st.markdown("#### 1. States & Alphabet")
            st.markdown(rf"- **States ($Q$):** `{', '.join(sorted(STATES))}`")
            st.markdown(rf"- **Initial State ($q_0$):** `{START_STATE}`")
            st.markdown(rf"- **Accept States ($F$):** `{', '.join(sorted(ACCEPT_STATES))}`")
            st.markdown(rf"- **Input Alphabet ($\Sigma$):** `{', '.join(sorted(ALPHABET))}`")

            st.markdown("#### 2. Regular Language Definition")
            st.markdown(
                r"A protocol session $w = \sigma_1 \sigma_2 \dots \sigma_n$ where $\sigma_i \in \Sigma$ belongs to "
                r"the regular language $L(M)$ if and only if:"
            )
            st.latex(r"\hat{\delta}(q_0, w) \in F \quad \text{and no undefined transition is traversed.}")

        with col_def2:
            st.markdown(r"#### 3. State Transition Table ($\delta: Q \times \Sigma \to Q$)")
            tt_rows = []
            for (q, sym), next_q in sorted(TRANSITION_TABLE.items()):
                tt_rows.append({"Current State (q)": q, "Input Symbol (σ)": sym, "Next State δ(q, σ)": next_q})
            st.dataframe(pd.DataFrame(tt_rows), height=320, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: DATASET & SUSPICIOUS TRAFFIC
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Synthetic Dataset & Suspicious Sessions")
        st.markdown(
            "Inspect the generated synthetic protocol dataset, feature distributions, and Random Forest feature importances."
        )

        c_dist1, c_dist2 = st.columns(2)

        with c_dist1:
            st.markdown("#### Class & Attack Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            type_counts = data["df"]["anomaly_type"].value_counts()
            type_counts.plot(kind="barh", ax=ax, color="#4299e1")
            ax.set_xlabel("Session Count")
            ax.set_ylabel("Anomaly Strategy")
            st.pyplot(fig)
            plt.close(fig)

        with c_dist2:
            st.markdown("#### Random Forest Feature Importances")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(data=feature_imp, x="importance", y="feature", palette="Blues_r", ax=ax)
            ax.set_xlabel("Gini Importance")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.markdown("#### 🚨 Suspicious Test Sessions Filter")

        filter_option = st.selectbox(
            "Filter Test Sessions By Type:",
            options=["All Sessions", "Anomalous Only", "Normal Only"] + list(data["df"]["anomaly_type"].unique()),
        )

        display_df = results_df[[
            "session_id", "anomaly_type", "is_anomalous", "pred_fsm", "pred_ml", "pred_hybrid",
            "num_invalid_transitions", "sequence_length", "auth_attempts", "num_data_events", "events_str"
        ]]

        if filter_option == "Anomalous Only":
            filtered = display_df[display_df["is_anomalous"] == 1]
        elif filter_option == "Normal Only":
            filtered = display_df[display_df["is_anomalous"] == 0]
        elif filter_option != "All Sessions":
            filtered = display_df[display_df["anomaly_type"] == filter_option]
        else:
            filtered = display_df

        st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()

