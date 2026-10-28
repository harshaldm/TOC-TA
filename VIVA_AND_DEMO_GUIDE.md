# NetGuard — Viva & Project Demonstration Guide

> **Quick Summary for Tomorrow:**  
> This project is a hybrid cybersecurity system. It uses **Theory of Computation (Finite-State Machines)** to catch illegal protocol actions (like hacking past a login screen), and **Machine Learning (Random Forest)** to catch statistical anomalies (like flooding or brute-force attacks).

---

## 1. The Problem Statement in Plain English

### The Real-World Situation
Every time two computers talk over the internet (e.g., your browser and a bank server), they follow a strict conversation sequence:
1. **Connect**
2. **Authenticate** (login with password)
3. **Data Transfer** (view account, send data)
4. **Disconnect** (logout)

### What Hackers Do
Intruders try to break or abuse this flow in different ways:
- **Attack 1 (Skip Login):** The attacker tries to send data packets directly without logging in.
- **Attack 2 (Out-of-Order Packets):** The attacker sends corrupted packets to crash the server.
- **Attack 3 (Brute Force):** The attacker sends 10 wrong passwords in a row trying to guess credentials.
- **Attack 4 (Flooding / DoS):** The attacker logs in legitimately, but then fires 100 identical requests in 2 seconds to crash the server.

---

## 2. Why Combine Theory of Computation (TOC) and Machine Learning (ML)?

This is the **#1 question your professor will ask you**. Here is the simple answer:

### Why TOC (Finite-State Machine)?
- In your TOC course, you studied **Deterministic Finite Automata (DFA)**.
- A DFA enforces a strict grammar (a *Regular Language*).
- We modeled network connection states: `DISCONNECTED`, `CONNECTING`, `AUTHENTICATED`, `DATA_TRANSFER`, `DISCONNECTING`.
- If someone tries to `data_send` while in `CONNECTING`, the DFA encounters an **undefined transition**.
- **Strength:** 100% deterministic mathematical proof. Zero false alarms for protocol violations.
- **Limitation:** The FSM is blind to volume. If an attacker logs in properly and then spams 50 data requests, every individual request is technically valid, so the FSM lets it pass!

### Why Machine Learning (Random Forest)?
- Random Forest doesn't care about the grammar; it analyzes **numbers and statistics** (session length, number of failures, repetition counts, duration).
- It catches high-volume attacks, brute force, and flooding easily.
- **Limitation:** It is a statistical approximation and can occasionally produce false alarms or miss novel subtle sequence violations.

### The Hybrid Solution (FSM ∨ ML)
We combine both using **OR-logic**:
$$\text{Intrusion Alert} = (\text{FSM flags an invalid transition}) \lor (\text{ML flags anomalous statistical features})$$
- If an attack violates protocol rules $\to$ **FSM catches it.**
- If an attack abuses volume or timing $\to$ **ML catches it.**
- **Result:** You achieve **maximum Recall (catching ~99.3% of all attacks)** while maintaining clear explainability.

---

## 3. How to Demonstrate the Project to Your Professor (Step-by-Step Script)

### Step 0: Open the Dashboard
Open your terminal in `C:\Users\Harshal\Desktop\TOC` and run:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Your browser will open at `http://localhost:8501`.

---

### Step 1: Introduce the Project (30 seconds)
> *"Good morning/afternoon Professor. Our project is NetGuard. It demonstrates how formal language theory from Theory of Computation—specifically Deterministic Finite Automata—can be combined with Machine Learning for network intrusion detection."*

Show the banner at the top of the dashboard. Point out the note explaining that traffic data is synthetically generated for academic demonstration.

---

### Step 2: Live Demo — Attack That FSM Catches (Tab 1: Live Sequence Inspector)
1. Go to **Tab 1: Live Sequence Inspector**.
2. In the dropdown, choose **"Attack: Skip Authentication (Direct Data Transfer)"**.
3. Click the red **"🚀 Analyze Protocol Sequence"** button.
4. **What to tell your professor:**
   > *"Notice here: the sequence is `connect -> data_send -> data_receive -> disconnect`. The user never authenticated."*
   > - Point to the **FSM Verdict Card**: It displays **ANOMALY DETECTED (1 invalid transition)**.
   > - Scroll down to the **FSM State Transition Trace Table**: Show Step 2 (`data_send` from state `CONNECTING`). Explain: *"The DFA transition function $\delta(\text{CONNECTING}, \text{data\_send})$ is undefined. The FSM caught this protocol violation mathematically."*

---

### Step 3: Live Demo — Attack That ML Catches (Tab 1: Live Sequence Inspector)
1. In the preset dropdown, choose **"Attack: Flooding / Repeated Requests"**.
2. Click **"🚀 Analyze Protocol Sequence"**.
3. **What to tell your professor:**
   > *"Now look at this scenario: the attacker connected, authenticated legally, but then flooded 10 identical `data_send` requests. Because every transition is technically allowed by the DFA rules, the FSM says VALID. BUT the ML Random Forest flags it as an anomaly because `max_repeated_consecutive` is 10 and `data_volume` is abnormally high. The Hybrid detector combines both and triggers an INTRUSION ALERT."*

---

### Step 4: Show the Theoretical Foundation (Tab 3: Formal TOC / FSM Theory)
1. Click on **Tab 3: Formal TOC / FSM Theory**.
2. **What to tell your professor:**
   > *"Here is the formal TOC definition of our DFA as a 5-tuple: $M = (Q, \Sigma, \delta, q_0, F)$."*
   > - State set $Q$: `{DISCONNECTED, CONNECTING, AUTHENTICATED, DATA_TRANSFER, DISCONNECTING}`
   > - Alphabet $\Sigma$: 8 protocol events.
   > - Initial state $q_0$: `DISCONNECTED`
   > - Accept states $F$: `{DISCONNECTED, DISCONNECTING}`
   > - Show the interactive **State Transition Table** on the right side.

---

### Step 5: Show the Metric Comparison (Tab 2: Comparative Evaluation)
1. Click on **Tab 2: Comparative Evaluation**.
2. **What to tell your professor:**
   > *"We tested all three detectors on the exact same held-out 500-sample test set (70% normal, 30% anomalous). Here are the real computed metrics:"*
   > - **Rule-Based FSM:** 100% Precision (0 False Positives!), but only 52% Recall because it cannot catch flooding.
   > - **ML Model:** 99.8% Accuracy and 99.3% Recall.
   > - **Hybrid:** Combines both to ensure zero protocol violations slip through while catching statistical anomalies.
   > - Point out the **Metric Comparison Bar Chart** and the **Confusion Matrix Heatmap**.

---

## 4. What Are the Automated Tests and Should You Show Them?

### Should you show them to your professor?
**YES, absolutely!** Showing automated unit tests in a viva proves you built a professional, robust software project rather than just a script. It leaves a very strong impression on evaluators.

### How to Run the Tests in Front of the Professor:
Open PowerShell and run:
```powershell
.\.venv\Scripts\pytest.exe -v
```

### What do the 15 tests actually check?
Here is a plain-English explanation of each test in `tests/test_netguard.py`:

| Test Name | What it Verifies |
|---|---|
| `test_fsm_valid_standard_sequence` | Checks that a normal session (`connect -> auth -> data -> disconnect`) reaches an accept state with 0 errors. |
| `test_fsm_valid_auth_retry` | Checks that entering 1 or 2 wrong passwords before a correct one is allowed by protocol rules. |
| `test_fsm_invalid_skip_authentication` | Confirms FSM rejects anyone trying to send data before logging in. |
| `test_fsm_invalid_out_of_order` | Confirms FSM rejects scrambled event sequences. |
| `test_fsm_unknown_symbol` | Confirms that foreign packets outside the alphabet $\Sigma$ are flagged immediately. |
| `test_fsm_empty_sequence` | Ensures the system doesn't crash if an empty list of events is passed. |
| `test_fsm_type_safety` | Ensures passing invalid types (e.g., numbers instead of lists) raises clean errors safely. |
| `test_dataset_generation_shape_and_distribution` | Confirms synthetic dataset contains 2000 rows with exact 70/30 class balance. |
| `test_dataset_reproducibility` | Confirms random seed 42 produces identical data every time (reproducibility). |
| `test_feature_extraction_contract` | Confirms all 10 features are extracted, numerical, and **no target label is leaked into the inputs**. |
| `test_train_test_split_stratification` | Confirms training (75%) and test (25%) splits preserve the 70/30 normal/attack ratio. |
| `test_ml_training_evaluation_and_persistence` | Confirms Random Forest trains, saves to `models/rf_model.joblib`, and reloads accurately. |
| `test_feature_importance_ranking` | Confirms feature importances are calculated properly. |
| `test_hybrid_decision_logic` | Verifies the OR-logic (`FSM_flag OR ML_flag`). |
| `test_compute_detector_metrics_integrity` | Verifies accuracy, precision, recall, and confusion matrix formulas with known values. |

---

## 5. Quick Viva "Cheat Sheet" (30-Second Answers)

**Q: What is the regular language in this project?**  
*A: The set of all syntactically valid protocol event sequences accepted by our DFA.*

**Q: Why is Random Forest used instead of Deep Learning?**  
*A: Because we have tabular features (10 numerical metrics), not raw image or audio data. Random Forest is fast, robust against overfitting with balanced class weights, and provides clear feature importances.*

**Q: Why is Recall more important than Precision in cybersecurity?**  
*A: A False Negative means an attacker breached your system undetected. A False Positive is just a security alert that an analyst checks. Missing an attack is far more dangerous, so we prioritize high recall.*

**Q: What is Target Leakage and how did you prevent it?**  
*A: Target leakage happens if the label `is_anomalous` or information derived from it is fed into the model during training. We prevented it because feature extraction only looks at the raw event list, and the label is assigned completely independently.*
