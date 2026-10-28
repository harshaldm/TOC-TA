"""fsm_detector.py - Deterministic Finite Automaton (DFA) Protocol Validator.

Theory of Computation Formal Definition:
A Deterministic Finite Automaton (DFA) is defined as a 5-tuple:
    M = (Q, Σ, δ, q0, F)
where:
    - Q: Finite set of states = {DISCONNECTED, CONNECTING, AUTHENTICATED, DATA_TRANSFER, DISCONNECTING}
    - Σ: Finite input alphabet = {connect, auth_success, auth_fail, data_send, data_receive, disconnect, timeout, reset}
    - δ: State transition function: Q × Σ -> Q
    - q0: Initial/start state = DISCONNECTED
    - F: Set of accept states = {DISCONNECTED, DISCONNECTING}
"""

from typing import Dict, List, Optional, Set, Tuple


# Formal DFA Components
STATES: Set[str] = {
    "DISCONNECTED",
    "CONNECTING",
    "AUTHENTICATED",
    "DATA_TRANSFER",
    "DISCONNECTING",
}

START_STATE: str = "DISCONNECTED"

ACCEPT_STATES: Set[str] = {
    "DISCONNECTED",
    "DISCONNECTING",
}

ALPHABET: Set[str] = {
    "connect",
    "auth_success",
    "auth_fail",
    "data_send",
    "data_receive",
    "disconnect",
    "timeout",
    "reset",
}

# δ: Transition Function Table
TRANSITION_TABLE: Dict[Tuple[str, str], str] = {
    ("DISCONNECTED", "connect"): "CONNECTING",
    ("CONNECTING", "auth_success"): "AUTHENTICATED",
    ("CONNECTING", "auth_fail"): "CONNECTING",
    ("CONNECTING", "timeout"): "DISCONNECTED",
    ("CONNECTING", "reset"): "DISCONNECTED",
    ("AUTHENTICATED", "data_send"): "DATA_TRANSFER",
    ("AUTHENTICATED", "data_receive"): "DATA_TRANSFER",
    ("AUTHENTICATED", "disconnect"): "DISCONNECTING",
    ("AUTHENTICATED", "timeout"): "DISCONNECTED",
    ("DATA_TRANSFER", "data_send"): "DATA_TRANSFER",
    ("DATA_TRANSFER", "data_receive"): "DATA_TRANSFER",
    ("DATA_TRANSFER", "disconnect"): "DISCONNECTING",
    ("DATA_TRANSFER", "timeout"): "DISCONNECTED",
    ("DISCONNECTING", "reset"): "DISCONNECTED",
    ("DISCONNECTING", "timeout"): "DISCONNECTED",
}


class ProtocolFSM:
    """Deterministic Finite Automaton for network protocol session validation.

    Tracks state transitions and detects violations of the formal regular language.
    """

    def __init__(self, start_state: str = START_STATE):
        self.start_state = start_state
        self.current_state = start_state
        self.history: List[Dict[str, object]] = []
        self.invalid_transitions: List[Dict[str, object]] = []

    def reset(self) -> None:
        """Reset the FSM to the initial start state."""
        self.current_state = self.start_state
        self.history.clear()
        self.invalid_transitions.clear()

    def step(self, symbol: str) -> Dict[str, object]:
        """Process a single input symbol through the transition function δ.

        Args:
            symbol: The protocol event symbol from Σ.

        Returns:
            Dictionary containing transition details and validity flag.
        """
        step_number = len(self.history) + 1
        from_state = self.current_state

        # Check if the symbol belongs to the alphabet Σ
        if symbol not in ALPHABET:
            is_valid = False
            to_state = from_state  # State remains unchanged on violation
            error_msg = f"Unknown symbol '{symbol}' not in alphabet Σ"
        elif (from_state, symbol) in TRANSITION_TABLE:
            is_valid = True
            to_state = TRANSITION_TABLE[(from_state, symbol)]
            error_msg = None
            self.current_state = to_state
        else:
            is_valid = False
            to_state = from_state  # State remains unchanged on violation
            error_msg = f"Undefined transition: cannot execute '{symbol}' from state '{from_state}'"

        step_info = {
            "step": step_number,
            "from_state": from_state,
            "symbol": symbol,
            "to_state": to_state,
            "is_valid": is_valid,
            "error": error_msg,
        }

        self.history.append(step_info)
        if not is_valid:
            self.invalid_transitions.append(step_info)

        return step_info

    def process_sequence(self, events: List[str]) -> Dict[str, object]:
        """Process a full sequence of protocol events.

        Args:
            events: List of event strings representing a network session.

        Returns:
            Detailed analysis dictionary with transition trace and verdict.
        """
        self.reset()

        if not isinstance(events, (list, tuple)):
            raise TypeError("Expected events to be a list or tuple of strings.")

        trace = []
        for event in events:
            trace.append(self.step(str(event)))

        num_invalid = len(self.invalid_transitions)
        is_valid_trace = (num_invalid == 0)
        is_accepted = is_valid_trace and (self.current_state in ACCEPT_STATES)
        is_anomalous = (num_invalid > 0)

        # Build clear human-readable explanation
        if is_anomalous:
            reasons = [f"Step {item['step']}: {item['error']}" for item in self.invalid_transitions]
            explanation = (
                f"Protocol violation detected ({num_invalid} invalid transition(s)). "
                f"Violations: {'; '.join(reasons)}"
            )
        elif not is_accepted and len(events) > 0:
            explanation = (
                f"Incomplete session: sequence finished in non-accept state '{self.current_state}', "
                f"expected one of {sorted(ACCEPT_STATES)}."
            )
        elif len(events) == 0:
            explanation = "Empty event sequence. FSM remains in start state DISCONNECTED."
        else:
            explanation = (
                f"Valid sequence accepted: followed legal protocol transitions and terminated "
                f"in accepted state '{self.current_state}'."
            )

        return {
            "events": list(events),
            "trace": trace,
            "final_state": self.current_state,
            "is_valid": is_valid_trace,
            "is_accepted": is_accepted,
            "num_invalid_transitions": num_invalid,
            "invalid_transitions": list(self.invalid_transitions),
            "is_anomalous": is_anomalous,
            "explanation": explanation,
        }


def validate_protocol_sequence(events: List[str]) -> Dict[str, object]:
    """Convenience functional interface to validate a protocol event sequence.

    Args:
        events: List of protocol event strings.

    Returns:
        Validation results dictionary from ProtocolFSM.process_sequence.
    """
    fsm = ProtocolFSM()
    return fsm.process_sequence(events)

