import logging
import math
import numpy as np
from typing import Union # For type hints like Optional
from config import settings # Import the global settings instance

# --- Utility Functions for Labeling ---
def error_map_to_class_label(error_map: dict, data_qubit_indices_list: list[int]) -> int:
    """Converts an error map (e.g., {data_idx: 'X'}) to a class label."""
    if not error_map:
        return 0 # Class 0 is Identity

    if len(error_map) > 1:
        logging.warning(f"error_map_to_class_label received multiple errors: {error_map}. Using first one.")

    data_idx, pauli_type = list(error_map.items())[0]

    try:
        list_pos = data_qubit_indices_list.index(data_idx)
    except ValueError:
        logging.error(f"Data qubit index {data_idx} not found in canonical list {data_qubit_indices_list}!")
        return 0

    pauli_offset = {'X': 1, 'Y': 2, 'Z': 3}
    if pauli_type not in pauli_offset:
        logging.error(f"Unknown Pauli type '{pauli_type}' in error map.")
        return 0

    label = 1 + list_pos * 3 + (pauli_offset[pauli_type] - 1)

    if label >= settings.NUM_ERROR_CLASSES:
        logging.error(f"Calculated label {label} exceeds NUM_ERROR_CLASSES {settings.NUM_ERROR_CLASSES}.")
        return 0

    return label

def class_label_to_error_map(label: int, data_qubit_indices_list: list[int]) -> dict:
    """Converts a class label back to an error map {data_index: pauli_char}."""
    if label == 0:
        return {}
    if not isinstance(label, int) or label < 0 or label >= settings.NUM_ERROR_CLASSES:
        logging.warning(f"Invalid label {label} for conversion to error map (expected 0-{settings.NUM_ERROR_CLASSES-1}).")
        return {}

    pauli_code = (label - 1) % 3
    data_list_pos = (label - 1) // 3

    if data_list_pos >= len(data_qubit_indices_list):
         logging.warning(f"Invalid data qubit position {data_list_pos} derived from label {label}. Max index is {len(data_qubit_indices_list)-1}.")
         return {}

    data_idx = data_qubit_indices_list[data_list_pos]
    pauli_map = {0: 'X', 1: 'Y', 2: 'Z'}
    pauli_type = pauli_map[pauli_code]

    return {data_idx: pauli_type}

def class_label_to_target_bitstring(label: int, num_vqc_qubits: int) -> str:
    """Maps a class label to its target VQC measurement bitstring (binary representation)."""
    if not isinstance(label, int) or label < 0 or label >= settings.NUM_ERROR_CLASSES:
        logging.warning(f"Label {label} out of bounds (0-{settings.NUM_ERROR_CLASSES-1}). Defaulting to label 0.")
        label = 0

    max_representable_value = (1 << num_vqc_qubits) - 1
    effective_label = label
    if label > max_representable_value:
         logging.warning(f"Label {label} exceeds max representable value {max_representable_value} for {num_vqc_qubits} qubits. Using modulo mapping.")
         effective_label = label % (max_representable_value + 1)

    # Return binary string, padded to length num_vqc_qubits
    return format(effective_label, f'0{num_vqc_qubits}b')


def measurement_to_class_label(counts: dict, num_vqc_qubits: int, num_classes: int) -> int:
    """Maps VQC measurement counts (dict) to a predicted class label."""
    if not counts:
        return 0 # Predict Identity if no counts

    most_frequent_bitstring = max(counts, key=counts.get)

    # Convert the bitstring (Qiskit format, e.g., '0101') to an integer
    predicted_label_raw = int(most_frequent_bitstring, 2)

    # Map the raw integer to the valid class label range [0, num_classes-1]
    max_representable_value = (1 << num_vqc_qubits) - 1
    if num_classes > (max_representable_value + 1):
        predicted_label = predicted_label_raw % num_classes
    elif predicted_label_raw >= num_classes:
        logging.warning(f"Raw predicted label {predicted_label_raw} >= num_classes {num_classes}. Mapping to label 0.")
        predicted_label = 0
    else:
         predicted_label = predicted_label_raw

    return predicted_label

# --- Data Conversion for Classical ML ---
def syndrome_str_to_feature_vector(syndrome_str: str) -> np.ndarray:
    """Converts a syndrome bitstring ('0110...') into a NumPy array of floats."""
    try:
        features = np.array([float(bit) for bit in syndrome_str], dtype=float)
        # Basic validation: check length against expected number of stabilizers
        num_stabilizers = settings.NUM_ANCILLA_QUBITS # For d=3, 8 stabilizers
        if len(features) != num_stabilizers:
             logging.warning(f"Syndrome string length {len(features)} does not match expected stabilizers {num_stabilizers}. Padding/truncating may occur implicitly later.")
             # We could optionally pad/truncate here if needed, but better to fix generation
        return features
    except ValueError as e:
        logging.error(f"Error converting syndrome string '{syndrome_str}' to feature vector: {e}")
        num_stabilizers = settings.NUM_ANCILLA_QUBITS
        return np.zeros(num_stabilizers, dtype=float) # Return zeros on error