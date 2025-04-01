import numpy as np
import random
import time
import logging
import math
import copy
import warnings
from typing import Union, Any # For type hints
import json
from config import settings, Config

# Qiskit imports
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.providers.backend import BackendV2
from qiskit_ibm_provider import IBMProvider, IBMBackend
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Aer for simulation
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error
# Qiskit Algorithms & Circuits
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import Parameter

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

# Project imports
from config import settings # Import the global settings instance
from qutils import (
    error_map_to_class_label,
    class_label_to_target_bitstring,
    measurement_to_class_label,
    syndrome_str_to_feature_vector,
    class_label_to_error_map # Import if needed for analysis
)

# --- Seeding ---
if settings.SEED is not None:
    np.random.seed(settings.SEED)
    random.seed(settings.SEED)
    logging.info(f"Global random seeds set to: {settings.SEED}")

# --- Backend Management ---
_qiskit_runtime_service: QiskitRuntimeService | None = None
_backend_instance: BackendV2 | None = None

def get_backend(force_reload=False) -> BackendV2:
    """Initializes and returns the appropriate Qiskit backend based on config."""
    global _backend_instance, _qiskit_runtime_service

    if _backend_instance is not None and not force_reload:
        logging.debug(f"Using cached backend: {_backend_instance.name}")
        return _backend_instance

    mode = settings.DEFAULT_BACKEND_MODE
    target = settings.IBM_TARGET_BACKEND
    ibm_instance = settings.IBM_INSTANCE

    logging.info(f"Initializing backend. Mode: {mode}, Target: {target or 'N/A'}, Instance: {ibm_instance or 'Default'}, Use Least Busy: {settings.USE_LEAST_BUSY_BACKEND}")

    try:
        if mode == 'aer_simulator':
            _backend_instance = AerSimulator()
        elif mode == 'simulator_stabilizer':
            _backend_instance = AerSimulator(method='stabilizer')
        elif mode in ['ibm_simulator', 'ibm_real_device']:
            if not settings.IBM_API_TOKEN:
                raise ValueError("IBM_API_TOKEN not set.")
            if settings.USE_LEAST_BUSY_BACKEND:
                if not _qiskit_runtime_service:
                    service_options = {
                        "channel": 'ibm_quantum',
                        "token": settings.IBM_API_TOKEN
                    }
                    if ibm_instance:
                        service_options["instance"] = ibm_instance
                    _qiskit_runtime_service = QiskitRuntimeService(**service_options)
                    logging.info(f"QiskitRuntimeService initialized. Instance:")

                # Get available operational backends
                #least_busy_backend = _qiskit_runtime_service.least_busy(operational=True, simulator=False, min_num_qubits=<num_qubits>)
                # backends = _qiskit_runtime_service.backends(filters=lambda b: b.status().operational and (not b.simulator if mode == 'ibm_real_device' else b.simulator))
                # if not backends:
                #     raise ValueError("No suitable operational backends found.")

                # # Select the least busy backend
                # least_busy_backend = min(backends, key=lambda b: b.status().pending_jobs)

                least_busy_backend = _qiskit_runtime_service.least_busy(operational=True, simulator=False,  min_num_qubits=120)
                _backend_instance = least_busy_backend
                print("least busy backend ", least_busy_backend)
                logging.info(f"Selected least busy backend: {least_busy_backend.name} with {least_busy_backend.status().pending_jobs} pending jobs.")
            else:
                if not target:
                    raise ValueError("IBM_TARGET_BACKEND not set.")
                if not _qiskit_runtime_service:
                    service_options = {
                        "channel": 'ibm_quantum',
                        "token": settings.IBM_API_TOKEN
                    }
                    if ibm_instance:
                        service_options["instance"] = ibm_instance
                    _qiskit_runtime_service = QiskitRuntimeService(**service_options)
                    logging.info(f"QiskitRuntimeService initialized. Instance: ")

                logging.info(f"Attempting to get IBM backend via service: {target}")
                _backend_instance = _qiskit_runtime_service.backend(target)
                logging.info(f"Successfully obtained backend: {_backend_instance.name} (Max Circuits: {_backend_instance.max_circuits})")

                print(f"Successfully obtained backend: {_backend_instance.name} (Max Circuits: {_backend_instance.max_circuits})")

                if mode == 'ibm_real_device' and _backend_instance.simulator:
                    logging.warning(f"Target backend '{target}' is a simulator, but mode was 'ibm_real_device'.")
                elif mode == 'ibm_simulator' and not _backend_instance.simulator:
                    logging.warning(f"Target backend '{target}' is a real device, but mode was 'ibm_simulator'.")

        else:
            raise ValueError(f"Unsupported backend mode: {mode}")

    except QiskitBackendNotFoundError:
        available_backends_str = "N/A"
        if _qiskit_runtime_service:
            try:
                available_backends_str = str([b.name for b in _qiskit_runtime_service.backends()])
            except Exception:
                pass
        logging.error(f"Backend '{target}' not found for service instance '{ibm_instance}'. Available backends might include: {available_backends_str}")
        raise LookupError(f"IBM Backend '{target}' not found.") from None
    except Exception as e:
        logging.error(f"Failed to initialize backend '{target or mode}': {e}", exc_info=True)
        raise ConnectionError(f"Could not initialize backend: {e}") from e

    logging.info(f"Backend '{_backend_instance.name}' initialized successfully.")
    return _backend_instance

# --- Rest of the file remains unchanged ---
# (Functions like initialize_d3_surface_code, get_d3_stabilizers, etc., are not modified)

# Surface Code Initialization
def initialize_d3_surface_code() -> tuple[dict, list, list, list, dict, dict]:
    """Initializes qubit labels, positions, indices, and QPU map for d=3."""
    d = settings.CODE_DISTANCE
    if d != 3: raise NotImplementedError("This function currently only supports distance d=3.")
    
    data_qubits = {}
    ancilla_qubits = {}
    q_idx = 0
    for r in range(d):
        for c in range(d):
            data_qubits[(r + 0.5, c + 0.5)] = f"D{q_idx}"
            q_idx += 1
    anc_idx = 0
    for r in range(d - 1):
        for c in range(d - 1):
            ancilla_qubits[(r + 1.0, c + 1.0)] = f"AX{anc_idx}"
            anc_idx += 1
    z_anc_map = { # Use the AZ4-AZ7 names consistently
        (0.5, 1.5): f"AZ{anc_idx}", (1.5, 0.5): f"AZ{anc_idx+1}",
        (1.5, 2.5): f"AZ{anc_idx+2}", (2.5, 1.5): f"AZ{anc_idx+3}",
    }
    ancilla_qubits.update(z_anc_map)
    lattice = {**data_qubits, **ancilla_qubits}
    data_q_list = sorted([name for name in data_qubits.values()], key=lambda x: int(x[1:]))
    ancilla_q_list = sorted([name for name in ancilla_qubits.values()], key=lambda x: (x[1], int(x[2:])))
    all_q_list = data_q_list + ancilla_q_list
    qubit_indices = {name: i for i, name in enumerate(all_q_list)}
    logging.info(f"Initialized d={d} code...") # Concise log
    # QPU Assignment
    qpus = [[] for _ in range(settings.NUM_QPUS)]
    num_qpus = settings.NUM_QPUS
    boundary_step = settings.LATTICE_DIM / num_qpus
    qpu_assignment_map = {}
    for pos, name in lattice.items():
        if name not in qubit_indices: continue
        idx = qubit_indices[name]
        qpu_idx = min(num_qpus - 1, int(pos[1] // boundary_step))
        qpus[qpu_idx].append(name)
        qpu_assignment_map[idx] = qpu_idx
    logging.info(f"QPU assignments complete ({settings.NUM_QPUS} QPUs).")
    return lattice, qpus, data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map

# Stabilizer Logic
def get_d3_stabilizers(qubit_indices: dict) -> tuple[list, list]:
    """Returns lists of (ancilla_index, [data_indices]) for X and Z stabilizers."""
    expected_qubits = [f"D{i}" for i in range(9)] + [f"AX{i}" for i in range(4)] + [f"AZ{i}" for i in range(4, 8)]
    missing = [q for q in expected_qubits if q not in qubit_indices]
    if missing: raise ValueError(f"Missing expected qubits for d=3 stabs: {missing}")
    x_stabilizers = [
        (qubit_indices['AX0'], [qubit_indices['D0'], qubit_indices['D1'], qubit_indices['D3'], qubit_indices['D4']]),
        (qubit_indices['AX1'], [qubit_indices['D1'], qubit_indices['D2'], qubit_indices['D4'], qubit_indices['D5']]),
        (qubit_indices['AX2'], [qubit_indices['D3'], qubit_indices['D4'], qubit_indices['D6'], qubit_indices['D7']]),
        (qubit_indices['AX3'], [qubit_indices['D4'], qubit_indices['D5'], qubit_indices['D7'], qubit_indices['D8']]),
    ]
    z_stabilizers = [ # Corrected vertex definitions
        (qubit_indices['AZ4'], [qubit_indices['D1'], qubit_indices['D2']]),
        (qubit_indices['AZ5'], [qubit_indices['D0'], qubit_indices['D3']]),
        (qubit_indices['AZ6'], [qubit_indices['D2'], qubit_indices['D5']]),
        (qubit_indices['AZ7'], [qubit_indices['D7'], qubit_indices['D8']]),
    ]
    logging.info(f"Defined {len(x_stabilizers)} X and {len(z_stabilizers)} Z stabilizers.")
    return x_stabilizers, z_stabilizers

# Remaining functions unchanged below
def apply_stabilizer_measurements(circuit: QuantumCircuit, stabilizers: list, basis: str, clbit_offset: int) -> QuantumCircuit:
    """Appends stabilizer measurement gates to the circuit."""
    num_stabilizers = len(stabilizers)
    if not circuit.cregs:
        raise ValueError("Circuit must have at least one classical register.")
    creg = circuit.cregs[0]
    if len(creg) < clbit_offset + num_stabilizers:
        raise ValueError(f"Classical register '{creg.name}' size ({len(creg)}) insufficient. Need {clbit_offset + num_stabilizers} bits.")

    op_name = f"{basis}Stabs"
    for i, (anc_idx, data_indices) in enumerate(stabilizers):
        creg_idx = clbit_offset + i
        if basis == 'X':
            circuit.h(anc_idx)
            for data_idx in data_indices: circuit.cx(data_idx, anc_idx)
            circuit.h(anc_idx)
        elif basis == 'Z':
            for data_idx in data_indices: circuit.cx(data_idx, anc_idx)
        else: raise ValueError("Basis must be 'X' or 'Z'")
        circuit.measure(anc_idx, creg[creg_idx])
    circuit.barrier(label=f"{basis}Stabs done")
    return circuit

def get_noise_model(num_total_qubits: int, error_rates: list[float], qpu_assignment_map: dict, readout_error_prob: float) -> NoiseModel | None:
    """Creates a Qiskit Aer NoiseModel based on configuration."""
    if not any(rate > 0 for rate in error_rates) and readout_error_prob <= 0:
        logging.info("No noise configured, returning None for noise model.")
        return None
    noise_model = NoiseModel(basis_gates=['id', 'rz', 'sx', 'x', 'cx', 'h', 'reset', 'measure'])
    logging.info("Building noise model...")
    for qubit_idx in range(num_total_qubits):
        qpu_idx = qpu_assignment_map.get(qubit_idx, 0)
        if qpu_idx >= len(error_rates): qpu_idx = 0
        error_rate = error_rates[qpu_idx]
        if error_rate > 0:
            prob_pauli = min(1.0/3.0, error_rate / 3.0)
            effective_rate = prob_pauli * 3
            pauli_channel = pauli_error([('X', prob_pauli), ('Y', prob_pauli), ('Z', prob_pauli), ('I', 1.0 - effective_rate)])
            noise_model.add_quantum_error(pauli_channel, ['x', 'sx', 'h'], [qubit_idx])
    if readout_error_prob > 0:
        p0g1 = min(1.0, max(0.0, readout_error_prob))
        p1g0 = min(1.0, max(0.0, readout_error_prob))
        readout_error = ReadoutError([[1 - p1g0, p1g0], [p0g1, 1 - p0g1]])
        noise_model.add_all_qubit_readout_error(readout_error, warnings=False)
        logging.info(f" Added readout error p(0|1)=p(1|0)={readout_error_prob:.4f}")
    logging.info("Noise model build complete.")
    return noise_model

def inject_single_pauli_error(circuit: QuantumCircuit, data_qubit_indices: list[int], error_prob_overall: float) -> tuple[QuantumCircuit, dict]:
    """Injects at most ONE single Pauli error onto ONE data qubit."""
    injected_errors = {}
    if random.random() < error_prob_overall and data_qubit_indices:
        target_qubit_idx = random.choice(data_qubit_indices)
        pauli_type = random.choice(['X', 'Y', 'Z'])
        if pauli_type == 'X': circuit.x(target_qubit_idx)
        elif pauli_type == 'Y': circuit.y(target_qubit_idx)
        elif pauli_type == 'Z': circuit.z(target_qubit_idx)
        injected_errors[target_qubit_idx] = pauli_type
    return circuit, injected_errors

def generate_labeled_syndrome_data(num_samples: int, data_q_list: list, ancilla_q_list: list,
                                   qubit_indices: dict, qpu_assignment_map: dict,
                                   stabilizers_x: list, stabilizers_z: list,
                                   noise_model: NoiseModel | None) -> list[tuple[str, int]]:
    """Generates labeled syndrome data (noisy_syndrome_str, error_class_label)."""
    dataset = []
    sim_backend = AerSimulator()
    num_total_qubits = len(qubit_indices)
    num_stabilizers = len(stabilizers_x) + len(stabilizers_z)
    data_qubit_indices_list = sorted([qubit_indices[q] for q in data_q_list])
    logging.info(f"Generating {num_samples} labeled syndrome samples...")
    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 3
    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1
        qc_run = QuantumCircuit(num_total_qubits, name=f"Sample_{attempts}")
        cr = ClassicalRegister(num_stabilizers, name="syndrome")
        qc_run.add_register(cr)
        qc_run, injected_errors_map = inject_single_pauli_error(qc_run, data_qubit_indices_list, settings.INJECTED_ERROR_PROB_OVERALL)
        error_class_label = error_map_to_class_label(injected_errors_map, data_qubit_indices_list)
        qc_run.barrier(label="Error")
        qc_run = apply_stabilizer_measurements(qc_run, stabilizers_x, 'X', 0)
        qc_run.barrier(label="Stabs")
        qc_run = apply_stabilizer_measurements(qc_run, stabilizers_z, 'Z', len(stabilizers_x))
        try:
            run_options = {'shots': 1}
            if noise_model: run_options['noise_model'] = noise_model
            if settings.SEED is not None: run_options['seed_simulator'] = settings.SEED + attempts

            pm = generate_preset_pass_manager(backend=sim_backend, optimization_level=1)
            isa_circuit = pm.run(qc_run)

            sampler = Sampler(sim_backend)
            job = sampler.run(isa_circuit, **run_options)
            result = job.result()
            counts = result[0].data.meas.get_counts()

            # t_qc_run = transpile(qc_run, sim_backend)
            # job = sim_backend.run(t_qc_run, **run_options)
            # result = job.result()
            # counts = result.get_counts()
        except Exception as e:
            logging.error(f"Simulation failed attempt {attempts}: {e}", exc_info=True)
            continue
        if not counts:
            logging.warning(f"Sample attempt {attempts} got no counts. Skipping.")
            continue
        syndrome_key = list(counts.keys())[0]
        if ' ' in syndrome_key: syndrome_key = syndrome_key.split(' ')[-1]
        syndrome_key_padded = syndrome_key.zfill(num_stabilizers)
        noisy_syndrome_str = syndrome_key_padded[::-1]
        dataset.append((noisy_syndrome_str, error_class_label))
        generated_count += 1
        if generated_count % max(1, num_samples // 10) == 0:
            logging.info(f" Generated {generated_count}/{num_samples} samples...")
    if generated_count < num_samples: logging.warning(f"Target samples: {num_samples}, Generated: {generated_count}")
    logging.info(f"Syndrome generation complete. Total samples: {len(dataset)}")
    return dataset

def add_syndrome_encoding(circuit: QuantumCircuit, syndrome_str: str) -> QuantumCircuit:
    """Encodes the syndrome string into the circuit using RY rotations."""
    syndrome_bits = [int(bit) for bit in syndrome_str]
    num_encode_qubits = min(circuit.num_qubits, len(syndrome_bits))
    if num_encode_qubits < len(syndrome_bits):
        logging.debug(f"Syndrome len ({len(syndrome_bits)}) > VQC encoding qubits ({num_encode_qubits}). Truncating.")
    for i in range(num_encode_qubits):
        angle = np.pi * syndrome_bits[i]
        if angle != 0: circuit.ry(angle, i)
    return circuit

def create_vqc_ansatz(num_qubits: int, reps: int, entanglement: str = 'linear') -> EfficientSU2:
    """Creates the EfficientSU2 variational form."""
    valid_entanglements = ['full', 'linear', 'circular', 'sca']
    if entanglement not in valid_entanglements:
        logging.warning(f"Unsupported entanglement '{entanglement}'. Using 'linear'.")
        entanglement = 'linear'
    return EfficientSU2(num_qubits, reps=reps, entanglement=entanglement, name=f"EffSU2_{entanglement}_r{reps}")

def create_local_vqc(qpu_qubit_names: list, syndrome_str: str, qpu_idx: int, qubit_indices_global: dict) -> tuple[QuantumCircuit, list[Parameter]]:
    """Creates a local VQC for a given QPU."""
    n_qubits = settings.LOCAL_VQC_QUBITS_PER_QPU
    if n_qubits <= 0: raise ValueError("LOCAL_VQC_QUBITS_PER_QPU must be > 0.")
    vqc_circuit = QuantumCircuit(n_qubits, n_qubits, name=f"LocalVQC_QPU{qpu_idx}")
    vqc_circuit = add_syndrome_encoding(vqc_circuit, syndrome_str)
    var_form = create_vqc_ansatz(n_qubits, settings.VQC_REPS, entanglement='linear')
    vqc_circuit.compose(var_form, inplace=True)
    vqc_circuit.measure_all(add_bits=False)
    return vqc_circuit, list(var_form.parameters)

def create_global_vqc(syndrome_str: str) -> tuple[QuantumCircuit, list[Parameter]]:
    """Creates a global VQC."""
    n_qubits = settings.GLOBAL_VQC_QUBITS
    if n_qubits <= 0: raise ValueError("GLOBAL_VQC_QUBITS must be > 0.")
    vqc_circuit = QuantumCircuit(n_qubits, n_qubits, name="GlobalVQC")
    vqc_circuit = add_syndrome_encoding(vqc_circuit, syndrome_str)
    var_form = create_vqc_ansatz(n_qubits, settings.VQC_REPS, entanglement='circular')
    vqc_circuit.compose(var_form, inplace=True)
    vqc_circuit.measure_all(add_bits=False)
    return vqc_circuit, list(var_form.parameters)

def create_centralized_vqc(syndrome_str: str) -> tuple[QuantumCircuit, list[Parameter]]:
    """Creates a centralized VQC."""
    n_qubits = settings.CENTRALIZED_VQC_QUBITS
    if n_qubits <= 0: raise ValueError("CENTRALIZED_VQC_QUBITS must be > 0.")
    vqc_circuit = QuantumCircuit(n_qubits, n_qubits, name="CentralizedVQC")
    vqc_circuit = add_syndrome_encoding(vqc_circuit, syndrome_str)
    var_form = create_vqc_ansatz(n_qubits, settings.VQC_REPS, entanglement='full')
    vqc_circuit.compose(var_form, inplace=True)
    vqc_circuit.measure_all(add_bits=False)
    return vqc_circuit, list(var_form.parameters)

def cost_function_decoder(params: np.ndarray, circuit_template: QuantumCircuit,
                          syndrome_str: str, target_label: int,
                          backend: BackendV2, shots: int) -> float:
    """Cost function for decoder: -log(Prob(target_bitstring)). Qiskit 1.x compatible."""
    num_vqc_qubits = circuit_template.num_qubits
    target_bitstring = class_label_to_target_bitstring(target_label, num_vqc_qubits)
    if len(params) != len(circuit_template.parameters):
        raise ValueError(f"Param length mismatch: Circuit '{circuit_template.name}' needs {len(circuit_template.parameters)}, got {len(params)}.")
    try:
        param_map = {p: v for p, v in zip(circuit_template.parameters, params)}
        bound_circuit = circuit_template.assign_parameters(param_map)
    except Exception as e:
        logging.error(f"Error binding parameters in cost function: {e}", exc_info=True)
        return -math.log(settings.EPSILON) * 10
    try:
        if isinstance(backend, AerSimulator):
            aer_basis_gates = ['id', 'rz', 'sx', 'x', 'cx', 'h', 'reset', 'measure', 'barrier']
            logging.debug(f"Transpiling for AerSimulator using explicit basis: {aer_basis_gates}")

            pm = generate_preset_pass_manager(backend=backend, basis_gates=aer_basis_gates, optimization_level=1)
            isa_circuit = pm.run(bound_circuit)

            # transpiled_circuit = transpile(
            #     bound_circuit,
            #     backend=backend,
            #     basis_gates=aer_basis_gates,
            #     optimization_level=1
            # )
        else:
            logging.debug(f"Transpiling for backend {backend.name} using its target implicitly.")
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            isa_circuit = pm.run(bound_circuit)

            # transpiled_circuit = transpile(
            #     bound_circuit,
            #     backend=backend,
            #     optimization_level=1
            # )

        run_options = {'shots': shots}
        if isinstance(backend, AerSimulator) and settings.SEED is not None:
            run_options['seed_simulator'] = settings.SEED

        sampler = Sampler(backend)
        job = sampler.run(isa_circuit, **run_options)
        result = job.result()
        counts = result[0].data.meas.get_counts()

        # print(f" > Counts: {result[0].data.meas.get_counts()}")   
        #job = backend.run(transpiled_circuit, **run_options)
    except Exception as e:
        logging.error(f"Error during circuit execution/transpilation in cost function "
                      f"(Label {target_label}, Target {target_bitstring}, Backend {backend.name}): {e}", exc_info=True)
        return -math.log(settings.EPSILON)
    total_shots = sum(counts.values())
    if total_shots == 0:
        target_prob = 0.0
    else:
        target_prob = counts.get(target_bitstring, 0) / total_shots
    cost = -math.log(target_prob + settings.EPSILON)
    if not np.isfinite(cost):
        logging.warning(f"Non-finite cost ({cost}) calculated for Label {target_label}. Clamping.")
        cost = -math.log(settings.EPSILON)
    return cost

def train_vqnn_supervised(circuit_template: QuantumCircuit, initial_params: np.ndarray,
                          training_data: list[tuple[str, int]], backend: BackendV2,
                          callback_func=None) -> tuple[np.ndarray, float, dict]:
    """Trains VQNN decoder using SPSA optimizer."""
    if not circuit_template.parameters:
        logging.warning(f"Circuit '{circuit_template.name}' has no parameters. Skipping SPSA.")
        return np.array(initial_params), 0.0, {'cost': [0.0]}
    if len(initial_params) != len(circuit_template.parameters):
        raise ValueError(f"Param length mismatch for '{circuit_template.name}'.")
    optimizer = SPSA(maxiter=settings.SPSA_MAX_ITER)
    params = np.array(initial_params)
    history = {'cost': []}
    current_iteration = 0

    def objective_function(current_params):
        nonlocal current_iteration
        current_iteration += 1
        batch_size = min(len(training_data), 32)
        if batch_size == 0: return 0.0
        sample_indices = random.sample(range(len(training_data)), batch_size)
        batch_total_cost = 0.0
        valid_samples_in_batch = 0
        for idx in sample_indices:
            syndrome_str, target_label = training_data[idx]
            try:
                cost = cost_function_decoder(current_params, circuit_template, syndrome_str, target_label, backend, settings.SHOTS)
                if not np.isfinite(cost): cost = -math.log(settings.EPSILON) * 10
                batch_total_cost += cost
                valid_samples_in_batch += 1
            except Exception as e:
                logging.error(f"Cost function error iter {current_iteration}, sample {idx}: {e}", exc_info=False)
                batch_total_cost += -math.log(settings.EPSILON) * 10
                valid_samples_in_batch += 1
        avg_cost = batch_total_cost / valid_samples_in_batch if valid_samples_in_batch > 0 else -math.log(settings.EPSILON) * 10
        history['cost'].append(avg_cost)
        if current_iteration % 10 == 0: logging.info(f" SPSA Iter {current_iteration}/{settings.SPSA_MAX_ITER}, Avg Batch Cost: {avg_cost:.4f}")
        if callback_func:
            try: callback_func({'iteration': current_iteration, 'max_iter': settings.SPSA_MAX_ITER, 'cost': avg_cost})
            except Exception as cb_e: logging.error(f"Callback error: {cb_e}", exc_info=False)
        return avg_cost
    
    try:
        logging.info(f"Starting SPSA for '{circuit_template.name}'...")
        result = optimizer.minimize(objective_function, params)
        optimal_params = np.array(result.x)
        final_cost = result.fun
        logging.info(f"Training finished for '{circuit_template.name}'. Final cost: {final_cost:.4f}, Iterations: {result.nit}.")
        return optimal_params, final_cost, history
    except Exception as e:
        logging.error(f"SPSA optimization error for '{circuit_template.name}': {e}", exc_info=True)
        return np.array(initial_params), np.inf, history

def evaluate_decoder(trained_params: np.ndarray, circuit_creator_func: callable,
                     evaluation_data: list[tuple[str, int]], backend: BackendV2) -> tuple[float, list[dict]]:
    """Evaluates the prediction accuracy of a trained VQNN decoder."""
    correct_predictions = 0
    total_samples = len(evaluation_data)
    results_detail = []
    if total_samples == 0: return 0.0, []
    logging.info(f"Evaluating decoder accuracy on {total_samples} samples using backend '{backend.name}'...")
    processed_samples = 0
    start_eval_time = time.time()

    for i, (syndrome_str, true_label) in enumerate(evaluation_data):
        predicted_label = -1; counts = {}; sample_status = "Processed"
        try:
            circuit_instance, circuit_params = circuit_creator_func(syndrome_str)
            num_vqc_qubits = circuit_instance.num_qubits
            if len(trained_params) != len(circuit_params):
                raise ValueError(f"Param length mismatch! Expected {len(circuit_params)}, got {len(trained_params)}.")
            param_map = {p: v for p, v in zip(circuit_params, trained_params)}
            bound_circuit = circuit_instance.assign_parameters(param_map)
            if isinstance(backend, AerSimulator):
                aer_basis_gates = ['id', 'rz', 'sx', 'x', 'cx', 'h', 'reset', 'measure', 'barrier']

                pm = generate_preset_pass_manager(backend=backend, basis_gates=aer_basis_gates, optimization_level=1)
                isa_circuit = pm.run(bound_circuit)

                # transpiled_circuit = transpile(
                #     bound_circuit,
                #     backend=backend,
                #     basis_gates=aer_basis_gates,
                #     optimization_level=1
                # )
            else:
                pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
                isa_circuit = pm.run(bound_circuit)
                # transpiled_circuit = transpile(
                #     bound_circuit,
                #     backend=backend,
                #     optimization_level=1
                # )

            run_options = {'shots': settings.EVAL_SHOTS}
            if isinstance(backend, AerSimulator) and settings.SEED is not None:
                run_options['seed_simulator'] = settings.SEED + i + total_samples


            sampler = Sampler(backend)
            job = sampler.run(isa_circuit, **run_options)
            result = job.result()
            counts = result[0].data.meas.get_counts()

            # print(f" > Counts: {result[0].data.meas.get_counts()}")   
            #job = backend.run(transpiled_circuit, **run_options)
            #counts = job.result().get_counts()

                
            predicted_label = measurement_to_class_label(counts, num_vqc_qubits, settings.NUM_ERROR_CLASSES)
            if predicted_label == true_label:
                correct_predictions += 1
            processed_samples += 1
            if (i + 1) % max(1, total_samples // 10) == 0: logging.info(f"  Evaluated {i+1}/{total_samples} samples...")
        except Exception as e:
            error_msg = f"Error evaluating sample {i} (Syndrome: {syndrome_str}, Label: {true_label}): {e}"
            logging.error(error_msg, exc_info=False)
            sample_status = f"Error: {str(e)[:100]}..."
            processed_samples += 1
        results_detail.append({
            'sample_index': i, 'syndrome': syndrome_str, 'true_label': true_label,
            'predicted_label': predicted_label, 'status': sample_status, 'counts': counts
        })
    eval_duration = time.time() - start_eval_time
    accuracy = correct_predictions / processed_samples if processed_samples > 0 else 0.0
    logging.info(f"Evaluation complete. Successfully processed: {processed_samples}/{total_samples}. Correct: {correct_predictions}. Accuracy: {accuracy:.4f}. Duration: {eval_duration:.2f}s")
    return accuracy, results_detail

def create_classical_models() -> dict:
    """Returns a dictionary of configured scikit-learn models."""
    models = {}
    models['classical_logreg'] = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=300, C=1.0, random_state=settings.SEED)
    models['classical_mlp'] = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.001, max_iter=500, random_state=settings.SEED, early_stopping=True, n_iter_no_change=10, learning_rate_init=0.001)
    logging.info(f"Defined classical models: {list(models.keys())}")
    return models

def train_classical_model(model_name: str, model, training_features: np.ndarray, training_labels: np.ndarray) -> tuple[Any, float]:
    """Trains a scikit-learn model."""
    logging.info(f" Training classical model: {model_name}...")
    start_train_time = time.time()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            model.fit(training_features, training_labels)
        train_time = time.time() - start_train_time
        logging.info(f" Training complete for {model_name} in {train_time:.2f}s")
        return model, train_time
    except Exception as e:
        logging.error(f"Error training classical model {model_name}: {e}", exc_info=True)
        return None, time.time() - start_train_time

def evaluate_classical_model(model_name: str, model, evaluation_features: np.ndarray, evaluation_labels: np.ndarray) -> tuple[float | None, float, list | None]:
    """Evaluates a trained scikit-learn model."""
    logging.info(f" Evaluating classical model: {model_name}...")
    if model is None: return None, 0.0, None
    start_eval_time = time.time()
    try:
        predictions = model.predict(evaluation_features)
        accuracy = accuracy_score(evaluation_labels, predictions)
        eval_time = time.time() - start_eval_time
        logging.info(f" Evaluation complete for {model_name}. Accuracy: {accuracy:.4f}, Time: {eval_time:.2f}s")
        return accuracy, eval_time, predictions.tolist()
    except Exception as e:
        logging.error(f"Error evaluating classical model {model_name}: {e}", exc_info=True)
        return None, time.time() - start_eval_time, None

def run_decoder_experiment_instance(config_override: dict | None = None, training_callback=None) -> dict:
    """Runs the full decoder training and evaluation pipeline."""
    current_settings = Config(**(config_override or {}))
    if current_settings.SEED is not None:
        np.random.seed(current_settings.SEED)
        random.seed(current_settings.SEED)
    start_overall_time = time.time()
    logging.info("="*30 + f"\n=== STARTING DECODER RUN (Seed: {current_settings.SEED}) ===\n" + "="*30)
    loggable_config = {k:v for k,v in vars(current_settings).items() if k != 'IBM_API_TOKEN' and not k.startswith('_')}
    logging.info(f"Effective Run Configuration:\n{json.dumps(loggable_config, indent=2)}")
    results = {
        'architectures': {}, 'execution_times': {'overall': 0, 'data_gen': 0, 'training': {}, 'evaluation': {}},
        'setup_info': {'config_used': loggable_config}, 'error': None, 'summary': {}
    }
    arch_results = results['architectures']
    exec_times = results['execution_times']
    try:
        backend = get_backend(force_reload=True)
        results['setup_info']['backend_name'] = backend.name
        print("backend name: ", backend.name)
        lattice, qpus, data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map = initialize_d3_surface_code()
        num_total_qubits = len(qubit_indices)
        results['setup_info'].update({
            'num_data_qubits': len(data_q_list), 'num_ancilla_qubits': len(ancilla_q_list),
            'num_total_qubits': num_total_qubits, 'num_error_classes': current_settings.NUM_ERROR_CLASSES,
            'min_vqc_qubits_needed': current_settings.MIN_VQC_QUBITS_NEEDED,
            'lattice_qubits': list(lattice.values()), 'qpu_assignments_list': qpus
        })
        noise_model = get_noise_model(num_total_qubits, current_settings.ERROR_RATES, qpu_assignment_map, current_settings.READOUT_ERROR_PROB) if current_settings.DEFAULT_BACKEND_MODE == 'aer_simulator' else None
        stabilizers_x, stabilizers_z = get_d3_stabilizers(qubit_indices)
    except Exception as e:
        results['error'] = f"Setup failed: {e}"; logging.error(results['error'], exc_info=True); return results
    start_data_time = time.time()
    data_qubit_indices_list = sorted([qubit_indices[q] for q in data_q_list])
    try:
        training_data = generate_labeled_syndrome_data(current_settings.NUM_SYNDROME_SAMPLES_TRAIN, data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map, stabilizers_x, stabilizers_z, noise_model)
        evaluation_data = generate_labeled_syndrome_data(current_settings.NUM_SYNDROME_SAMPLES_EVAL, data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map, stabilizers_x, stabilizers_z, noise_model)
    except Exception as e:
        results['error'] = f"Data generation failed: {e}"; logging.error(results['error'], exc_info=True); return results
    exec_times['data_gen'] = time.time() - start_data_time
    logging.info(f"Data generation complete ({exec_times['data_gen']:.2f}s). Train: {len(training_data)}, Eval: {len(evaluation_data)}")
    if not training_data or not evaluation_data:
        results['error'] = "No training or evaluation data generated."; logging.error(results['error']); return results
    try:
        X_train = np.array([syndrome_str_to_feature_vector(s) for s, l in training_data])
        y_train = np.array([l for s, l in training_data])
        X_eval = np.array([syndrome_str_to_feature_vector(s) for s, l in evaluation_data])
        y_eval = np.array([l for s, l in evaluation_data])
    except Exception as e:
        logging.error(f"Failed to prepare classical data: {e}", exc_info=True)
    model_definitions = {
        "centralized_vqnn": {"creator": lambda syndrome: create_centralized_vqc(syndrome), "config_qubits": current_settings.CENTRALIZED_VQC_QUBITS, "type": "vqnn"},
        "distributed_vqnn": {"creator": lambda syndrome: create_global_vqc(syndrome), "config_qubits": current_settings.GLOBAL_VQC_QUBITS, "type": "vqnn"},
        "localized_vqnn": {"is_local": True, "config_qubits": current_settings.LOCAL_VQC_QUBITS_PER_QPU, "type": "vqnn"},
        "classical_logreg": {"is_classical": True, "type": "classical"},
        "classical_mlp": {"is_classical": True, "type": "classical"},
        "baseline_predict_identity": {"is_baseline": True, "type": "baseline"}
    }
    classical_model_instances = create_classical_models()
    model_definitions['classical_logreg']['instance'] = classical_model_instances['classical_logreg']
    model_definitions['classical_mlp']['instance'] = classical_model_instances['classical_mlp']
    trained_classical_models = {}
    for arch_name, definition in model_definitions.items():
        model_type = definition['type']
        clean_arch_name = arch_name.replace('_decoder', '').replace('_vqnn', '')
        logging.info(f"\n--- Processing: {clean_arch_name} ({model_type}) ---")
        arch_results[clean_arch_name] = {'params': None, 'accuracy': None, 'training_history': None, 'status': 'Pending', 'error': None, 'eval_details': None, 'type': model_type}
        exec_times['training'][clean_arch_name] = 0
        exec_times['evaluation'][clean_arch_name] = 0
        if definition.get("is_baseline"): continue
        if definition.get("is_classical") and X_train is None:
            arch_results[clean_arch_name]['status'] = 'Skipped (Classical Data Failed)'
            logging.warning(f"Skipping {clean_arch_name} due to classical data prep failure.")
            continue
        if model_type == 'vqnn':
            min_q = current_settings.MIN_VQC_QUBITS_NEEDED
            config_q = definition.get("config_qubits")
            if config_q < min_q:
                warn_msg = f"Config qubits ({config_q}) < min required ({min_q})."
                arch_results[clean_arch_name]['status'] = 'Skipped (Insufficient Qubits)'
                arch_results[clean_arch_name]['error'] = warn_msg
                logging.warning(f"Skipping {clean_arch_name}: {warn_msg}")
                continue
        start_train_time = time.time()
        arch_results[clean_arch_name]['status'] = 'Training'
        try:
            if model_type == 'vqnn':
                def specific_callback(info):
                    if training_callback: info['arch'] = clean_arch_name; training_callback(info)
                if definition.get("is_local", False):
                    optimal_params_list = [] ; history_list = []
                    for qpu_idx, qpu_names in enumerate(qpus):
                        dummy_s = '0' * settings.NUM_ANCILLA_QUBITS
                        tmpl_c, tmpl_p = create_local_vqc(qpu_names, dummy_s, qpu_idx, qubit_indices)
                        if not tmpl_p: opt_p, hist = np.array([]), {'cost': []}
                        else:
                            init_p = get_initial_params(tmpl_p)
                            def local_cb(info):
                                if training_callback: info['arch']=clean_arch_name; info['detail']=f"QPU {qpu_idx}"; training_callback(info)
                            opt_p, _, hist = train_vqnn_supervised(tmpl_c, init_p, training_data, backend, callback_func=local_cb)
                        optimal_params_list.append(opt_p.tolist())
                        history_list.append(hist)
                    arch_results[clean_arch_name]['params'] = optimal_params_list
                    arch_results[clean_arch_name]['training_history'] = history_list
                else:
                    creator = definition["creator"]
                    dummy_s = '0' * settings.NUM_ANCILLA_QUBITS
                    tmpl_c, tmpl_p = creator(dummy_s)
                    if not tmpl_p: opt_p, hist = np.array([]), {'cost': []}
                    else:
                        init_p = get_initial_params(tmpl_p)
                        opt_p, _, hist = train_vqnn_supervised(tmpl_c, init_p, training_data, backend, callback_func=specific_callback)
                    arch_results[clean_arch_name]['params'] = opt_p.tolist()
                    arch_results[clean_arch_name]['training_history'] = hist
            elif model_type == 'classical':
                model_instance = definition['instance']
                trained_model, train_time = train_classical_model(clean_arch_name, model_instance, X_train, y_train)
                if trained_model is None: raise RuntimeError("Classical training function returned None.")
                trained_classical_models[clean_arch_name] = trained_model
            exec_times['training'][clean_arch_name] = time.time() - start_train_time
            arch_results[clean_arch_name]['status'] = 'Trained'
            logging.info(f" Training complete ({exec_times['training'][clean_arch_name]:.2f}s)")
        except Exception as train_err:
            error_msg = f"Training failed: {train_err}"
            logging.error(f" {error_msg}", exc_info=True)
            arch_results[clean_arch_name]['status'] = 'Error (Training)'
            arch_results[clean_arch_name]['error'] = error_msg
            exec_times['training'][clean_arch_name] = time.time() - start_train_time
            continue
        start_eval_time = time.time()
        arch_results[clean_arch_name]['status'] = 'Evaluating'
        logging.info(f" Evaluating {clean_arch_name}...")
        try:
            accuracy = None; details = None
            if model_type == 'vqnn':
                if definition.get("is_local", False):
                    accuracies_local = []
                    for qpu_idx, qpu_names in enumerate(qpus):
                        def local_eval_creator(syndrome_str_arg): return create_local_vqc(qpu_names, syndrome_str_arg, qpu_idx, qubit_indices)
                        qpu_params = np.array(arch_results[clean_arch_name]['params'][qpu_idx])
                        acc, det = evaluate_decoder(qpu_params, local_eval_creator, evaluation_data, backend)
                        accuracies_local.append(acc)
                    accuracy = np.mean(accuracies_local) if accuracies_local else 0.0
                else:
                    trained_params = np.array(arch_results[clean_arch_name]['params'])
                    eval_creator_func = definition["creator"]
                    accuracy, details = evaluate_decoder(trained_params, eval_creator_func, evaluation_data, backend)
            elif model_type == 'classical':
                trained_model = trained_classical_models.get(clean_arch_name)
                if trained_model:
                    accuracy, _, predictions = evaluate_classical_model(clean_arch_name, trained_model, X_eval, y_eval)
            if accuracy is not None:
                arch_results[clean_arch_name]['accuracy'] = accuracy
                arch_results[clean_arch_name]['status'] = 'Complete'
            else:
                raise RuntimeError("Evaluation function returned None accuracy.")
            exec_times['evaluation'][clean_arch_name] = time.time() - start_eval_time
            logging.info(f" Evaluation complete ({exec_times['evaluation'][clean_arch_name]:.2f}s). Accuracy: {accuracy:.4f}")
        except Exception as eval_err:
            error_msg = f"Evaluation failed: {eval_err}"
            logging.error(f" {error_msg}", exc_info=True)
            arch_results[clean_arch_name]['status'] = 'Error (Evaluation)'
            arch_results[clean_arch_name]['error'] = error_msg
            exec_times['evaluation'][clean_arch_name] = time.time() - start_eval_time
    base_name = 'baseline_predict_identity'
    logging.info(f"\n--- Calculating {base_name} ---")
    arch_results[base_name] = model_definitions[base_name]
    arch_results[base_name]['status'] = 'Calculating'
    exec_times['training'][base_name] = 0
    exec_times['evaluation'][base_name] = 0
    try:
        eval_labels = y_eval if y_eval is not None else [label for _, label in evaluation_data]
        if isinstance(eval_labels, np.ndarray) and eval_labels.size > 0:
            num_eval_samples = eval_labels.size
            identity_correct = np.sum(eval_labels == 0)
            baseline_accuracy = identity_correct / num_eval_samples
            arch_results[base_name]['accuracy'] = baseline_accuracy
            arch_results[base_name]['status'] = 'Complete'
            logging.info(f" Baseline Accuracy: {baseline_accuracy:.4f}")
        elif isinstance(eval_labels, list) and len(eval_labels) > 0:
            num_eval_samples = len(eval_labels)
            identity_correct = sum(1 for label in eval_labels if label == 0)
            baseline_accuracy = identity_correct / num_eval_samples
            arch_results[base_name]['accuracy'] = baseline_accuracy
            arch_results[base_name]['status'] = 'Complete'
            logging.info(f" Baseline Accuracy: {baseline_accuracy:.4f}")
        else:
            raise ValueError("No evaluation labels available for baseline calculation.")
    except Exception as base_err:
        error_msg = f"Baseline calculation failed: {base_err}"
        logging.error(error_msg, exc_info=True)
        arch_results[base_name]['status'] = 'Error'
        arch_results[base_name]['error'] = error_msg
        arch_results[base_name]['accuracy'] = None
    exec_times['overall'] = time.time() - start_overall_time
    logging.info("="*30 + f"\n=== Run Completed in {exec_times['overall']:.2f} seconds ===\n" + "="*30)
    results['summary'] = { name: {"status": data.get('status'), "accuracy": data.get('accuracy'), "train_time": exec_times['training'].get(name), "eval_time": exec_times['evaluation'].get(name), "type": data.get('type'), "error": data.get('error')} for name, data in arch_results.items() }
    logging.info("Final Summary:")
    for name, summary_data in sorted(results['summary'].items()):
        acc_str = f"{summary_data.get('accuracy', 0.0):.4f}" if isinstance(summary_data.get('accuracy'), float) else 'N/A'
        logging.info(f"  {name:<20}: Acc={acc_str}, Status={summary_data.get('status')}, Type={summary_data.get('type')}")
    return results

def get_initial_params(circuit_params: list[Parameter]) -> np.ndarray:
    """Generates random initial parameters (angles) for a VQC."""
    num_params = len(circuit_params)
    if num_params == 0:
        return np.array([])
    initial_values = np.random.uniform(-np.pi, np.pi, num_params)
    return initial_values