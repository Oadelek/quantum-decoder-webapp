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
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService 
from qiskit_ibm_runtime.exceptions import IBMRuntimeError
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
# To make sure qutils can be found (e.g., same directory or added to sys.path)
try:
    from qutils import (
        error_map_to_class_label,
        class_label_to_target_bitstring,
        measurement_to_class_label,
        syndrome_str_to_feature_vector,
        class_label_to_error_map 
    )
except ImportError:
    logging.error("Failed to import from qutils. Ensure qutils.py is accessible.")
    raise

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
    use_least_busy = settings.USE_LEAST_BUSY_BACKEND

    logging.info(f"Initializing backend. Mode: {mode}, Target: {target or 'N/A'}, Instance: {ibm_instance or 'Default'}, Use Least Busy: {use_least_busy}")

    try:
        if mode == 'aer_simulator':
            _backend_instance = AerSimulator()
            logging.info(f"Initialized AerSimulator.")
        elif mode == 'simulator_stabilizer':
            # Note: Stabilizer simulator might not be able to support all noise models or operations
            _backend_instance = AerSimulator(method='stabilizer')
            logging.info(f"Initialized AerSimulator (stabilizer method).")
        elif mode in ['ibm_simulator', 'ibm_real_device']:
            if not settings.IBM_API_TOKEN:
                raise ValueError("IBM_API_TOKEN not set for IBM backend mode.")

            if not _qiskit_runtime_service or force_reload:
                logging.info("Initializing QiskitRuntimeService...")
                service_options = {
                    "channel": 'ibm_quantum',
                    "token": settings.IBM_API_TOKEN
                }
                if ibm_instance:
                    service_options["instance"] = ibm_instance
                try:
                    _qiskit_runtime_service = QiskitRuntimeService(**service_options)
                    logging.info(f"QiskitRuntimeService initialized targeting instance: {ibm_instance or 'Default/Not Specified'}")

                except Exception as service_err:
                     logging.error(f"Failed to initialize QiskitRuntimeService: {service_err}", exc_info=True)
                     raise ConnectionError(f"QiskitRuntimeService initialization failed: {service_err}") from service_err


            if use_least_busy:
                # Calculate minimum qubits needed based on config
                # Surface code + VQC (take max needed by any single circuit)
                surface_code_qubits = settings.NUM_DATA_QUBITS + settings.NUM_ANCILLA_QUBITS
                min_req_qubits = max(
                    surface_code_qubits,
                    settings.CENTRALIZED_VQC_QUBITS,
                    settings.GLOBAL_VQC_QUBITS,
                    settings.LOCAL_VQC_QUBITS_PER_QPU
                )
                logging.info(f"Searching for least busy backend with >= {min_req_qubits} qubits...")
                print("min qubits is: ", min_req_qubits)
                try:
                    # Filter for operational, non-simulator if real device needed
                    is_simulator = (mode == 'ibm_simulator')
                    print("simulator?? ", is_simulator)
                    least_busy_backend = _qiskit_runtime_service.least_busy(
                        operational=True,
                        simulator=is_simulator,
                        min_num_qubits=min_req_qubits
                    )
                    _backend_instance = least_busy_backend
                    # Check if status attribute exists and has pending_jobs
                    pending_jobs = "N/A"
                    if hasattr(least_busy_backend, 'status') and callable(least_busy_backend.status) and hasattr(least_busy_backend.status(), 'pending_jobs'):
                         pending_jobs = least_busy_backend.status().pending_jobs

                    logging.info(f"Selected least busy backend: {least_busy_backend.name} (Simulator: {least_busy_backend.simulator}, Pending Jobs: {pending_jobs})")
                except IBMRuntimeError as e:
                    logging.error(f"Error finding least busy backend: {e}. Trying available backends...", exc_info=True)
                    # Fallback or re-raise depending on desired robustness
                    available_sims = [b.name for b in _qiskit_runtime_service.backends(simulator=True, operational=True)]
                    available_reals = [b.name for b in _qiskit_runtime_service.backends(simulator=False, operational=True)]
                    logging.info(f"Available Simulators: {available_sims}")
                    logging.info(f"Available Real Devices: {available_reals}")
                    raise LookupError(f"Could not find a suitable least busy backend: {e}") from e
                except Exception as e: # Catch other potential errors
                     logging.error(f"Unexpected error finding least busy backend: {e}", exc_info=True)
                     raise LookupError(f"Could not find a suitable least busy backend: {e}") from e

            else: # Use specific target backend
                if not target:
                    raise ValueError("IBM_TARGET_BACKEND must be set if USE_LEAST_BUSY_BACKEND is False.")
                logging.info(f"Attempting to get specific IBM backend via service: {target}")
                try:
                    _backend_instance = _qiskit_runtime_service.backend(target)
                    logging.info(f"Successfully obtained backend: {_backend_instance.name} (Simulator: {_backend_instance.simulator}, Max Circuits: {_backend_instance.max_circuits})")

                    # Warning if mode mismatches backend type
                    # Likely wont
                    if mode == 'ibm_real_device' and _backend_instance.simulator:
                        logging.warning(f"Target backend '{target}' is a simulator, but mode is 'ibm_real_device'.")
                    elif mode == 'ibm_simulator' and not _backend_instance.simulator:
                        logging.warning(f"Target backend '{target}' is a real device, but mode is 'ibm_simulator'.")

                except IBMRuntimeError as e:
                    logging.error(f"Backend '{target}' not found or error accessing it via runtime service: {e}", exc_info=True)
                    available_sims = [b.name for b in _qiskit_runtime_service.backends(simulator=True, operational=True)]
                    available_reals = [b.name for b in _qiskit_runtime_service.backends(simulator=False, operational=True)]
                    logging.info(f"Available Simulators: {available_sims}")
                    logging.info(f"Available Real Devices: {available_reals}")
                    raise LookupError(f"IBM Backend '{target}' not found or unavailable via QiskitRuntimeService.") from e

        else:
            raise ValueError(f"Unsupported backend mode: {mode}")

    except Exception as e:
        # Catch-all for unexpected errors during initialization
        logging.error(f"Failed to initialize backend '{target or mode}': {e}", exc_info=True)

        # Try to list available backends if service was initialized
        available_backends_str = "N/A (Service not initialized)"
        if _qiskit_runtime_service:
            try:
                sims = [b.name for b in _qiskit_runtime_service.backends(simulator=True)]
                reals = [b.name for b in _qiskit_runtime_service.backends(simulator=False)]
                available_backends_str = f"Sims: {sims}, Reals: {reals}"
            except Exception as list_err:
                available_backends_str = f"N/A (Error listing backends: {list_err})"
        logging.error(f"Context: Mode='{mode}', Target='{target}', Instance='{ibm_instance}'. Available backends might include: {available_backends_str}")
        # Reraise as a more specific error type if possible, or a generic one
        if isinstance(e, (LookupError, ValueError, ConnectionError)):
            raise e
        else:
            raise ConnectionError(f"Could not initialize backend due to an unexpected error: {e}") from e


    logging.info(f"Backend '{_backend_instance.name}' initialized successfully.")
    return _backend_instance


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
        # print(f"Injecting {pauli_type} on qubit {target_qubit_idx}")
        # input("stopping after injection")

    return circuit, injected_errors

def generate_labeled_syndrome_data(num_samples: int, data_q_list: list, ancilla_q_list: list,
                                   qubit_indices: dict, qpu_assignment_map: dict,
                                   stabilizers_x: list, stabilizers_z: list,
                                   noise_model: NoiseModel | None) -> list[tuple[str, int]]:
    """Generates labeled syndrome data (noisy_syndrome_str, error_class_label). Uses AerSimulator directly."""
    dataset = []
    # Use AerSimulator directly for data generation as it's much faster.
    # Apply noise model if provided.
    sim_backend_options = {}

    if noise_model:
        # Stabilizer method might not support full noise model; default method usually needed.
        # Check if noise model has instructions not supported by stabilizer sim
        requires_density_matrix = any(isinstance(err, (ReadoutError)) for q_errs in noise_model._local_quantum_errors.values() for err in q_errs) or \
                                  noise_model._local_readout_errors 
 
        if requires_density_matrix:
             sim_backend = AerSimulator(**sim_backend_options) # Default method handles noise
             logging.info("Using AerSimulator (default method) for data generation due to noise model.")
             sim_backend.set_options(noise_model=noise_model)
        else:
             # If only Pauli errors, stabilizer might be faster if compatible
             try:
                sim_backend = AerSimulator(method='stabilizer', **sim_backend_options)
                sim_backend.set_options(noise_model=noise_model)
                logging.info("Using AerSimulator (stabilizer method) with compatible noise model.")
             except Exception: # Fallback if stabilizer + noise fails
                sim_backend = AerSimulator(**sim_backend_options)
                sim_backend.set_options(noise_model=noise_model)
                logging.info("Fallback to AerSimulator (default method) for data gen with noise.")
    else:
        sim_backend = AerSimulator(method='stabilizer', **sim_backend_options) # No noise, stabilizer is fast
        logging.info("Using AerSimulator (stabilizer method) for data generation (no noise).")


    num_total_qubits = len(qubit_indices)
    num_stabilizers = len(stabilizers_x) + len(stabilizers_z)
    data_qubit_indices_list = sorted([qubit_indices[q] for q in data_q_list])

    logging.info(f"Generating {num_samples} labeled syndrome samples using {sim_backend.name}...")
    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 5 # Increased max attempts

    # Setup SamplerV2 with the chosen AerSimulator backend
    sampler = Sampler(sim_backend)
    # Pre-generate pass manager
    pm = generate_preset_pass_manager(backend=sim_backend, optimization_level=1)

    circuits_to_run = []
    target_labels = []
    seed_values = [] # Store seeds per circuit if needed

    # print(f"target labels: {target_labels}  length of target labels: {len(target_labels)}")
    # print(f"generated_count {generated_count} num_samples: {num_samples} attempts: {attempts} max_attempts: {max_attempts}")
    # Prepare all circuits first
    prep_start_time = time.time()
    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1
        qc_run = QuantumCircuit(num_total_qubits, name=f"Sample_{attempts}")
        cr = ClassicalRegister(num_stabilizers, name="syndrome")
        qc_run.add_register(cr)

        qc_run, injected_errors_map = inject_single_pauli_error(qc_run, data_qubit_indices_list, settings.INJECTED_ERROR_PROB_OVERALL)
        
        print(f"injected errors map: {injected_errors_map},data_qubit_indices_list {data_qubit_indices_list} ")
        error_class_label = error_map_to_class_label(injected_errors_map, data_qubit_indices_list)
        print(f"error class label: {error_class_label}")
        input("rahhhhhh")

        qc_run.barrier(label="Error")
        qc_run = apply_stabilizer_measurements(qc_run, stabilizers_x, 'X', 0)
        # No barrier needed between X/Z stab measurements typically
        qc_run = apply_stabilizer_measurements(qc_run, stabilizers_z, 'Z', len(stabilizers_x))

        try:
            # Transpile here before adding to list
            isa_circuit = pm.run(qc_run)
            circuits_to_run.append(isa_circuit)
            #print("circuits to run is given: ", isa_circuit)
            #input("stopping after inpjut")
            target_labels.append(error_class_label)
            if settings.SEED is not None:
                # Generate a unique seed for each simulation run
                 seed_values.append(settings.SEED + attempts)
            generated_count += 1

        except Exception as e:
            logging.warning(f"Failed to prepare or transpile circuit attempt {attempts}: {e}", exc_info=False)
            continue # Skip this attempt

        if generated_count % max(1, num_samples // 10) == 0:
            logging.info(f" Prepared {generated_count}/{num_samples} circuits for generation...")

    logging.info(f"Circuit preparation took {time.time() - prep_start_time:.2f}s. Running {len(circuits_to_run)} circuits.")

    if not circuits_to_run:
        logging.error("No circuits were successfully prepared for data generation.")
        return []
    
    print(f"generated_count {generated_count} num_samples: {num_samples} attempts: {attempts} max_attempts: {max_attempts}")
    print(f"target labels: {target_labels}  length of target labels: {len(target_labels)} length of circuit {len(circuits_to_run)}")
    input("PAUSEDDDD")

    # Run simulations in batches if many circuits
    total_circuits = len(circuits_to_run)
    batch_size = min(500, total_circuits)  # Adjust batch size based on memory/performance
    run_start_time = time.time()
    for i in range(0, len(circuits_to_run), batch_size):
        batch_circuits = circuits_to_run[i:i+batch_size]
        batch_labels = target_labels[i:i+batch_size]
        batch_seeds = seed_values[i:i+batch_size] if seed_values else []
        logging.info(f" Running batch {i//batch_size + 1}/{(len(circuits_to_run) + batch_size - 1)//batch_size} (size {len(batch_circuits)})...")

        try:
            run_options = {'shots': 1} # 1 shot per sample is standard for syndrome extraction

            # Set seed for the batch if applicable
            if isinstance(sim_backend, AerSimulator) and settings.SEED is not None and batch_seeds:
                # Qiskit Aer Sampler takes seed in run options now
                # Using a single seed for the batch run might be sufficient for Aer's handling
                # Or pass per-circuit seeds if the interface supports it (check specific Aer version)
                # Let's try setting a base seed for the batch run via options

                sim_backend.set_options(seed_simulator=batch_seeds[0]) # Set seed for the backend instance
                logging.debug(f"Set AerSimulator seed to {batch_seeds[0]} for batch.")


            # Run the batch of circuits
            # SamplerV2 takes list of circuits, optional params, optional shots override per circuit
            # We are using default shots=1 for all. No parameters needed here.
            job = sampler.run(batch_circuits, **run_options)
            results_batch = job.result() # This is a PrimitiveResult

            # Process results for the batch
            for j, pub_result in enumerate(results_batch):
                 # Access data for the j-th circuit in the batch
                 counts = pub_result.data.syndrome.get_counts()   # we use synndrome here because that is the name of the classical register
                 if not counts:
                     logging.warning(f"Sample attempt (original index ~{i+j}) got no counts. Skipping.")
                     continue

                 # Assuming the key is the measurement outcome '0bxxxx' or 'xxxx'
                 # Let's take the first (and only, for shots=1) outcome
                 syndrome_key_bin = list(counts.keys())[0]

                 print(f"syndrome {syndrome_key_bin}")
                 print(f"num of stabilizers {num_stabilizers}")
                 print(f"batch label: {batch_labels[j]}")
                 input("Paused. Press Enter to continue...")


                 # Remove potential '0b' prefix if present
                 if syndrome_key_bin.startswith('0b'):
                       syndrome_key = syndrome_key_bin[2:]
                 else:
                     syndrome_key = syndrome_key_bin

                 # Pad with leading zeros if necessary
                 syndrome_key_padded = syndrome_key.zfill(num_stabilizers)
                 noisy_syndrome_str = syndrome_key_padded
                 dataset.append((noisy_syndrome_str, batch_labels[j]))
        except Exception as e:
            logging.error(f"Simulation batch failed (starting index {i}): {e}", exc_info=True)

    logging.info(f"Syndrome generation runs took {time.time() - run_start_time:.2f}s.")

    if len(dataset) < num_samples:
        logging.warning(f"Target samples: {num_samples}, Successfully generated: {len(dataset)}")
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
    """Cost function for decoder: -log(Prob(target_bitstring)). Qiskit 1.x / SamplerV2 compatible."""
    num_vqc_qubits = circuit_template.num_qubits
    target_bitstring = class_label_to_target_bitstring(target_label, num_vqc_qubits)

    if len(params) != len(circuit_template.parameters):
        # Log detailed info for debugging
        logging.error(f"Param length mismatch in cost_function_decoder:"
                      f" Circuit '{circuit_template.name}' needs {len(circuit_template.parameters)} params "
                      f"(Params: {circuit_template.parameters}), got {len(params)} params.")
        # Return a high cost to penalize this state
        return -math.log(settings.EPSILON) * 100 # Significantly high cost


    try:
        # Check for unbound parameters BEFORE binding
        unbound_params = [p for p in circuit_template.parameters if p not in circuit_template.parameters] # This logic seems wrong - should check against provided params dict keys implicitly
        if len(params) != len(circuit_template.parameters): # Re-check just in case
             raise ValueError(f"Parameter count mismatch. Circuit: {len(circuit_template.parameters)}, Provided: {len(params)}")

        # Parameter binding: Use assign_parameters which is generally robust
        # Ensure params are in the same order as circuit_template.parameters
        param_map = dict(zip(circuit_template.parameters, params))
        bound_circuit = circuit_template.assign_parameters(param_map)

    except Exception as e:
        logging.error(f"Error binding parameters in cost function for circuit '{circuit_template.name}': {e}", exc_info=True)
        logging.error(f"Circuit parameters: {circuit_template.parameters}")
        logging.error(f"Provided params shape: {params.shape}, first few: {params[:5]}")
        # Return high cost
        return -math.log(settings.EPSILON) * 100

    try:
        # Transpilation using Preset Pass Manager (Correct for Qiskit 1.3.1 but may break with change in version(beware of Qiskit!))
        logging.debug(f"Transpiling circuit '{bound_circuit.name}' for backend '{backend.name}' in cost function.")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1) # Opt level 1 is usually good balance
        isa_circuit = pm.run(bound_circuit)
        logging.debug(f"Transpilation complete for '{bound_circuit.name}'.")

        # Setup SamplerV2
        sampler = Sampler(backend) # Use the provided backend

        run_options = {'shots': shots}
        # Seed handling: Set on backend instance if Aer, otherwise ignore for Runtime
        if isinstance(backend, AerSimulator) and settings.SEED is not None:
            # Using a fixed seed per cost evaluation might be okay, or vary it slightly
            # Let's use a base seed for reproducibility within an optimization step
            current_seed = settings.SEED # Or add hash(params.tobytes()) etc. if necessary
            backend.set_options(seed_simulator=current_seed)
            logging.debug(f"Set AerSimulator seed to {current_seed} for cost evaluation.")

        # Execute using SamplerV2
        logging.debug(f"Running SamplerV2 job for cost function (Shots: {shots})...")
        job = sampler.run([isa_circuit], **run_options) # Pass circuit as a list
        result = job.result() # Get PrimitiveResult
        logging.debug(f"SamplerV2 job completed.")

        # Extract counts from the first (and only) PubResult
        pub_result = result[0]
        counts = pub_result.data.c.get_counts()      # We use c here because that is the name of the classical register
        logging.debug(f"Counts obtained: {counts}")
    except Exception as e:
        logging.error(f"Error during circuit execution/transpilation in cost function "
                      f"(Label {target_label}, Target {target_bitstring}, Backend {backend.name}): {e}", exc_info=True)
        # Return high cost on execution failure
        return -math.log(settings.EPSILON) * 50

    # Calculate cost based on probability of the target bitstring
    total_shots = sum(counts.values())
    if total_shots == 0:
        target_prob = 0.0
        logging.warning(f"Cost function evaluation resulted in 0 total shots for label {target_label}.")
    else:
        # Target bitstring from class_label_to_target_bitstring should match counts keys format (binary string)
        target_prob = counts.get(target_bitstring, 0) / total_shots

    # Calculate cost: -log(probability)
    cost = -math.log(target_prob + settings.EPSILON) # Add epsilon for numerical stability

    if not np.isfinite(cost):
        logging.warning(f"Non-finite cost ({cost}) calculated for Label {target_label} "
                        f"(Prob: {target_prob}, Target: {target_bitstring}, Counts: {counts}). Clamping.")
        cost = -math.log(settings.EPSILON) # Clamp to a large finite value

    logging.debug(f"Cost for Label {target_label}, Target {target_bitstring}: {cost:.4f} (Prob: {target_prob:.4f})")
    return cost


def train_vqnn_supervised(circuit_template: QuantumCircuit, initial_params: np.ndarray,
                          training_data: list[tuple[str, int]], backend: BackendV2,
                          callback_func=None) -> tuple[np.ndarray, float, dict]:
    """Trains VQNN decoder using SPSA optimizer. Compatible with SamplerV2 cost function."""
    if not circuit_template.parameters:
        logging.warning(f"Circuit '{circuit_template.name}' has no parameters. Skipping SPSA.")
        # Return initial params (empty array), cost 0, empty history
        return np.array([]), 0.0, {'cost': []}

    num_params = len(circuit_template.parameters)
    if len(initial_params) != num_params:
         raise ValueError(f"Parameter length mismatch for '{circuit_template.name}'. "
                          f"Circuit needs {num_params}, got {len(initial_params)} initial params.")

    # SPSA Optimizer from qiskit_algorithms
    optimizer = SPSA(maxiter=settings.SPSA_MAX_ITER)

    params = np.array(initial_params) # Ensure it's a numpy array
    history = {'cost': []}
    current_iteration = 0 # Use instance variable for iteration tracking

    # Objective function for SPSA (calculates batch cost)
    def objective_function(current_params):
        nonlocal current_iteration # Modify the outer scope variable
        current_iteration += 1

        # --- Batching Logic ---
        # Ensure batch size doesn't exceed available data
        batch_size = min(len(training_data), 32) # Or use a config setting
        if batch_size == 0:
            logging.warning("Objective function called with empty training data.")
            #return 0.0 # Or raise error? Returning 0 might stall optimizer. I will return high cost.
            return -math.log(settings.EPSILON) * 100

        # Sample a batch of data points
        sample_indices = random.sample(range(len(training_data)), batch_size)
        batch_total_cost = 0.0
        valid_samples_in_batch = 0

        # --- Cost Calculation per Sample ---
        for idx in sample_indices:
            syndrome_str, target_label = training_data[idx]
            try:
                # Call the SamplerV2-compatible cost function
                cost = cost_function_decoder(
                    current_params,
                    circuit_template,
                    syndrome_str,
                    target_label,
                    backend,
                    settings.SHOTS # Use training shots
                )

                # Check for non-finite cost again, though cost_function should handle it
                if not np.isfinite(cost):
                    logging.warning(f"SPSA objective received non-finite cost for sample {idx}. Clamping.")
                    cost = -math.log(settings.EPSILON) * 10

                batch_total_cost += cost
                valid_samples_in_batch += 1

            except Exception as e:
                 # Catch errors from cost_function_decoder (e.g., binding, execution)
                 logging.error(f"Cost function error during SPSA iter {current_iteration}, sample index {idx}: {e}", exc_info=False) # Avoid flooding logs
                 # Penalize errors with a high cost
                 batch_total_cost += -math.log(settings.EPSILON) * 50
                 valid_samples_in_batch += 1 # Count as processed, but with high cost


        # --- Average Cost & Logging ---
        if valid_samples_in_batch > 0:
             avg_cost = batch_total_cost / valid_samples_in_batch
        else:
             logging.error(f"SPSA iteration {current_iteration}: No valid samples processed in the batch.")
             avg_cost = -math.log(settings.EPSILON) * 100 # Return high cost if batch failed entirely

        # Store cost history
        history['cost'].append(avg_cost)

        # Log progress periodically
        if current_iteration % 10 == 0:
            logging.info(f" SPSA Iter {current_iteration}/{settings.SPSA_MAX_ITER}, Avg Batch Cost: {avg_cost:.4f}")

        # --- Callback ---
        if callback_func:
            try:
                 callback_func({
                     'iteration': current_iteration,
                     'max_iter': settings.SPSA_MAX_ITER,
                     'cost': avg_cost,
                     'params': current_params # Optional: provide params to callback
                 })
            except Exception as cb_e:
                 logging.error(f"Callback error during SPSA iteration {current_iteration}: {cb_e}", exc_info=False)

        return avg_cost

    # --- Run the Optimization ---
    try:
        logging.info(f"Starting SPSA optimization for '{circuit_template.name}' with {settings.SPSA_MAX_ITER} iterations...")
        # SPSA minimize takes objective function and initial parameters
        result = optimizer.minimize(objective_function, params)

        optimal_params = np.array(result.x) # Best parameters found
        final_cost = result.fun         # Final cost value
        num_iterations_run = result.nit # Number of iterations completed

        logging.info(f"SPSA Training finished for '{circuit_template.name}'.")
        logging.info(f"  Final Cost: {final_cost:.4f}")
        logging.info(f"  Iterations Run: {num_iterations_run}")
        logging.info(f"  Optimal Params (first 5): {optimal_params[:5]}") # Optionally log params

        return optimal_params, final_cost, history

    except Exception as e:
        logging.error(f"SPSA optimization failed for '{circuit_template.name}': {e}", exc_info=True)
        # Return initial parameters and infinite cost on failure
        return np.array(initial_params), np.inf, history
    

def evaluate_decoder(trained_params: np.ndarray, circuit_creator_func: callable,
                     evaluation_data: list[tuple[str, int]], backend: BackendV2) -> tuple[float, list[dict]]:
    """Evaluates the prediction accuracy of a trained VQNN decoder using SamplerV2."""
    total_samples = len(evaluation_data)
    if total_samples == 0:
        logging.warning("Evaluation called with empty data.")
        return 0.0, []

    logging.info(f"Evaluating decoder accuracy on {total_samples} samples using backend '{backend.name}'...")
    start_eval_time = time.time()

    # --- Prepare Circuits for Evaluation ---
    circuits_to_run = []
    true_labels = []
    circuit_details = [] # Store index, syndrome etc. for matching results

    prep_start_time = time.time()
    processed_for_run = 0
    for i, (syndrome_str, true_label) in enumerate(evaluation_data):
        try:
            # 1. Create the circuit instance for this syndrome
            circuit_instance, circuit_params_list = circuit_creator_func(syndrome_str)
            num_vqc_qubits = circuit_instance.num_qubits

            # 2. Check parameter consistency BEFORE binding
            if len(trained_params) != len(circuit_params_list):
                 # Log specific details for mismatch
                 logging.error(f"Param length mismatch during evaluation prep for sample {i}!")
                 logging.error(f"  Syndrome: {syndrome_str}, True Label: {true_label}")
                 logging.error(f"  Circuit '{circuit_instance.name}' needs {len(circuit_params_list)} params.")
                 logging.error(f"  Trained params length: {len(trained_params)}")
                 # Skip this sample if params don't match
                 raise ValueError(f"Param length mismatch evaluation sample {i}")

            # 3. Bind the trained parameters
            param_map = {p: v for p, v in zip(circuit_params_list, trained_params)}
            bound_circuit = circuit_instance.assign_parameters(param_map)

            # 4. Transpile the circuit (using pass manager)
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
            isa_circuit = pm.run(bound_circuit)

            # 5. Add to lists for batch execution
            circuits_to_run.append(isa_circuit)
            true_labels.append(true_label)
            circuit_details.append({'original_index': i, 'syndrome': syndrome_str, 'true_label': true_label, 'num_vqc_qubits': num_vqc_qubits})
            processed_for_run += 1

            if (i + 1) % max(1, total_samples // 10) == 0:
                 logging.info(f"  Prepared {i+1}/{total_samples} circuits for evaluation...")

        except Exception as e:
            # Log errors during preparation (binding, transpiling)
            error_msg = f"Error preparing evaluation sample {i} (Syndrome: {syndrome_str}, Label: {true_label}): {e}"
            logging.error(error_msg, exc_info=False) # Keep log concise
            # Store error info, but don't run this circuit
            circuit_details.append({'original_index': i, 'syndrome': syndrome_str, 'true_label': true_label, 'status': f"Error (Prep): {str(e)[:100]}...", 'predicted_label': -1, 'counts': {}})
            # Don't increment processed_for_run


    logging.info(f"Circuit preparation for evaluation took {time.time() - prep_start_time:.2f}s. Prepared {processed_for_run} circuits.")

    if not circuits_to_run:
        logging.error("No circuits were successfully prepared for evaluation.")
        # Need to construct results_detail from circuit_details which may contain only errors
        results_detail = [cd for cd in circuit_details if cd.get('status', '').startswith('Error')]
        return 0.0, results_detail # Return 0 accuracy if nothing ran


    # --- Execute Circuits in Batches using SamplerV2 ---
    sampler = Sampler(backend)
    run_options = {'shots': settings.EVAL_SHOTS}
    results_detail_map = {cd['original_index']: cd for cd in circuit_details if 'status' not in cd} # Map results back using original index

    correct_predictions = 0
    processed_samples_count = 0 # Count samples successfully run and processed

    eval_run_start_time = time.time()
    batch_size = 100 # Adjust as needed

    for i in range(0, len(circuits_to_run), batch_size):
        batch_circuits = circuits_to_run[i:i+batch_size]
        # Get corresponding original indices for this batch
        batch_indices = [cd['original_index'] for cd in circuit_details if 'status' not in cd][i:i+batch_size]

        logging.info(f" Running evaluation batch {i//batch_size + 1}/{(len(circuits_to_run) + batch_size - 1)//batch_size} (size {len(batch_circuits)})...")

        try:
            # Set seed for the batch if applicable (using varying seed for eval)
            if isinstance(backend, AerSimulator) and settings.SEED is not None:
                # Use a seed that varies per batch/sample for independent evaluation runs
                batch_base_seed = settings.SEED + total_samples + i # Offset seed
                backend.set_options(seed_simulator=batch_base_seed)
                logging.debug(f"Set AerSimulator seed to {batch_base_seed} for eval batch starting at index {i}.")

            # Run the batch
            job = sampler.run(batch_circuits, **run_options)
            results_batch = job.result() # PrimitiveResult

            # Process results for the batch
            for j, pub_result in enumerate(results_batch):
                 original_index = batch_indices[j]
                 detail = results_detail_map[original_index]
                 processed_samples_count += 1

                 try:
                     # Extract counts
                     counts = pub_result.data.c.get_counts()
                     detail['counts'] = counts # Store counts

                     # Predict label from counts
                     predicted_label = measurement_to_class_label(
                         counts,
                         detail['num_vqc_qubits'],
                         settings.NUM_ERROR_CLASSES
                     )
                     detail['predicted_label'] = predicted_label

                     # Check accuracy
                     if predicted_label == detail['true_label']:
                         correct_predictions += 1
                         detail['status'] = 'Complete (Correct)'
                     else:
                         detail['status'] = 'Complete (Incorrect)'

                 except Exception as proc_err:
                     logging.error(f"Error processing result for sample index {original_index}: {proc_err}", exc_info=False)
                     detail['status'] = f'Error (Processing): {str(proc_err)[:100]}...'
                     detail['predicted_label'] = -1
                     detail['counts'] = {}
        except Exception as e:
            logging.error(f"Evaluation batch run failed (starting index {i}): {e}", exc_info=True)
            # Mark all samples in this batch as errored in the results map
            for k in range(len(batch_circuits)):
                 original_index = batch_indices[k]
                 detail = results_detail_map[original_index]
                 detail['status'] = f'Error (Batch Run): {str(e)[:100]}...'
                 detail['predicted_label'] = -1
                 detail['counts'] = {}
                 processed_samples_count += 1 # Count as processed (with error)

        if processed_samples_count % max(1, total_samples // 5) == 0:
              logging.info(f"  Processed {processed_samples_count}/{processed_for_run} running samples...")


    logging.info(f"Evaluation runs took {time.time() - eval_run_start_time:.2f}s.")

    # Consolidate results (combine successfully run ones with prep errors)
    final_results_detail = sorted(list(results_detail_map.values()) + [cd for cd in circuit_details if cd.get('status', '').startswith('Error')], key=lambda x: x['original_index'])


    eval_duration = time.time() - start_eval_time
    # Calculate accuracy based on successfully processed samples where prediction was possible
    # successful_runs = [r for r in final_results_detail if 'Complete' in r.get('status', '')]
    successful_processed_count = len([r for r in final_results_detail if not r.get('status','').startswith('Error')])


    if successful_processed_count > 0:
        accuracy = correct_predictions / successful_processed_count
    else:
        accuracy = 0.0

    logging.info(f"Evaluation complete. Duration: {eval_duration:.2f}s")
    logging.info(f"  Total samples provided: {total_samples}")
    logging.info(f"  Samples prepared for run: {processed_for_run}")
    logging.info(f"  Samples successfully processed (run+result): {successful_processed_count}")
    logging.info(f"  Correct predictions: {correct_predictions}")
    logging.info(f"  Accuracy (Correct / Successfully Processed): {accuracy:.4f}")

    return accuracy, final_results_detail

def get_initial_params(circuit_params: list[Parameter]) -> np.ndarray:
    """Generates random initial parameters (angles) for a VQC."""
    num_params = len(circuit_params)
    if num_params == 0:
        return np.array([])
    # Initialize parameters typically between -pi and pi or 0 and 2pi for rotational gates
    initial_values = np.random.uniform(-np.pi, np.pi, num_params)
    logging.debug(f"Generated {num_params} initial random parameters.")
    return initial_values

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


def run_decoder_experiment_instance(config_override: dict | None = None, training_callback=None) -> tuple[dict, dict]:
    """
    Runs the full decoder training and evaluation pipeline.
    Uses SamplerV2 compatible functions.

    Returns:
        tuple: (results_dict, setup_details_dict)
               results_dict contains performance metrics and training outputs.
               setup_details_dict contains static info like lattice, qubit maps, etc., for visualization.
    """
    # --- Configuration Setup ---
    effective_config = copy.deepcopy(settings) # Start with default settings
    if config_override:
        # Update attributes of the copied config object
        for key, value in config_override.items():
            if hasattr(effective_config, key):
                 # Use the validation/type conversion logic from Config if needed,
                 # or assume overrides are correct type for simplicity here.
                 setattr(effective_config, key, value)
            else:
                 logging.warning(f"Config override key '{key}' not found in default settings. Ignored.")
    current_settings = effective_config # Use the potentially updated settings

    # Re-apply seeding if specified in the final config
    if current_settings.SEED is not None:
        np.random.seed(current_settings.SEED)
        random.seed(current_settings.SEED)
        logging.info(f"Applied SEED {current_settings.SEED} for this run.")
    else:
        logging.info("Running with random seed.")


    start_overall_time = time.time()
    logging.info("="*30 + f"\n=== STARTING DECODER RUN (Seed: {current_settings.SEED}) ===\n" + "="*30)

    # Log effective config (masking token)
    loggable_config = {k: v for k, v in vars(current_settings).items() if k != 'IBM_API_TOKEN' and not k.startswith('_')}
    logging.info(f"Effective Run Configuration:\n{json.dumps(loggable_config, indent=2)}")

    # --- Results & Setup Details Structure ---
    results = {
        'architectures': {},
        'execution_times': {'overall': 0, 'setup': 0, 'data_gen': 0, 'training': {}, 'evaluation': {}},
        'setup_info': {'config_used': loggable_config}, # Basic info stored directly in results
        'error': None,
        'summary': {}
    }
    setup_details = { # New dictionary for visualization-related setup info
        'lattice_data': None,
        'qpus': None,
        'qubit_indices': None,
        'data_q_list': None,
        'ancilla_q_list': None,
        'stabilizers_x': None,
        'stabilizers_z': None,
        'data_qubit_indices_list': None, # Canonical list of data qubit indices
        'vqc_templates': {}, # To store example VQC circuits for plotting
        'classical_model_defs': {} # Store definitions like layer sizes
    }
    arch_results = results['architectures']
    exec_times = results['execution_times']
    setup_info = results['setup_info'] # Keep basic info here

    # --- Setup Phase ---
    start_setup_time = time.time()
    backend = None # Initialize backend to None
    try:
        # Get backend (force reload based on current_settings)
        backend = get_backend(force_reload=True) # Force reload ensures correct config is used
        setup_info['backend_name'] = backend.name
        setup_info['backend_mode_used'] = current_settings.DEFAULT_BACKEND_MODE
        logging.info(f"Using backend: {backend.name} (Mode: {current_settings.DEFAULT_BACKEND_MODE})")

        # Initialize surface code parameters and store in setup_details
        # Ensure these use current_settings if they were overridable (they aren't directly here)
        lattice, qpus, data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map = initialize_d3_surface_code()
        num_total_qubits = len(qubit_indices)
        setup_details.update({
            'lattice_data': lattice,
            'qpus': qpus,
            'qubit_indices': qubit_indices,
            'data_q_list': data_q_list,
            'ancilla_q_list': ancilla_q_list,
             'qpu_assignment_map_idx': qpu_assignment_map # Map qubit_idx -> qpu_idx
        })
        # Also store some basic info in results['setup_info'] for quick viewing
        setup_info.update({
             'code_distance': current_settings.CODE_DISTANCE,
             'num_data_qubits': len(data_q_list),
             'num_ancilla_qubits': len(ancilla_q_list),
             'num_total_qubits': num_total_qubits,
             'num_error_classes': current_settings.NUM_ERROR_CLASSES,
             'min_vqc_qubits_needed': current_settings.MIN_VQC_QUBITS_NEEDED,
        })

        # Get noise model ONLY if using Aer simulator (real backends have inherent noise)
        noise_model = None
        if current_settings.DEFAULT_BACKEND_MODE == 'aer_simulator':
             noise_model = get_noise_model(
                 num_total_qubits,
                 current_settings.ERROR_RATES,
                 qpu_assignment_map,
                 current_settings.READOUT_ERROR_PROB
             )
             setup_info['noise_model_enabled'] = (noise_model is not None)
             logging.info(f"Noise model prepared for Aer: {'Enabled' if noise_model else 'Disabled'}")
        else:
            setup_info['noise_model_enabled'] = False # Noise comes from backend
            logging.info("Noise model not applicable (using IBM backend or non-Aer sim).")

        # Get stabilizers
        stabilizers_x, stabilizers_z = get_d3_stabilizers(qubit_indices)
        setup_details.update({ # Store stabilizers for potential use
            'stabilizers_x': stabilizers_x,
            'stabilizers_z': stabilizers_z,
        })
        setup_info['num_x_stabilizers'] = len(stabilizers_x)
        setup_info['num_z_stabilizers'] = len(stabilizers_z)

        exec_times['setup'] = time.time() - start_setup_time
        logging.info(f"Setup phase complete ({exec_times['setup']:.2f}s)")
    except Exception as e:
        exec_times['setup'] = time.time() - start_setup_time
        error_msg = f"Setup failed: {e}"
        results['error'] = error_msg
        logging.error(error_msg, exc_info=True)
        # Add backend name if it was successfully fetched before error
        if backend: results['setup_info']['backend_name'] = backend.name
        # Return both dictionaries, even on error, with error flagged in results
        return results, setup_details # Critical setup failed, cannot continue

    # --- Data Generation Phase ---
    start_data_time = time.time()
    training_data = []
    evaluation_data = []

    try:
        # Ensure data_qubit_indices_list is generated and stored
        data_qubit_indices_list = sorted([qubit_indices[q] for q in data_q_list]) # Needed for label mapping
        setup_details['data_qubit_indices_list'] = data_qubit_indices_list # Store for reference

        # Generate data using AerSimulator
        logging.info("Generating training data...")
        training_data = generate_labeled_syndrome_data(
            current_settings.NUM_SYNDROME_SAMPLES_TRAIN,
            data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map,
            stabilizers_x, stabilizers_z,
            noise_model # Pass the generated noise model (or None)
        )
        logging.info("Generating evaluation data...")
        evaluation_data = generate_labeled_syndrome_data(
            current_settings.NUM_SYNDROME_SAMPLES_EVAL,
            data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map,
            stabilizers_x, stabilizers_z,
            noise_model # Pass the same noise model (or None)
        )

        exec_times['data_gen'] = time.time() - start_data_time
        setup_info['num_training_samples_generated'] = len(training_data)
        setup_info['num_evaluation_samples_generated'] = len(evaluation_data)
        logging.info(f"Data generation complete ({exec_times['data_gen']:.2f}s). Train: {len(training_data)}, Eval: {len(evaluation_data)}")

        if not training_data or not evaluation_data:
             raise RuntimeError("Failed to generate sufficient training or evaluation data.")

    except Exception as e:
        exec_times['data_gen'] = time.time() - start_data_time
        error_msg = f"Data generation failed: {e}"
        results['error'] = error_msg
        logging.error(error_msg, exc_info=True)
        return results, setup_details # Cannot proceed without data


    # --- Prepare Data for Classical Models ---
    X_train, y_train, X_eval, y_eval = None, None, None, None
    try:
        # Convert syndrome strings to feature vectors (numpy arrays)
        X_train = np.array([syndrome_str_to_feature_vector(s) for s, l in training_data])
        y_train = np.array([l for s, l in training_data])
        X_eval = np.array([syndrome_str_to_feature_vector(s) for s, l in evaluation_data])
        y_eval = np.array([l for s, l in evaluation_data])
        logging.info("Classical feature vectors prepared.")
        setup_info['classical_data_prepared'] = True
    except Exception as e:
        logging.error(f"Failed to prepare data for classical models: {e}", exc_info=True)
        setup_info['classical_data_prepared'] = False
        # Continue with VQNNs, but classical models will be skipped

    # --- Model Definitions & Template Generation ---
    # Define creators first
    # Dummy syndrome string needed for template creation
    num_stabilizers = len(stabilizers_x) + len(stabilizers_z)
    dummy_syndrome = '0' * num_stabilizers

    def _create_centralized_template():
        try: return create_centralized_vqc(dummy_syndrome)[0] # Return only circuit
        except Exception: logging.error("Failed to create centralized template", exc_info=True); return None
    def _create_global_template():
        try: return create_global_vqc(dummy_syndrome)[0]
        except Exception: logging.error("Failed to create global template", exc_info=True); return None
    def _create_local_template(qpu_idx=0): # Create template for first QPU
        try: return create_local_vqc(qpus[qpu_idx], dummy_syndrome, qpu_idx, qubit_indices)[0]
        except Exception: logging.error(f"Failed to create local template qpu={qpu_idx}", exc_info=True); return None


    model_definitions = {
         "centralized_vqnn": {"creator": create_centralized_vqc, "config_qubits": current_settings.CENTRALIZED_VQC_QUBITS, "type": "vqnn", "template": _create_centralized_template()},
         "distributed_vqnn": {"creator": create_global_vqc, "config_qubits": current_settings.GLOBAL_VQC_QUBITS, "type": "vqnn", "template": _create_global_template()},
         "localized_vqnn": {"is_local": True, "config_qubits": current_settings.LOCAL_VQC_QUBITS_PER_QPU, "type": "vqnn", "template": _create_local_template()}, # Template for QPU 0
         "classical_logreg": {"is_classical": True, "type": "classical"},
         "classical_mlp": {"is_classical": True, "type": "classical"},
         "baseline_predict_identity": {"is_baseline": True, "type": "baseline"}
     }

    # Store templates in setup_details
    for name, definition in model_definitions.items():
        if 'template' in definition and definition['template'] is not None:
             setup_details['vqc_templates'][name] = definition['template']
             logging.info(f"Stored VQC template for: {name}")

    # Add classical model instances and definitions if preparation succeeded
    trained_classical_models = {}
    if setup_info['classical_data_prepared']:
        try:
            classical_model_instances = create_classical_models()
            model_definitions['classical_logreg']['instance'] = classical_model_instances['classical_logreg']
            model_definitions['classical_mlp']['instance'] = classical_model_instances['classical_mlp']
            # Store MLP definition for plotting
            setup_details['classical_model_defs']['mlp'] = {
                'hidden_layers': (64, 32), # Hardcoded in create_classical_models
                'input_size': num_stabilizers,
                'output_size': current_settings.NUM_ERROR_CLASSES
            }
        except Exception as e:
             logging.error(f"Failed to create/store classical model instances/defs: {e}", exc_info=True)
             setup_info['classical_data_prepared'] = False # Mark as failed if instances couldn't be made


    # --- Training and Evaluation Loop ---
    for arch_name, definition in model_definitions.items():
        model_type = definition['type']
        # Use a cleaner name for logging/reporting if desired
        clean_arch_name = arch_name.replace('_vqnn', '').replace('_decoder', '').replace('classical_','').replace('baseline_','') # Make name cleaner
        logging.info(f"\n--- Processing Architecture: {clean_arch_name} ({model_type}) ---")

        # Initialize results structure for this architecture
        arch_results[arch_name] = { # Use original key here
            'params': None, 'accuracy': None, 'training_history': None,
            'status': 'Pending', 'error': None, 'eval_details': None, 'type': model_type
        }
        exec_times['training'][arch_name] = 0
        exec_times['evaluation'][arch_name] = 0

        # --- Skip Conditions ---
        if definition.get("is_baseline"):
            logging.info(f"Skipping training/evaluation for baseline model '{clean_arch_name}'.")
            arch_results[arch_name]['status'] = 'Skipped (Baseline)'
            continue # Baseline calculated later

        if definition.get("is_classical") and not setup_info['classical_data_prepared']:
            arch_results[arch_name]['status'] = 'Skipped (Classical Data Failed)'
            logging.warning(f"Skipping classical model '{clean_arch_name}' due to classical data preparation failure.")
            continue

        if model_type == 'vqnn':
            min_q = current_settings.MIN_VQC_QUBITS_NEEDED
            config_q = definition.get("config_qubits", 0) # Default to 0 if missing
            if config_q < min_q:
                warn_msg = f"Config qubits ({config_q}) < min required ({min_q})."
                arch_results[arch_name]['status'] = 'Skipped (Insufficient Qubits)'
                arch_results[arch_name]['error'] = warn_msg
                logging.warning(f"Skipping VQNN '{clean_arch_name}': {warn_msg}")
                continue

        # --- Training Phase ---
        start_train_time = time.time()
        arch_results[arch_name]['status'] = 'Training'
        try:
            if model_type == 'vqnn':
                # Define the callback for this specific VQNN training
                def specific_training_callback(info):
                     if training_callback:
                         # Add architecture info before passing to the main callback
                         info['arch'] = clean_arch_name # Use clean name for callback display
                         training_callback(info)

                if definition.get("is_local", False):
                    # --- Local VQNN Training (Per QPU) ---
                    optimal_params_list = []
                    history_list = []
                    num_qpus_local = len(qpus)
                    logging.info(f"Training localized VQNNs for {num_qpus_local} QPUs...")

                    for qpu_idx, qpu_names in enumerate(qpus):
                        logging.info(f" Training Local VQNN for QPU {qpu_idx}...")
                        # Create a template circuit for this QPU to get parameter structure
                        try:
                            # Recreate specific template to ensure correct params list for THIS QPU
                            local_tmpl_c, local_tmpl_p_list = create_local_vqc(qpu_names, dummy_syndrome, qpu_idx, qubit_indices)
                            logging.info(f"  QPU {qpu_idx}: Template circuit '{local_tmpl_c.name}' created with {len(local_tmpl_p_list)} parameters.")
                        except Exception as local_create_err:
                             logging.error(f"  Failed to create local VQC template for QPU {qpu_idx}: {local_create_err}", exc_info=True)
                             raise RuntimeError(f"Local VQC creation failed for QPU {qpu_idx}") from local_create_err

                        if not local_tmpl_p_list: # No parameters in the template
                            logging.warning(f"  Local VQC for QPU {qpu_idx} has no parameters. Assigning empty params.")
                            opt_p, final_cost, hist = np.array([]), 0.0, {'cost': []}
                        else:
                            # Get initial parameters
                            init_p = get_initial_params(local_tmpl_p_list)
                            logging.info(f"  QPU {qpu_idx}: Initial params shape {init_p.shape}")

                            # Define a more specific callback for local training progress
                            def local_progress_callback(info):
                                if training_callback:
                                    info['arch'] = clean_arch_name # Use clean name
                                    info['detail'] = f"QPU {qpu_idx}"
                                    training_callback(info) # Pass enriched info

                            # Train this local VQNN
                            opt_p, final_cost, hist = train_vqnn_supervised(
                                local_tmpl_c,
                                init_p,
                                training_data, # Use full training data for each local model
                                backend,       # Use the main backend
                                callback_func=local_progress_callback
                            )
                            logging.info(f"  QPU {qpu_idx}: Training complete. Final cost: {final_cost:.4f}")


                        optimal_params_list.append(opt_p.tolist()) # Store as list
                        history_list.append(hist) # Store history

                    # Store results for the overall localized architecture
                    arch_results[arch_name]['params'] = optimal_params_list
                    arch_results[arch_name]['training_history'] = history_list

                else:
                    # --- Global / Centralized VQNN Training ---
                    logging.info(f"Training {clean_arch_name} VQNN...")
                    # Retrieve the template created earlier
                    tmpl_c = definition.get('template')
                    if tmpl_c is None:
                         # Attempt to recreate if missing
                         logging.warning(f"Template not found for {clean_arch_name}, attempting recreation.")
                         creator = definition["creator"]
                         try: tmpl_c, tmpl_p_list = creator(dummy_syndrome)
                         except Exception as create_err:
                             logging.error(f" Failed to create VQC template for {clean_arch_name}: {create_err}", exc_info=True)
                             raise RuntimeError(f"VQC template creation failed for {clean_arch_name}") from create_err
                         if tmpl_c is None: # Check again after recreation attempt
                             raise RuntimeError(f"Failed to obtain template for {clean_arch_name}")
                    else:
                        # Get parameters from stored template if it exists
                        tmpl_p_list = tmpl_c.parameters

                    logging.info(f" Using template circuit '{tmpl_c.name}' with {len(tmpl_p_list)} parameters.")

                    if not tmpl_p_list: # No parameters
                        logging.warning(f" VQNN {clean_arch_name} has no parameters. Assigning empty params.")
                        opt_p, final_cost, hist = np.array([]), 0.0, {'cost': []}
                    else:
                        init_p = get_initial_params(tmpl_p_list)
                        logging.info(f" Initial params shape {init_p.shape}")
                        opt_p, final_cost, hist = train_vqnn_supervised(
                            tmpl_c,
                            init_p,
                            training_data,
                            backend,
                            callback_func=specific_training_callback # Use the arch-specific callback
                        )
                        logging.info(f" Training complete for {clean_arch_name}. Final cost: {final_cost:.4f}")

                    arch_results[arch_name]['params'] = opt_p.tolist() # Store as list
                    arch_results[arch_name]['training_history'] = hist

            elif model_type == 'classical':
                # --- Classical Model Training ---
                if not setup_info['classical_data_prepared']:
                     # This check is technically redundant due to the skip condition, but safer
                     raise RuntimeError(f"Cannot train classical model {clean_arch_name}, data not available.")

                model_instance = definition.get('instance')
                if model_instance is None:
                     raise RuntimeError(f"Model instance not found for classical model {clean_arch_name}.")

                logging.info(f"Training classical model: {clean_arch_name}...")
                trained_model, train_time_classical = train_classical_model(
                    clean_arch_name, model_instance, X_train, y_train
                )

                if trained_model is None:
                    # Handle training failure reported by train_classical_model
                    raise RuntimeError(f"Classical training function failed for {clean_arch_name}.")

                # Store the trained model instance for evaluation
                trained_classical_models[arch_name] = trained_model # Use original arch_name key
                # Note: We don't store classical model parameters directly in results['params']
                logging.info(f" Classical model {clean_arch_name} trained in {train_time_classical:.2f}s.")


            # Record training time and update status
            exec_times['training'][arch_name] = time.time() - start_train_time
            arch_results[arch_name]['status'] = 'Trained'
            logging.info(f" Training phase complete for {clean_arch_name} ({exec_times['training'][arch_name]:.2f}s)")

        except Exception as train_err:
            # Catch any error during the training block
            error_msg = f"Training failed for {clean_arch_name}: {train_err}"
            logging.error(f" {error_msg}", exc_info=True) # Log full traceback for training errors
            arch_results[arch_name]['status'] = 'Error (Training)'
            arch_results[arch_name]['error'] = str(error_msg) # Store simplified error message
            exec_times['training'][arch_name] = time.time() - start_train_time
            continue # Skip evaluation if training failed


        # --- Evaluation Phase ---
        start_eval_time = time.time()
        arch_results[arch_name]['status'] = 'Evaluating'
        logging.info(f" Evaluating {clean_arch_name}...")
        try:
            accuracy = None
            eval_details = None # Store detailed per-sample results if available

            if model_type == 'vqnn':
                if definition.get("is_local", False):
                    # --- Local VQNN Evaluation ---
                    accuracies_local = []
                    eval_details_local = [] # Collect details from each QPU eval if needed
                    num_qpus_local = len(qpus)
                    all_qpu_params = arch_results[arch_name].get('params', []) # Use original key

                    if len(all_qpu_params) != num_qpus_local:
                         raise RuntimeError(f"Mismatch between number of QPUs ({num_qpus_local}) and stored local params ({len(all_qpu_params)}) for {clean_arch_name}.")

                    logging.info(f"Evaluating localized VQNNs for {num_qpus_local} QPUs...")
                    for qpu_idx, qpu_names in enumerate(qpus):
                         logging.info(f"  Evaluating Local VQNN for QPU {qpu_idx}...")
                         # Define the circuit creator for this specific QPU
                         def local_eval_creator(syndrome_str_arg):
                              # Closure captures qpu_names, qpu_idx, qubit_indices
                              return create_local_vqc(qpu_names, syndrome_str_arg, qpu_idx, qubit_indices)

                         # Get the trained parameters for this QPU
                         qpu_params = np.array(all_qpu_params[qpu_idx])
                         logging.info(f"   Params shape for QPU {qpu_idx}: {qpu_params.shape}")


                         # Evaluate using the generic decoder evaluator
                         acc, details = evaluate_decoder(
                             qpu_params,
                             local_eval_creator,
                             evaluation_data, # Use full eval data
                             backend
                         )
                         logging.info(f"  QPU {qpu_idx}: Evaluation complete. Accuracy: {acc:.4f}")
                         accuracies_local.append(acc)
                         # Store details with QPU index if needed later
                         # for d in details: d['qpu_index'] = qpu_idx
                         # eval_details_local.extend(details) # This could get large


                    # Calculate average accuracy across local QPUs
                    if accuracies_local:
                        accuracy = np.mean(accuracies_local)
                        logging.info(f" Localized VQNN average accuracy: {accuracy:.4f}")
                    else:
                        accuracy = 0.0
                        logging.warning(" No local accuracies recorded for localized VQNN.")
                    # For simplicity, don't store combined eval_details for local VQNNs now
                    eval_details = {"average_accuracy": accuracy, "individual_accuracies": accuracies_local} # Store summary

                else:
                     # --- Global / Centralized VQNN Evaluation ---
                     trained_params_np = np.array(arch_results[arch_name].get('params', [])) # Use original key
                     # Check parameter presence vs circuit needs
                     template_circuit_check = definition.get('template')
                     expected_params_count = len(template_circuit_check.parameters) if template_circuit_check else 0

                     if trained_params_np.size == 0 and expected_params_count > 0:
                          # Handle case where params should exist but are empty
                           logging.warning(f" Trained parameters for {clean_arch_name} are empty, but circuit requires {expected_params_count} parameters. Evaluation may fail or yield poor results.")
                           # Proceed, evaluate_decoder might handle empty params if circuit expects none.
                     elif trained_params_np.size != expected_params_count:
                          logging.warning(f"Parameter count mismatch during evaluation for {clean_arch_name}. Expected {expected_params_count}, got {trained_params_np.size}. Evaluation may fail.")


                     logging.info(f" Evaluating {clean_arch_name} VQNN (Params shape: {trained_params_np.shape})...")
                     eval_creator_func = definition["creator"]
                     accuracy, eval_details = evaluate_decoder(
                         trained_params_np,
                         eval_creator_func,
                         evaluation_data,
                         backend
                     )
                     logging.info(f" {clean_arch_name} evaluation complete. Accuracy: {accuracy:.4f}")
            elif model_type == 'classical':
                 # --- Classical Model Evaluation ---
                 if not setup_info['classical_data_prepared']:
                     raise RuntimeError(f"Cannot evaluate classical model {clean_arch_name}, data not available.")

                 trained_model = trained_classical_models.get(arch_name) # Use original key
                 if trained_model is None:
                      raise RuntimeError(f"Trained model instance not found for classical model {clean_arch_name} during evaluation.")

                 logging.info(f"Evaluating classical model: {clean_arch_name}...")
                 acc, eval_time_classical, predictions = evaluate_classical_model(
                     clean_arch_name, trained_model, X_eval, y_eval
                 )

                 if acc is None:
                      # Handle evaluation failure reported by evaluate_classical_model
                      raise RuntimeError(f"Classical evaluation function failed for {clean_arch_name}.")

                 accuracy = acc # Assign accuracy
                 # Store predictions if needed (convert to simple list for JSON)
                 eval_details = predictions # Store predictions as details for classical
                 logging.info(f" Classical model {clean_arch_name} evaluated. Accuracy: {accuracy:.4f}, Time: {eval_time_classical:.2f}s")


            # --- Store Evaluation Results ---
            if accuracy is not None:
                 arch_results[arch_name]['accuracy'] = float(accuracy) # Ensure float type
                 arch_results[arch_name]['status'] = 'Complete'
                 if eval_details is not None:
                      # Store eval details (can be large,we might consider limiting size or saving separately)
                      # Convert numpy arrays/objects in details to JSON serializable types if necessary
                      try:
                           # Attempt basic JSON serialization check/conversion
                           processed_details = None
                           if isinstance(eval_details, np.ndarray):
                               processed_details = eval_details.tolist()
                           elif isinstance(eval_details, list) and eval_details:
                               # Check elements within the list if it's complex (like list of dicts from VQNN)
                               if isinstance(eval_details[0], dict):
                                   processed_details = []
                                   for item in eval_details:
                                        serializable_item = {}
                                        for k, v in item.items():
                                            if isinstance(v, np.integer): serializable_item[k] = int(v)
                                            elif isinstance(v, np.floating): serializable_item[k] = float(v)
                                            elif isinstance(v, dict) and k == 'counts': # Special handle counts
                                                serializable_item[k] = {bitstr: int(count) for bitstr, count in v.items()}
                                            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                                serializable_item[k] = v # Already serializable
                                            else:
                                                serializable_item[k] = str(v) # Fallback to string
                                        processed_details.append(serializable_item)
                               else: # Assume simple list (like classical predictions)
                                   processed_details = eval_details # Already a list

                           elif isinstance(eval_details, dict): # Handle dict case (e.g., local VQNN summary)
                               processed_details = eval_details # Assume dict is serializable for now

                           # Limit size if too large
                           MAX_DETAILS_LEN = 500 # Store details for first N samples only
                           if isinstance(processed_details, list) and len(processed_details) > MAX_DETAILS_LEN:
                                logging.warning(f"Evaluation details for {clean_arch_name} truncated to first {MAX_DETAILS_LEN} entries.")
                                processed_details = processed_details[:MAX_DETAILS_LEN]

                           arch_results[arch_name]['eval_details'] = processed_details # Use original key
                      except Exception as json_err:
                           logging.warning(f"Could not make eval_details JSON serializable for {clean_arch_name}: {json_err}. Skipping details storage.")
                           arch_results[arch_name]['eval_details'] = "Error serializing details" # Use original key
            else:
                 # Should not happen if evaluation functions return accuracy or raise error
                 logging.error(f"Evaluation accuracy for {clean_arch_name} resulted in None unexpectedly.")
                 arch_results[arch_name]['accuracy'] = None # Ensure it's None if evaluation fails softly
                 # Keep status as 'Evaluating' or update based on error? Let error handler below catch it.
                 # raise RuntimeError(f"Evaluation function returned None accuracy for {clean_arch_name}.") # Or maybe don't raise?

            exec_times['evaluation'][arch_name] = time.time() - start_eval_time
            # Log accuracy if available, otherwise indicate issue
            acc_log_str = f"{arch_results[arch_name]['accuracy']:.4f}" if arch_results[arch_name]['accuracy'] is not None else "N/A"
            logging.info(f" Evaluation phase complete for {clean_arch_name} ({exec_times['evaluation'][arch_name]:.2f}s). Accuracy: {acc_log_str}")
        except Exception as eval_err:
            # Catch any error during the evaluation block
            error_msg = f"Evaluation failed for {clean_arch_name}: {eval_err}"
            logging.error(f" {error_msg}", exc_info=True) # Log full traceback for eval errors
            arch_results[arch_name]['status'] = 'Error (Evaluation)'
            arch_results[arch_name]['error'] = str(error_msg)
            arch_results[arch_name]['accuracy'] = None # Ensure accuracy is None on error
            exec_times['evaluation'][arch_name] = time.time() - start_eval_time
            # Continue to the next architecture

    # --- Baseline Calculation ---
    base_name = 'baseline_predict_identity'
    clean_base_name = base_name.replace('baseline_','') # Already clean
    logging.info(f"\n--- Calculating Baseline: {clean_base_name} ---")
    arch_results[base_name] = {'params': None, 'accuracy': None, 'training_history': None, 'status': 'Calculating', 'error': None, 'eval_details': None, 'type': 'baseline'} # Use original key
    exec_times['training'][base_name] = 0
    exec_times['evaluation'][base_name] = 0

    try:
        # Use the prepared evaluation labels (y_eval) if available
        eval_labels_for_baseline = None
        if y_eval is not None:
            eval_labels_for_baseline = y_eval
        elif evaluation_data: # Fallback to original data structure if numpy conversion failed
            eval_labels_for_baseline = [label for _, label in evaluation_data]

        if eval_labels_for_baseline is not None and len(eval_labels_for_baseline) > 0:
            # Calculate the frequency of the 'Identity' class (label 0)
            num_eval_samples = len(eval_labels_for_baseline)
            # Use np.sum if it's a numpy array, otherwise standard sum
            if isinstance(eval_labels_for_baseline, np.ndarray):
                 identity_correct = np.sum(eval_labels_for_baseline == 0)
            else: # Assume list
                 identity_correct = sum(1 for label in eval_labels_for_baseline if label == 0)

            baseline_accuracy = identity_correct / num_eval_samples
            arch_results[base_name]['accuracy'] = float(baseline_accuracy) # Use original key
            arch_results[base_name]['status'] = 'Complete'
            logging.info(f" Baseline Accuracy (Predict Identity): {baseline_accuracy:.4f}")
        else:
            raise ValueError("No evaluation labels available for baseline calculation.")

    except Exception as base_err:
        error_msg = f"Baseline calculation failed: {base_err}"
        logging.error(error_msg, exc_info=True)
        arch_results[base_name]['status'] = 'Error' # Use original key
        arch_results[base_name]['error'] = error_msg
        arch_results[base_name]['accuracy'] = None


    # --- Finalization ---
    exec_times['overall'] = time.time() - start_overall_time
    logging.info("\n" + "="*30 + f"\n=== Run Completed in {exec_times['overall']:.2f} seconds ===\n" + "="*30)

    # --- Summary Generation ---
    results['summary'] = {}
    # Iterate through original architecture keys used in results dict
    for name, data in arch_results.items():
         # Use the clean name only for the summary display key
         clean_name_summary = name.replace('_vqnn', '').replace('_decoder', '').replace('classical_','').replace('baseline_','')
         results['summary'][clean_name_summary] = { # Key is clean name here
             "status": data.get('status'),
             "accuracy": data.get('accuracy'), # Will be None if error or not calculated
             "train_time": exec_times['training'].get(name), # Use original name to look up time
             "eval_time": exec_times['evaluation'].get(name), # Use original name to look up time
             "type": data.get('type'),
             "error": data.get('error') # Will be None if no error
         }

    # Log the summary table
    logging.info("Final Summary:")
    logging.info(f"{'Architecture':<25} | {'Type':<10} | {'Status':<20} | {'Accuracy':<10} | {'Train (s)':<10} | {'Eval (s)':<10} | {'Error'}")
    logging.info("-" * 100)
    # Sort summary by the clean display name for logging
    for name, summary_data in sorted(results['summary'].items()):
        acc_str = f"{summary_data.get('accuracy', 0.0):.4f}" if isinstance(summary_data.get('accuracy'), float) else 'N/A'
        train_t_str = f"{summary_data.get('train_time', 0.0):.2f}" if isinstance(summary_data.get('train_time'), float) else 'N/A'
        eval_t_str = f"{summary_data.get('eval_time', 0.0):.2f}" if isinstance(summary_data.get('eval_time'), float) else 'N/A'
        error_str = summary_data.get('error') or ""
        logging.info(f"{name:<25} | {summary_data.get('type','N/A'):<10} | {summary_data.get('status','N/A'):<20} | {acc_str:<10} | {train_t_str:<10} | {eval_t_str:<10} | {error_str[:50]}") # Truncate long errors

    # --- Return the comprehensive results dictionary AND the setup details dictionary ---
    logging.info("Returning results and setup details dictionaries.")
    return results, setup_details # MODIFIED RETURN VALUE


# --- Optional: Main execution block for testing this file directly ---
if __name__ == '__main__':
    # Configure logging for direct script execution
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    logging.info("Running qis_logic.py as main script for testing...")

    # Example: Test backend fetching
    try:
        logging.info("Testing get_backend()...")
        # Use test-specific config overrides if needed
        test_config_be = {"DEFAULT_BACKEND_MODE": "aer_simulator"} # Test with Aer first
        # Temporarily override settings for the test
        original_settings = copy.deepcopy(settings)
        settings.__init__(**test_config_be)

        backend = get_backend(force_reload=True)
        logging.info(f"Successfully got backend: {backend.name}")

        # Test with IBM backend (if token is available)
        if settings.IBM_API_TOKEN:
             logging.info("Testing IBM backend fetch (using least busy)...")
             test_config_ibm = {
                 "DEFAULT_BACKEND_MODE": "ibm_simulator", # or ibm_real_device
                 "USE_LEAST_BUSY_BACKEND": True
             }
             settings.__init__(**test_config_ibm) # Re-init settings
             try:
                  backend_ibm = get_backend(force_reload=True)
                  logging.info(f"Successfully got IBM backend: {backend_ibm.name}")
             except Exception as ibm_err:
                  logging.error(f"Failed to get IBM backend: {ibm_err}")
        else:
            logging.warning("Skipping IBM backend test: IBM_API_TOKEN not found in environment/config.")

        # Restore original settings
        settings.__init__(**vars(original_settings))


    except Exception as e:
        logging.error(f"Error during basic backend test: {e}", exc_info=True)


    # Sample test: Test running the full experiment (using test config)
    # logging.info("\n--- Testing run_decoder_experiment_instance ---")
    # test_config_exp = {
    #     "NUM_SYNDROME_SAMPLES_TRAIN": 10,
    #     "NUM_SYNDROME_SAMPLES_EVAL": 5,
    #     "SPSA_MAX_ITER": 3,
    #     "DEFAULT_BACKEND_MODE": 'aer_simulator', # Use simulator for quick test
    #     "SHOTS": 64,
    #     "EVAL_SHOTS": 128,
    #     "SEED": 54321
    # }
    # try:
    #     results = run_decoder_experiment_instance(config_override=test_config_exp)
    #     logging.info("--- Experiment Test Run Finished ---")
    #     if results.get("error"):
    #          logging.error(f"Experiment test failed with error: {results['error']}")
    #     else:
    #          logging.info("Experiment test completed successfully. Summary:")
    #          print(json.dumps(results.get("summary", {}), indent=2))
    #
    # except Exception as exp_err:
    #      logging.error(f"An error occurred during the experiment test run: {exp_err}", exc_info=True)

    logging.info("--- qis_logic.py testing finished ---")
