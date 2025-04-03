import math
import os
from dotenv import load_dotenv
from typing import Union

# Load environment variables from a .env file in the same directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

class Config:
    """Holds all configuration parameters for the experiment and application."""

    # --- Code Parameters ---
    CODE_DISTANCE: int = 3
    NUM_DATA_QUBITS: int = CODE_DISTANCE**2
    NUM_ANCILLA_QUBITS: int = CODE_DISTANCE**2 - 1
    LATTICE_DIM: int = CODE_DISTANCE # For visualization grid

    # --- Error Classes (Decoder Target) ---
    NUM_ERROR_CLASSES: int = 1 + NUM_DATA_QUBITS * 3 # Identity + (X, Y, Z per data qubit)
    MIN_VQC_QUBITS_NEEDED: int = math.ceil(math.log2(NUM_ERROR_CLASSES)) if NUM_ERROR_CLASSES > 0 else 1

    # --- Simulation Parameters ---
    QPUS_GRID_SIZE: int = 2 # For visualization partitioning logic
    NUM_QPUS: int = 2       # Number of simulated QPUs
    ERROR_RATES: list[float] = [0.01, 0.015] # MUST have NUM_QPUS elements
    READOUT_ERROR_PROB: float = 0.03
    INJECTED_ERROR_PROB_OVERALL: float = 0.10 # Probability of *one* single error

    # --- Execution Parameters ---
    SHOTS: int = 1024 # Shots for VQC execution during training cost evaluation
    EVAL_SHOTS: int = 2048 # Shots for final evaluation runs
    SEED: int | None = 42 # Set to integer for reproducibility, None for random

    # --- VQC Architecture Parameters ---
    VQC_REPS: int = 3 # Repetitions in EfficientSU2 layers
    CENTRALIZED_VQC_QUBITS: int = 6 # Should be >= MIN_VQC_QUBITS_NEEDED
    GLOBAL_VQC_QUBITS: int = 5      # Should be >= MIN_VQC_QUBITS_NEEDED
    LOCAL_VQC_QUBITS_PER_QPU: int = 5 # Should be >= MIN_VQC_QUBITS_NEEDED

    # --- Training Parameters ---
    SPSA_MAX_ITER: int = 50 # Max iterations for SPSA optimizer
    EPSILON: float = 1e-9 # Small value for numerical stability (e.g., log(0))
    NUM_SYNDROME_SAMPLES_TRAIN: int = 200
    NUM_SYNDROME_SAMPLES_EVAL: int = 100
    MAX_EXECUTION_TIME: int = 300 

    # --- Backend Selection ---
    DEFAULT_BACKEND_MODE: str = 'aer_simulator' # 'aer_simulator', 'simulator_stabilizer', 'ibm_simulator', 'ibm_real_device'
    DEFAULT_IBM_BACKEND:str = "ibm_brisbane" 

    IBM_API_TOKEN: str | None = os.getenv("IBM_API_TOKEN")
    IBM_INSTANCE: str | None = os.getenv("IBM_INSTANCE", "ibm-q/open/main") # Default instance
    IBM_TARGET_BACKEND: str | None = os.getenv("IBM_TARGET_BACKEND") # e.g., "ibm_brisbane", "simulator_stabilizer"
    USE_LEAST_BUSY_BACKEND: bool = True # Enable dynamic selection of the least busy backend

    
       

    def __init__(self, **kwargs):
        """Initialize and validate config, allowing overrides from kwargs."""
        # Apply overrides first
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Basic type validation/conversion
                try:
                    attr_type = self.__annotations__.get(key)
                    is_optional = False
                    if hasattr(attr_type, '__origin__') and attr_type.__origin__ is Union:
                        if type(None) in attr_type.__args__:
                            is_optional = True
                            non_none_types = [t for t in attr_type.__args__ if t is not type(None)]
                            if len(non_none_types) == 1:
                                attr_type = non_none_types[0]
                    if is_optional and value is None:
                        setattr(self, key, None)
                        continue
                    if attr_type and value is not None and not isinstance(value, attr_type):
                        if attr_type is int: value = int(value)
                        elif attr_type is float: value = float(value)
                        elif attr_type is bool: value = str(value).lower() in ['true', '1', 'yes']
                        elif attr_type is list[float]: value = [float(v) for v in value]
                        else:
                            print(f"Warning: Type mismatch for key '{key}'. Expected {attr_type}, got {type(value)}. Attempting to keep original value.")
                    setattr(self, key, value)
                except Exception as e:
                    print(f"Warning: Could not process override for key '{key}' with value '{value}'. Error: {e}. Keeping default.")
            else:
                print(f"Warning: Override key '{key}' not found in Config class.")

        # --- Sanity Checks (Run after potential overrides) ---
        if len(self.ERROR_RATES) != self.NUM_QPUS:
            raise ValueError(f"Config Error: Length of ERROR_RATES ({len(self.ERROR_RATES)}) must match NUM_QPUS ({self.NUM_QPUS})")
        min_q = self.MIN_VQC_QUBITS_NEEDED
        if self.CENTRALIZED_VQC_QUBITS < min_q: print(f"Config Warning: CENTRALIZED_VQC_QUBITS ({self.CENTRALIZED_VQC_QUBITS}) < min needed ({min_q}).")
        if self.GLOBAL_VQC_QUBITS < min_q: print(f"Config Warning: GLOBAL_VQC_QUBITS ({self.GLOBAL_VQC_QUBITS}) < min needed ({min_q}).")
        if self.LOCAL_VQC_QUBITS_PER_QPU < min_q: print(f"Config Warning: LOCAL_VQC_QUBITS_PER_QPU ({self.LOCAL_VQC_QUBITS_PER_QPU}) < min needed ({min_q}).")
        if self.DEFAULT_BACKEND_MODE in ['ibm_simulator', 'ibm_real_device']:
            if not self.IBM_API_TOKEN: raise ValueError("Config Error: IBM mode selected, but IBM_API_TOKEN is not set.")
            if not self.USE_LEAST_BUSY_BACKEND and not self.IBM_TARGET_BACKEND:
                raise ValueError("Config Error: IBM mode selected, but IBM_TARGET_BACKEND is not set and USE_LEAST_BUSY_BACKEND is False.")

# Global default settings instance (can be updated via API)
settings = Config()