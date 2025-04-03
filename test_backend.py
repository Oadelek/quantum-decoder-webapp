import sys
import os
import json
import logging
from time import time, sleep
from qiskit_ibm_runtime import QiskitRuntimeService, IBMRuntimeError
from qiskit.providers.exceptions import QiskitBackendNotFoundError

# --- Configuration: Add backend directory to Python path ---
# Ensure the script can find the 'backend' directory relative to its own location
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, 'backend'))

if backend_dir not in sys.path:
    # Insert at the beginning to prioritize local backend over installed packages if names clash
    sys.path.insert(0, backend_dir)
    print(f"DEBUG: Added '{backend_dir}' to sys.path") # Temporary print for debugging path issues
else:
    print(f"DEBUG: '{backend_dir}' already in sys.path")

# --- Import Backend Modules ---
try:
    from config import settings, Config
    from qis_logic import run_decoder_experiment_instance
    print("DEBUG: Successfully imported backend modules.") # Temporary print
except ImportError as e:
    print(f"FATAL: Error importing backend modules from '{backend_dir}'. Exception: {e}")
    print(f"Current sys.path: {sys.path}")
    # Attempt to list contents of backend_dir to help diagnose
    try:
        print(f"Contents of '{backend_dir}': {os.listdir(backend_dir)}")
    except FileNotFoundError:
        print(f"Error: Directory '{backend_dir}' not found.")
    except Exception as list_e:
        print(f"Error listing directory '{backend_dir}': {list_e}")
    sys.exit(1)
except Exception as e:
    # Catch other potential exceptions during import
    print(f"FATAL: An unexpected error occurred during backend module import: {e}")
    sys.exit(1)


# --- Logging Configuration ---
# Ensure the log file is placed within the potentially dynamically located backend_dir
log_file = os.path.join(backend_dir, 'test_run_decoder_log.log')
logging.basicConfig(
    level=logging.DEBUG,  # Capture detailed information for debugging
    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(filename)s:%(lineno)d] - %(message)s',
    filename=log_file,
    filemode='w'  # Overwrite log file each run for clean testing logs
)
# Console Handler for INFO level messages and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # Show INFO level messages on console
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# --- Test Configuration Overrides ---
# These settings are optimized for a quick test run, especially on hardware
test_config_overrides = {
    "NUM_SYNDROME_SAMPLES_TRAIN": 20,    # Drastically reduced for faster training phase
    "NUM_SYNDROME_SAMPLES_EVAL": 10,      # Drastically reduced for faster evaluation phase
    "SPSA_MAX_ITER": 5,                  # Very few SPSA iterations to save time
    "DEFAULT_BACKEND_MODE": 'aer_simulator', # Start with the simulator for training
    "SHOTS": 128,                         # Reduced shots for training (faster)
    "EVAL_SHOTS": 256,                   # Slightly more shots for evaluation, but still low
    "SEED": 12345,                       # Reproducibility
    "INJECTED_ERROR_PROB_OVERALL": 0.5,  # Increased to get error samples quickly (may impact accuracy)
    "USE_LEAST_BUSY_BACKEND": True,      # Important when targeting real hardware
    "DEFAULT_IBM_BACKEND": "ibm_brisbane", # Example: Specify a default preferred backend if USE_LEAST_BUSY is False or fails
    "MAX_EXECUTION_TIME": 300            # Set max execution time for jobs (useful for IBM Runtime) - Check if your qis_logic uses this
}

# --- IBM Runtime Usage Check Function ---
def check_runtime_service(service: QiskitRuntimeService) -> bool:
    """
    Performs basic checks on the QiskitRuntimeService instance.
    Currently checks if the service instance is valid and lists available backends.
    NOTE: A precise pre-check of *available runtime seconds* is not directly
          available via a simple API call. This function serves as a basic
          service health check before submitting jobs. Time limits should be
          enforced via job options (e.g., max_execution_time) or by timing
          the execution block.
    """
    if not service:
        logging.error("QiskitRuntimeService instance is not valid.")
        return False
    try:
        backends = service.backends()
        logging.info(f"Successfully connected to IBM Quantum. Available backends: {[b.name for b in backends]}")
        # Add any specific backend checks if needed, e.g., operational status
        # operational_backends = service.backends(operational=True)
        # logging.info(f"Operational backends: {[b.name for b in operational_backends]}")
        # if not operational_backends:
        #     logging.warning("No operational backends found.")
        #     # Depending on strictness, you might return False here
        return True
    except IBMRuntimeError as e:
        logging.error(f"Error interacting with IBM Runtime Service: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during service check: {e}")
        return False

# --- Main Execution Block ---
if __name__ == "__main__":
    start_run_time = time()
    logging.info("--- Starting Backend Test Run ---")
    logging.debug(f"Using Backend Directory: {backend_dir}")
    logging.info(f"Test Configuration Overrides: {json.dumps(test_config_overrides, indent=2)}")

    # == Step 1: Train on Aer Simulator ==
    logging.info("STEP 1: Training models on Aer simulator...")
    train_config = test_config_overrides.copy()
    train_config["DEFAULT_BACKEND_MODE"] = 'aer_simulator' # Ensure Aer is used

    try:
        train_start_time = time()
        results_aer = run_decoder_experiment_instance(config_override=train_config)
        train_elapsed = time() - train_start_time
        logging.info(f"Aer training phase completed in {train_elapsed:.2f}s")

        if results_aer.get("error"):
            logging.error(f"Aer training failed: {results_aer['error']}")
            # Optionally print full results for debugging even on error
            logging.debug(f"Full Aer results (error): {json.dumps(results_aer, indent=2, default=str)}")
            sys.exit(1) # Exit if training fails

        # Log key training results
        trained_params = {}
        if "architectures" in results_aer:
             trained_params = {arch: data.get('params') for arch, data in results_aer['architectures'].items() if data.get('params') is not None}
             logging.info(f"Successfully trained models on Aer: {list(trained_params.keys())}")
             logging.debug(f"Aer Training Summary: {json.dumps(results_aer.get('summary', {}), indent=2)}")
        else:
            logging.warning("No 'architectures' key found in Aer results. Cannot confirm trained models.")
            logging.debug(f"Full Aer results (no architectures key): {json.dumps(results_aer, indent=2, default=str)}")


    except Exception as e:
        logging.exception("An unexpected error occurred during the Aer training phase.")
        sys.exit(1)

    # == Step 2: Evaluate on IBM Hardware (Conditional) ==
    logging.info("STEP 2: Attempting evaluation on IBM hardware...")
    ibm_eval_config = test_config_overrides.copy()
    ibm_eval_config["DEFAULT_BACKEND_MODE"] = "ibm_real_device" # Switch to real device mode
    # Ensure EVAL_SHOTS is used if different from SHOTS
    ibm_eval_config["SHOTS"] = test_config_overrides.get("EVAL_SHOTS", ibm_eval_config["SHOTS"])

    service = None
    can_run_on_ibm = True
    try:
        # Retrieve token securely - Ensure 'settings.IBM_API_TOKEN' is correctly loaded
        ibm_api_token = getattr(settings, 'IBM_API_TOKEN', None)
        if not ibm_api_token:
            logging.error("IBM_API_TOKEN not found in settings. Cannot connect to IBM Quantum.")
        else:
            logging.info("Initializing QiskitRuntimeService...")
            # Consider adding instance parameter if needed: service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_token, instance="ibm-q/open/main")
            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_token)
            logging.info("Checking IBM Runtime Service status...")
            if check_runtime_service(service):
                 can_run_on_ibm = True
            else:
                logging.warning("IBM Runtime Service check failed. Skipping hardware evaluation.")

    except IBMRuntimeError as e:
        logging.error(f"Failed to initialize IBMRuntimeService: {e}. Check API token and network connection.")
    except Exception as e:
        logging.exception("An unexpected error occurred during IBM Service initialization.")


    if can_run_on_ibm:
        logging.info("Proceeding with IBM hardware evaluation...")
        try:
            eval_start_time = time()
            results_ibm = run_decoder_experiment_instance(config_override=ibm_eval_config)
            eval_elapsed = time() - eval_start_time
            logging.info(f"IBM evaluation attempt finished in {eval_elapsed:.2f}s") # Log time regardless of success

            if results_ibm.get("error"):
                logging.error(f"IBM evaluation phase failed: {results_ibm['error']}")
                # Log full results for debugging
                logging.debug(f"Full IBM results (error): {json.dumps(results_ibm, indent=2, default=str)}")
                # Decide whether to fall back or exit. Here we fall back to Aer results.
                logging.warning("Falling back to Aer results due to IBM evaluation error.")
                print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
                print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))

            else:
                logging.info("IBM evaluation completed successfully.")
                print("\n" + "="*20 + " IBM RESULTS SUMMARY " + "="*20)
                # Use default=str to handle potential non-serializable types like numpy arrays
                print(json.dumps(results_ibm.get("summary", {"message": "No summary available"}), indent=2, default=str))
                # Optionally log full IBM results
                logging.debug(f"Full IBM results (success): {json.dumps(results_ibm, indent=2, default=str)}")

        except QiskitBackendNotFoundError as e:
             logging.error(f"Backend not found during IBM evaluation: {e}. Check backend name and availability.")
             logging.warning("Falling back to Aer results due to backend not found error.")
             print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
             print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))
        except IBMRuntimeError as e:
             logging.error(f"An IBM Runtime error occurred during evaluation: {e}")
             logging.warning("Falling back to Aer results due to IBM Runtime error.")
             print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
             print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))
        except Exception as e:
            logging.exception("An unexpected error occurred during the IBM evaluation phase.")
            logging.warning("Falling back to Aer results due to unexpected error.")
            print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
            print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))

    else:
        logging.warning("Skipped IBM hardware evaluation based on service check or initialization failure.")
        print("\n" + "="*20 + " AER RESULTS SUMMARY (IBM Skipped) " + "="*20)
        print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))

    end_run_time = time()
    total_elapsed = end_run_time - start_run_time
    logging.info(f"--- Backend Test Run Finished --- Total Time: {total_elapsed:.2f}s")