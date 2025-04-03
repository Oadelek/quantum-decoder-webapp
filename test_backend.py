import sys
import os
import json
import logging
from time import time, sleep
from qiskit_ibm_runtime import QiskitRuntimeService, IBMRuntimeError 
from qiskit.providers.exceptions import QiskitBackendNotFoundError 

print("--- test_backend.py START ---")

# --- Configuration: Add backend directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, 'backend'))
print(f"DEBUG: Calculated script_dir: {script_dir}") 
print(f"DEBUG: Calculated backend_dir: {backend_dir}")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    print(f"DEBUG: Added '{backend_dir}' to sys.path")
else:
    print(f"DEBUG: '{backend_dir}' already in sys.path")
print(f"DEBUG: Current sys.path: {sys.path}") 

# --- Import Backend Modules ---
try:
    print("DEBUG: Attempting to import backend modules...") 
    from config import settings, Config
    from qis_logic import run_decoder_experiment_instance
    print("DEBUG: Successfully imported backend modules.")
except ImportError as e:
    print(f"FATAL: Error importing backend modules from '{backend_dir}'. Exception: {e}")
    try:
        print(f"Contents of '{backend_dir}': {os.listdir(backend_dir)}")
    except FileNotFoundError:
        print(f"Error: Directory '{backend_dir}' not found.")
    except Exception as list_e:
        print(f"Error listing directory '{backend_dir}': {list_e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: An unexpected error occurred during backend module import: {e}")
    sys.exit(1)


# --- Logging Configuration ---
print("DEBUG: Setting up logging...")
log_file = os.path.join(backend_dir, 'test_run_decoder_log.log')
print(f"DEBUG: Attempting to configure logging to file: {log_file}")

try:
    # Use force=True to remove potentially conflicting handlers
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - [%(threadName)s:%(filename)s:%(lineno)d] - %(message)s',
        filename=log_file,
        filemode='w',
        force=True  # <--- FORCE RECONFIGURATION
    )

    root_logger = logging.getLogger()

    # Add console handler (basicConfig might not add StreamHandler if filename is given)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) 
    console_formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    console_handler.setFormatter(console_formatter)

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
         root_logger.addHandler(console_handler)
         print("DEBUG: Console handler added.")
    else:
         # If basicConfig added one, ensure its level is DEBUG
         for h in root_logger.handlers:
              if isinstance(h, logging.StreamHandler):
                   h.setLevel(logging.DEBUG)
                   print("DEBUG: Existing StreamHandler level set to DEBUG.")


    print("DEBUG: Logging setup complete.")

    # --- !!! Diagnostic: Check effective levels !!! ---
    print(f"DEBUG: Root logger effective level: {logging.getLevelName(root_logger.getEffectiveLevel())}")
    print(f"DEBUG: Console handler level: {logging.getLevelName(console_handler.level)}")
    file_handler = next((h for h in root_logger.handlers if isinstance(h, logging.FileHandler)), None)
    if file_handler:
        print(f"DEBUG: File handler level: {logging.getLevelName(file_handler.level)}")
        print(f"DEBUG: File handler stream path: {file_handler.baseFilename}")
    else:
        print("DEBUG: File handler NOT found after basicConfig!")
    # --- End Diagnostic ---

    logging.info("--- This is the FIRST log message via logging (INFO) ---") # Test INFO
    logging.debug("--- This is the FIRST log message via logging (DEBUG) ---") # Test DEBUG

except Exception as log_ex:
    print(f"FATAL: Error during logging setup: {log_ex}")
    # Also log the exception to stderr just in case stdout is weird
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


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
    """
    if not service:
        logging.error("QiskitRuntimeService instance is not valid.")
        return False
    try:
        backends = service.backends()
        logging.info(f"Successfully connected to IBM Quantum. Available backends: {[b.name for b in backends]}")
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
    # Use logging now that it's configured (hopefully)
    logging.info("--- Starting Backend Test Run ---")
    logging.debug(f"Using Backend Directory: {backend_dir}")
    logging.info(f"Test Configuration Overrides: {json.dumps(test_config_overrides, indent=2)}")
    print("DEBUG: Entering main execution block (__name__ == '__main__')...")
    logging.debug("DEBUG log: Entering main execution block.") # Test logging again

    # == Step 1: Train on Aer Simulator ==
    logging.info("STEP 1: Training models on Aer simulator...")
    print("DEBUG: Preparing Step 1 (Aer Training)...") # Add print
    train_config = test_config_overrides.copy()
    train_config["DEFAULT_BACKEND_MODE"] = 'aer_simulator' # Ensure Aer is used

    results_aer = None # Initialize
    try:
        print("DEBUG: Calling run_decoder_experiment_instance for Aer...") # Add print
        logging.info("Calling run_decoder_experiment_instance for Aer...") # Add logging
        train_start_time = time()
        results_aer = run_decoder_experiment_instance(config_override=train_config)
        train_elapsed = time() - train_start_time
        print(f"DEBUG: run_decoder_experiment_instance for Aer returned. Elapsed: {train_elapsed:.2f}s") # Add print
        logging.info(f"Aer training phase call completed in {train_elapsed:.2f}s")

        if results_aer is None:
             logging.error("Aer results are None after execution!")
             print("ERROR: Aer results are None!")
             sys.exit(1)
        elif results_aer.get("error"):
            logging.error(f"Aer training failed: {results_aer['error']}")
            logging.debug(f"Full Aer results (error): {json.dumps(results_aer, indent=2, default=str)}")
            sys.exit(1) # Exit if training fails
        else:
             trained_params = {}
             if "architectures" in results_aer:
                  trained_params = {arch: data.get('params') for arch, data in results_aer['architectures'].items() if data.get('params') is not None}
                  logging.info(f"Successfully trained models on Aer: {list(trained_params.keys())}")
                  logging.debug(f"Aer Training Summary: {json.dumps(results_aer.get('summary', {}), indent=2)}")
             else:
                  logging.warning("No 'architectures' key found in Aer results.")
                  logging.debug(f"Full Aer results (no architectures key): {json.dumps(results_aer, indent=2, default=str)}")


    except Exception as e:
        print(f"FATAL: An unexpected error occurred during the Aer training phase call: {e}") # Add print
        logging.exception("An unexpected error occurred during the Aer training phase.") # Use logging.exception to capture traceback
        sys.exit(1)

    # == Step 2: Evaluate on IBM Hardware (Conditional) ==
    logging.info("STEP 2: Attempting evaluation on IBM hardware...")
    print("DEBUG: Preparing Step 2 (IBM Evaluation)...") # Add print
    ibm_eval_config = test_config_overrides.copy()
    ibm_eval_config["DEFAULT_BACKEND_MODE"] = "ibm_real_device" # Switch to real device mode
    # Ensure EVAL_SHOTS is used if different from SHOTS
    ibm_eval_config["SHOTS"] = test_config_overrides.get("EVAL_SHOTS", ibm_eval_config["SHOTS"])

    service = None
    can_run_on_ibm = False # Default to False unless service check succeeds
    try:
        # Retrieve token securely - Ensure 'settings.IBM_API_TOKEN' is correctly loaded
        # This relies on the successful import of 'settings' from config.py earlier
        ibm_api_token = getattr(settings, 'IBM_API_TOKEN', None)
        if not ibm_api_token:
            logging.error("IBM_API_TOKEN not found in settings. Cannot connect to IBM Quantum.")
            print("ERROR: IBM_API_TOKEN not found.") # Also print error
        else:
            logging.info("Initializing QiskitRuntimeService...")
            print("DEBUG: Initializing QiskitRuntimeService...") # Add print
            # Consider adding instance parameter if needed: service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_token, instance="ibm-q/open/main")
            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_token)
            logging.info("Checking IBM Runtime Service status...")
            print("DEBUG: Checking IBM Runtime Service status...") # Add print
            if check_runtime_service(service):
                 can_run_on_ibm = True
                 print("DEBUG: IBM Runtime Service check PASSED.") # Add print
            else:
                logging.warning("IBM Runtime Service check failed. Skipping hardware evaluation.")
                print("WARNING: IBM Runtime Service check FAILED.") # Add print

    except IBMRuntimeError as e:
        logging.error(f"Failed to initialize IBMRuntimeService: {e}. Check API token and network connection.")
        print(f"ERROR: Failed to initialize IBMRuntimeService: {e}") # Add print
    except Exception as e:
        logging.exception("An unexpected error occurred during IBM Service initialization.")
        print(f"ERROR: An unexpected error occurred during IBM Service initialization: {e}") # Add print


    if can_run_on_ibm:
        logging.info("Proceeding with IBM hardware evaluation...")
        print("DEBUG: Proceeding with IBM hardware evaluation...") # Add print
        try:
            print("DEBUG: Calling run_decoder_experiment_instance for IBM...") # Add print
            logging.info("Calling run_decoder_experiment_instance for IBM...") # Add logging
            eval_start_time = time()
            results_ibm = run_decoder_experiment_instance(config_override=ibm_eval_config)
            eval_elapsed = time() - eval_start_time
            print(f"DEBUG: run_decoder_experiment_instance for IBM returned. Elapsed: {eval_elapsed:.2f}s") # Add print
            logging.info(f"IBM evaluation attempt finished in {eval_elapsed:.2f}s")

            if results_ibm.get("error"):
                logging.error(f"IBM evaluation phase failed: {results_ibm['error']}")
                logging.debug(f"Full IBM results (error): {json.dumps(results_ibm, indent=2, default=str)}")
                logging.warning("Falling back to Aer results due to IBM evaluation error.")
                print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
                print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))

            else:
                logging.info("IBM evaluation completed successfully.")
                print("\n" + "="*20 + " IBM RESULTS SUMMARY " + "="*20)
                print(json.dumps(results_ibm.get("summary", {"message": "No summary available"}), indent=2, default=str))
                logging.debug(f"Full IBM results (success): {json.dumps(results_ibm, indent=2, default=str)}")

        except QiskitBackendNotFoundError as e:
             logging.error(f"Backend not found during IBM evaluation: {e}. Check backend name and availability.")
             logging.warning("Falling back to Aer results due to backend not found error.")
             print(f"ERROR: Backend not found during IBM evaluation: {e}") # Add print
             print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
             print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))
        except IBMRuntimeError as e:
             logging.error(f"An IBM Runtime error occurred during evaluation: {e}")
             logging.warning("Falling back to Aer results due to IBM Runtime error.")
             print(f"ERROR: IBM Runtime error during evaluation: {e}") # Add print
             print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
             print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))
        except Exception as e:
            logging.exception("An unexpected error occurred during the IBM evaluation phase.")
            print(f"FATAL: Unexpected error during IBM evaluation phase: {e}") # Add print
            logging.warning("Falling back to Aer results due to unexpected error.")
            print("\n" + "="*20 + " AER RESULTS SUMMARY (Fallback) " + "="*20)
            print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))

    else:
        logging.warning("Skipped IBM hardware evaluation based on service check or initialization failure.")
        print("INFO: Skipped IBM hardware evaluation.") # Add print
        print("\n" + "="*20 + " AER RESULTS SUMMARY (IBM Skipped) " + "="*20)
        print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))

    end_run_time = time()
    total_elapsed = end_run_time - start_run_time
    logging.info(f"--- Backend est Run Finished --- Total Time: {total_elapsed:.2f}s")

    # Add a final print statement
    print("--- test_backend.py END ---")
    logging.info("--- test_backend.py END ---")