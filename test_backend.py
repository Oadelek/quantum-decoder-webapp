import sys
import os
import json
import logging

# --- Add backend directory to Python path ---
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# --- Import backend components ---
try:
    from config import settings, Config
    from qis_logic import run_decoder_experiment_instance
    log_file = os.path.join(backend_dir, 'test_run_decoder_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        filename=log_file,
        filemode='w'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
except ImportError as e:
    print(f"Error importing backend modules. Make sure you are in the project root directory.")
    print(f"Sys path: {sys.path}")
    print(f"Error details: {e}")
    sys.exit(1)

# --- Test Configuration ---
test_config_overrides = {
    "NUM_SYNDROME_SAMPLES_TRAIN": 20,
    "NUM_SYNDROME_SAMPLES_EVAL": 10,
    "SPSA_MAX_ITER": 5,
    "DEFAULT_BACKEND_MODE": 'ibm_real_device',
    "SHOTS": 128,
    "EVAL_SHOTS": 256,
    "SEED": 12345,
    "USE_LEAST_BUSY_BACKEND": True  
}

# --- Main Test Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Backend Test Run ---")
    logging.info(f"Using test overrides: {json.dumps(test_config_overrides, indent=2)}")

    try:
        results = run_decoder_experiment_instance(config_override=test_config_overrides)
        logging.info("\n--- Experiment Run Finished ---")
        if results.get("error"):
            logging.error(f"Experiment failed with error: {results['error']}")
        else:
            logging.info("Experiment completed successfully.")
            print("\n" + "="*20 + " RESULTS SUMMARY " + "="*20)
            if "summary" in results:
                print(json.dumps(results["summary"], indent=2))
            else:
                print("No summary found in results.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the test run: {e}", exc_info=True)
        print(f"\nTEST RUN FAILED: {e}")

    logging.info("--- Backend Test Run Finished ---")