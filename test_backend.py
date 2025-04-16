import sys
import os
import json
import logging
from time import time, sleep
import numpy as np # Needed for error map creation

# --- Qiskit Imports ---
from qiskit_ibm_runtime import QiskitRuntimeService, IBMRuntimeError
from qiskit.providers.exceptions import QiskitBackendNotFoundError
# Import QuantumCircuit if needed for type hints or direct use (though unlikely here)
# from qiskit import QuantumCircuit

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
    # Import visualization functions needed
    from visualization_gen import (
        generate_lattice_plot,
        generate_accuracy_plot,
        generate_time_plot,
        generate_training_history_plot,
        generate_bloch_sphere_plot,
        generate_lattice_with_error_plot,
        generate_classical_mlp_structure_plot,
        generate_vqc_structure_plot
    )
    print("DEBUG: Successfully imported backend modules.")
except ImportError as e:
    print(f"FATAL: Error importing backend modules from '{backend_dir}'. Exception: {e}")
    # (Keep existing error reporting for imports)
    try: print(f"Contents of '{backend_dir}': {os.listdir(backend_dir)}")
    except Exception: pass
    sys.exit(1)
except Exception as e:
    print(f"FATAL: An unexpected error occurred during backend module import: {e}")
    sys.exit(1)


# --- Logging Configuration ---
# (Keep existing logging setup - it seems reasonable)
print("DEBUG: Setting up logging...")
log_file = os.path.join(backend_dir, 'test_run_decoder_log.log')
print(f"DEBUG: Attempting to configure logging to file: {log_file}")
try:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(threadName)s:%(filename)s:%(lineno)d] - %(message)s', filename=log_file, filemode='w', force=True)
    root_logger = logging.getLogger()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Set console to INFO level for less noise
    console_formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    console_handler.setFormatter(console_formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers): root_logger.addHandler(console_handler)
    else:
         for h in root_logger.handlers:
              if isinstance(h, logging.StreamHandler): h.setLevel(logging.INFO) # Ensure level is INFO

    print("DEBUG: Logging setup complete.")
    logging.info("--- This is the FIRST log message via logging (INFO) ---")
    logging.debug("--- This is the FIRST log message via logging (DEBUG, check file) ---")
except Exception as log_ex:
    print(f"FATAL: Error during logging setup: {log_ex}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


# --- Test Configuration Overrides ---
test_config_overrides = {
    "NUM_SYNDROME_SAMPLES_TRAIN": 100,    # Reduced but not tiny
    "NUM_SYNDROME_SAMPLES_EVAL": 50
    ,
    "SPSA_MAX_ITER": 15,                 # More than 5 for some training effect
    "DEFAULT_BACKEND_MODE": 'aer_simulator',
    "SHOTS": 512,
    "EVAL_SHOTS": 1024,
    "SEED": 12345,
    "INJECTED_ERROR_PROB_OVERALL": 0.1,
    "USE_LEAST_BUSY_BACKEND": True,
    "DEFAULT_IBM_BACKEND": "ibm_brisbane",
    "PLOTS_OUTPUT_DIR": "run_plots"      # Specify output directory for plots
}

# --- IBM Runtime Usage Check Function ---
# (Keep existing check_runtime_service function)
def check_runtime_service(service: QiskitRuntimeService) -> bool:
    if not service: logging.error("QiskitRuntimeService instance is not valid."); return False
    try:
        backends = service.backends(); logging.info(f"Successfully connected. Backends: {[b.name for b in backends]}"); return True
    except IBMRuntimeError as e: logging.error(f"Error interacting IBM Service: {e}"); return False
    except Exception as e: logging.error(f"Unexpected error during service check: {e}"); return False

# --- Main Execution Block ---
if __name__ == "__main__":
    start_run_time = time()
    logging.info("--- Starting Backend Test Run with Plot Generation ---")
    logging.debug(f"Using Backend Directory: {backend_dir}")
    logging.info(f"Test Configuration Overrides: {json.dumps(test_config_overrides, indent=2)}")
    logging.debug("DEBUG log: Entering main execution block.")

    # Ensure plot directory exists using the setting from config
    plot_dir = test_config_overrides.get("PLOTS_OUTPUT_DIR", "simulation_plots_default")
    # Make it absolute relative to the script directory for clarity
    abs_plot_dir = os.path.abspath(os.path.join(script_dir, plot_dir))
    try:
        os.makedirs(abs_plot_dir, exist_ok=True)
        logging.info(f"Ensured plot output directory exists: {abs_plot_dir}")
    except OSError as e:
        logging.error(f"Could not create plot directory '{abs_plot_dir}': {e}")
        # Decide if this is fatal or if base64 fallback is okay (let's make it fatal for now)
        sys.exit(1)


    # == Step 1: Run Simulation on Aer Simulator ==
    logging.info("STEP 1: Running experiment on Aer simulator...")
    train_config = test_config_overrides.copy()
    train_config["DEFAULT_BACKEND_MODE"] = 'aer_simulator'

    results_aer = None
    setup_details = None # To store the setup info returned
    try:
        logging.info("Calling run_decoder_experiment_instance for Aer...")
        train_start_time = time()
        # Capture both return values
        results_aer, setup_details = run_decoder_experiment_instance(config_override=train_config)
        train_elapsed = time() - train_start_time
        logging.info(f"Aer experiment phase call completed in {train_elapsed:.2f}s")

        # Check results_aer first
        if results_aer is None:
             logging.error("Aer results are None after execution!")
             sys.exit(1)
        elif results_aer.get("error"):
            logging.error(f"Aer run failed: {results_aer['error']}")
            logging.debug(f"Full Aer results (error): {json.dumps(results_aer, indent=2, default=str)}")
            sys.exit(1) # Exit if run fails
        else:
             logging.info("Aer run completed successfully.")
             logging.debug(f"Aer Run Summary: {json.dumps(results_aer.get('summary', {}), indent=2)}")

        # Check setup_details
        if setup_details is None:
             logging.error("Setup details are None after execution! Cannot generate plots.")
             sys.exit(1)

    except Exception as e:
        logging.exception("An unexpected error occurred during the Aer run phase call.")
        sys.exit(1)

    # == Step 1.5: Generate Plots based on Aer Results and Setup Details ==
    logging.info("STEP 1.5: Generating plots from Aer run...")
    plot_generation_errors = False
    try:
        # --- Generate Basic/Static Plots ---
        generate_bloch_sphere_plot(filepath=os.path.join(abs_plot_dir, "bloch_sphere_generic.png"))

        if setup_details.get('lattice_data') and setup_details.get('qpus') and setup_details.get('qubit_indices'):
             generate_lattice_plot(setup_details['lattice_data'], setup_details['qpus'], setup_details['qubit_indices'],
                                   filepath=os.path.join(abs_plot_dir, "lattice_base.png"))
             # Generate lattice with example errors (e.g., on first data qubit)
             data_indices = setup_details.get('data_qubit_indices_list')
             if data_indices:
                 target_qubit_idx = data_indices[0] # Example: first data qubit index
                 target_qubit_name = {v: k for k, v in setup_details['qubit_indices'].items()}.get(target_qubit_idx, f'Idx {target_qubit_idx}')
                 # X Error
                 error_map_x = {target_qubit_idx: 'X'}
                 generate_lattice_with_error_plot(setup_details['lattice_data'], setup_details['qpus'], setup_details['qubit_indices'],
                                                  error_map_x, f"with X Error on {target_qubit_name}",
                                                  filepath=os.path.join(abs_plot_dir, f"lattice_error_X_{target_qubit_name}.png"))
                 # Y Error
                 error_map_y = {target_qubit_idx: 'Y'}
                 generate_lattice_with_error_plot(setup_details['lattice_data'], setup_details['qpus'], setup_details['qubit_indices'],
                                                  error_map_y, f"with Y Error on {target_qubit_name}",
                                                  filepath=os.path.join(abs_plot_dir, f"lattice_error_Y_{target_qubit_name}.png"))
                 # Z Error
                 error_map_z = {target_qubit_idx: 'Z'}
                 generate_lattice_with_error_plot(setup_details['lattice_data'], setup_details['qpus'], setup_details['qubit_indices'],
                                                  error_map_z, f"with Z Error on {target_qubit_name}",
                                                  filepath=os.path.join(abs_plot_dir, f"lattice_error_Z_{target_qubit_name}.png"))
             else: logging.warning("Could not find data qubit indices list in setup_details to generate error plots.")
        else: logging.warning("Missing lattice data/qpus/indices in setup_details for lattice plot.")

        # --- Generate Result-Dependent Plots ---
        if results_aer:
            generate_accuracy_plot(results_aer, filepath=os.path.join(abs_plot_dir, "comparison_accuracy.png"))
            generate_time_plot(results_aer, filepath=os.path.join(abs_plot_dir, "comparison_time.png"))

            # Generate training history plots for VQNNs if history exists
            for arch_name, arch_data in results_aer.get('architectures', {}).items():
                if arch_data.get('type') == 'vqnn' and 'training_history' in arch_data and arch_data['training_history']:
                    clean_name = arch_name.replace('_vqnn', '').replace('_decoder','')
                    generate_training_history_plot(arch_data['training_history'], clean_name,
                                                   filepath=os.path.join(abs_plot_dir, f"training_history_{clean_name}.png"))

        # --- Generate Structure Plots ---
        if setup_details.get('classical_model_defs', {}).get('mlp'):
            generate_classical_mlp_structure_plot(filepath=os.path.join(abs_plot_dir, "structure_classical_mlp.png"))
        else: logging.info("MLP definition not found in setup_details, skipping structure plot.")

        # Generate VQC structure plots from templates
        vqc_templates = setup_details.get('vqc_templates', {})
        for arch_name, template_circuit in vqc_templates.items():
             if template_circuit: # Check if template exists
                 clean_name = arch_name.replace('_vqnn', '').replace('_decoder','')
                 generate_vqc_structure_plot(template_circuit, clean_name.capitalize(),
                                             filepath=os.path.join(abs_plot_dir, f"structure_vqc_{clean_name}.png"))
             else: logging.warning(f"Template circuit missing for {arch_name}, cannot generate structure plot.")

        logging.info("Plot generation attempt complete.")

    except Exception as plot_err:
        logging.exception("An error occurred during plot generation.")
        plot_generation_errors = True # Flag that errors occurred


    # == Step 2: Evaluate on IBM Hardware (Conditional) ==
    # (Keep the IBM evaluation logic mostly the same)
    # Note: We are *not* generating plots based on IBM results in this example,
    # but you could add another plotting block here if desired, similar to Step 1.5.
    logging.info("STEP 2: Attempting evaluation on IBM hardware (plots based on Aer results)...")
    ibm_eval_config = test_config_overrides.copy()
    # Choose simulator or real device for the IBM step
    ibm_eval_config["DEFAULT_BACKEND_MODE"] = "ibm_simulator" # Use simulator for faster testing
    # ibm_eval_config["DEFAULT_BACKEND_MODE"] = "ibm_real_device" # Or use real device if intended
    ibm_eval_config["SHOTS"] = test_config_overrides.get("EVAL_SHOTS", ibm_eval_config["SHOTS"]) # Use eval shots

    service = None
    can_run_on_ibm = False
    try:
        ibm_api_token = getattr(settings, 'IBM_API_TOKEN', None)
        if not ibm_api_token:
            logging.warning("IBM_API_TOKEN not found in settings. Skipping IBM Quantum evaluation.")
        else:
            logging.info("Initializing QiskitRuntimeService for IBM evaluation...")
            instance = getattr(settings, 'IBM_INSTANCE', None)
            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_token, instance=instance)
            logging.info("Checking IBM Runtime Service status...")
            if check_runtime_service(service):
                 can_run_on_ibm = True
            else:
                logging.warning("IBM Runtime Service check failed. Skipping hardware evaluation.")
    except Exception as e:
        logging.exception("An error occurred during IBM Service initialization for evaluation.")

    # --- Execute IBM Run (if possible) ---
    results_ibm = None
    if can_run_on_ibm:
        logging.info("Proceeding with IBM evaluation...")
        try:
            logging.info("Calling run_decoder_experiment_instance for IBM...")
            eval_start_time = time()
            # Note: IBM run doesn't need to return setup_details again
            results_ibm, _ = run_decoder_experiment_instance(config_override=ibm_eval_config)
            eval_elapsed = time() - eval_start_time
            logging.info(f"IBM evaluation attempt finished in {eval_elapsed:.2f}s")

            if results_ibm is None:
                 logging.error("IBM results are None after execution!")
            elif results_ibm.get("error"):
                logging.error(f"IBM evaluation phase failed: {results_ibm['error']}")
                logging.debug(f"Full IBM results (error): {json.dumps(results_ibm, indent=2, default=str)}")
                logging.warning("Displaying Aer results due to IBM evaluation error.")
            else:
                logging.info("IBM evaluation completed successfully.")
                # You could generate plots based on IBM results here if needed

        except Exception as e:
             logging.exception("An unexpected error occurred during the IBM evaluation phase.")
             logging.warning("Displaying Aer results due to unexpected IBM evaluation error.")
             results_ibm = None # Ensure we fallback to Aer summary

    # --- Final Summary Output ---
    if results_ibm and not results_ibm.get("error"):
         print("\n" + "="*20 + " IBM RESULTS SUMMARY " + "="*20)
         print(json.dumps(results_ibm.get("summary", {"message": "No summary available"}), indent=2, default=str))
    elif results_aer: # Fallback to showing Aer results if IBM failed or was skipped
         status_reason = "IBM Skipped/Failed" if not can_run_on_ibm or results_ibm is None or results_ibm.get("error") else "Unknown"
         print(f"\n" + "="*20 + f" AER RESULTS SUMMARY ({status_reason}) " + "="*20)
         print(json.dumps(results_aer.get("summary", {"message": "No summary available"}), indent=2))
    else:
         print("\n" + "="*20 + " NO RESULTS AVAILABLE " + "="*20)


    end_run_time = time()
    total_elapsed = end_run_time - start_run_time
    logging.info(f"--- Backend Test Run Finished --- Total Time: {total_elapsed:.2f}s")
    if plot_generation_errors:
        logging.warning("There were errors during plot generation. Please check the log file.")
    else:
        logging.info(f"Plots generated successfully in: {abs_plot_dir}")

    print(f"--- test_backend.py END (Plots saved in '{abs_plot_dir}') ---")
    logging.info(f"--- test_backend.py END (Plots saved in '{abs_plot_dir}') ---")