# backend/app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import threading
import json
from uuid import uuid4
import numpy as np # For NumPy type checking/serialization
import os # For port env var

# --- Project Imports ---
from config import settings, Config # Import class for type checking/instance creation
from qis_logic import (
    run_decoder_experiment_instance, create_centralized_vqc, create_global_vqc,
    create_local_vqc, initialize_d3_surface_code, get_d3_stabilizers
)
from visualization_gen import (
    generate_lattice_plot, generate_accuracy_plot, generate_time_plot,
    generate_circuit_image, generate_training_history_plot
)

# --- App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Adjust origins for production

# --- Logging Setup ---
log_file = 'decoder_log.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    filename=log_file,
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - [%(threadName)s] - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)
logging.info("Flask application starting...")

# --- In-memory Job Store ---
jobs = {} # {job_id: {'status': str, 'result': dict|None, ...}}

# --- JSON Serialization Helper ---
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)): return obj.tolist() # Fallback for arrays
        elif isinstance(obj, (np.bool_)): return bool(obj)
        elif isinstance(obj, (np.void)): return None
        return super(NumpyJSONEncoder, self).default(obj)

def make_json_response(data, status_code=200):
     try:
          json_string = json.dumps(data, cls=NumpyJSONEncoder, indent=2)
          return Response(json_string, status=status_code, mimetype='application/json')
     except TypeError as e:
          logging.error(f"JSON Serialization Error: {e}", exc_info=True)
          error_data = {"error": "Failed to serialize response.", "details": str(e)}
          return Response(json.dumps(error_data), status=500, mimetype='application/json')

# --- Training Progress Callback ---
def training_update_callback(job_id):
    def callback(info):
        # Use Flask app context if needed for logging within thread
        # with app.app_context():
        if job_id in jobs:
            jobs[job_id]['status'] = 'training'
            jobs[job_id]['last_update'] = info
            iter_num = info.get('iteration', 0)
            if iter_num % 20 == 0 or iter_num == 1: # Log less frequently
                 cost_str = f"{info.get('cost', '?'):.4f}" if isinstance(info.get('cost'), float) else '?'
                 logging.info(f" Job {job_id} Prg: {info.get('arch','?')} {info.get('detail','')} Iter {iter_num}/{info.get('max_iter','?')}, Cost {cost_str}")
        # else: logging.warning(f"Callback for missing job {job_id}") # Can be noisy
    return callback

# --- Background Experiment Runner ---
def run_experiment_background(job_id, config_override):
    thread_name = threading.current_thread().name
    logging.info(f"Background task started for job {job_id} on {thread_name}")
    results = None # Initialize results variable
    try:
        callback = training_update_callback(job_id)
        # Run the core logic
        results = run_decoder_experiment_instance(
            config_override=config_override,
            training_callback=callback
        )

        # Check for fatal errors during the run before plotting
        if results.get('error'):
             jobs[job_id]['status'] = 'error'
             jobs[job_id]['result'] = results
             jobs[job_id]['error_message'] = results['error']
             logging.error(f"Job {job_id} failed during experiment: {results['error']}")
             return # Stop processing this job

        # --- Post-processing: Generate Plots ---
        logging.info(f"Generating plots for job {job_id}...")
        results['plots'] = {}
        setup_info = results.get('setup_info', {})
        try:
            # Re-init lattice info needed only for plotting if not already sent back
            # This avoids sending potentially large structures in results['setup_info']
            lattice_p, qpus_p, _, _, q_indices_p, _ = initialize_d3_surface_code()

            results['plots']['lattice'] = generate_lattice_plot(lattice_p, qpus_p, q_indices_p)
            results['plots']['accuracy'] = generate_accuracy_plot(results)
            results['plots']['time'] = generate_time_plot(results)

            # Generate training history plots
            for arch, data in results.get('architectures', {}).items():
                history_key = f'history_{arch}'
                if 'training_history' in data and data['training_history']:
                     history_plot = generate_training_history_plot(data['training_history'])
                     results['plots'][history_key] = history_plot # Will be None if generation failed

        except Exception as plot_err:
             logging.error(f"Plot generation failed for job {job_id}: {plot_err}", exc_info=True)
             # Store partial plots if needed, or just log error

        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['result'] = results
        logging.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logging.error(f"Critical error in background task {job_id}: {e}", exc_info=True)
        if job_id in jobs:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error_message'] = f"Unexpected background task error: {e}"
            # Store partial results if available
            jobs[job_id]['result'] = results if results else {"error": jobs[job_id]['error_message']}


# --- API Endpoints ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check."""
    return jsonify({"status": "ok", "message": "Backend is running."})

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the current *default* configuration settings."""
    logging.info("GET /api/config request")
    current_config = {k:v for k,v in vars(settings).items() if k != 'IBM_API_TOKEN' and not k.startswith('_')}
    return make_json_response(current_config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Updates the *global default* configuration."""
    global settings
    try:
        new_config_data = request.get_json()
        if not new_config_data: return make_json_response({"error": "Empty JSON data"}, 400)
        logging.info(f"POST /api/config request: {new_config_data}")
        # Create temporary instance for validation
        validated_settings = Config(**new_config_data)
        # If validation passes (no exception raised in __init__), update global
        settings = validated_settings
        logging.info("Global configuration updated.")
        current_config = {k:v for k,v in vars(settings).items() if k != 'IBM_API_TOKEN' and not k.startswith('_')}
        return make_json_response({"message": "Global configuration updated", "new_config": current_config})
    except (ValueError, TypeError) as e:
         logging.error(f"Config validation failed: {e}")
         return make_json_response({"error": f"Config validation failed: {e}"}, 400)
    except Exception as e:
        logging.error(f"Error POST /api/config: {e}", exc_info=True)
        return make_json_response({"error": "Server error updating config"}, 500)

@app.route('/api/run_experiment', methods=['POST'])
def start_experiment():
    """Starts the experiment in a background thread."""
    job_id = str(uuid4())
    config_override = request.get_json() if request.is_json else {}
    logging.info(f"POST /api/run_experiment. Job ID: {job_id}. Overrides: {config_override}")
    jobs[job_id] = {'status': 'queued', 'result': None, 'error_message': None, 'last_update': None}
    thread = threading.Thread(target=run_experiment_background, args=(job_id, config_override), name=f"Job-{job_id}")
    thread.daemon = True
    thread.start()
    logging.info(f"Job {job_id} thread started.")
    return make_json_response({"message": "Experiment submitted", "job_id": job_id}, 202)

@app.route('/api/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Gets the status and latest progress of a job."""
    logging.debug(f"GET /api/job_status/{job_id}")
    job = jobs.get(job_id)
    if not job: return make_json_response({"error": "Job not found"}, 404)
    response_data = {"job_id": job_id, "status": job['status'], "last_update": job.get('last_update'), "error_message": job.get('error_message')}
    return make_json_response(response_data)

@app.route('/api/results/<job_id>', methods=['GET'])
def get_job_results(job_id):
    """Gets the final results of a completed or errored job."""
    logging.info(f"GET /api/results/{job_id}")
    job = jobs.get(job_id)
    if not job: return make_json_response({"error": "Job not found"}, 404)
    status_code = 200; response_data = {"job_id": job_id, "status": job['status'], "error_message": job.get('error_message'), "data": job.get('result')}
    if job['status'] == 'error': status_code = 500
    elif job['status'] not in ['complete', 'error']: status_code = 202; response_data["message"] = "Job processing."; response_data["data"] = None # No final data yet
    if response_data["data"] is None and status_code != 202 : response_data["data"] = {} # Ensure data key exists for finished jobs
    return make_json_response(response_data, status_code=status_code)

@app.route('/api/example_circuit/<arch_type>', methods=['GET'])
def get_example_circuit_image(arch_type):
    """Generates and returns an image of an example initial VQC circuit."""
    # ... (Implementation from previous app.py - verified) ...
    logging.info(f"GET /api/example_circuit/{arch_type}")
    circuit = None; img_data_base64 = None
    try:
        example_settings = Config() # Use default config for example circuit
        lattice, qpus, _, _, qubit_indices, _ = initialize_d3_surface_code()
        num_syndrome_bits = example_settings.NUM_ANCILLA_QUBITS
        dummy_syndrome = '0' * num_syndrome_bits
        if arch_type == 'centralized': circuit, _ = create_centralized_vqc(dummy_syndrome)
        elif arch_type == 'distributed' or arch_type == 'global': circuit, _ = create_global_vqc(dummy_syndrome)
        elif arch_type == 'localized':
             if not qpus: raise ValueError("QPU assignments missing."); qpu_idx = 0; qpu_names = qpus[qpu_idx]
             circuit, _ = create_local_vqc(qpu_names, dummy_syndrome, qpu_idx, qubit_indices)
        else: return make_json_response({"error": f"Unknown arch type: {arch_type}"}, 400)
        if circuit: img_data_base64 = generate_circuit_image(circuit)
        if img_data_base64: return make_json_response({"image_base64": img_data_base64})
        else: return make_json_response({"error": "Failed to generate circuit image"}, 500)
    except Exception as e: logging.error(f"Error example circuit '{arch_type}': {e}", exc_info=True); return make_json_response({"error": f"Server error: {e}"}, 500)

# --- Main Entry Point ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logging.info(f"Starting Flask server on host 0.0.0.0, port {port}")
    # Use threaded=True for basic concurrency handling multiple requests
    # Set debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)