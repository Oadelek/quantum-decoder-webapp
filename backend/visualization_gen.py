# backend/visualization_gen.py
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import logging

# Import settings from config.py
from config import settings

# Use updated colormap access
try:
    cmap_category = matplotlib.colormaps['tab10'] # For discrete categories
    cmap_seq = matplotlib.colormaps['viridis'] # For sequential data if needed
except AttributeError:
    cmap_category = plt.cm.get_cmap('tab10')
    cmap_seq = plt.cm.get_cmap('viridis')
    logging.warning("Using legacy plt.cm.get_cmap. Update Matplotlib if possible.")
except KeyError:
    cmap_category = plt.cm.get_cmap('viridis')
    cmap_seq = plt.cm.get_cmap('viridis')
    logging.warning("Colormap 'tab10' not found, using 'viridis'.")


def _plot_to_base64(fig) -> str | None:
    """Converts a matplotlib figure to a base64 encoded PNG string."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig) # Close the figure!
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        if 'fig' in locals(): plt.close(fig) # Ensure close on error
        logging.error(f"Error converting plot to base64: {e}", exc_info=True)
        return None

def generate_lattice_plot(lattice_data: dict, qpus: list, qubit_indices: dict) -> str | None:
    """Generates the lattice plot as a base64 encoded PNG image."""
    logging.info("Generating lattice plot...")
    try:
        d = settings.CODE_DISTANCE
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = cmap_category
        # Data qubits
        data_positions = {p: n for p, n in lattice_data.items() if n.startswith('D')}
        if data_positions:
            ax.scatter([p[1] for p in data_positions], [-p[0] for p in data_positions], c='black', marker='o', s=150, label='Data', zorder=3)
            for p, n in data_positions.items(): ax.text(p[1], -p[0], n, ha='center', va='center', color='white', fontsize=8, zorder=4)
        # Ancilla qubits by QPU
        anc_positions = {p: n for p, n in lattice_data.items() if n.startswith('A')}
        for qpu_idx, qpu_names in enumerate(qpus):
            qpu_ancs = {p: n for p, n in anc_positions.items() if n in qpu_names}
            if not qpu_ancs: continue
            qpu_color = colors(qpu_idx % colors.N)
            first_anc_name = list(qpu_ancs.values())[0]
            marker = 's' if 'AX' in first_anc_name else 'd'
            ax.scatter([p[1] for p in qpu_ancs], [-p[0] for p in qpu_ancs], color=qpu_color, marker=marker, s=150, label=f'QPU {qpu_idx+1}', zorder=3)
            for p, n in qpu_ancs.items(): ax.text(p[1], -p[0], n, ha='center', va='center', color='black', fontsize=7, zorder=4)
        # Styling
        ax.set_title(f'Surface Code Lattice (d={d}) & QPU Map'); ax.set_xlabel('Column'); ax.set_ylabel('Row')
        ax.set_xlim(-0.5, d + 0.5); ax.set_ylim(-(d + 0.5), 0.5); ax.set_aspect('equal', adjustable='box')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        return _plot_to_base64(fig)
    except Exception as e: logging.error(f"Lattice plot error: {e}", exc_info=True); return None


def generate_accuracy_plot(results_data: dict) -> str | None:
    """Generates accuracy plot including VQNN, Classical, Baseline."""
    logging.info("Generating accuracy plot...")
    try:
        if 'architectures' not in results_data: return None
        plot_data = []
        for name, data in results_data['architectures'].items():
             if data.get('status') == 'Complete' or name.startswith('baseline'):
                  plot_data.append({
                       'name': name.replace('_decoder', '').replace('_vqnn', '').replace('classical_', ''),
                       'Accuracy': data.get('accuracy', 0.0),
                       'type': data.get('type', 'unknown')
                  })
        if not plot_data: return None
        plot_data.sort(key=lambda x: (x['type'] != 'baseline', x['type'] != 'classical', -x['Accuracy']))
        arch_names = [item['name'] for item in plot_data]; accuracies = [item['Accuracy'] for item in plot_data]; types = [item['type'] for item in plot_data]
        type_colors = {'vqnn': 'cornflowerblue', 'classical': 'lightcoral', 'baseline': 'grey'}
        bar_colors = [type_colors.get(t, 'lightgrey') for t in types]
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(arch_names, accuracies, color=bar_colors)
        ax.set_ylabel('Prediction Accuracy'); ax.set_title('Decoder Architecture Comparison - Accuracy')
        ax.tick_params(axis='x', rotation=35, labelsize=10); ax.set_ylim(0, max(1.05, ax.get_ylim()[1])); ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.bar_label(bars, padding=3, fmt='%.3f', fontsize=9)
        handles = [plt.Rectangle((0,0),1,1, color=type_colors[typename]) for typename in type_colors if any(t == typename for t in types)]
        labels = [typename.upper() for typename in type_colors if any(t == typename for t in types)]
        ax.legend(handles, labels, title="Model Type", fontsize=10)
        fig.tight_layout()
        return _plot_to_base64(fig)
    except Exception as e: logging.error(f"Accuracy plot error: {e}", exc_info=True); return None

def generate_time_plot(results_data: dict) -> str | None:
    """Generates execution time plot including VQNN, Classical, Baseline."""
    logging.info("Generating execution time plot...")
    try:
        if 'architectures' not in results_data or 'execution_times' not in results_data: return None
        exec_times = results_data['execution_times']
        plot_data = []
        for name, data in results_data['architectures'].items():
             if data.get('status') != 'Skipped (Insufficient Qubits)':
                  plot_data.append({
                       'name': name.replace('_decoder', '').replace('_vqnn', '').replace('classical_', ''),
                       'TrainingTime': exec_times.get('training', {}).get(name.replace('classical_','').replace('baseline_',''), 0), # Match original keys
                       'EvaluationTime': exec_times.get('evaluation', {}).get(name.replace('classical_','').replace('baseline_',''), 0),
                       'type': data.get('type', 'unknown')
                  })
        if not plot_data: return None
        plot_data.sort(key=lambda x: x['name']) # Sort alphabetically for consistency
        arch_names = [item['name'] for item in plot_data]
        training_times = [item['TrainingTime'] for item in plot_data]
        evaluation_times = [item['EvaluationTime'] for item in plot_data]
        fig, ax = plt.subplots(figsize=(12, 7))
        bar_width = 0.35; index = np.arange(len(arch_names))
        rects1 = ax.bar(index - bar_width/2, training_times, bar_width, label='Training Time (s)', color='salmon')
        rects2 = ax.bar(index + bar_width/2, evaluation_times, bar_width, label='Evaluation Time (s)', color='skyblue')
        ax.set_ylabel('Execution Time (seconds - Log Scale)'); ax.set_xlabel('Architecture')
        ax.set_title('Decoder Architecture Comparison - Execution Time'); ax.set_xticks(index); ax.set_xticklabels(arch_names)
        ax.tick_params(axis='x', rotation=35, labelsize=10); ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_yscale('log') # Log scale often better for time comparisons
        # Adjust y-limit slightly for log scale if values are very small
        min_time = min(filter(lambda x: x > 0, training_times + evaluation_times)) if any(t > 0 for t in training_times + evaluation_times) else 0.01
        ax.set_ylim(bottom=min_time * 0.5) # Start slightly below min non-zero time

        # Labels on log scale can be tricky, maybe omit or format carefully
        # ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=8)
        # ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=8)
        fig.tight_layout()
        return _plot_to_base64(fig)
    except Exception as e: logging.error(f"Time plot error: {e}", exc_info=True); return None


from qiskit import QuantumCircuit # Import for type hint
def generate_circuit_image(circuit: QuantumCircuit | None) -> str | None:
    """Generates a circuit diagram as a base64 encoded PNG image."""
    if not circuit or not isinstance(circuit, QuantumCircuit): return None
    logging.info(f"Generating circuit image for: {circuit.name}")
    try:
        fig = circuit.draw(output='mpl', style={'name': 'bw'}, fold=-1, initial_state=True) # Show initial state
        if fig is None: raise RuntimeError("circuit.draw returned None")
        return _plot_to_base64(fig)
    except ImportError: logging.error("Circuit plotting error: Optional dependency (e.g., pylatexenc) might be missing."); return None
    except Exception as e: logging.error(f"Circuit plot error for '{circuit.name}': {e}", exc_info=True); return None


def generate_training_history_plot(history_data: dict | list) -> str | None:
    """Generates plot of training cost history."""
    logging.info("Generating training history plot...")
    if not history_data: return None
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_title = 'Training Cost History'
    if isinstance(history_data, dict) and 'cost' in history_data: # Single history
        costs = history_data['cost']
        if costs: ax.plot(range(1, len(costs) + 1), costs, marker='.', linestyle='-', label='Avg Batch Cost', alpha=0.8)
    elif isinstance(history_data, list): # List of histories (localized)
        plot_title += ' (Localized QPUs)'
        colors = cmap_category
        for i, history in enumerate(history_data):
              if isinstance(history, dict) and 'cost' in history:
                   costs = history['cost']
                   if costs: ax.plot(range(1, len(costs) + 1), costs, marker='.', linestyle='-', label=f'QPU {i}', alpha=0.7, color=colors(i % colors.N))
        if any(isinstance(h, dict) and 'cost' in h for h in history_data): ax.legend(fontsize=9) # Add legend only if multiple lines plotted
    else: logging.warning("Unsupported history data format."); plt.close(fig); return None
    ax.set_title(plot_title); ax.set_xlabel('SPSA Iteration'); ax.set_ylabel('Average Batch Cost (-log P)')
    ax.set_yscale('log'); ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout()
    return _plot_to_base64(fig)