import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import logging
import os
from typing import Dict, List, Tuple, Any, Callable # Added more types
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.patches as patches
import random
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

# Qiskit imports for plotting circuits and Bloch sphere
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_vector, circuit_drawer

# Import settings from config.py
from config import settings

try:
    from qis_logic import (
        initialize_d3_surface_code
    )
except ImportError:
    logging.error("Failed to import from qis_logic. Ensure qis_logic.py is accessible.")
    raise

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


def _save_or_encode_plot(fig, filepath: str | None = None, use_tight_layout: bool = True) -> str | None:
    """
    Saves a matplotlib figure to a file or encodes it as base64.

    Args:
        fig: The matplotlib figure object.
        filepath: The full path to save the file (e.g., 'plots/accuracy.png').
                  If None, returns base64 string.
        use_tight_layout: Whether to apply tight_layout before saving/encoding.

    Returns:
        Base64 encoded string if filepath is None, otherwise None.
    """
    if fig is None:
        logging.error("Attempted to save/encode a None figure object.")
        return None
    try:
        # Apply tight layout if requested and if axes exist
        if use_tight_layout and fig.get_axes():
             try:
                 fig.tight_layout()
             except Exception as tle:
                 logging.warning(f"Could not apply tight_layout: {tle}")

        if filepath:
            # Ensure directory exists
            dir_name = os.path.dirname(filepath)
            if dir_name: # Check if directory part exists
                 os.makedirs(dir_name, exist_ok=True)
            # Save the figure
            fig.savefig(filepath, format='png', bbox_inches='tight', dpi=150)
            logging.info(f"Plot saved to: {filepath}")
            plt.close(fig) # Close the figure after saving
            return None # Indicate saving was successful
        else:
            # Encode to base64
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig) # Close the figure after encoding
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig) # Ensure close on error
        logging.error(f"Error saving/encoding plot to '{filepath}': {e}", exc_info=True)
        return None



def generate_lattice_plot(lattice_data: dict, qpus: list, qubit_indices: dict, filepath: str | None = None) -> str | None:
    """Generates the lattice plot, saves to file or returns base64."""
    logging.info(f"Generating lattice plot {'(saving to file)' if filepath else '(encoding)'}...")
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
            label_prefix = 'X' if 'AX' in first_anc_name else 'Z'
            ax.scatter([p[1] for p in qpu_ancs], [-p[0] for p in qpu_ancs], color=qpu_color, marker=marker, s=150, label=f'{label_prefix}-Anc (QPU {qpu_idx})', zorder=3)
            for p, n in qpu_ancs.items(): ax.text(p[1], -p[0], n, ha='center', va='center', color='black', fontsize=7, zorder=4)
        # Styling
        ax.set_title(f'Surface Code Lattice (d={d}) & QPU Map'); ax.set_xlabel('Column Index'); ax.set_ylabel('Row Index')
        ax.set_xlim(-0.5, d + 0.5); ax.set_ylim(-(d + 0.5), 0.5); ax.set_aspect('equal', adjustable='box')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        # Adjust layout slightly *before* saving/encoding if possible (tight_layout in helper)
        fig.subplots_adjust(right=0.75) # Make space for legend

        return _save_or_encode_plot(fig, filepath, use_tight_layout=False) # Use pre-adjusted layout
    except Exception as e: logging.error(f"Lattice plot error: {e}", exc_info=True); return None

def generate_accuracy_plot(results_data: dict, filepath: str | None = None) -> str | None:
    """Generates accuracy plot, saves to file or returns base64."""
    logging.info(f"Generating accuracy plot {'(saving to file)' if filepath else '(encoding)'}...")
    try:
        if 'architectures' not in results_data: return None
        plot_data = []
        for name, data in results_data['architectures'].items():
             # Include completed models and baselines
             if data.get('status') == 'Complete' or 'baseline' in name:
                 # Handle potential None accuracy for baseline if calculation failed
                 accuracy = data.get('accuracy', None)
                 if accuracy is not None: # Only plot if accuracy exists
                      plot_data.append({
                           'name': name.replace('_decoder', '').replace('_vqnn', '').replace('classical_', '').replace('baseline_',''),
                           'Accuracy': accuracy,
                           'type': data.get('type', 'unknown')
                      })
        if not plot_data:
            logging.warning("No plottable accuracy data found.")
            return None
        # Sort by type (baseline, classical, vqnn) then by accuracy descending
        plot_data.sort(key=lambda x: (x['type'] == 'baseline', x['type'] == 'classical', -x['Accuracy']))
        arch_names = [item['name'] for item in plot_data]; accuracies = [item['Accuracy'] for item in plot_data]; types = [item['type'] for item in plot_data]
        type_colors = {'vqnn': 'cornflowerblue', 'classical': 'lightcoral', 'baseline': 'grey', 'unknown': 'lightgrey'}
        bar_colors = [type_colors.get(t, 'lightgrey') for t in types]
        fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted size slightly
        bars = ax.bar(arch_names, accuracies, color=bar_colors)
        ax.set_ylabel('Prediction Accuracy'); ax.set_title('Decoder Architecture Comparison - Accuracy')
        ax.tick_params(axis='x', rotation=30, ha='right', labelsize=9) # Adjusted rotation
        ax.set_ylim(0, max(1.05, ax.get_ylim()[1]))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.bar_label(bars, padding=3, fmt='%.3f', fontsize=9)
        # Create legend handles only for types present in the data
        present_types = sorted(list(set(types)))
        handles = [plt.Rectangle((0,0),1,1, color=type_colors.get(typename, 'lightgrey')) for typename in present_types]
        labels = [typename.upper() for typename in present_types]
        ax.legend(handles, labels, title="Model Type", fontsize=9)

        return _save_or_encode_plot(fig, filepath)
    except Exception as e: logging.error(f"Accuracy plot error: {e}", exc_info=True); return None

def generate_time_plot(results_data: dict, filepath: str | None = None) -> str | None:
    """Generates execution time plot, saves to file or returns base64."""
    logging.info(f"Generating execution time plot {'(saving to file)' if filepath else '(encoding)'}...")
    try:
        if 'architectures' not in results_data or 'execution_times' not in results_data: return None
        exec_times = results_data['execution_times']
        plot_data = []
        # Use architecture keys from results, not hardcoded names
        arch_keys = list(results_data['architectures'].keys())
        for name in arch_keys:
             data = results_data['architectures'][name]
             # Include if not skipped due to qubit count etc. (Training/Eval time might be 0)
             if data.get('status') not in ['Skipped (Insufficient Qubits)', 'Pending']:
                  # Clean name for display
                  clean_name = name.replace('_decoder', '').replace('_vqnn', '').replace('classical_', '').replace('baseline_','')
                  plot_data.append({
                       'name': clean_name,
                       'TrainingTime': exec_times.get('training', {}).get(name, 0) or 0, # Use arch name, default 0
                       'EvaluationTime': exec_times.get('evaluation', {}).get(name, 0) or 0, # Use arch name, default 0
                       'type': data.get('type', 'unknown')
                  })
        if not plot_data:
            logging.warning("No plottable time data found.")
            return None

        plot_data.sort(key=lambda x: x['name']) # Sort alphabetically
        arch_names = [item['name'] for item in plot_data]
        training_times = np.array([item['TrainingTime'] for item in plot_data])
        evaluation_times = np.array([item['EvaluationTime'] for item in plot_data])

        fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted size
        bar_width = 0.35; index = np.arange(len(arch_names))

        # Filter out zero times for log scale minimum calculation
        non_zero_times = np.concatenate((training_times[training_times > 0], evaluation_times[evaluation_times > 0]))
        min_time = np.min(non_zero_times) if len(non_zero_times) > 0 else 0.01
        log_threshold = 1e-3 # Threshold below which log scale might fail/look bad

        use_log_scale = min_time > 0 and np.max(np.concatenate((training_times, evaluation_times))) / min_time > 50 # Use log if significant range

        rects1 = ax.bar(index - bar_width/2, training_times, bar_width, label='Training Time (s)', color='salmon')
        rects2 = ax.bar(index + bar_width/2, evaluation_times, bar_width, label='Evaluation Time (s)', color='skyblue')

        ax.set_ylabel(f'Execution Time (seconds{" - Log Scale" if use_log_scale else ""})')
        ax.set_xlabel('Architecture'); ax.set_title('Decoder Architecture Comparison - Execution Time')
        ax.set_xticks(index); ax.set_xticklabels(arch_names)
        ax.tick_params(axis='x', rotation=30, ha='right', labelsize=9); ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        if use_log_scale:
             ax.set_yscale('log')
             ax.set_ylim(bottom=max(min_time * 0.5, log_threshold)) # Adjust bottom limit for log
        else:
            ax.set_ylim(bottom=0) # Linear scale starts at 0

        # Add labels if not using log scale or if values are reasonably large
        if not use_log_scale:
             ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
             ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)

        return _save_or_encode_plot(fig, filepath)
    except Exception as e: logging.error(f"Time plot error: {e}", exc_info=True); return None

def generate_training_history_plot(history_data: dict | list, arch_name: str, filepath: str | None = None) -> str | None:
    """Generates plot of training cost history, saves to file or returns base64."""
    logging.info(f"Generating training history plot for {arch_name} {'(saving to file)' if filepath else '(encoding)'}...")
    if not history_data: return None

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_title = f'{arch_name} Training Cost History'
    lines_plotted = 0

    if isinstance(history_data, dict) and 'cost' in history_data: # Single history (Centralized, Global)
        costs = history_data.get('cost', [])
        if costs and isinstance(costs, list) and all(isinstance(c, (int, float)) for c in costs):
            valid_costs = [c for c in costs if np.isfinite(c)] # Filter out non-finite costs
            if valid_costs:
                iterations = range(1, len(valid_costs) + 1)
                ax.plot(iterations, valid_costs, marker='.', linestyle='-', label='Avg Batch Cost', alpha=0.8)
                lines_plotted += 1
            else: logging.warning(f"No valid finite costs found in history for {arch_name}.")
        else: logging.warning(f"Invalid or empty cost history for {arch_name}: {costs}")

    elif isinstance(history_data, list): # List of histories (Localized)
        plot_title += ' (Localized QPUs)'
        colors = cmap_category
        max_len = 0
        plotted_indices = []
        for i, history in enumerate(history_data):
              if isinstance(history, dict) and 'cost' in history:
                   costs = history.get('cost', [])
                   if costs and isinstance(costs, list) and all(isinstance(c, (int, float)) for c in costs):
                        valid_costs = [c for c in costs if np.isfinite(c)]
                        if valid_costs:
                            iterations = range(1, len(valid_costs) + 1)
                            ax.plot(iterations, valid_costs, marker='.', linestyle='-', label=f'QPU {i}', alpha=0.7, color=colors(i % colors.N))
                            max_len = max(max_len, len(valid_costs))
                            lines_plotted += 1
                            plotted_indices.append(i)
                        else: logging.warning(f"No valid finite costs found in history for QPU {i} in {arch_name}.")
                   else: logging.warning(f"Invalid or empty cost history for QPU {i} in {arch_name}.")

        if lines_plotted > 1:
            ax.legend(title="QPU Index", fontsize=9)

    else:
        logging.warning(f"Unsupported history data format for {arch_name}: {type(history_data)}")
        plt.close(fig)
        return None

    if lines_plotted == 0:
        logging.warning(f"No data plotted for training history of {arch_name}.")
        ax.text(0.5, 0.5, 'No plottable training data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # Still save/encode the empty plot with the message
    else:
        ax.set_xlabel('SPSA Iteration')
        ax.set_ylabel('Average Batch Cost (-log P)')
        # Use log scale if costs span several orders of magnitude, else linear
        all_costs_flat = []
        if isinstance(history_data, dict): all_costs_flat = [c for c in history_data.get('cost',[]) if np.isfinite(c)]
        elif isinstance(history_data, list): all_costs_flat = [c for h in history_data if isinstance(h,dict) for c in h.get('cost',[]) if np.isfinite(c)]

        if all_costs_flat and (np.max(all_costs_flat) / np.min(all_costs_flat) > 100):
             ax.set_yscale('log')
             # Ensure y-lim starts slightly below the minimum finite cost
             min_cost = np.min(all_costs_flat)
             ax.set_ylim(bottom=min_cost * 0.8)
        else:
            # For linear scale, ensure y starts at or below 0 if costs can be negative (unlikely here) or near 0
             min_cost = np.min(all_costs_flat) if all_costs_flat else 0
             ax.set_ylim(bottom=min(0, min_cost * 1.1 if min_cost > 0 else min_cost * 0.9 ))
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

    ax.set_title(plot_title)
    return _save_or_encode_plot(fig, filepath)


def generate_bloch_sphere_plot(filepath: str | None = None) -> str | None:
    """Generates a simple Bloch sphere plot (e.g., |0> state), saves or returns base64."""
    logging.info(f"Generating generic Bloch sphere plot {'(saving to file)' if filepath else '(encoding)'}...")
    try:
        # Example: Plot the |0> state vector [1, 0, 0] in Bloch sphere coordinates (x,y,z)
        # |0> corresponds to the North Pole, so vector is [0, 0, 1]
        state_vector = [0, 0, 1]
        fig = plot_bloch_vector(state_vector, title="Generic Bloch Sphere (|0⟩ state)")
        return _save_or_encode_plot(fig, filepath, use_tight_layout=False) # Bloch plot handles layout
    except ImportError:
        logging.error("Failed to generate Bloch sphere: Qiskit visualization components not fully available.")
        return None
    except Exception as e:
        logging.error(f"Bloch sphere plot error: {e}", exc_info=True)
        return None

def generate_lattice_with_error_plot(lattice_data: dict, qpus: list, qubit_indices: dict,
                                     error_map: dict, title_suffix: str,
                                     filepath: str | None = None) -> str | None:
    """
    Generates the lattice plot highlighting specific Pauli errors.

    Args:
        lattice_data: Dictionary mapping (row, col) to qubit name.
        qpus: List of lists containing qubit names per QPU.
        qubit_indices: Dictionary mapping qubit name to its index.
        error_map: Dictionary {qubit_index: 'Pauli'} e.g., {0: 'X', 5: 'Z'}.
        title_suffix: String to append to the plot title (e.g., "with X Error on D0").
        filepath: Optional path to save the plot.

    Returns:
        Base64 string or None if saved to file.
    """
    logging.info(f"Generating lattice plot {title_suffix} {'(saving to file)' if filepath else '(encoding)'}...")
    try:
        d = settings.CODE_DISTANCE
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = cmap_category

        # Reverse lookup: index to name and name to position
        index_to_name = {v: k for k, v in qubit_indices.items()}
        name_to_pos = {v: k for k, v in lattice_data.items()}

        # Plot all qubits normally first
        data_positions = {p: n for p, n in lattice_data.items() if n.startswith('D')}
        anc_positions = {p: n for p, n in lattice_data.items() if n.startswith('A')}

        if data_positions:
            ax.scatter([p[1] for p in data_positions], [-p[0] for p in data_positions], c='grey', marker='o', s=150, label='Data', alpha=0.5, zorder=2)
            for p, n in data_positions.items(): ax.text(p[1], -p[0], n, ha='center', va='center', color='black', fontsize=8, alpha=0.7, zorder=2)

        for qpu_idx, qpu_names in enumerate(qpus):
            qpu_ancs = {p: n for p, n in anc_positions.items() if n in qpu_names}
            if not qpu_ancs: continue
            qpu_color = colors(qpu_idx % colors.N)
            first_anc_name = list(qpu_ancs.values())[0]
            marker = 's' if 'AX' in first_anc_name else 'd'
            label_prefix = 'X' if 'AX' in first_anc_name else 'Z'
            ax.scatter([p[1] for p in qpu_ancs], [-p[0] for p in qpu_ancs], color=qpu_color, marker=marker, s=150, label=f'{label_prefix}-Anc (QPU {qpu_idx})', alpha=0.5, zorder=2)
            for p, n in qpu_ancs.items(): ax.text(p[1], -p[0], n, ha='center', va='center', color='black', fontsize=7, alpha=0.7, zorder=2)

        # Highlight qubits with errors
        error_markers = {'X': ('X', 'red'), 'Y': ('Y', 'blue'), 'Z': ('Z', 'green'), 'I': ('I', 'grey')}
        plotted_error_labels = set()

        for q_idx, pauli in error_map.items():
            if q_idx not in index_to_name: continue
            q_name = index_to_name[q_idx]
            if q_name not in name_to_pos: continue
            pos = name_to_pos[q_name]
            marker, color = error_markers.get(pauli, ('?', 'black'))

            # Use a larger marker or outline for the error location
            label = f"Error: {pauli}" if pauli not in plotted_error_labels else ""
            ax.scatter(pos[1], -pos[0], c=color, marker=marker, s=350, label=label, zorder=5, edgecolors='black', linewidth=1.5)
            # Optional: Add text label next to the error marker
            ax.text(pos[1] + 0.1, -pos[0] + 0.1, f"{q_name}:{pauli}", color=color, fontsize=9, zorder=6)
            plotted_error_labels.add(pauli)


        # Styling
        ax.set_title(f'Surface Code Lattice (d={d})\n{title_suffix}'); ax.set_xlabel('Column Index'); ax.set_ylabel('Row Index')
        ax.set_xlim(-0.5, d + 0.5); ax.set_ylim(-(d + 0.5), 0.5); ax.set_aspect('equal', adjustable='box')
        # Combine legends smartly if possible, or just show error legend prominently
        handles, labels = ax.get_legend_handles_labels()
        # Filter out duplicate labels if any
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l not in unique_labels: unique_labels[l] = h
        ax.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        fig.subplots_adjust(right=0.75) # Make space for legend

        return _save_or_encode_plot(fig, filepath, use_tight_layout=False)
    except Exception as e: logging.error(f"Lattice with error plot error: {e}", exc_info=True); return None


def generate_classical_mlp_structure_plot(filepath: str | None = None) -> str | None:
    """Generates a *schematic* representation of the MLP structure."""
    logging.info(f"Generating schematic MLP structure plot {'(saving to file)' if filepath else '(encoding)'}...")
    try:
        # --- Get MLP parameters from Config ---
        # These are hardcoded in create_classical_models, retrieve them
        hidden_layer_sizes = (64, 32) # As defined in create_classical_models
        activation = 'relu'
        # Input size depends on number of stabilizers
        num_stabilizers = settings.NUM_ANCILLA_QUBITS
        # Output size depends on number of error classes
        num_outputs = settings.NUM_ERROR_CLASSES

        layer_sizes = [num_stabilizers] + list(hidden_layer_sizes) + [num_outputs]
        num_layers = len(layer_sizes)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off') # No axes needed

        v_spacing = 1.0 / num_layers
        h_spacing = 0.8 # Relative horizontal space for nodes

        node_radius = 0.08
        max_nodes_display = 8 # Max nodes to draw per layer

        for i, layer_size in enumerate(layer_sizes):
            layer_y = 1.0 - (i + 0.5) * v_spacing
            num_nodes_to_draw = min(layer_size, max_nodes_display)
            node_xs = np.linspace((1 - h_spacing)/2, 1 - (1 - h_spacing)/2, num_nodes_to_draw)

            for j, node_x in enumerate(node_xs):
                circle = plt.Circle((node_x, layer_y), node_radius, color='skyblue', ec='black', zorder=2)
                ax.add_patch(circle)
                if layer_size > max_nodes_display and j == num_nodes_to_draw - 2: # Indicate truncation
                    ax.text(node_x, layer_y, '...', ha='center', va='center', fontsize=12, zorder=3)
                # Add node count label below layer
                if j == num_nodes_to_draw // 2:
                     ax.text(node_x, layer_y - node_radius*1.8, f'({layer_size})', ha='center', va='top', fontsize=8, zorder=3)

            # Layer Label
            layer_name = ""
            if i == 0: layer_name = "Input (Syndrome)"
            elif i == num_layers - 1: layer_name = "Output (Class Probs)"
            else: layer_name = f"Hidden {i} ({activation})"
            ax.text(0.5, layer_y + node_radius*1.8, layer_name, ha='center', va='bottom', fontsize=9, zorder=3)

            # Draw connections (schematic - only between center nodes for clarity)
            if i > 0:
                prev_layer_y = 1.0 - (i - 1 + 0.5) * v_spacing
                prev_layer_size = layer_sizes[i-1]
                num_prev_nodes_to_draw = min(prev_layer_size, max_nodes_display)
                prev_node_xs = np.linspace((1 - h_spacing)/2, 1 - (1 - h_spacing)/2, num_prev_nodes_to_draw)

                # Draw lines from all prev drawn nodes to all current drawn nodes (can get dense)
                # Or simplify: draw lines only between the central nodes
                # Simplified approach:
                prev_center_idx = num_prev_nodes_to_draw // 2
                curr_center_idx = num_nodes_to_draw // 2
                ax.plot([prev_node_xs[prev_center_idx], node_xs[curr_center_idx]],
                        [prev_layer_y - node_radius, layer_y + node_radius], color='grey', alpha=0.6, zorder=1)
                # Add a couple more connections if layers aren't too big
                if num_prev_nodes_to_draw > 1 and num_nodes_to_draw > 1:
                     ax.plot([prev_node_xs[0], node_xs[0]], [prev_layer_y - node_radius, layer_y + node_radius], color='grey', alpha=0.4, zorder=1)
                     ax.plot([prev_node_xs[-1], node_xs[-1]], [prev_layer_y - node_radius, layer_y + node_radius], color='grey', alpha=0.4, zorder=1)


        ax.set_title("Schematic MLP Structure", fontsize=12)
        ax.set_aspect('equal') # Keep circles round
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1) # Extra space for title/labels

        return _save_or_encode_plot(fig, filepath, use_tight_layout=False) # No axes, layout managed manually
    except Exception as e:
        logging.error(f"MLP structure plot error: {e}", exc_info=True)
        return None

def generate_vqc_structure_plot(circuit: QuantumCircuit | None, title: str, filepath: str | None = None) -> str | None:
    """
    Generates a circuit diagram for a VQC, saves or returns base64.
    Uses generate_circuit_image internally.
    """
    logging.info(f"Generating VQC structure plot for {title} {'(saving to file)' if filepath else '(encoding)'}...")
    if not circuit or not isinstance(circuit, QuantumCircuit):
        logging.error(f"Invalid circuit provided for VQC structure plot: {circuit}")
        return None
    try:
        # Use Qiskit's drawer directly
        # Choose style options: 'mpl', 'text', 'latex', 'latex_source'
        # 'mpl' is good for PNG output
        # Use fold=-1 to prevent folding for potentially wider circuits
        fig = circuit_drawer(circuit, output='mpl', style={'name': 'iqp'}, fold=-1, initial_state=False, plot_barriers=True)

        if fig is None:
            raise RuntimeError(f"circuit_drawer returned None for {title}")

        # Set a title on the figure
        fig.suptitle(f"VQC Structure: {title}\n({circuit.num_qubits} Qubits, Reps={settings.VQC_REPS})", fontsize=10) # Add reps from config

        return _save_or_encode_plot(fig, filepath, use_tight_layout=True) # Let helper handle layout
    except ImportError as ie:
        logging.error(f"Circuit plotting error for '{title}': Missing optional dependencies (e.g., pylatexenc, Pillow). {ie}")
        return None
    except Exception as e:
        logging.error(f"VQC structure plot error for '{title}': {e}", exc_info=True)
        # Ensure figure is closed if an error occurs after creation
        if 'fig' in locals() and fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None
    

# Function to visualize the surface code
def visualize_surface_code():
    lattice, qpus, data_q_list, ancilla_q_list, qubit_indices, qpu_assignment_map = initialize_d3_surface_code()
    
    # Set up plot with better styling
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # Define colors for different qubit types and QPUs
    data_color = '#3498db'  # Blue
    x_ancilla_color = '#e74c3c'  # Red
    z_ancilla_color = '#2ecc71'  # Green
    qpu_colors = ['#f1c40f', '#9b59b6']  # Yellow, Purple
    
    # Background for QPU regions
    d = settings.CODE_DISTANCE
    for qpu_idx in range(settings.NUM_QPUS):
        boundary_step = settings.LATTICE_DIM / settings.NUM_QPUS
        left = 0
        right = d
        top = d
        bottom = 0
        
        # Calculate the QPU region boundaries
        qpu_left = left
        qpu_right = right
        qpu_bottom = bottom
        qpu_top = top
        
        if settings.NUM_QPUS > 1:
            qpu_left = left
            qpu_right = right
            qpu_bottom = qpu_idx * boundary_step
            qpu_top = min(top, (qpu_idx + 1) * boundary_step)
        
        # Draw the QPU region
        rect = patches.Rectangle((qpu_left, qpu_bottom), 
                                qpu_right - qpu_left, 
                                qpu_top - qpu_bottom, 
                                linewidth=1, 
                                edgecolor='gray', 
                                facecolor=qpu_colors[qpu_idx], 
                                alpha=0.1,
                                zorder=0)
        ax.add_patch(rect)
        
        # Add QPU label
        ax.text(qpu_left + 0.1, (qpu_top + qpu_bottom) / 2, 
                f"QPU {qpu_idx}", 
                fontsize=12, 
                color=qpu_colors[qpu_idx],
                fontweight='bold')
    
    # Add stabilizer faces (plaquettes)
    for r in range(d-1):
        for c in range(d-1):
            # Draw X stabilizer face
            rect = patches.Rectangle((c+0.5, r+0.5), 1, 1, 
                                    linewidth=2, 
                                    edgecolor=x_ancilla_color, 
                                    facecolor='none', 
                                    linestyle='--',
                                    alpha=0.7,
                                    zorder=1)
            ax.add_patch(rect)
    
    # Draw Z stabilizer faces
    z_faces = [
        [(0, 1), (0, 2), (1, 2), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 0)],
        [(1, 2), (1, 3), (2, 3), (2, 2)],
        [(2, 1), (2, 2), (3, 2), (3, 1)]
    ]
    
    for face in z_faces:
        x_coords = [point[1] for point in face]
        y_coords = [point[0] for point in face]
        ax.fill(x_coords, y_coords, 
                facecolor='none', 
                edgecolor=z_ancilla_color, 
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                zorder=1)
    
    # Draw connections between qubits
    for pos, name in lattice.items():
        if name.startswith('AX'):
            # X ancilla connects to data qubits in a + pattern
            r, c = pos
            neighbors = [(r-0.5, c-0.5), (r-0.5, c+0.5), (r+0.5, c-0.5), (r+0.5, c+0.5)]
            for neighbor in neighbors:
                if neighbor in lattice and lattice[neighbor].startswith('D'):
                    ax.plot([pos[1], neighbor[1]], [pos[0], neighbor[0]], 
                            color=x_ancilla_color, linewidth=1.5, alpha=0.7, zorder=2)
        elif name.startswith('AZ'):
            # Z ancilla connects to neighboring data qubits
            r, c = pos
            neighbors = []
            
            # Determine neighbors based on position
            if abs(r - 0.5) < 0.1 and abs(c - 1.5) < 0.1:  # Top Z ancilla
                neighbors = [(0.5, 0.5), (0.5, 1.5), (0.5, 2.5)]
            elif abs(r - 1.5) < 0.1 and abs(c - 0.5) < 0.1:  # Left Z ancilla
                neighbors = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)]
            elif abs(r - 1.5) < 0.1 and abs(c - 2.5) < 0.1:  # Right Z ancilla
                neighbors = [(0.5, 2.5), (1.5, 2.5), (2.5, 2.5)]
            elif abs(r - 2.5) < 0.1 and abs(c - 1.5) < 0.1:  # Bottom Z ancilla
                neighbors = [(2.5, 0.5), (2.5, 1.5), (2.5, 2.5)]
            
            for neighbor in neighbors:
                if neighbor in lattice and lattice[neighbor].startswith('D'):
                    ax.plot([pos[1], neighbor[1]], [pos[0], neighbor[0]], 
                            color=z_ancilla_color, linewidth=1.5, alpha=0.7, zorder=2)
    
    # Draw the qubits
    for pos, name in lattice.items():
        r, c = pos
        if name.startswith('D'):
            color = data_color
            marker = 'o'
            size = 300
        elif name.startswith('AX'):
            color = x_ancilla_color
            marker = 's'  # Square for X ancilla
            size = 250
        elif name.startswith('AZ'):
            color = z_ancilla_color
            marker = 'd'  # Diamond for Z ancilla
            size = 250
        
        # Get the qubit's QPU assignment
        idx = qubit_indices[name]
        qpu_idx = qpu_assignment_map.get(idx, 0)
        
        # Draw the qubit
        ax.scatter(c, r, color=color, s=size, marker=marker, 
                   edgecolor='black', linewidth=1.5, alpha=0.8, zorder=3)
        
        # Add qubit label
        ax.text(c, r, name, fontsize=9, ha='center', va='center', 
                color='black', fontweight='bold', zorder=4)
        
        # Add small indicator of QPU assignment
        ax.scatter(c + 0.2, r + 0.2, color=qpu_colors[qpu_idx], s=80, marker='o', 
                   edgecolor='black', linewidth=1, alpha=1.0, zorder=4)
    
    # Add grid lines
    for i in range(d+1):
        ax.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=i, color='gray', linestyle='-', alpha=0.3)
    
    # Title and other settings
    plt.title('d=3 Surface Code Qubit Layout', fontsize=16, fontweight='bold')
    
    # Add legend
    data_marker = plt.Line2D([], [], color=data_color, marker='o', linestyle='None', 
                           markersize=10, markeredgecolor='black', label='Data Qubit')
    x_marker = plt.Line2D([], [], color=x_ancilla_color, marker='s', linestyle='None', 
                         markersize=10, markeredgecolor='black', label='X Ancilla')
    z_marker = plt.Line2D([], [], color=z_ancilla_color, marker='d', linestyle='None', 
                         markersize=10, markeredgecolor='black', label='Z Ancilla')
    
    qpu0_marker = plt.Line2D([], [], color=qpu_colors[0], marker='o', linestyle='None', 
                            markersize=8, markeredgecolor='black', label='QPU 0')
    qpu1_marker = plt.Line2D([], [], color=qpu_colors[1], marker='o', linestyle='None', 
                            markersize=8, markeredgecolor='black', label='QPU 1')
    
    plt.legend(handles=[data_marker, x_marker, z_marker, qpu0_marker, qpu1_marker], 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add explanatory annotation
    info_text = (f"Surface Code (d={settings.CODE_DISTANCE}):\n"
                f"• {len(data_q_list)} Data Qubits\n"
                f"• {len([q for q in ancilla_q_list if q.startswith('AX')])} X Syndrome Ancillas\n"
                f"• {len([q for q in ancilla_q_list if q.startswith('AZ')])} Z Syndrome Ancillas\n"
                f"• {settings.NUM_QPUS} Quantum Processing Units")
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust plot limits
    plt.xlim(-0.5, d+0.5)
    plt.ylim(-0.5, d+0.5)
    
    # Hide axis ticks but keep grid
    plt.tick_params(axis='both', which='both', length=0)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_3d_mlp():
    # Set style for a clean, modern look
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure and 3D axis
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Custom color gradients
    input_cmap = LinearSegmentedColormap.from_list("input", [(0.1, 0.5, 0.8), (0.2, 0.7, 1.0)])
    hidden1_cmap = LinearSegmentedColormap.from_list("hidden1", [(0.8, 0.2, 0.6), (1.0, 0.4, 0.8)])
    hidden2_cmap = LinearSegmentedColormap.from_list("hidden2", [(0.9, 0.5, 0.1), (1.0, 0.7, 0.3)])
    output_cmap = LinearSegmentedColormap.from_list("output", [(0.1, 0.8, 0.4), (0.3, 1.0, 0.6)])
    cmaps = [input_cmap, hidden1_cmap, hidden2_cmap, output_cmap]

    # Network architecture
    layer_sizes = [6, 10, 8, 4]
    layer_depths = [0, 4, 8, 12]  # X-axis positions for layers
    layer_names = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]

    # Function to create 3D sphere coordinates
    def get_sphere_coordinates(center, radius, resolution=15):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    # Function to draw neuron connections with gradient color
    def draw_connection(ax, p1, p2, color1, color2, alpha=0.6, linewidth=1.5, segments=10):
        # Create gradient line with multiple segments
        for i in range(segments):
            t = i / segments
            t_next = (i + 1) / segments
            
            pt1 = (p1[0] * (1-t) + p2[0] * t, 
                p1[1] * (1-t) + p2[1] * t, 
                p1[2] * (1-t) + p2[2] * t)
            
            pt2 = (p1[0] * (1-t_next) + p2[0] * t_next, 
                p1[1] * (1-t_next) + p2[1] * t_next, 
                p1[2] * (1-t_next) + p2[2] * t_next)
            
            # Interpolate color
            r = color1[0] * (1-t) + color2[0] * t
            g = color1[1] * (1-t) + color2[1] * t
            b = color1[2] * (1-t) + color2[2] * t
            
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                    color=(r, g, b), alpha=alpha * (0.5 + 0.5*t), linewidth=linewidth)

    # Store neuron positions
    neuron_positions = []
    neuron_colors = []

    # Create "active" signals that will animate through the network
    signal_paths = []
        
    # Draw each layer
    for layer_idx, (size, depth, cmap) in enumerate(zip(layer_sizes, layer_depths, cmaps)):
        layer_positions = []
        layer_neuron_colors = []
        
        # Calculate positions for this layer (spiral arrangement)
        radius = 2.5 if layer_idx in (1, 2) else 2.0  # Make hidden layers slightly larger
        
        for i in range(size):
            if size == 1:
                angle = 0
            else:
                angle = i * 2 * np.pi / size
                
            # Neurons arranged in a circle on the YZ plane
            if layer_idx == 0:  # Input layer - flat circle
                x, y, z = depth, radius * np.cos(angle), radius * np.sin(angle)
            elif layer_idx == len(layer_sizes) - 1:  # Output layer - flat circle
                x, y, z = depth, radius * np.cos(angle), radius * np.sin(angle)
            else:  # Hidden layers - slight spiral
                spiral_offset = 0.5 * np.sin(i * np.pi / size)
                x, y, z = depth + spiral_offset, radius * np.cos(angle), radius * np.sin(angle)
            
            neuron_radius = 0.4 if layer_idx in (0, 3) else 0.5  # Slightly larger hidden neurons
            
            # Get color intensity based on position
            color_val = 0.2 + 0.8 * (i / size)
            color = cmap(color_val)
            layer_neuron_colors.append(color)
            
            # Draw the neuron as a sphere
            sphere_x, sphere_y, sphere_z = get_sphere_coordinates((x, y, z), neuron_radius)
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color, alpha=0.9, 
                            shade=True, antialiased=True)
            
            # Add label for input and output neurons
            if layer_idx == 0:
                ax.text(x - 0.8, y, z, f"x{i+1}", color='white', fontsize=10)
            elif layer_idx == len(layer_sizes) - 1:
                ax.text(x + 0.5, y, z, f"y{i+1}", color='white', fontsize=10)
                
            # Store position for connections
            layer_positions.append((x, y, z))
            
            # Create signal paths for animation
            if layer_idx < len(layer_sizes) - 1:
                # Each neuron connects to every neuron in the next layer
                for j in range(layer_sizes[layer_idx + 1]):
                    # We'll fill these with the next layer's positions later
                    signal_paths.append({"start": (x, y, z), 
                                        "start_color": color,
                                        "layer": layer_idx,
                                        "from_idx": i,
                                        "to_idx": j})
                    
        neuron_positions.append(layer_positions)
        neuron_colors.append(layer_neuron_colors)

    # Update signal paths with destination positions
    for path in signal_paths:
        layer, from_idx, to_idx = path["layer"], path["from_idx"], path["to_idx"]
        path["end"] = neuron_positions[layer + 1][to_idx]
        path["end_color"] = neuron_colors[layer + 1][to_idx]

    # Draw connections between layers
    for l in range(len(layer_sizes) - 1):
        for i, pos1 in enumerate(neuron_positions[l]):
            for j, pos2 in enumerate(neuron_positions[l + 1]):
                # Only connect some neurons for cleaner visualization in hidden layers
                if l == 1 and (i + j) % 3 != 0:  # Skip some connections in hidden layers
                    continue
                    
                color1 = neuron_colors[l][i]
                color2 = neuron_colors[l + 1][j]
                
                # Draw the connection
                draw_connection(ax, pos1, pos2, color1[:3], color2[:3], 
                            alpha=0.4, linewidth=1.0)

    # Add layer labels
    for i, (depth, name) in enumerate(zip(layer_depths, layer_names)):
        ax.text(depth, -4, 0, name, color='white', fontsize=14, 
                horizontalalignment='center', verticalalignment='center')

    # Set title and labels
    ax.set_title('Neural Network Architecture - 3D Visualization', 
                color='white', fontsize=16, pad=20)

    # Add annotation text
    info_text = (f"Architecture: {layer_sizes[0]}-{layer_sizes[1]}-{layer_sizes[2]}-{layer_sizes[3]}\n"
                f"Total Neurons: {sum(layer_sizes)}\n"
                f"Total Connections: {sum([layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1)])}")
                
    ax.text(6, 0, -4.5, info_text, color='white', fontsize=12, 
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5', 
                    edgecolor='white', linewidth=1))

    # Set view angle
    ax.view_init(elev=25, azim=-35)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Set axis labels
    ax.set_xlabel('Layer Depth', color='white', labelpad=10)
    ax.set_ylabel('', labelpad=10)
    ax.set_zlabel('', labelpad=10)

    # Set axis limits
    ax.set_xlim([-1, layer_depths[-1] + 1])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    # Add a subtle grid
    ax.grid(True, alpha=0.2, linestyle='--')

    # Remove panes and spines for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.set_box_aspect([2, 1, 1])

    plt.tight_layout()
    plt.show()


def visualize_2d_mlp():
    # Set up the figure with a white background
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Create a custom colormap for the neurons
    colors = [(0.2, 0.4, 0.6, 1.0), (0.3, 0.5, 0.7, 1.0)]
    cmap = LinearSegmentedColormap.from_list("MLP_cmap", colors, N=100)

    # Define the network architecture
    layer_sizes = [4, 8, 6, 3]  # Number of neurons in each layer
    layer_names = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
    n_layers = len(layer_sizes)

    # Define spacing parameters
    x_spacing = 3.0
    y_spacing = 1.0
    max_neurons = max(layer_sizes)

    # Calculate vertical positions for neurons in each layer
    y_positions = []
    for n_neurons in layer_sizes:
        positions = np.linspace(0, (n_neurons - 1) * y_spacing, n_neurons)
        positions -= np.mean(positions)  # Center around 0
        y_positions.append(positions)

    # Draw the neurons and layer labels
    neuron_radius = 0.3
    for i, (n_neurons, y_pos, name) in enumerate(zip(layer_sizes, y_positions, layer_names)):
        x = i * x_spacing
        
        # Add layer name
        ax.text(x, -max_neurons * y_spacing / 1.5, name, 
                horizontalalignment='center', fontsize=14, fontweight='bold')
        
        # Draw neurons
        for j, y in enumerate(y_pos):
            color_intensity = 0.4 + 0.6 * (j / n_neurons)
            circle = Circle((x, y), radius=neuron_radius, 
                        facecolor=cmap(color_intensity), 
                        edgecolor='black', linewidth=1.5, 
                        alpha=0.9, zorder=2)
            ax.add_patch(circle)
            
            # Add small label inside neuron
            if i == 0:
                ax.text(x, y, f"x{j+1}", ha='center', va='center', fontsize=10, fontweight='bold')
            elif i == n_layers - 1:
                ax.text(x, y, f"y{j+1}", ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw the connections between neurons
    for i in range(n_layers - 1):
        for j, y1 in enumerate(y_positions[i]):
            for k, y2 in enumerate(y_positions[i + 1]):
                # Calculate the intensity based on position (just for visual variation)
                intensity = 0.4 + 0.2 * (j / len(y_positions[i])) + 0.4 * (k / len(y_positions[i + 1]))
                
                # Draw the arrow with variable transparency and thickness
                arrow = FancyArrowPatch(
                    (i * x_spacing + neuron_radius, y1),
                    ((i + 1) * x_spacing - neuron_radius, y2),
                    arrowstyle='-|>', 
                    mutation_scale=15,
                    linewidth=1.0 + 0.5 * intensity,
                    color=cmap(intensity),
                    alpha=0.6,
                    connectionstyle="arc3,rad=0.1",
                    zorder=1
                )
                ax.add_patch(arrow)

    # Add a title and annotation boxes
    plt.title('Multilayer Perceptron (MLP) Architecture', fontsize=16, fontweight='bold', pad=20)

    # Add annotation for input size
    ax.text(0, max_neurons * y_spacing / 1.2, 
            f"Input Size: {layer_sizes[0]}", 
            ha='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    # Add annotation for output size
    ax.text((n_layers-1) * x_spacing, max_neurons * y_spacing / 1.2, 
            f"Output Size: {layer_sizes[-1]}", 
            ha='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    # Add formula annotation for hidden layers
    ax.text((n_layers-1) * x_spacing / 2, -max_neurons * y_spacing / 1.1, 
            r"$h = \sigma(W \cdot x + b)$", 
            ha='center', fontsize=14, 
            bbox=dict(boxstyle="round,pad=0.6", facecolor='lightblue', alpha=0.7))

    # Add grid for better visualization
    ax.grid(True, linestyle='--', alpha=0.3)

    # Set axis properties
    ax.set_xlim(-1, (n_layers) * x_spacing + 1)
    max_y = max_neurons * y_spacing / 1.5
    ax.set_ylim(-max_y, max_y)
    ax.set_axis_off()

    # Set the aspect ratio to be equal
    ax.set_aspect('equal', adjustable='datalim')

    # Add a legend/explanation box
    legend_text = (
        "Network Details:\n"
        f"• Layers: {n_layers}\n"
        f"• Hidden Neurons: {sum(layer_sizes[1:-1])}\n"
        "• Activation: ReLU (hidden), Softmax (output)"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()