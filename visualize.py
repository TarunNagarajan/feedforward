import numpy as np
import matplotlib.pyplot as plt

'''
PLOT I: mixing_time vs. n
PLOT II: minimax_fidelity vs. n

'''

def plot_metrics(n_values, mixing_time, minimax_fidelity, theoretical_minimax_fidelity=None, normalized_minimax_fidelity_line=None, normalized_minimax_fidelity_fc=None):
    plt.figure(figsize=(12, 10)) # Increased figure size for 4 subplots

    plt.suptitle("Graph Metrics")
    
    plt.subplot(2, 2, 1) # Changed to 2 rows, 2 columns, 1st plot
    plt.plot(n_values, mixing_time, marker = 'o', color = 'orchid')
    plt.title("Mixing Times")
    plt.xlabel("Nodes")
    plt.ylabel("Mixing Time")
    plt.grid(True)

    plt.subplot(2, 2, 2) # Changed to 2 rows, 2 columns, 2nd plot
    plt.plot(n_values, minimax_fidelity, marker = 'o', color = 'teal', label='Calculated')
    if theoretical_minimax_fidelity is not None:
        plt.plot(n_values, theoretical_minimax_fidelity, marker='x', color='red', linestyle='--', label='Raw Minimax Fidelity: $1/\sqrt{\pi(n-1)}')
        plt.legend()
    plt.title("Raw Minimax Fidelity (Line Graph)")
    plt.xlabel("Nodes")
    plt.ylabel("Minimax Fidelity")
    plt.grid(True)

    plt.subplot(2, 2, 3) # New subplot for Normalized Minimax Fidelity (Line Graph)
    plt.plot(n_values, normalized_minimax_fidelity_line, marker='o', color='blue', label='Line Graph (Normalized)')
    plt.title("Normalized Minimax Fidelity (Line Graph)")
    plt.xlabel("Nodes")
    plt.ylabel("Normalized Fidelity")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4) # New subplot for Normalized Minimax Fidelity (FC Graph)
    plt.plot(n_values, normalized_minimax_fidelity_fc, marker='s', color='green', label='FC Graph (Normalized)')
    plt.title("Normalized Minimax Fidelity (FC Graph)")
    plt.xlabel("Nodes")
    plt.ylabel("Normalized Fidelity")
    plt.ylim(0, 1.2) # Set y-axis limit from 0 to 1.2 to clearly show FC graph at 1
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    