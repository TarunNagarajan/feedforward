import numpy as np
import matplotlib.pyplot as plt

from visualize import plot_metrics
from graph_generators import line_graph, fc_graph
from metrics import compute_walk_matrix, compute_diffusion_matrix, mixing_time, minimax_fidelity, compute_normalized_minimax_fidelity

if __name__ == "__main__":
    n_values = [16, 32, 64, 128, 256, 512]

    # start evaluating line graph type
    mixing_time_line = np.zeros_like(n_values, dtype=float)
    minimax_fidelity_line = np.zeros_like(n_values, dtype=float)
    theoretical_minimax_fidelity_line = np.zeros_like(n_values, dtype=float)
    normalized_minimax_fidelity_line = np.zeros_like(n_values, dtype=float)

    mixing_time_fc = np.zeros_like(mixing_time_line, dtype=float)
    minimax_fidelity_fc = np.zeros_like(minimax_fidelity_line, dtype=float)
    normalized_minimax_fidelity_fc = np.zeros_like(n_values, dtype=float)

    for i in range (0, len(n_values)):
        # generate the line graphs's adjacency matrix
        adj_line = line_graph(n_values[i])
        adj_fc = fc_graph(n_values[i])

        # compute it's walk matrix
        W_line = compute_walk_matrix(adj_line)
        W_fc = compute_walk_matrix(adj_fc)

        # compute it's diffusion matrix
        Delta_line = compute_diffusion_matrix(adj_line)
        Delta_fc = compute_diffusion_matrix(adj_fc)

        # compute metrics
        # Use a sufficiently large steps_bound for minimax_fidelity
        max_n_value = max(n_values)
        sufficient_steps_bound = max_n_value * 2 # Ensure enough steps for signal propagation

        mixing_line = mixing_time(W_line)
        mixing_fc = mixing_time(W_fc)

        fidelity_line = minimax_fidelity(Delta_line, steps_bound=sufficient_steps_bound)
        fidelity_fc = minimax_fidelity(Delta_fc, steps_bound=sufficient_steps_bound)

        # store them in the storage lists
        mixing_time_line[i] = mixing_line
        minimax_fidelity_line[i] = fidelity_line
        normalized_minimax_fidelity_line[i] = compute_normalized_minimax_fidelity(fidelity_line, n_values[i], "line")
        # Calculate theoretical minimax fidelity for line graph
        if n_values[i] > 1:
            theoretical_minimax_fidelity_line[i] = 1 / np.sqrt(np.pi * (n_values[i] - 1))
        else:
            theoretical_minimax_fidelity_line[i] = 0 # Handle n=1 case

        mixing_time_fc[i] = mixing_fc
        minimax_fidelity_fc[i] = fidelity_fc
        normalized_minimax_fidelity_fc[i] = compute_normalized_minimax_fidelity(fidelity_fc, n_values[i], "fc")

    # visualize
    print("Calculated Line Graph Minimax Fidelities:", minimax_fidelity_line)
    print("Plotting Line Graph Metrics [mixing_time, minimax_fidelity]...")
    plot_metrics(n_values, mixing_time_line, minimax_fidelity_line, theoretical_minimax_fidelity_line, normalized_minimax_fidelity_line, normalized_minimax_fidelity_fc)

    print("Calculated Fully Connected Graph Minimax Fidelities:", minimax_fidelity_fc)
    print("Plotting Fully Complete Graph Metrics [mixing_time, minimax_fidelity]...")
    plot_metrics(n_values, mixing_time_fc, minimax_fidelity_fc, None, normalized_minimax_fidelity_line, normalized_minimax_fidelity_fc)
        



