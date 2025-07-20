# TODO List for Computational Graph Prototype

This document outlines the remaining tasks to bring the prototype more in line with the "What Makes a Good Feedforward Computational Graph?" paper.

## 1. Implement Additional Graph Generators

Implement the following graph generators in `graph_generators.py`:

-   **Erdős-Rényi Graphs:** As discussed in Section 5.2.3 of the paper.
-   **Oriented Expander Graphs:** As discussed in Section 5.2.3 of the paper.
-   **Poisson(p) Graphs:** As detailed in Section 5.2.4 of the paper.

After implementing each, integrate them into `evaluate_graphs.py` to calculate and plot their mixing time and minimax fidelity (both raw and normalized).

## 2. Implement Machine Learning Models and Evaluation Tasks

To fully replicate the paper's experimental setup and evaluate the performance of different computational graphs, implement the following:

-   **Neural Network Framework Setup:** Choose a framework (e.g., PyTorch, TensorFlow) and set up the basic environment.
-   **Graph Attention Network (GAT) Architecture:** Implement the GAT-style model as described in Section 7 and Appendix E of the paper.
-   **Evaluation Tasks:** Create datasets and implement the logic for the three tasks mentioned in the paper:
    -   Finding the highest valued node (max retrieval).
    -   Finding the second highest valued node (second max retrieval).
    -   Computing the parity value of a bitstring.
-   **Training and Evaluation Pipeline:** Develop a pipeline to train the GAT models on these tasks using different computational graphs and measure their performance (e.g., test accuracy).
