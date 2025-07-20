import numpy as np

def line_graph(n):
    """
    Generates the adjacency matrix for a line graph.
    
    """
    # A[i, j] = 1 means edge from j to i
    adj = np.zeros((n, n), dtype=int)
    # Add self-edges
    np.fill_diagonal(adj, 1)
    # Add forward edges from i to i+1
    for i in range(n - 1):
        adj[i + 1, i] = 1
    return adj

def fc_graph(n):
    """
    Generates the adjacency matrix for a fully connected feedforward graph.
    
    """
    # A[i, j] = 1 means edge from j to i
    adj = np.zeros((n, n), dtype=int)

    # An edge from j to i is allowed if j < i. Also add self-edges (j=i)
    # (lower-triangular matrix)
    adj = np.tril(np.ones((n, n), dtype=int))
    return adj