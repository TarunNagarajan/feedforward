import numpy as np

"""
    Here, we produce normalized versions of the adjacency matrix, 
    which serve different modelling goals.

    1. Confused about the use of np.newaxis:
    In the case of the walk matrix W:
        we're dividing each row of adj by the corresponding out_degree
    In the case of the diffusion matrix Delta:
        we're dividing each col of adj by the corresponding in_degree
"""

def compute_walk_matrix(adj):
    # out degree of node j is the sum of col j
    out_degrees = np.sum(adj, axis = 0)
    # just in case...
    out_degrees[out_degrees == 0] = 1
    # walk matrix
    W = adj / out_degrees[np.newaxis, :]
    return W

def compute_diffusion_matrix(adj):
    # in degree of node j is the sum of row j
    in_degrees = np.sum(adj, axis = 1)
    # just in case
    in_degrees[in_degrees == 0] = 1
    # delta
    Delta = adj / in_degrees[:, np.newaxis]
    return Delta

def mixing_time(W, step_bound = 10000):
    # calculates the average mixing time for the walk matrix W.
    n = W.shape[0] 

    # walk matrix at time_step t
    W_t = W.copy() 

    # let's say that the stationary distribution is a one-hot vector at the sink node n - 1
    pi = np.zeros(n)
    pi[n - 1] = 1 

    # the columns of W_t represent the probability of random walkers starting at each respective node.  

    for t in range(1, step_bound + 1):
        # we'll broadcast and subtract the standard distribution
        distances = np.sum(np.abs(W_t - pi[:, np.newaxis]), axis = 0) 
        avg_distances = np.mean(distances)

        # Section 4.3, "There is nothing special in the use of 1/4 here. Any fixed 0 < ε < 1/2 would work."
        # changing it doesn't change overall scaling or relative ranking.  
        if (avg_distances < 0.25):
            return t
        
        # increase the power of the walk
        W_t = W_t @ W
        
        """
        here, we are effectively performing W @ W
        I'd like to break this down in terms of the dot-products that take place during the matmul

        the second 'W' is the departure board, and the col 'k' (say) is the subject of interest here
        (if I am at node k, where can I possibly go in the very next step?)
        that column k is a complete probability distribution, with the sum = 1 (except the sink node, where there is no escape)

        the first 'W' is the arrivals board, and the row 'i' (say) is the subject of interest here
        this is not a probability distribution, but rather a set of conditional probabilities for:
        (if I wanted to land on node i, where could I have come from?)

        """
    
    # the loop finishes, and we know that the condition was not met.
    return step_bound

def minimax_fidelity(Delta, steps_bound = 100):
    # calculate the minimax fidelity for a given diffusion matrix delta

    # fidelity refers to the signal purity
    # high fidelity -> signal is clear and distinct
    # low fidelity -> signal is overly blended

    # what fraction of total effect came from the node i?

    #  Δ_ij = 1 / d_in(i) if there's an edge from j to i.

    # fidelity issues:
    # 1. for the nodes nearest to the sink, the sound is pure in the first step, but after that, it gets averaged due to the sink itself, and starts averaging it in.
    # 2. for the nodes farthest away from the sink, the sound takes a lot of time to even reach the sink.

    '''a graph with high minimax fidelity is good because it guarantees that for every input node, there exists some layer in the
        network where its information is represented clearly. This provides a rich set of useful intermediate representations that the final
        layers can draw upon via skip connections to make a better decision.'''
        
    # this goes on to say that we SHOULD care about the moment of maximum impact

    '''
    * Column `k` of `Delta`: This column represents how node k's signal (or value) is distributed to all other nodes in one step. If node k has an
     outgoing edge to i, then Δ_ik will be non-zero.
       * del_0k: How much of k's signal goes to 0 in one step.
       * del_1k: How much of k's signal goes to 1 in one step.
       * ...and so on.


    * Row `i` of `Delta`: This row represents how node i's new value is formed by averaging its inputs in one step.
       * del_i0: How much node 0 contributes to i's value.
       * del_i1: How much node 1 contributes to i's value.
       * ...and so on.
    '''

    n = Delta.shape[0]
    sink_node = n - 1

    # store the fidelities for all nodes
    max_fidelities = np.zeros(n) 

    # a copy for the time step version
    Delta_t = Delta.copy()

    print(f"--- Minimax Fidelity Debug for n={n} ---")
    print(f"Initial Delta matrix: {Delta}")

    for t in range(1, steps_bound + 1):
        fidelities_t = Delta_t[sink_node, :] # convert to row vector
        # preparing for the next step
        max_fidelities = np.maximum(fidelities_t, max_fidelities);
        Delta_t = Delta_t @ Delta
        
        if t % 100 == 0 or t == 1 or t == steps_bound: # Print at intervals or important steps
            print(f"Step {t}: fidelities_t (row {sink_node} of Delta_t): {fidelities_t}")
            print(f"Step {t}: max_fidelities: {max_fidelities}")
    
    minimax_fidelity_val = np.min(max_fidelities[:-1]) # excluding the sink node (ie the last element)
    print(f"Final minimax_fidelity (excluding sink node): {minimax_fidelity_val}")
    print(f"--- End Minimax Fidelity Debug for n={n} ---")
    return minimax_fidelity_val

def compute_normalized_minimax_fidelity(raw_fidelity, n, graph_type):
    if graph_type == "fc":
        return 1.0
    elif graph_type == "line":
        if n > 0:
            return np.sqrt(n) / np.pi
        else:
            return 0.0 # Handle n=0 case, though not expected
    else:
        # For other graph types, we might not have a defined normalization
        return raw_fidelity # Return raw fidelity if normalization is unknown



