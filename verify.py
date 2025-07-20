import numpy as np
from graph_generators import line_graph, fc_graph
from metrics import compute_walk_matrix, compute_diffusion_matrix

def test_walk_matrix():
    # Test with a simple line graph
    adj = line_graph(3)
    W = compute_walk_matrix(adj)
    expected_W = np.array([
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 1.]
    ])
    # Normalize expected_W by out-degrees
    # out_degrees for line_graph(3) are [1, 2, 1]
    expected_W_normalized = np.array([
        [0.5, 0., 0.],
        [0.5, 0.5, 0.],
        [0., 0.5, 1.]
    ])
    assert np.allclose(W, expected_W_normalized), f"Expected:\n{expected_W_normalized}\nGot:\n{W}"
    print("Walk matrix test passed!")

def test_diffusion_matrix():
    # Test with a simple line graph
    adj = line_graph(3)
    Delta = compute_diffusion_matrix(adj)
    expected_Delta = np.array([
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 1.]
    ])
    # Normalize expected_Delta by in-degrees
    # in_degrees for line_graph(3) are [1, 2, 1]
    expected_Delta_normalized = np.array([
        [1., 0., 0.],
        [0.5, 0.5, 0.],
        [0., 0.5, 0.5]
    ])
    assert np.allclose(Delta, expected_Delta_normalized), f"Expected:\n{expected_Delta_normalized}\nGot:\n{Delta}"
    print("Diffusion matrix test passed!")

if __name__ == "__main__":
    test_walk_matrix()
    test_diffusion_matrix()
