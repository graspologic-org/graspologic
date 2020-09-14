import numpy as np
import pytest
from numpy.testing import assert_equal

from graspy.pipeline import anomaly_detection


def load_data():
    f = np.load("tests/pipeline/data/anomaly.npz")
    graphs = f["graphs"]
    return graphs


def test_inputs():
    graphs = load_data()

    with pytest.raises(TypeError):
        anomaly = anomaly_detection(graphs, method=1)

    with pytest.raises(ValueError):
        anomaly = anomaly_detection(graphs, method="LaplacianSpectralEmbed")

    with pytest.raises(TypeError):
        anomaly = anomaly_detection(graphs, time_window="3")

    with pytest.raises(ValueError):
        anomaly = anomaly_detection(graphs, time_window=20)  # There are 12 input graphs

    with pytest.raises(ValueError):
        anomaly = anomaly_detection(graphs, time_window=1)

    with pytest.raises(TypeError):
        anomaly = anomaly_detection(graphs, use_lower_line="True")


def test_pipeline():
    """Based on the example in Guodong's repo"""
    graphs = load_data()

    (graph_indices, graph_dict, vertex_indices, vertex_dict) = anomaly_detection(
        graphs, method="omni", time_window=4, n_components=1
    )

    # Expected graphs
    assert_equal(graph_indices, [5, 10])

    # All 100 vertices in graphs 5 and 6 should be different
    assert len(vertex_indices[1][1]) == 100
    assert len(vertex_indices[2][1]) == 100