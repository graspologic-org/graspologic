import numpy as np
from graspy.embed import MultipleASE, OmnibusEmbed
from scipy.special import gamma
from sklearn.base import BaseEstimator


class AnomalyDetection(BaseEstimator):
    """

    """

    def __init__(self, method="omni", time_window=3, diag_aug=True, **svd_kwargs):
        self.method = method
        self.time_window = time_window
        # self.n_components = n_components
        self.diag_aug = diag_aug
        self.svd_kwargs = svd_kwargs

    def _embed(self, X):
        """
        Computes sequential pairwise embeddings
        """
        if self.method == "omni":
            embedder = OmnibusEmbed(**self.svd_kwargs)  # TODO: add diag_aug
        elif self.method == "mase":
            embedder = MultipleASE(**self.svd_kwargs)

        embeddings = [
            embedder.fit_transform(X[i : i + 2]) for i in range(self.n_graphs_ - 1)
        ]

        return embeddings

    def _compute_statistics(self, embeddings):
        """
        For graph anomaly detection, computes spectral norm between each pair-wise
        embeddings. For vertex anomaly detection, computes 2-norm between each
        pair-wise embeddings for each vertex.

        Parameters
        ----------
        embeddings : list of ndarray

        Returns
        -------

        """
        graph_stats = np.array(
            [
                np.linalg.norm(embeddings[i][0] - embeddings[i][1], ord=2)
                for i in range(len(embeddings))
            ]
        )

        vertex_stats = np.array(
            [
                np.linalg.norm(embeddings[i][0] - embeddings[i][1], ord=None, axis=1)
                for i in range(len(embeddings))
            ]
        )

        return graph_stats, vertex_stats

    def _compute_graph_control_chart(self, stats):
        el = self.time_window
        m = self.n_graphs_

        # Compute moving means
        means = np.empty(m - el)
        for i in range(m - el):
            means[i] = stats[i : i + el - 1].sum() / (el - 1)

        # Compute moving standard deviations
        diffs = np.abs(np.diff(stats))
        stds = np.empty(m - el)
        for i in range(m - el):
            stds[i] = diffs[i : i + el - 2].sum() / (1.128 * (el - 2))

        upper_central_line = means + 3 * stds
        lower_central_line = means - 3 * stds

        return means, stds, upper_central_line, lower_central_line

    def _compute_vertex_control_chart(self, stats):
        el = self.time_window
        m = self.n_graphs_
        n = self.n_vertices_

        # Compute moving means
        means = np.empty(m - el)
        for i in range(m - el):
            means[i] = stats[i : i + el - 1].sum() / (n * (el - 1))

        # Compute moving standard deviations
        sample_stds = np.std(stats, axis=1, ddof=1)
        constant = gamma(n / 2) * np.sqrt(2 / (n - 1)) / gamma((n - 1) / 2)

        stds = np.empty(m - el)
        for i in range(m - el):
            stds[i] = sample_stds[i : i + el - 1].sum() / (constant * (el - 1))

        upper_central_line = means + 3 * stds
        lower_central_line = means - 3 * stds

        return means, stds, upper_central_line, lower_central_line

    def fit_predict(self, graphs):
        """

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
        """
        self.n_graphs_ = len(graphs)
        self.n_vertices_ = graphs[0].shape[0]

        embeddings = self._embed(graphs)
        graph_stats, vertex_stats = self._compute_statistics(embeddings)

        res = self._compute_vertex_control_chart(vertex_stats)

        return vertex_stats, res
