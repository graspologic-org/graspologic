# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from typing import Dict, List, Tuple
from collections import defaultdict
from customleiden.partition.leiden import leiden_with_context, _compute_embeddings, compute_node2vec_embeddings
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import random
import numpy as np
import pytest
import scipy
from beartype.roar import BeartypeCallHintParamViolation

from customleiden.partition import (
    HierarchicalCluster,
    HierarchicalClusters,
    hierarchical_leiden,
    leiden,
)
from customleiden.partition.leiden import (
    _adjacency_matrix_to_edge_list,
    _edge_list_to_edge_list,
    _from_native,
    _IdentityMapper,
    _nx_to_edge_list,
)
from tests.utils import data_file

def precision_at_k(query_node, retrieved_nodes, ground_truth_labels, k=5):
    if query_node not in ground_truth_labels:
        return 0.0
    query_label = ground_truth_labels[query_node]
    correct = sum(1 for n in retrieved_nodes if ground_truth_labels.get(n) == query_label)
    return correct / k


class TestContextAwareLeiden(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Prepare Karate Club graph for precision@k evaluation
        cls.graph = nx.karate_club_graph()
        cls.orig_partitions = leiden(cls.graph)
        cls.new_result = leiden_with_context(
            cls.graph, embedding_method="node2vec", random_seed=42
        )
        cls.embeddings = compute_node2vec_embeddings(cls.graph)
        cls.ground_truth = {
            n: cls.graph.nodes[n]['club'] for n in cls.graph.nodes
        }

    def test_precision_at_k_vs_ground_truth(self):
        """
        This test ensures context-aware Leiden does not significantly degrade in-group precision@k.

        We compute precision@k for each node's nearest neighbors within its original cluster vs.
        its extended cluster plus context nodes. The context-aware version should maintain or improve precision.
        """
        k = 5
        orig_scores, new_scores = [], []

        for node in self.graph.nodes:
            orig_cluster = self.orig_partitions[node]
            new_cluster = self.new_result.partitions[node]

            orig_cands = [n for n, c in self.orig_partitions.items() if c == orig_cluster]
            new_cands = [
                n for n, c in self.new_result.partitions.items() if c == new_cluster
            ] + list(self.new_result.context_nodes.get(new_cluster, ()))

            def top_k(n, cands):
                vec = self.embeddings[str(n)]
                sims = [
                    (m, cosine_similarity([vec], [self.embeddings[str(m)]])[0,0])
                    for m in cands if m != n
                ]
                return [m for m, _ in sorted(sims, key=lambda x: -x[1])[:k]]

            orig_scores.append(precision_at_k(node, top_k(node, orig_cands), self.ground_truth, k))
            new_scores.append(precision_at_k(node, top_k(node, new_cands), self.ground_truth, k))

        avg_orig = np.mean(orig_scores)
        avg_new = np.mean(new_scores)

        print("\n--- Precision@k Evaluation ---")
        print(f"k = {k}")
        print(f"Original Leiden avg precision@{k}: {avg_orig:.3f}")
        print(f"Context-Aware Leiden avg precision@{k}: {avg_new:.3f}")
        print(f"Improvement: {avg_new - avg_orig:+.4f}")

        self.assertTrue(
            avg_new >= avg_orig - 0.01,
            f"Precision degraded: original={avg_orig:.3f}, new={avg_new:.3f}"
        )

    def test_context_nodes_have_higher_cross_cluster_similarity(self):
        """
        This test evaluates how well context nodes capture cross-cluster semantic relationships.

        We compare:
        - Plain cluster nodes (randomly sampled from each community)
        - Context nodes (selected by the context-aware Leiden)

        Metric:
        For each node, we compute average cosine similarity to nodes in **other** clusters.
        The better the context selection, the higher this "semantic bridging" score should be.
        """
        random.seed(42)
        np.random.seed(42)

        # Build a synthetic graph with 4 semantic groups
        G = nx.Graph()
        N = 40
        sem = {
            0: "apple red sweet fruit juice",
            1: "banana yellow tropical smoothie ripe",
            2: "grape purple fresh wine vineyard",
            3: "orange citrus tangy vitamin c"
        }
        for i in range(N):
            main = i % 4
            mix = (main + random.choice([1,2])) % 4
            text = (
                f"{sem[main]} {sem[mix]}"
                if random.random() < 0.3 else sem[main]
            )
            G.add_node(i, text=text)

        # Add intra- and inter-group edges
        for i in range(N):
            for j in range(i+1, N):
                if i%4 == j%4 and random.random() < 0.6:
                    G.add_edge(i, j)
        for _ in range(80):
            u, v = random.sample(range(N), 2)
            if u%4 != v%4 and random.random() < 0.3:
                G.add_edge(u, v)

        emb = _compute_embeddings(G, method="node2vec")
        plain_part = leiden(G, random_seed=42)
        ctx_res = leiden_with_context(
            G, embedding_method="node2vec",
            random_seed=42
        )
        ctx_part, ctx_nodes = ctx_res.partitions, ctx_res.context_nodes

        def avg_other_sim(n, part, embedding):
            own = part[n]
            others = [m for m in G.nodes if part[m] != own]
            if not others:
                return 0
            sims = [
                cosine_similarity([embedding[str(n)]], [embedding[str(m)]])[0,0]
                for m in others
            ]
            return float(np.mean(sims))

        ctx_scores, plain_scores = [], []
        for cluster, nodes in ctx_nodes.items():
            members = [n for n in G.nodes if ctx_part[n] == cluster]
            if len(members) < 2 or not nodes:
                continue
            sample_plain = np.random.choice(members, size=min(len(nodes), len(members)), replace=False)
            ctx_scores.extend(avg_other_sim(n, ctx_part, emb) for n in nodes)
            plain_scores.extend(avg_other_sim(n, plain_part, emb) for n in sample_plain)

        self.assertGreater(
            np.mean(ctx_scores),
            np.mean(plain_scores),
            "Context nodes do not outperform plain nodes in cross-cluster similarity."
        )

class TestLeidenSemantic(unittest.TestCase):
    def test_context_semantics(self):
        """
        Unit test for verifying semantic and structural correctness of context-aware Leiden.

        This test ensures that the context nodes identified by `leiden_with_context` meet
        three key criteria:

        1. **Context nodes truly bridge clusters**  
        Every selected context node must connect to nodes from at least one other community
        in the original graph.

        2. **Cluster-level graph reflects these bridges**  
        The high-level `cluster_graph` must contain weighted edges between all clusters
        that are connected via context nodes.

        3. **All inter-cluster connections are captured**  
        Every real inter-cluster link observed via context nodes must correspond to
        an edge in the `cluster_graph`.

        Procedure
        ---------
        - Loads the standard Zachary Karate Club graph.
        - Runs `leiden_with_context()` using the "sbert" embedding method for semantic scoring.
        - Asserts the following:
            a. Each context node has at least one external neighbor from a different community.
            b. The cluster-level graph contains only positive edge weights.
            c. No inter-community relationship is missing from the cluster-level graph.

        Output
        ------
        Prints helpful diagnostics for debugging and educational purposes, including:
        - Number of communities, context-node sets, and cluster edges
        - Context nodes per community and their external neighbors
        - All cluster-graph edges with weights
        - Missing cluster-graph edges, if any (should be zero)

        Raises
        ------
        AssertionError
            If any of the above conditions are violated, the test fails with a clear message.
        """
        graph = nx.karate_club_graph()

        print("\n=== Running leiden_with_context on Karate Club graph ===")
        result = leiden_with_context(
            graph,
            random_seed=42,
            embedding_method="sbert"
        )

        communities = result.partitions
        context_nodes = result.context_nodes
        cluster_graph = result.cluster_graph

        # Group nodes by community
        community_to_nodes = defaultdict(list)
        for node, comm in communities.items():
            community_to_nodes[comm].append(node)

        print(f"#Communities: {len(community_to_nodes)}")
        print(f"#Context-node sets: {len(context_nodes)}")
        print(f"#Cluster-graph edges: {cluster_graph.number_of_edges()}")

        for comm_id, nodes in sorted(community_to_nodes.items()):
            print(f"\nCommunity {comm_id} → {sorted(nodes)}")

        # Every context node bridges communities
        for comm_id, nodes in context_nodes.items():
            print(f"\nCommunity {comm_id} context nodes → {sorted(nodes)}")
            for node in nodes:
                self.assertEqual(communities[node], comm_id)
                ext_neigh = [
                    nbr for nbr in graph.neighbors(node)
                    if communities[nbr] != comm_id
                ]
                print(f"  Node {node} external neighbours → {sorted(ext_neigh)}")
                self.assertGreater(
                    len(ext_neigh), 0,
                    f"Context node '{node}' has no bridging neighbors."
                )

        # Validate cluster-graph edge weights
        print("\nCluster‑level edges with weights:")
        for u, v, data in cluster_graph.edges(data=True):
            print(f"  {u} — {v}  (weight={data['weight']})")
            self.assertGreater(data["weight"], 0)

        # Check that all bridging edges are reflected
        cluster_edges = {(min(u, v), max(u, v)) for u, v in cluster_graph.edges()}
        missing = []
        for comm_id, nodes in context_nodes.items():
            for node in nodes:
                for nbr in graph.neighbors(node):
                    nbr_comm = communities[nbr]
                    if nbr_comm != comm_id:
                        pair = (min(comm_id, nbr_comm), max(comm_id, nbr_comm))
                        if pair not in cluster_edges:
                            missing.append((node, pair))

        print(f"\n#Missing cluster edges (should be 0) → {len(missing)}")
        for node, pair in missing:
            print(f"  Missing edge {pair} caused by node {node}")

        self.assertEqual(
            len(missing), 0,
            "Some inter-community connections are not represented in cluster_graph."
        )

class TestHierarchicalCluster(unittest.TestCase):
    def test_from_native(self):
        with self.assertRaises(TypeError):
            _from_native(1, {"1": 1})

        # note: it is impossible to create a native instance of a HierarchicalCluster.  We will
        # test from_native indirectly through calling customleiden.partition.hierarchical_leiden()

    def test_final_hierarchical_clustering(self):
        hierarchical_clusters = HierarchicalClusters([
            HierarchicalCluster("1", 0, None, 0, False),
            HierarchicalCluster("2", 0, None, 0, False),
            HierarchicalCluster("3", 0, None, 0, False),
            HierarchicalCluster("4", 1, None, 0, True),
            HierarchicalCluster("5", 1, None, 0, True),
            HierarchicalCluster("1", 2, 0, 1, True),
            HierarchicalCluster("2", 2, 0, 1, True),
            HierarchicalCluster("3", 3, 0, 1, True),
        ])

        expected = {
            "1": 2,
            "2": 2,
            "3": 3,
            "4": 1,
            "5": 1,
        }
        self.assertEqual(
            expected,
            hierarchical_clusters.final_level_hierarchical_clustering(),
        )


def _create_edge_list() -> List[Tuple[str, str, float]]:
    edges = []
    with open(data_file("large-graph.csv"), "r") as edges_io:
        for line in edges_io:
            source, target, weight = line.strip().split(",")
            edges.append((source, target, float(weight)))
    return edges


class TestLeiden(unittest.TestCase):
    def test_correct_types(self):
        # both leiden and hierarchical_leiden require the same types and mostly the same value range restrictions
        good_args = {
            "starting_communities": {"1": 2},
            "extra_forced_iterations": 0,
            "resolution": 1.0,
            "randomness": 0.001,
            "use_modularity": True,
            "random_seed": None,
            "is_weighted": True,
            "weight_default": 1.0,
            "check_directed": True,
        }

        graph = nx.Graph()
        graph.add_edge("1", "2", weight=3.0)
        graph.add_edge("2", "3", weight=4.0)

        leiden(graph=graph, **good_args)
        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["starting_communities"] = 123
            leiden(graph=graph, **args)

        args = good_args.copy()
        args["starting_communities"] = None
        leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["extra_forced_iterations"] = 1234.003
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["extra_forced_iterations"] = -4003
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["resolution"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["resolution"] = 0
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["randomness"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["randomness"] = 0
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["use_modularity"] = 1234
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["trials"] = "hotdog"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["trials"] = 0
            leiden(graph=graph, **args)

        args = good_args.copy()
        args["random_seed"] = 1234
        leiden(graph=graph, **args)
        args["random_seed"] = None
        leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["random_seed"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["random_seed"] = -1
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["is_weighted"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["weight_default"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["check_directed"] = "leiden"
            leiden(graph=graph, **args)

        # one extra parameter hierarchical needs
        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["max_cluster_size"] = "leiden"
            hierarchical_leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["max_cluster_size"] = 0
            hierarchical_leiden(graph=graph, **args)

        cleared_partitions = good_args.copy()
        del cleared_partitions["starting_communities"]
        as_csr = nx.to_scipy_sparse_array(graph)
        partitions = leiden(graph=as_csr, **cleared_partitions)
        node_ids = partitions.keys()
        for node_id in node_ids:
            self.assertTrue(
                isinstance(
                    node_id, np.integer
                ),  # this is the preferred numpy typecheck
                f"{node_id} has {type(node_id)} should be an int",
            )

    def test_hierarchical(self):
        # most of leiden is tested in unit / integration tests in graspologic-native.
        # All we're trying to test through these unit tests are the python conversions
        # prior to calling, so type and value validation and that we got a result
        edges = _create_edge_list()
        results = hierarchical_leiden(edges, random_seed=1234)

        total_nodes = len([item for item in results if item.level == 0])

        partitions = results.final_level_hierarchical_clustering()
        self.assertEqual(total_nodes, len(partitions))

    # Github issue: 738
    def test_matching_return_types(self):
        graph = nx.erdos_renyi_graph(20, 0.4, seed=1234)
        partitions = leiden(graph)
        for node_id in partitions:
            self.assertTrue(isinstance(node_id, int))

    # Github issue: 901
    def test_hashable_nonstr_with_starting_communities(self):
        seed = 1234
        first_graph = nx.erdos_renyi_graph(20, 0.4, seed=seed)
        second_graph = nx.erdos_renyi_graph(21, 0.4, seed=seed)
        third_graph = nx.erdos_renyi_graph(19, 0.4, seed=seed)

        first_partitions = leiden(first_graph)
        second_partitions = leiden(second_graph, starting_communities=first_partitions)
        third_partitions = leiden(third_graph, starting_communities=second_partitions)


class TestLeidenIsolates(unittest.TestCase):
    """
    Tests to verify fix for Github issue: 803 - isolate nodes are dropped silently
    """

    def setUp(self) -> None:
        # prepare a graph with an isolate node
        self.graph: nx.Graph = nx.complete_graph(10)
        nodelist = sorted(self.graph.nodes)
        for node in nodelist[1:]:
            self.graph.remove_edge(0, node)

    def assert_isolate_not_in_result(self, partitions: Dict[str, int]):
        """verify that isolate node was not returned"""
        self.assertTrue(
            0 not in partitions, "the isolate node is not in the result from leiden"
        )
        self.assertTrue(
            3 in partitions, "a node that was not removed is in the result from leiden"
        )
        self.assertEqual(
            9,
            len(partitions),
            "the result contains all nodes in the connected component",
        )

    def assert_isolate_not_in_hierarchical_result(
        self, partitions: List[HierarchicalCluster]
    ):
        """verify that isolate node was not returned"""
        all_nodes = {p.node for p in partitions}

        self.assertTrue(
            0 not in all_nodes, "the isolate node is not in the result from leiden"
        )
        self.assertTrue(
            3 in all_nodes, "a node that was not removed is in the result from leiden"
        )
        self.assertEqual(
            9,
            len(all_nodes),
            "the result contains all nodes in the connected component",
        )

    def test_isolate_nodes_in_nx_graph_are_not_returned(self):
        self.assertEqual(
            10,
            len(self.graph.nodes),
            "the input graph contains all nodes including isolate",
        )

        with pytest.warns(UserWarning, match="isolate"):
            partitions = leiden(self.graph)

        self.assert_isolate_not_in_result(partitions)

        with pytest.warns(UserWarning, match="isolate"):
            hierarchical_partitions = hierarchical_leiden(self.graph)

        self.assert_isolate_not_in_hierarchical_result(hierarchical_partitions)

    def test_isolate_nodes_in_ndarray_are_not_returned(self):
        ndarray_adj_matrix = nx.to_numpy_array(self.graph)

        self.assertEqual(
            10,
            ndarray_adj_matrix.shape[0],
            "the input array contains all nodes including isolate",
        )

        with pytest.warns(UserWarning, match="isolate"):
            partitions = leiden(ndarray_adj_matrix)

        self.assert_isolate_not_in_result(partitions)

        with pytest.warns(UserWarning, match="isolate"):
            hierarchical_partitions = hierarchical_leiden(ndarray_adj_matrix)

        self.assert_isolate_not_in_hierarchical_result(hierarchical_partitions)

    def test_isolate_nodes_in_csr_array_are_not_returned(self):
        sparse_adj_matrix = nx.to_scipy_sparse_array(self.graph)

        self.assertEqual(
            10,
            sparse_adj_matrix.shape[0],
            "the input csr contains all nodes including isolate",
        )

        with pytest.warns(UserWarning, match="isolate"):
            partitions = leiden(sparse_adj_matrix)

        self.assert_isolate_not_in_result(partitions)

        with pytest.warns(UserWarning, match="isolate"):
            hierarchical_partitions = hierarchical_leiden(sparse_adj_matrix)

        self.assert_isolate_not_in_hierarchical_result(hierarchical_partitions)


def add_edges_to_graph(graph: nx.Graph) -> nx.Graph:
    graph.add_edge("nick", "dwayne", weight=1.0)
    graph.add_edge("nick", "dwayne", weight=3.0)
    graph.add_edge("dwayne", "nick", weight=2.2)
    graph.add_edge("dwayne", "ben", weight=4.2)
    graph.add_edge("ben", "dwayne", weight=0.001)
    return graph


class TestValidEdgeList(unittest.TestCase):
    def test_empty_edge_list(self):
        edges = []
        results = _edge_list_to_edge_list(
            edges=edges,
            identifier=_IdentityMapper(),
        )
        self.assertEqual([], results[1])

    def test_assert_list_does_not_contain_tuples(self):
        edges = ["invalid"]
        with self.assertRaises(BeartypeCallHintParamViolation):
            _edge_list_to_edge_list(
                edges=edges,
                identifier=_IdentityMapper(),
            )

    def test_assert_list_contains_misshapen_tuple(self):
        edges = [(1, 2, 1.0, 1.0)]
        with self.assertRaises(BeartypeCallHintParamViolation):
            _edge_list_to_edge_list(
                edges=edges,
                identifier=_IdentityMapper(),
            )

    def test_assert_wrong_types_in_tuples(self):
        edges = [(True, 4, "sandwich")]
        with self.assertRaises(BeartypeCallHintParamViolation):
            _edge_list_to_edge_list(
                edges=edges,
                identifier=_IdentityMapper(),
            )

        edges = [(True, False, 3.2)]
        _nodes, results = _edge_list_to_edge_list(
            edges=edges,
            identifier=_IdentityMapper(),
        )
        self.assertEqual([("True", "False", 3.2)], results)

    def test_empty_nx(self):
        expected = 0, []
        results = _nx_to_edge_list(
            graph=nx.Graph(),
            identifier=_IdentityMapper(),
            is_weighted=None,
            weight_attribute="weight",
            weight_default=1.0,
        )
        self.assertEqual(expected, results)
        with self.assertRaises(ValueError):
            _nx_to_edge_list(
                graph=nx.DiGraph(),
                identifier=_IdentityMapper(),
                is_weighted=None,
                weight_attribute="weight",
                weight_default=1.0,
            )
        with self.assertRaises(ValueError):
            _nx_to_edge_list(
                graph=nx.MultiGraph(),
                identifier=_IdentityMapper(),
                is_weighted=None,
                weight_attribute="weight",
                weight_default=1.0,
            )
        with self.assertRaises(ValueError):
            _nx_to_edge_list(
                graph=nx.MultiDiGraph(),
                identifier=_IdentityMapper(),
                is_weighted=None,
                weight_attribute="weight",
                weight_default=1.0,
            )

    def test_valid_nx(self):
        graph = add_edges_to_graph(nx.Graph())
        expected = [("nick", "dwayne", 2.2), ("dwayne", "ben", 0.001)]
        _, edges = _nx_to_edge_list(
            graph=graph,
            identifier=_IdentityMapper(),
            is_weighted=None,
            weight_attribute="weight",
            weight_default=1.0,
        )
        self.assertEqual(expected, edges)

    def test_unweighted_nx(self):
        graph = nx.Graph()
        graph.add_edge("dwayne", "nick")
        graph.add_edge("nick", "ben")

        with self.assertRaises(TypeError):
            _, edges = _nx_to_edge_list(
                graph=graph,
                identifier=_IdentityMapper(),
                is_weighted=True,
                weight_attribute="weight",
                weight_default=1.0,
            )

        _, edges = _nx_to_edge_list(
            graph=graph,
            identifier=_IdentityMapper(),
            is_weighted=False,
            weight_attribute="weight",
            weight_default=3.33333,
        )
        self.assertEqual(
            [("dwayne", "nick", 3.33333), ("nick", "ben", 3.33333)],
            edges,
        )

        graph.add_edge("salad", "sandwich", weight=100)
        _, edges = _nx_to_edge_list(
            graph=graph,
            identifier=_IdentityMapper(),
            is_weighted=False,
            weight_attribute="weight",
            weight_default=3.33333,
        )
        self.assertEqual(
            [
                ("dwayne", "nick", 3.33333),
                ("nick", "ben", 3.33333),
                ("salad", "sandwich", 100),
            ],
            edges,
        )

    def test_matrices(self):
        graph = add_edges_to_graph(nx.Graph())
        di_graph = add_edges_to_graph(nx.DiGraph())

        dense_undirected = nx.to_numpy_array(graph)
        dense_directed = nx.to_numpy_array(di_graph)

        sparse_undirected = nx.to_scipy_sparse_array(graph)
        sparse_directed = nx.to_scipy_sparse_array(di_graph)

        expected = [("0", "1", 2.2), ("1", "2", 0.001)]
        _, edges = _adjacency_matrix_to_edge_list(
            matrix=dense_undirected,
            identifier=_IdentityMapper(),
            check_directed=True,
            is_weighted=True,
            weight_default=1.0,
        )
        self.assertEqual(expected, edges)
        _, edges = _adjacency_matrix_to_edge_list(
            matrix=sparse_undirected,
            identifier=_IdentityMapper(),
            check_directed=True,
            is_weighted=True,
            weight_default=1.0,
        )
        self.assertEqual(expected, edges)

        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=dense_directed,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=sparse_directed,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

    def test_empty_adj_matrices(self):
        dense = np.array([])
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=dense,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

        sparse = scipy.sparse.csr_array([])
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=sparse,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

    def test_misshapen_matrices(self):
        data = [[3, 2, 0], [2, 0, 1]]  # this is utter gibberish
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=np.array(data),
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=scipy.sparse.csr_array(data),
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

    def test_nx_graph_node_str_collision(self):
        graph = nx.Graph()
        graph.add_edge("1", 1, weight=1.0)
        with self.assertRaisesRegex(ValueError, "collision"):
            _nx_to_edge_list(
                graph=graph,
                identifier=_IdentityMapper(),
                is_weighted=True,
                weight_attribute="weight",
                weight_default=1.0,
            )

    def test_edgelist_node_str_collision(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            _edge_list_to_edge_list(edges=[("1", 1, 1.0)], identifier=_IdentityMapper())
