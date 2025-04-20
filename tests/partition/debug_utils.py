import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def draw_node_and_neighbors(graph, node, partitions):
    neighbors = list(graph.neighbors(node)) + [node]
    subgraph = graph.subgraph(neighbors)

    pos = nx.spring_layout(subgraph, seed=42)
    colors = [partitions[n] for n in subgraph.nodes()]
    cmap = plt.cm.get_cmap("Set1")

    plt.figure(figsize=(6, 4))
    nx.draw(subgraph, pos, with_labels=True, node_color=colors, cmap=cmap, edge_color='gray')
    plt.title(f"Node {node} and its Neighborhood (Community {partitions[node]})")
    plt.show()


def compare_embedding_similarity(node, partitions, embeddings, community_id):
    node_vec = embeddings[str(node)]
    other_nodes = [n for n, comm in partitions.items() if comm == community_id and n != node]
    if not other_nodes:
        return 0.0
    sims = [cosine_similarity([node_vec], [embeddings[str(n)]])[0, 0] for n in other_nodes]
    return np.mean(sims)


def print_node_structure(graph, node, partitions, misaligned, embeddings):
    print(f"\n Analyzing Node {node}")
    print(f"- Assigned Community: {partitions[node]}")
    print(f"- Suggested by node2vec: {misaligned.get(node)}")
    print(f"- Degree: {graph.degree[node]}")
    print(f"- Neighbors: {list(graph.neighbors(node))}")

    sim_assigned = compare_embedding_similarity(node, partitions, embeddings, partitions[node])
    sim_misaligned = compare_embedding_similarity(node, partitions, embeddings, misaligned.get(node))

    print(f"- Avg sim to assigned community: {sim_assigned:.4f}")
    print(f"- Avg sim to suggested community: {sim_misaligned:.4f}")

    draw_node_and_neighbors(graph, node, partitions)
