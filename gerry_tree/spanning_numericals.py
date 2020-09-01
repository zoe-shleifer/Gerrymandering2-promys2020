import networkx as nx
import numpy as np
import pickle

def random_weight_spanning(G):
    T = G.copy()
    rand = np.random.uniform(0, 1, T.size())
    for i, edge in enumerate(T.edges()):
        T.edges[edge]["weight"] = rand[i]
    return nx.minimum_spanning_tree(T)

def spanning_numericals(G,reps):
    all_tress = []
    for i in range(reps):
        all_trees.append(random_weight_spanning(G).edges())
    return all_trees

