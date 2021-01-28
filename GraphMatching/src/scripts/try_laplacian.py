from load_data import load_dataset
import networkx as nx
import numpy as np
from numpy import linalg as LA

name = 'aids700nef'
dataset = load_dataset(name, 'all', 'mcs', 'bfs')

g = dataset.gs[0].nxgraph

# adj = nx.laplacian_matrix(g).todense()
# print(adj)
#
# values, vectors = LA.eig(adj)
# print(values)

adj = nx.normalized_laplacian_matrix(g).todense()
print(adj)

values, vectors = LA.eig(adj)
print(sorted(values))