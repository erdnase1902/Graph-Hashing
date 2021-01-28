import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def plot_simple(g1, g2, mapping=None):
    colors_1 = ['red'] * g1.number_of_nodes()
    colors_2 = ['red'] * g2.number_of_nodes()
    if mapping is not None:
        for i in mapping.keys():
            colors_1[i] = 'blue'
        for i in mapping.values():
            colors_2[i] = 'blue'
    plt.clf()
    plt.subplot(211)
    nx.draw(g1, with_labels=True, node_color=colors_1)
    plt.subplot(212)
    nx.draw(g2, with_labels=True, node_color=colors_2)
    plt.show()
    plt.close()

def plot_many(g1, g2, g3, g4, g5, g6):
    plt.subplot(231)
    nx.draw(g1, with_labels=True)
    plt.subplot(232)
    nx.draw(g2, with_labels=True)
    plt.subplot(233)
    nx.draw(g3, with_labels=True)
    plt.subplot(234)
    nx.draw(g4, with_labels=True)
    plt.subplot(235)
    nx.draw(g5, with_labels=True)
    plt.subplot(236)
    nx.draw(g6, with_labels=True)
    plt.show()
    plt.close()

def _gen_nids(y_mat, axis):
    if axis == 0:
        rtn = np.where(y_mat == 1)[1]
    elif axis == 1:
        rtn = np.where(y_mat.T == 1)[1]
    else:
        assert False
    # assert len(set(rtn)) == len(rtn), 'There is duplicate nid {}'.format(rtn)
    return list(rtn)