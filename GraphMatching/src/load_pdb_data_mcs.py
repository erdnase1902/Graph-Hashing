from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair

from os.path import join
import os
import csv
import networkx as nx
from networkx import empty_graph
import itertools
import random

from tqdm import tqdm

LABEL = 'type' # TODO merge with dataset_config

def load_pdb_data_mcs(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == ['type', 'x', 'y', 'z', 'ss_type'] and eatts == ['dist', 'id'] and tvt in ['train', 'test'] and \
           align_metric == 'random' and glabel == 'random'

    path_graphs = '/home/yba/Documents/GraphMatching/data/pdb_preprocess/graphs'
    path_pairs = '/home/yba/Documents/GraphMatching/data/pdb_preprocess/pairs.txt'
    pair_key_list = load_pair_key_list(path_pairs)
    graph_dict = load_graph_dict(path_graphs) # gid -> nxgraph

    graph_list = []
    for g in graph_dict.values():
        graph_list.append(RegularGraph(g))

    pairs = {}
    for pair_key in pair_key_list:
        mapping = {}
        gid0, gid1 = pair_key
        # g0, g1 = graph_dict[gid0], graph_dict[gid1]
        pairs[(gid0, gid1)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                        running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

def load_pair_key_list(path):
    fp = open(path)
    reader = csv.reader(fp, delimiter=',')

    pair_key_list = []
    for row in reader:
        assert len(row) == 2
        pair_key = tuple([int(gid) for gid in row])
        pair_key_list.append(pair_key)
    return pair_key_list

def load_graph_dict(path):
    graph_dict = {}
    print('loading pdb graphs (might take a while)...')
    for graph_file in tqdm(os.listdir(path)):
        gid = int(graph_file.split('.')[0])
        g = nx.read_gexf(join(path, graph_file))
        g = relabel_nodes_str2int(g)
        add_pdb_edges(g)
        rm_node_attr(g)
        g.graph['gid'] = gid
        graph_dict[gid] = g
    return graph_dict

def relabel_nodes_str2int(g):
    mapping = {}
    for v in g.nodes:
        mapping[v] = int(v)
    return nx.relabel_nodes(g, mapping)

def rm_node_attr(g):
    for node in g.nodes:
        del g.nodes[node]['label']

import numpy as np
def compute_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

def add_pdb_edges(g):
    # threshold = 0
    # new_edge_list = []
    # for edge in nx.non_edges(g):
    #     i,j = edge
    #     dist = compute_dist((g.nodes[i]['x'],g.nodes[i]['y'], g.nodes[i]['z']),
    #                  (g.nodes[j]['x'],g.nodes[j]['y'], g.nodes[j]['z']))
    #     if dist > threshold:
    #         new_edge_list.append(edge)
    # g.add_edges_from(new_edge_list, dist=0, id=0)
    return g

# print('creating pdb dataset')
# d = PPIDataset(name, graphs, natts, interaction_edge_labels, eatts,
#                pairs, tvt, string_pdb_mapping_filtered, sid_to_int_map,
#                pdb_to_idx_map, string_seq_map)
#
# ###############################################################################
# # TODO: remove this code (used for MCS)
# path_graphs = '/home/yba/Documents/GraphMatching/data/pdb_preprocess/graphs'
# path_pairs = '/home/yba/Documents/GraphMatching/data/pdb_preprocess/pairs.txt'
#
# cur_gid = 0
# gid2nxgid = {}
# for graph in graphs:
#     for nx_graph in graph:
#         if graph.gid not in gid2nxgid:
#             gid2nxgid[graph.gid] = []
#         gid2nxgid[graph.gid].append(cur_gid)
#         nx_graph.graph['gid'] = cur_gid
#         nx.write_gexf(nx_graph, join(path_graphs, '{}.gexf'.format(cur_gid)))
#         cur_gid += 1
#
# fp_pair = open(path_pairs)
# pair_key_list_nx = []
# for pair_key in pairs.keys():
#     gid0, gid1 = pair_key
#     gid0_nx_list = gid2nxgid[gid0]
#     gid1_nx_list = gid2nxgid[gid1]
#     for gid0_nx in gid0_nx_list:
#         for gid1_nx in gid1_nx_list:
#             pair_key_list_nx.append('{},{}'.format(gid0_nx, gid1_nx))
# fp_pair.writelines(pair_key_list_nx)
# fp_pair.close()
# ###############################################################################