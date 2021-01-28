from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair

import os
import csv
import random
import networkx as nx
from os.path import join

def read_graph(fn, delimiter, num_lines_header):
    fp = open(fn)
    csv_reader = csv.reader(fp, delimiter=delimiter)
    g = nx.Graph()
    for _ in range(num_lines_header):
        next(csv_reader)
    i = 0
    for line in csv_reader:
        if i < 1:
            print(line)
            i += 1
        u,v = line
        g.add_edge(u,v)
    return g

def get_random_new_nids(g, seed):
    new_nids = list(range(g.number_of_nodes()))
    seed.shuffle(new_nids)
    return new_nids

def apply_node_relabel(g, seed):
    new_nids = get_random_new_nids(g, seed)
    assert len(new_nids) == g.number_of_nodes()
    mapping = {}
    i = 0
    for nid in g.nodes():
        mapping[nid] = new_nids[i]
        i += 1
    return nx.relabel_nodes(g, mapping)

def get_file_path(fn, ext):
    fnp = join('/', 'home', 'yba', 'Documents', 'GraphMatching', 'file', 'MCSRL_files', 'TODO', f'{fn}.{ext}')
    return fnp

def load_duogexf_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == ['type'] and eatts == [] and tvt in ['train', 'test']
    data_name, label_name, fns = name.split(':')
    fn1, fn2 = fns.split(';')
    assert data_name in ['duogexf', 'duogexf_tsv', 'duogexf_csv']

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    graph_list, pairs, mapping = [], {}, {}

    gid1, gid2 = 0,1
    if data_name == 'duogexf':
        fn1 = get_file_path(fn1,'gexf')
        fn2 = get_file_path(fn2,'gexf')
        g1 = nx.read_gexf(fn1).to_undirected()
        g2 = nx.read_gexf(fn2).to_undirected()
    elif data_name == 'duogexf_tsv':
        fn1 = get_file_path(fn1,'tsv')
        fn2 = get_file_path(fn2,'tsv')
        delimiter = '\t'
        num_lines_header = int(input('How many lines in header? '))
        g1 = read_graph(fn1, delimiter, num_lines_header)
        g2 = read_graph(fn2, delimiter, num_lines_header)
    elif data_name == 'duogexf_csv':
        fn1 = get_file_path(fn1,'csv')
        fn2 = get_file_path(fn2,'csv')
        delimiter = ','
        num_lines_header = int(input('How many lines in header? '))
        g1 = read_graph(fn1, delimiter, num_lines_header)
        g2 = read_graph(fn2, delimiter, num_lines_header)
    else:
        assert False
    _assign_labels(g1, label_name)
    _assign_labels(g2, label_name)
    g1 = max(nx.connected_component_subgraphs(g1), key=len)
    g2 = max(nx.connected_component_subgraphs(g2), key=len)
    g1.graph['gid'] = gid1
    g2.graph['gid'] = gid2
    g1 = apply_node_relabel(g1, seed)
    g2 = apply_node_relabel(g2, seed)
    graph_list.append(RegularGraph(g1))
    graph_list.append(RegularGraph(g2))
    pairs[(gid1, gid2)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                    running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

def _assign_labels(g, label_name):
    for node in g.nodes:
        if label_name == '':
            label_val = 0
        else:
            label_val = g.nodes[node][label_name]
        label_keys = list(g.nodes[node].keys())
        for label_key in label_keys:
            del g.nodes[node][label_key]
        g.nodes[node]['type'] = label_val
    for edge in g.edges:
        label_keys = list(g.edges[edge].keys())
        for label_key in label_keys:
            del g.edges[edge][label_key]

def load_isogexf_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == [] and eatts == [] and tvt in ['train', 'test']
    data_name, fn = name.split(':')
    fn = join('/','home','yba','Documents','GraphMatching','data', fn)
    assert data_name == 'isogexf'

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    delimiter = input('What is the delimiter? ')
    num_lines_header = int(input('How many lines in header? '))

    # initialize graph parameters
    gid = 0
    pairs = {}
    graph_list = []
    for graph_file in os.listdir(fn):
        mapping = {}
        gid1, gid2 = gid, gid+1
        g1 = read_graph(join(fn, graph_file), delimiter, num_lines_header)
        g2 = read_graph(join(fn, graph_file), delimiter, num_lines_header)
        g1 = max(nx.connected_component_subgraphs(g1), key=len)
        g2 = max(nx.connected_component_subgraphs(g2), key=len)
        g1.graph['gid'] = gid1
        g2.graph['gid'] = gid2
        g1 = apply_node_relabel(g1, seed)
        g2 = apply_node_relabel(g2, seed)
        graph_list.append(RegularGraph(g1))
        graph_list.append(RegularGraph(g2))
        pairs[(gid1, gid2)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                        running_time=0)
        gid += 2

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

