from utils import get_data_path, get_temp_path, get_file_path, create_dir_if_not_exists
from os.path import join
import pandas as pd
import networkx as nx
from tqdm import tqdm
import csv
import json
import random

import numpy as np
import networkx as nx


class SRW_RWF_ISRW:

    def __init__(self, seed):
        self.growth_size = 2
        self.T = 100  # number of iterations
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.15
        self.seed = seed

    def random_walk_sampling_simple(self, complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.node[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = self.seed.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.node[index_of_first_random_node]['id'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = self.seed.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = self.seed.randint(0, nr_nodes - 1)
                edges_before_t_iter = sampled_graph.number_of_edges()
        return sampled_graph

    def random_walk_sampling_with_fly_back(self, complete_graph, nodes_to_sample, fly_back_prob):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.node[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample

        index_of_first_random_node = self.seed.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.node[index_of_first_random_node]['id'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = self.seed.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = self.seed.randint(0, nr_nodes - 1)
                    print("Choosing another random node to continue random walk ")
                edges_before_t_iter = sampled_graph.number_of_edges()

        return sampled_graph

    def random_walk_induced_graph_sampling(self, complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.node[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = self.seed.randint(0, nr_nodes - 1)

        Sampled_nodes = {complete_graph.node[index_of_first_random_node]['id']}

        iteration = 1
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = self.seed.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.node[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < self.growth_size):
                    curr_node = self.seed.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        sampled_graph = complete_graph.subgraph(Sampled_nodes)

        return sampled_graph

def _bfs(fn, start_node, g, max_len):
    # I think it shouldn't be the largest one otherwise it would juts be 1 node with 2999 neighbookrsay got it anyway just try a bucnh of seed then
    # seed = 10 # find largest degree node; also turn into undirectedl oehteise tricky for neighbors
    # start_node = list(g.nodes)[seed]
    assert g.has_node(start_node)
    queue = [start_node]

    undired_g = nx.Graph(g)
    print('getting subg')
    nodes_subg = [start_node]
    # max_len = 100  # let;s decreasr to 10 for debugging! Go t it
    for _ in tqdm(range(max_len)):
        if len(queue) == 0:
            break
        node = queue.pop()
        neighbors = set(undired_g.neighbors(node)) - set(nodes_subg)
        for node_out in neighbors:
            nodes_subg.append(node_out)
            queue.append(node_out)
    subg = g.subgraph(nodes_subg)

    print(start_node, 'subg', _g_str(subg))
    # print('saving subg')
    fn = join(get_file_path(), f'{fn}_small_bfs_{start_node}_{max_len}.gexf')
    nx.write_gexf(subg, fn)
    print(fn)
    return subg

def xtract(g):
    return max(nx.connected_component_subgraphs(g), key=len)

def random_walk(fn, g, max_len, srw_rwf_isrw, i):
    subg = srw_rwf_isrw.random_walk_induced_graph_sampling(g, max_len)
    fn = join(get_file_path(), f'{fn}_rw_{max_len}_{i}.gexf')
    nx.write_gexf(subg, fn)
    print(fn)
    return xtract(subg)

def _g_str(g):
    nn = g.number_of_nodes()
    ne = g.number_of_edges()
    return f'g {type(g)} has {nn} nodes {ne} edges ({ne / nn})'

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

# file:///home/yba/Documents/GraphMatching/file/as19971201.tsv
# file:///home/yba/Documents/GraphMatching/file/as19971223.tsv

# file:///home/yba/Documents/GraphMatching/file/web-Stanford.tsv
# file:///home/yba/Documents/GraphMatching/file/web-NotreDame.tsv
# file:///home/yba/Documents/GraphMatching/file/web-Google.tsv
# file:///home/yba/Documents/GraphMatching/file/web-BerkStan.tsv

#DBP15K_en_zh_1
#DBP15K_en_zh_2
# DD-Miner_miner-disease-disease
file_name = 'DD-Miner_miner-disease-disease'#''web-BerkStan'#'com-youtube-ungraph'#'roadNet-CA'#'ChG-Miner_miner-chem-gene'
ext = 'tsv'

'''
roadNet-CA:             4
musae_facebook_edges:   1
Email-Enron:            4
PP-Pathways_ppi:        0
DD-Miner_miner-disease-disease: 1
'''
perc = 50
subg_mode = 'rw'
seed = random.Random(345)#234
fn = join(get_file_path(), f'{file_name}.{ext}')
print('loading graph')
if ext == 'gexf':
    g = nx.read_gexf(fn)
elif ext == 'tsv':
    delimiter = '\t'
    num_lines_header = int(input('How many lines in header? '))
    g = read_graph(fn, delimiter, num_lines_header)
elif ext == 'csv':
    delimiter = ','
    num_lines_header = int(input('How many lines in header? '))
    g = read_graph(fn, delimiter, num_lines_header)
elif ext == 'json':
    fp = open(fn)
    lns = fp.readlines()
    assert len(lns) == 1
    ln = lns[0]
    j = json.loads(ln)
    dir_name = join(file_name, 'graphs')
    folder = join(get_file_path(), dir_name)
    create_dir_if_not_exists(folder)
    for gid, edge_list in tqdm(j.items()):
        g = nx.Graph()
        g.add_edges_from(edge_list)
        nx.write_gexf(g, join(get_file_path(), dir_name, f'{gid}_{g.number_of_nodes()}.gexf'))
    exit(-1)

g=xtract(g)
print('graph loaded')
print(_g_str(g))
all_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)
print(f'{len(all_nodes)} nodes in total')
# fn = join(get_file_path(), f'{file_name}_all.gexf')
# nx.write_gexf(g, fn)
# exit(-1)

max_len = int(perc/100. * len(all_nodes))
print(max_len)

if subg_mode == 'bfs':
    # start_node1 = all_nodes[0][0]
    start_node1 = all_nodes[100][0]
    print('number of nodes', start_node1)
    # start_node2 = all_nodes[2][0]
    start_node2 = all_nodes[-1][0]
    print('number of nodes', start_node2)

    subg1 = _bfs(file_name, start_node1, g, max_len)
    # print('subg1', _g_str(subg1))
    subg2 = _bfs(file_name, start_node2, g, max_len)
    # print('subg2', _g_str(subg2))
elif subg_mode == 'rw':
    srw_rwf_isrw = SRW_RWF_ISRW(seed)
    subg1 = random_walk(file_name, g, max_len, srw_rwf_isrw, 1)
    subg2 = random_walk(file_name, g, max_len, srw_rwf_isrw, 2)
else:
    assert False

print(subg1.number_of_nodes())
print(subg2.number_of_nodes())
# final_g = nx.compose(subg1, subg2)
#
# print('final_g', _g_str(final_g))
# print('saving subg')
# fn = join(get_file_path(), f'amazon_small_bfs_{start_node1}_{start_node2}.gexf')
# nx.write_gexf(final_g, fn)
# print(fn)
# df = pd.read_csv(join(get_data_path(), 'Amazon', 'ratings_Musical_Instruments.csv'), names=['uid', 'iid', 'score', "?"])
# print(df)#
#
# g = nx.DiGraph()
#
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     g.add_edge(row['uid'], row['iid'])
#     g.add_node(row['uid'], type='user')
#     g.add_node(row['iid'], type='item')
#
# nx.write_gexf(g, join(get_temp_path(), 'amazon.gexf'))
# print('done')
#
