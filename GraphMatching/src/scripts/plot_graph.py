from collections import defaultdict
from collections import OrderedDict
import matplotlib

# Fix font type for ACM paper submission.
matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 0})  # turn off tick labels
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from os.path import join, dirname
from warnings import warn

from utils import create_dir_if_not_exists, append_ext_to_filepath


TYPE_COLOR_MAP = {
    'C': '#ff6666',
    'O': 'lightskyblue',
    'N': 'yellowgreen',
    'H': '#ffcc99',
    'movie': '#ff6666',
    'tvSeries': '#ff6666',
    'actor': 'lightskyblue',
    'actress': '#ffb3e6',
    'director': 'yellowgreen',
    'composer': '#c2c2f0',
    'producer': '#ffcc99',
    'cinematographer': 'gold'}

FAVORITE_COLORS = ['#ff6666', 'lightskyblue', 'yellowgreen', '#c2c2f0', 'gold',
                   '#ffb3e6', '#ffcc99', '#E0FFFF', '#7FFFD4', '#20B2AA',
                   '#FF8C00', '#ff1493',
                   '#FFE4B5', '#e6e6fa', '#7CFC00']

def plot_graphs(graphs, node_feat_name, dir, fn, print_path=True):

    graphs_list = []
    feat_dict_list = []
    for graph in graphs:
        graph, feat_dict = _gen_feat_dict(graph, node_feat_name)
        graphs_list.append(graph)
        feat_dict_list.append(feat_dict)

    graphs_pos = [_sorted_dict(graphviz_layout(graph, prog='neato')) for graph in graphs]

    _orig(graphs, graphs_pos, feat_dict_list, node_feat_name,
          dir, fn, print_path)

def _orig(graphs, graphs_pos, graphs_feats, node_feat_name,
          dir, fn, print_path):

    ntypes = defaultdict(int)
    if node_feat_name is not None and node_feat_name is not "gid":
        for graph in graphs:
            for node, ndata in graph.nodes(data=True):
                ntypes[ndata[node_feat_name]] += 1
    color_map = _gen_color_map(ntypes)

    graph_colors = [_gen_orig_node_colors(graph, node_feat_name, color_map) for graph in graphs]

    _plot(graphs, graph_colors, graphs_pos, graphs_feats,
          dir, fn + '_orig', print_path, color_map)


def _get_node(node_mapping, node):
    for i in range(len(node_mapping)):
        if list(node_mapping)[i] == node:
            return i
    #assert False


def _gen_feat_dict(g, node_feat_name):
    feat_dict = {}
    node_mapping = {}
    for node in range(g.number_of_nodes()):
        node_mapping[node] = node
        if node_feat_name != "gid" and "gid" in g.nodes[node].keys():
            feat = '{}'.format(g.nodes[node]['gid'])
        else:
            feat = '{}'.format(node)
        if node_feat_name is not None:
            try:
                feat += '_{}'.format(g.nodes[node][node_feat_name])
            except KeyError:
                feat += '_{}'.format(g.nodes[str(node)][node_feat_name])
        feat_dict[node] = feat
    g = nx.relabel_nodes(g, node_mapping)
    return g, _sorted_dict(feat_dict)


def _gen_orig_node_colors(g, node_label_name, color_map):
    if node_label_name is not None and node_label_name != "gid":
        color_values = []
        node_color_labels = _sorted_dict(nx.get_node_attributes(g, node_label_name))
        for node_label in node_color_labels.values():
            color = TYPE_COLOR_MAP.get(node_label, None)
            if color is None:
                color = color_map[node_label]
            color_values.append(color)
    else:
        color_values = ['lightskyblue'] * g.number_of_nodes()
    # print(color_values)
    return color_values


def _gen_color_map(ntypes_count_map):
    fl = len(FAVORITE_COLORS)
    rtn = {}
    # ntypes = defaultdict(int)
    # for g in gs:
    #     for nid, node in g.nodes(data=True):
    #         ntypes[node.get('type')] += 1
    secondary = {}
    for i, (ntype, cnt) in enumerate(
            sorted(ntypes_count_map.items(), key=lambda x: x[1], reverse=True)):
        if ntype is None:
            color = None
            rtn[ntype] = color
        elif i >= fl:
            cmaps = plt.cm.get_cmap('hsv')
            color = cmaps((i - fl) / (len(ntypes_count_map) - fl))
            secondary[ntype] = color
        else:
            color = mcolors.to_rgba(FAVORITE_COLORS[i])[:3]
            rtn[ntype] = color
    if secondary:
        rtn.update(secondary)
    return rtn


def _plot(graphs, graphs_colors, graphs_pos, graphs_feats,
          dir, fn, print_path, color_map):
    num_graphs = len(graphs)
    num_grid_rows = int(num_graphs / 4) + 1
    num_grid_cols = num_graphs % 4 if num_graphs < 4 else 4

    plt.figure(figsize=(11*num_grid_cols, 10*num_grid_rows))
    gs = gridspec.GridSpec(num_grid_rows, num_grid_cols)
    gs.update(wspace=0, hspace=0.05)  # set the spacing between axes

    graph_num_nodes = [g.number_of_nodes() for g in graphs]
    #max_num_nodes = max(graph_num_nodes)#int(sum(graph_num_nodes) / len(graph_num_nodes))
    avg_num_nodes = int(sum(graph_num_nodes) / len(graph_num_nodes))
    node_size_reduce_factor = avg_num_nodes / 20
    node_size = int(900/node_size_reduce_factor)

    for i, (graph, graph_pos, graph_color, graph_feats) in enumerate(zip(graphs, graphs_pos, graphs_colors, graphs_feats)):
        ax = plt.subplot(gs[i])
        ax.axis('off')
        title = _get_graph_label(graph)
        ax.set_title(title, fontsize=25)
        if i == 0:
            legend_elements = []
            for label, color in color_map.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color),)

            ax.legend(handles=legend_elements, loc='lower right', fontsize=15)

        if avg_num_nodes > 50:
            graph_feats = None
        _plot_one_graph(graph, graph_color, graph_pos, graph_feats, node_size)
    _save_fig(plt, dir, fn, print_path)
    plt.close()


def _plot_one_graph(g, color, pos, feat_dict, node_size):
    with_labels = True
    arrows = True
    if feat_dict is None: # when the graph is too big
        with_labels = False
        arrows = False
    label = _get_graph_label(g)
    edge_colors = "black"
    if "color" in g.edges[list(g.edges)[0]].keys():
        e_colors = nx.get_edge_attributes(g, 'color')
        edge_colors = [e_colors[(u, v)]for u, v in g.edges()]
    nx.draw_networkx(
        g, node_color=color, pos=pos, edge_color=edge_colors, arrows=arrows, edgecolors="black",
        with_labels=with_labels, labels=feat_dict, node_size=node_size, width=node_size/300,
        label=label
    )


def _get_graph_label(nx_graph):
    graph_label_dict = nx_graph.graph
    if "gid" in graph_label_dict.keys():
        label = "gid {}".format(graph_label_dict["gid"])
    elif "label" in graph_label_dict.keys():
        label = graph_label_dict["label"]
    else:
        label = "graph"
    return label

def _save_fig(plt, dir, fn, print_path=False):
    plt_cnt = 0
    if dir is None or fn is None:
        return plt_cnt
    final_path_without_ext = join(dir, fn)
    for ext in ['.png']:# '.eps']:
        final_path = append_ext_to_filepath(ext, final_path_without_ext)
        create_dir_if_not_exists(dirname(final_path))
        try:
            plt.savefig(final_path, bbox_inches='tight')
        except:
            warn('savefig')
        if print_path:
            print('Saved to {}'.format(final_path))
        plt_cnt += 1
    return plt_cnt


def _sorted_dict(d):
    rtn = OrderedDict()
    for k in sorted(d.keys()):
        rtn[k] = d[k]
    return rtn

def plot_graph(nx_graph, natt_name, f_name, log_dir):
    plot_graphs([nx_graph], natt_name, log_dir, f_name)

if __name__ == '__main__':
    from load_data import load_dataset
    from utils import get_temp_path, load_klepto
    from data_model import load_train_test_data
    dataset_name = 'imdbmulti'
    gids = [257, 1483, 959, 1187, 1490, 539]
    dataset = load_dataset(dataset_name, 'all', 'mcs', 'bfs')
    # dataset.return_gids_by_graph_label()
    # graphs = [dataset.look_up_graph_by_gid(gid).get_nxgraph() for gid in gids]

    tp = '/home/kengu13/PycharmProjects/GraphMatching/save/OurModelData/imdbmulti_train_test_mcs_bfs_subgraphs_rooted_k=3_one_hot'
    sgids = [115, 69, 5, 0, 5, 0]
    # rtn = load_klepto(tp, True)
    # if rtn:
    train_data, test_data = load_train_test_data()
    sub_graphs = [train_data.dataset.subgraph_map[gid].get_nxgraph() for gid in sgids]

    dir = get_temp_path()
    # fn = dataset_name + "_" + "_".join(list(map(lambda x: str(x), gids)))
    fn = dataset_name + "_subgraphs_" + "_".join(list(map(lambda x: str(x), sgids)))
    print_path = True
    # plot_graphs(graphs, dataset.natts[0] if len(dataset.natts) > 0 else None , dir, fn, print_path)
    plot_graphs(sub_graphs, None, dir, fn, print_path)
