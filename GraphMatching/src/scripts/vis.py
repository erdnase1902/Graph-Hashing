from utils import append_ext_to_filepath, create_dir_if_not_exists
import matplotlib

# Fix font type for ACM paper submission.
# matplotlib.use('Agg')
# matplotlib.rc('font', **{'family': 'serif', 'size': 0})  # turn off tick labels
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from collections import OrderedDict, defaultdict
from os.path import join, dirname
from warnings import warn

TYPE_COLOR_MAP = {
    'C': '#ff6666',
    'O': 'lightskyblue',
    'N': 'yellowgreen',
    'S': 'yellow',
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
NODE_SIZE = 300


def plot_node_mapping(g1, g2, mapping, node_feat_name, fix_match_pos,
                      dir, fn, need_eps, print_path, need_node_id=False,
                      addi_node_text_dict=None, prog='neato'):
    assert type(mapping) is dict
    g1, feat_dict_1 = _gen_feat_dict(g1, node_feat_name, addi_node_text_dict)
    g2, feat_dict_2 = _gen_feat_dict(g2, node_feat_name, addi_node_text_dict)

    pos_g1 = _sorted_dict(graphviz_layout(g1, prog=prog))
    pos_g2 = _sorted_dict(graphviz_layout(g2, prog=prog))

    # _orig(g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2, node_feat_name,
    #       dir, fn, need_eps, print_path)

    # _blue_red(mapping, g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2,
    #           fix_match_pos, dir, fn, need_eps, print_path)

    _detail(mapping, node_feat_name, fix_match_pos, g1, pos_g1, g2, pos_g2, dir, fn, need_eps,
            print_path, need_node_id, addi_node_text_dict)

    # pos_g1 = _sorted_dict(graphviz_layout(g1, prog='neato'))
    # pos_g2 = _sorted_dict(graphviz_layout(g2, prog='neato'))

    # _paper_style(mapping, node_feat_name, fix_match_pos, g1, pos_g1, g2, pos_g2, dir, fn,
    #              need_eps, print_path, need_node_id, addi_node_text_dict)


def plot_single_graph(g, color_dict, node_label_dict, dir, fn, need_eps, print_path, prog): # 'neato', 'dot'
    # plt.figure()
    plt.figure(figsize=(6, 14))
    # gs = gridspec.GridSpec(1, 2)
    # gs.update(wspace=0, hspace=0)  # set the spacing between axes
    # ax = plt.subplot(gs[0])
    plt.axis('off')
    pos_dict = _sorted_dict(graphviz_layout(g, prog=prog))
    color_list = _convert_to_color_list(g, color_dict)
    _plot_one_graph(g, color_list, pos_dict, node_label_dict)
    if dir is not None:
        _save_fig(plt, dir, fn, need_eps, print_path)
    else:
        plt.show()
    plt.close()


def _convert_to_color_list(g, color_dict):
    color_g = ['lightgray'] * g.number_of_nodes()
    for node in color_dict:
        color = color_dict[node]
        color_g[_get_node(g.nodes, node)] = color
    return color_g


def _orig(g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2, node_feat_name,
          dir, fn, need_eps, print_path):
    ntypes = defaultdict(int)
    if node_feat_name is not None:
        for node, ndata in g1.nodes(data=True):
            ntypes[ndata[node_feat_name]] += 1
        for node, ndata in g2.nodes(data=True):
            ntypes[ndata[node_feat_name]] += 1
    color_map = _gen_color_map(ntypes)  # ntypes are the actual node features such as C, N, Cl, ...

    color_g1 = _gen_orig_node_colors(g1, node_feat_name, color_map)
    color_g2 = _gen_orig_node_colors(g2, node_feat_name, color_map)

    _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn + '_orig', need_eps, print_path)


def _blue_red(mapping, g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2,
              fix_match_pos, dir, fn, need_eps, print_path):
    color_g1 = []
    color_g2 = []

    for node in range(g1.number_of_nodes()):
        color_g1.append('lightskyblue')
    for node in range(g2.number_of_nodes()):
        color_g2.append('lightskyblue')

    for node in mapping.keys():
        if fix_match_pos:
            pos_g2[mapping[node]] = pos_g1[node]  # matched nodes are in the same position
        color_g1[_get_node(g1.nodes, node)] = 'coral'
        color_g2[_get_node(g2.nodes, mapping[node])] = 'coral'

    _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn + '_blue_red', need_eps, print_path)


def _paper_style(mapping, node_feat_name, fix_match_pos, g1, pos_g1, g2, pos_g2, dir, fn,
                 need_eps, print_path, need_node_id, addi_node_text_dict):
    color_g1 = []
    color_g2 = []

    for node in range(g1.number_of_nodes()):
        color_g1.append('lightgray')
    for node in range(g2.number_of_nodes()):
        color_g2.append('lightgray')

    _, feat_dict_1 = _gen_feat_dict(g1, node_feat_name, need_node_id, addi_node_text_dict['g1'])
    _, feat_dict_2 = _gen_feat_dict(g2, node_feat_name, need_node_id, addi_node_text_dict['g2'])
    fix_match_pos_list = _get_fix_match_pos_list(fix_match_pos)
    for fix_node_pos in fix_match_pos_list:  # first False, then True so that updates later
        for node in mapping.keys():
            color_g1[_get_node(g1.nodes, node)] = 'coral'
            color_g2[_get_node(g2.nodes, mapping[node])] = 'coral'
            if fix_node_pos:
                pos_g2[mapping[node]] = pos_g1[node]  # matched nodes are in the same position

        new_fn = None if fn is None else fn + '_paper_style{}'.format(
            '_fix' if fix_node_pos else '')
        _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
              dir, new_fn, need_eps, print_path)


def _get_fix_match_pos_list(fix_match_pos):
    if fix_match_pos is None:
        fix_match_pos_list = [True, False]
    else:
        assert type(fix_match_pos) is bool
        fix_match_pos_list = [fix_match_pos]
    return fix_match_pos_list


def _detail(mapping, node_feat_name, fix_match_pos, g1, pos_g1, g2, pos_g2, dir, fn, need_eps,
            print_path, need_node_id, addi_node_text_dict):
    color_g1 = []
    color_g2 = []

    for node in range(g1.number_of_nodes()):
        color_g1.append('lightgray')
    for node in range(g2.number_of_nodes()):
        color_g2.append('lightgray')

    ntypes = defaultdict(int)
    for node in mapping.keys():
        ntypes[node] += 1
    for x in ntypes.values():
        assert x == 1
    color_map = _gen_color_map(ntypes)  # ntypes are the original node ids

    _, feat_dict_1 = _gen_feat_dict(g1, node_feat_name, need_node_id, addi_node_text_dict['g1'])
    _, feat_dict_2 = _gen_feat_dict(g2, node_feat_name, need_node_id, addi_node_text_dict['g2'])

    fix_match_pos_list = _get_fix_match_pos_list(fix_match_pos)
    for fix_node_pos in fix_match_pos_list:  # first False, then True so that updates later
        for node in mapping.keys():
            if fix_node_pos:
                pos_g2[mapping[node]] = pos_g1[node]
            color_g1[_get_node(g1.nodes, node)] = color_map[node]
            color_g2[_get_node(g2.nodes, mapping[node])] = color_map[node]

        new_fn = None if fn is None else fn + '_detail{}'.format('_fix' if fix_node_pos else '')
        _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
              dir, new_fn, need_eps, print_path)


def _get_node(g_nodes, node):
    # node_mapping may look like {'cat': 'dog', 'cake': 'food'} (string ids),
    # but the returned index must be wan INTEGER ith respect to the original nodes in g.
    # This is to ensure consistency node id mechanism with nx.draw_networkx()
    for i in range(len(g_nodes)):
        if list(g_nodes)[i] == node:
            return i
    raise ValueError('{} not in {}'.format(node, g_nodes))


def _gen_feat_dict(g, node_feat_name, need_node_id=True, addi_node_text_dict=None):
    feat_dict = {}
    node_mapping = {}
    for node in range(g.number_of_nodes()):
        node_mapping[node] = node
        if need_node_id:
            feat = '{}'.format(node)
            if node_feat_name is not None:
                feat += '_{}'.format(g.nodes[node][node_feat_name])
        else:
            feat = ''
            if node_feat_name is not None:
                feat += '{}'.format(g.nodes[node][node_feat_name])
        if addi_node_text_dict is not None:
            addi_node_t = addi_node_text_dict.get(node, '')  # {'0': 'q val=0.1', '7': 'q'}
            feat += addi_node_t
        feat_dict[node] = feat
    g = nx.relabel_nodes(g, node_mapping)
    return g, _sorted_dict(feat_dict)


def _gen_orig_node_colors(g, node_label_name, color_map):
    if node_label_name is not None:
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


def _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn, need_eps, print_path):
    plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes
    ax = plt.subplot(gs[0])
    ax.axis('off')
    _plot_one_graph(g1, color_g1, pos_g1, feat_dict_1)
    ax = plt.subplot(gs[1])
    ax.axis('off')
    _plot_one_graph(g2, color_g2, pos_g2, feat_dict_2)
    if dir is not None:
        _save_fig(plt, dir, fn, need_eps, print_path)
    else:
        plt.show()
    plt.close()

    # plt.figure(figsize=(8, 8))
    # plt.tight_layout()
    # plt.axis('off')
    # _plot_one_graph(g1, color_g1, pos_g1, feat_dict_1, 1500)
    # _save_fig(plt, dir, fn + '_g1', print_path)
    # plt.close()
    #
    # plt.figure(figsize=(8, 8))
    # plt.tight_layout()
    # plt.axis('off')
    # _plot_one_graph(g2, color_g2, pos_g2, feat_dict_2, 1500)
    # _save_fig(plt, dir, fn + '_g2', print_path)
    # plt.close()


def _plot_one_graph(g, color, pos, feat_dict):
    nx.draw_networkx(
        g, node_color=color, pos=pos,  # color is a list; pos is a dict
        with_labels=True, labels=feat_dict, node_size=NODE_SIZE, width=1)


def _save_fig(plt, dir, fn, need_eps=False, print_path=False):
    plt_cnt = 0
    if dir is None or fn is None:
        return plt_cnt
    final_path_without_ext = join(dir, fn)
    exts = ['.png']
    if need_eps:
        exts += '.eps'
    for ext in exts:
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


if __name__ == '__main__':
    from dataset_config import get_dataset_conf
    from load_data import load_dataset
    from utils import get_temp_path

    dataset_name = 'alchemy'
    gid1 = 1600
    gid2 = 403
    dataset = load_dataset(dataset_name, 'all', 'mcs', 'bfs')
    natts, *_ = get_dataset_conf(dataset_name)
    node_feat_name = natts[0] if len(natts) >= 1 else None
    g1 = dataset.look_up_graph_by_gid(gid1).get_nxgraph()
    g2 = dataset.look_up_graph_by_gid(gid2).get_nxgraph()
    pair = dataset.look_up_pair_by_gids(gid1, gid2)
    mapping = pair.get_y_true_list_dict_view()[0]
    print(mapping)
    dir = get_temp_path()
    fn = '{}_{}_{}'.format(dataset_name, g1.graph['gid'], g2.graph['gid'])
    need_eps = True
    print_path = True
    plot_node_mapping(g1, g2, mapping, node_feat_name, False, dir, fn, need_eps,
                      print_path)
