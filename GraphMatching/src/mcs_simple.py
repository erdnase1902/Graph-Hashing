import sys
sys.path.insert(1, '/project/Graph-Hashing/GraphMatching/src')
from utils2 import get_root_path, exec_cmd, sorted_nicely, get_ts, create_dir_if_not_exists
from os.path import isfile, join, basename
from nx_to_mivia import convert_to_mivia, get_current_label
from os import getpid
from time import time
import networkx as nx
from glob import glob
import pickle5 as pickle

def node_id_map_to_label_map(g1, g2, node_id_map):
    node_label_map = {}
    for (source1, target1), (source2, target2) in node_id_map.items():
        g1_edge = (g1.node[source1]['label'], g1.node[target1]['label'])
        g2_edge = (g2.node[source2]['label'], g2.node[target2]['label'])
        node_label_map[g1_edge] = g2_edge
    return node_label_map

def get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping):
    node_map = {}
    for edge1, edge2 in edge_mapping.items():
        nodes_edge1 = id_edge_map1[edge1]
        nodes_edge2 = id_edge_map2[edge2]
        nodes1 = (nodes_edge1[0], nodes_edge1[1])
        nodes2 = (nodes_edge2[0], nodes_edge2[1])
        node_map[nodes1] = nodes2
    return node_map


def get_id_edge_map(graph):
    id_edge_map = {}
    for u, v, edge_data in graph.edges(data=True):
        edge_id = edge_data['id']
        assert edge_id not in id_edge_map
        id_edge_map[edge_id] = (u, v)
    return id_edge_map

def get_mcs_info(g1, g2, edge_mappings):
    id_edge_map1 = get_id_edge_map(g1)
    id_edge_map2 = get_id_edge_map(g2)

    mcs_node_id_maps = []
    mcs_node_label_maps = []
    for edge_mapping in edge_mappings:
        node_id_map = get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping)
        node_label_map = node_id_map_to_label_map(g1, g2, node_id_map)
        mcs_node_id_maps.append(node_id_map)
        mcs_node_label_maps.append(node_label_map)
    return mcs_node_id_maps, mcs_node_label_maps


def mcis_edge_map_from_nodes(g1, g2, node_mapping):
    edge_map = {}
    induced_g1 = g1.subgraph([key for key in node_mapping.keys()])
    induced_g2 = g2.subgraph([key for key in node_mapping.values()])

    used_edge_ids_g2 = set()
    for u1, v1, edge1_attr in induced_g1.edges(data=True):
        u2 = node_mapping[int(u1)]
        v2 = node_mapping[int(v1)]
        edge1_id = edge1_attr['id']
        found = False
        for temp1, temp2, edge2_attr in induced_g2.edges(nbunch=[u2, v2], data=True):
            if (u2 == temp1 and v2 == temp2) or (u2 == temp2 and v2 == temp1):
                edge2_id = edge2_attr['id']
                if edge2_id in used_edge_ids_g2:
                    continue
                used_edge_ids_g2.add(edge2_id)
                edge_map[edge1_id] = edge2_id
                found = True
        # if not found:
        #     raise ValueError('X')

    return edge_map

def _get_label_map(g1, g2, label_key):
    # Need this function because the two graphs needs consistent labelings in the mivia format. If they are called
    # separately, then they will likely have wrong labelings.
    label_dict = {}
    label_counter = 0
    # We make the labels into ints so that they can fit in the 16 bytes needed
    # for the labels in the mivia format. Each unique label encountered just gets a
    # unique label from 0 to edge_num - 1
    for g in [g1, g2]:
        for node, attr in g.nodes(data=True):
            current_label = get_current_label(attr,label_key)
            if current_label not in label_dict:
                label_dict[current_label] = label_counter
                label_counter += 1
    return label_dict
def setup_temp_data_folder(gp, append_str, fp_prepend_info=''):
    dir = gp + '/data'
    create_dir_if_not_exists(dir)
    if fp_prepend_info != '':
        append_str = fp_prepend_info.replace(';', '_')
    tp = dir + '/temp_{}'.format(append_str)
    exec_cmd('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec_cmd('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, append_str))
    return src, tp
def write_mivia_input_file(graph, filepath, labeled, label_key, label_map):
    bytes, idx_to_node = convert_to_mivia(graph, labeled, label_key, label_map)
    with open(filepath, 'wb') as writefile:
        for byte in bytes:
            writefile.write(byte)
    return idx_to_node
def get_mcs_path():
    return get_root_path() + '/model/mcs'
def get_append_str(g1, g2):
    return '{}_{}_{}_{}'.format(
        get_ts(), getpid(), g1.graph['gid'], g2.graph['gid'])
def mcs_simple(g1, g2, algo, labeled, label_key, recursion_threshold,
                   save_every_seconds, save_every_iter,
                   debug=False, timeit=False, timeout=None,
                   computer_name='', fp_prepend_info=''):
    """See mcs function. Must match return format."""
    # Input format is ./model/mcs/data/temp_<ts>_<pid>_<gid1>_<gid2>/<gid1>.<extension>
    # Prepare both graphs to be read by the program.
    commands = []
    if algo == 'mccreesh2016':
        binary_name = 'solve_max_common_subgraph'
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('' if labeled else '--unlabelled')
        commands.append('--connected')
        commands.append('--undirected')
    elif algo == 'mccreesh2017' or algo == 'mcsp+rl':
        # print('computer_name', computer_name)
        # binary_name = 'mcsp'  # 'mcsp_scai1'
        # if 'scai1' in computer_name:
        #     binary_name = 'mcsp_scai1'
        if algo == 'mccreesh2017':
            binary_name = 'code/james-cpp-periodic-save/mcsp'
        elif algo == 'mcsp+rl':
            binary_name = 'mcsp+rl'
        else:
            assert False
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('--labelled' if labeled else '')
        commands.append('--connected')
        commands.append('--quiet')
        if timeout:
            commands.append('--timeout={}'.format(timeout))
        commands.append('min_product')
    else:
        raise RuntimeError('{} not yet implemented in mcs_cpp_helper'.format(algo))

    gp = get_mcs_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str, fp_prepend_info)
    filepath_g1 = '{base}/{g1}.{extension}'.format(
        base=t_datapath,
        g1=g1.graph['gid'],
        extension=extension)
    filepath_g2 = '{base}/{g2}.{extension}'.format(
        base=t_datapath,
        g2=g2.graph['gid'],
        extension=extension)

    if labeled:
        label_map = _get_label_map(g1, g2, label_key)
    else:
        label_map = {}
    idx_to_node_1 = write_fn(g1, filepath_g1, labeled, label_key, label_map)
    idx_to_node_2 = write_fn(g2, filepath_g2, labeled, label_key, label_map)

    cpp_binary = '{mcs_path}/{algo}/{binary}'.format(
        mcs_path=get_mcs_path(), algo=algo,
        binary=binary_name)

    # Run the solver.
    t = time()
    if algo in ['mccreesh2017', 'mcsp+rl']:
        commands.append('--recursion_threshold=' + str(recursion_threshold))
        commands.append('--save_every_seconds=' + str(save_every_seconds))
        if algo == 'mcsp+rl' and save_every_iter:
            commands.append('--save_every_iter')

    # runthis = '{bin} {commands} {g1} {g2}'.format(
    #     bin=cpp_binary, commands=' '.join(commands),
    #     g1=filepath_g1, g2=filepath_g2)
    exec_result = exec_cmd('{bin} {commands} {g1} {g2}'.format(
        bin=cpp_binary, commands=' '.join(commands),
        g1=filepath_g1, g2=filepath_g2), timeout)
    elapsed_time = time() - t
    elapsed_time *= 1000  # sec to msec

    # # Get out immediately with a -1 so the csv file logs failed test.
    # # mcs_size = -1 means failed in the time limit.
    # # mcs_size = -2 means failed by memory limit or other error.
    # if not exec_result:
    #     return -1, -1, -1, timeout * 1000

    # # Check if the output file exists, otherwise something failed with no output.
    output_filepath = join(t_datapath, 'output.csv')
    # if not isfile(output_filepath):
    #     return -2, -1, -1, 0

    # Process the output data.

    all_csv_files = sorted_nicely(glob(join(t_datapath, 'output_*.csv')))
    if isfile(output_filepath):
        all_csv_files += [output_filepath]
    mcs_size_list = []
    idx_mapping_list = []
    refined_mcs_node_label_maps_list = []
    refined_edge_mapping_list = []
    time_list = []

    for f in all_csv_files:
        with open(f, 'r') as readfile:
            x = None
            try:
                x = readfile.readline().strip()
                num_nodes_mcis = int(x)
            except:
                print(f)
                print(x)
                exit(-1)
            idx_mapping = eval(readfile.readline().strip())
            mcs_node_id_mapping = {idx_to_node_1[idx1]: idx_to_node_2[idx2] for idx1, idx2 in
                                   idx_mapping.items()}
            # elapsed_time = int(readfile.readline().strip())

        idx_mapping_list.append(idx_mapping)

        # Sanity Check 1: connectedneses
        indices_left = idx_mapping.keys()
        indices_right = idx_mapping.values()
        subgraph_left = g1.subgraph(indices_left)
        subgraph_right = g2.subgraph(indices_right)
        is_connected_left = nx.is_empty(subgraph_left) or nx.is_connected(subgraph_left)
        is_connected_right = nx.is_empty(subgraph_right) or nx.is_connected(subgraph_right)
        assert is_connected_left and is_connected_right, \
            'Unconnected result for pair ={}\n{}'.format(f, idx_mapping)
        # # Sanity Check 2: isomorphism (NOTE: labels not considered!)
        # # import networkx.algorithms.isomorphism as iso
        # # natts = ['type']
        # # nm = iso.categorical_node_match(natts, [''] * len(natts))
        # assert nx.is_isomorphic(subgraph_left, subgraph_right)#, node_match=nm)

        refined_edge_mapping = mcis_edge_map_from_nodes(g1, g2, mcs_node_id_mapping)
        mcs_size_list.append(num_nodes_mcis)
        mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [refined_edge_mapping])
        refined_mcs_node_label_maps_list.append(mcs_node_label_maps)
        refined_edge_mapping_list.append(refined_edge_mapping)
        bfn = basename(f)
        if 'output_' in bfn:
            # print('Output', bfn)
            if algo == 'mcsp+rl':
                time_list.append(int(bfn.split('output_')[1].split('_')[1].split('.csv')[0])) # iter
            else:
                time_list.append(float(bfn.split('output_')[1].split('.csv')[0]))

    # clean_up([t_datapath])  # TODO: remove me if you are debugging the solver! Derek: Search here

    if not debug:
        return mcs_size_list

    if timeit:
        return mcs_size_list, idx_mapping_list, refined_mcs_node_label_maps_list, \
               refined_edge_mapping_list, time_list + [elapsed_time]
    else:
        return mcs_size_list, idx_mapping_list, refined_mcs_node_label_maps_list, \
               refined_edge_mapping_list

def mcs_simple_default_args(g1, g2):
    algo = 'mccreesh2017'
    labeled = True
    label_key = 'type'
    recursion_threshold = 7500
    save_every_seconds = -1
    save_every_iter = False
    debug = False
    timeit = True
    timeout = 1
    computer_name = 'yba'
    fp_prepend_info = 'mccreesh2017_iters_aids700nef'
    return mcs_simple(g1, g2, algo, labeled, label_key, recursion_threshold,
               save_every_seconds, save_every_iter,
               debug, timeit, timeout,
               computer_name, fp_prepend_info)[0]
if __name__ == '__main__':
    f1 = open('/home/user/g1.pkl', 'rb')
    f2 = open('/home/user/g2.pkl', 'rb')
    g1 = pickle.load(f1)
    g2 = pickle.load(f2)
    f1.close()
    f2.close()
    algo = 'mccreesh2017'
    labeled = True
    label_key = 'type'
    recursion_threshold = 7500
    save_every_seconds = -1
    save_every_iter = False
    debug = False
    timeit = True
    timeout = 1
    computer_name = 'yba'
    fp_prepend_info = 'mccreesh2017_iters_aids700nef'
    mcs_value = mcs_simple_default_args(g1, g2)
    mcs_simple(g1, g2, algo, labeled, label_key, recursion_threshold,
                              save_every_seconds, save_every_iter,
                              debug, timeit, timeout,
                              computer_name, fp_prepend_info)
