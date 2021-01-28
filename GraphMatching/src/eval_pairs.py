from eval_mne import max_emb_sim
from eval_mcs import mcs_size
from graph_pair import GraphPair
from scipy.optimize import linear_sum_assignment
from numpy import linalg as LA
from sklearn.metrics import average_precision_score, precision_score, \
    recall_score, f1_score, roc_auc_score
import numpy as np
from collections import OrderedDict
from warnings import warn
from tqdm import tqdm

#############################################
from scripts.vis import plot_node_mapping
from utils import get_temp_path
from os.path import join

#############################################

# For each pair, there can be multiple predictions, and the following getter
# functions are used to find the max/min result indicating the best one.
METRIC_BEST_GETTER_MAP = {
    'L1_dist': np.argmin,
    'L2_dist': np.argmin,
    'p': np.argmax,
    'r': np.argmax,
    'f1': np.argmax,
    'avg_p': np.argmax,
    'auc': np.argmax,
    'max_sim_left': np.argmax,
    'max_sim_right': np.argmax,
    'mcs_iso': np.argmax,
    'mcs_size_frac': np.argmax,
    'mcs_size_dev': np.argmin,
    'mcs_exact': np.argmax,
    'mcs_size': np.argmax,  # very important, since we want MCS to be as large as posible
    'reward': np.argmax,
    'time_pred(msec)': np.argmin,
    'time_true(msec)': np.argmin,
    'num_iters_pred': np.argmin,
}

# MODE_LIST = ['regular', 'lsap', 'threshold', 'rc_thres_lsap', 'tree_search',
#              'unknwon_y_true']

MODE_LIST = ['regular', 'unknwon_y_true']


def eval_pair_list(pair_list, reward_calculator, MCS_eval, FLAGS):
    assert MCS_eval is not None
    if type(pair_list) is not list or not pair_list:
        raise ValueError('pair_list must be a non-empty list')
    global_result = OrderedDict()
    global_result['final'] = _create_global_result_mode_dict()
    # hits = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    # cur_hit = 0
    for i, pair in enumerate(tqdm(pair_list)):
        # perc = i / len(pair_list)
        # if cur_hit < len(hits) and abs(perc - hits[cur_hit]) <= 0.05:
        #     print('{}/{}={:.1%}'.format(i, len(pair_list), i / len(pair_list)))
        #     cur_hit += 1
        if type(pair) is not GraphPair:
            raise ValueError('pair_list can only contain GraphPairs')
        """
        match_eval_result:
            {'regular': { 'avg': { 'f1': 0.5, ... },
                          'bestid': { 'f1': 7, ... },
                          'best': { 'f1': 0.9, ... }}
                        },
             'lsap': { 'avg': { 'f1': 0.9, ... },
                       'bestid': { 'f1': 8, ... },
                       'best': { 'f1': 0.99, ... }}
                     },
             ...
            }
        """
        match_eval_result = _match_result_single_graph_pair(
            pair, reward_calculator, global_result, MCS_eval, FLAGS)
        pair.add_match_eval_result(match_eval_result)
    for timestamp, d in global_result.items():
        for mode, mode_dict in d.items():
            num_pairs_in_this_mode = 0
            for avg_best, metric_num_dict in mode_dict.items():
                for metric in metric_num_dict:
                    num_list = metric_num_dict[metric]
                    if len(num_list) != len(pair_list):  # TODO: consider unknown pairs
                        # if FLAGS.only_iters_for_debug is not None:
                        warn('\nonly_iters_for_debug={} and #tested pairs {} '
                             '!= #total testing pairs {} '
                             '(consider unknown y_true)'.format(
                            FLAGS.only_iters_for_debug, len(num_list),
                            len(pair_list)))
                        assert len(num_list) < len(pair_list)
                        # else:
                        #     assert False, 'Should have gone through all the testing pairs'
                    # Turn the list into the average across all data points,
                    # i.e. pairs (do NOT do the min/max that gives the best).
                    metric_num_dict[metric] = np.mean(num_list)
                    num_pairs_in_this_mode = len(num_list)
            global_result[timestamp][mode]['num_pairs_in_this_mode'] = num_pairs_in_this_mode
    global_result.move_to_end('final')
    return global_result


def _create_global_result_mode_dict():
    rtn = OrderedDict()
    # Three modes: regular + lsap + threshold
    for mode in MODE_LIST:
        rtn[mode] = OrderedDict()
        for avg_best in ['avg', 'best']:
            rtn[mode][avg_best] = OrderedDict()
    return rtn


def _match_result_single_graph_pair(
        pair, reward_calculator, global_result, MCS_eval, FLAGS):
    rtn = OrderedDict()
    if pair.has_alignment_true_pred():
        rtn.update(_match_result_single_graph_pair_alignment(
            pair, reward_calculator, global_result, MCS_eval, FLAGS))
    # else:
    #     print('No true alignment matrices; Skip evaluating...')
    if pair.has_ds_score_true_pred():
        rtn.update(_match_result_single_graph_pair_ds_score(
            pair, reward_calculator, global_result, FLAGS))
    # else:
    #     print('No true+pred ds scores; Skip evaluating...')
    return rtn


def _match_result_single_graph_pair_ds_score(
        pair, reward_calculator, global_result, FLAGS):
    rtn = OrderedDict()
    result = OrderedDict()
    ds_true_trans = pair.get_ds_true(
        FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)
    for metric in ['mse', 'dev']:
        delta = pair.get_ds_pred() - ds_true_trans
        if metric == 'mse':
            num = delta ** 2 / 2
        elif metric == 'dev':
            num = np.abs(delta)
        else:
            raise NotImplementedError()
        result[metric] = num
        for avg_best in ['avg', 'best']:
            if metric not in global_result['regular'][avg_best]:
                global_result['regular'][avg_best][metric] = []
            global_result['regular'][avg_best][metric].append(num)
    rtn['regular'] = result  # only regular mode has ds_pred as continuous pred
    return rtn


def _match_result_single_graph_pair_alignment(
        pair, reward_calculator, global_result, MCS_eval, FLAGS):
    rtn = OrderedDict()
    y_pred_mat_list = pair.get_y_pred_list_mat_view(format='numpy')
    if not pair.has_y_true_dict_list():
        # d = global_result['unknwon_y_true']
        # d['avg']['mcs_size'] = [55]
        # d['best']['mcs_size'] = [55]
        # return rtn # TODO: finish rtn update
        y_true_mat_list = [None]
    else:
        y_true_mat_list = pair.get_y_true_list_mat_view(format='numpy')
    rtn['y_pred'] = _match_result_single_graph_pair_alignment_helper(
        pair, reward_calculator, global_result, FLAGS, y_pred_mat_list, y_true_mat_list, MCS_eval, False)
    if 'search' in FLAGS.model.lower():  # TODO: assumes the model has name 'search'
        tree_pred_mat_list = pair.get_tree_pred_list_mat_view(format='numpy')
        tree_true_mat_list = y_true_mat_list  # TODO: fix naming later
        rtn['tree_pred'] = _match_result_single_graph_pair_alignment_helper(
            pair, reward_calculator, global_result, FLAGS, tree_pred_mat_list, tree_true_mat_list, MCS_eval, True)
    return rtn


def _match_result_single_graph_pair_alignment_helper(
        pair, reward_calculator, global_result, FLAGS, pred_mat_list, true_mat_list, MCS_eval, is_tree):
    rtn = OrderedDict()
    if true_mat_list[0] is None:
        mode_list = ['unknwon_y_true']
    else:
        mode_list = [x for x in MODE_LIST if x != 'unknwon_y_true']
    for mode in mode_list:
        if not is_tree and mode == 'tree_search':
            continue
        if is_tree and mode != 'tree_search':
            continue
        # [(x, 1), (x, 1), (x, 1), (y, 2), (y, 2)]
        # {1: [x, x, x], 2: [y, y]}
        time_pred_map = OrderedDict()
        for i, pred_mat in enumerate(pred_mat_list):  # [(x, 1), (x, 1), (x, 1), (y, 2), (y, 2)]
            if type(pred_mat) is tuple:
                # Has timestamp.
                assert len(pred_mat) == 2
                pred_mat, timestamp_tuple = pred_mat
                assert type(timestamp_tuple) == tuple
                timestamp_iter, timestamp_real_time, iter_or_time = timestamp_tuple
                if iter_or_time == 't':
                    type(timestamp_real_time) is int, \
                    '{} {} {} {}'.format(type(timestamp_iter), type(timestamp_real_time),
                                         timestamp_iter, timestamp_real_time)
                assert iter_or_time in ['i', 't'], iter_or_time
                assert timestamp_iter != mode, 'Shouldn\'t occur'
                if timestamp_tuple not in time_pred_map:
                    time_pred_map[timestamp_tuple] = []
                time_pred_map[timestamp_tuple].append(pred_mat)
            else:
                if 'final' not in time_pred_map:
                    time_pred_map[('final', None, 'i')] = []
                time_pred_map[('final', None, 'i')].append(pred_mat)
        for timestamp_tuple, pred_mat_list_proc in time_pred_map.items():
            timestamp_iter, timestamp_real_time, iter_or_time = timestamp_tuple
            result_list = []
            for i, pred_mat in enumerate(pred_mat_list_proc):
                if not is_tree:
                    if mode == 'regular' or mode == 'unknwon_y_true':
                        pred_mat_trans = pred_mat
                    elif mode == 'lsap':
                        pred_mat_trans = lsap(pred_mat)
                    elif mode == 'threshold':
                        pred_mat_trans = threshold(pred_mat, FLAGS.theta)
                    elif mode == 'rc_thres_lsap':
                        pred_mat_trans = rc_thres_lsap(pred_mat, FLAGS.theta)
                    else:
                        raise NotImplementedError()
                else:
                    pred_mat_trans = pred_mat  # tree search already generates 0-1
                for j, true_mat in enumerate(true_mat_list):
                    result = _match_result_single_mat_pair(
                        pred_mat_trans, true_mat, mode, pair, MCS_eval,
                        reward_calculator, timestamp_iter, timestamp_real_time,
                        FLAGS)
                    result_list.append(result)
            if iter_or_time == 'i':
                ts_iter_s = '{}{}'.format(timestamp_iter,
                                          ' iter' if timestamp_iter != 'final' else '')
                d = global_result.get(ts_iter_s)
                if d is None:
                    new_dict = _create_global_result_mode_dict()
                    global_result[ts_iter_s] = new_dict  # highest-level: timestamp
                    d = new_dict
                rtn[(mode, ts_iter_s)] = _gen_avg_best(result_list, d[mode])
            elif iter_or_time == 't':
                assert len(result_list) != 0
                ts_real_s = '{} msec'.format(timestamp_real_time)
                d = global_result.get(ts_real_s)
                if d is None:
                    new_dict = _create_global_result_mode_dict()
                    global_result[ts_real_s] = new_dict
                    d = new_dict
                rtn[(mode, ts_real_s)] = _gen_avg_best(result_list, d[mode])
            else:
                assert False
    return rtn


def gen_nids(y_mat, axis):
    if axis == 0:
        rtn = np.where(y_mat == 1)[1]
    elif axis == 1:
        rtn = np.where(y_mat.T == 1)[1]
    else:
        assert False
    return list(rtn)


'''
def save_to_disk(pair, y_pred, i, j, mode, result, FLAGS):
    if not (mode == FLAGS.mode_save and i == 0 and j == 0):
        return 0
    g1, g2 = pair.g1.nxgraph, pair.g2.nxgraph
    g1_nodes = gen_nids(y_pred, 1)
    g2_nodes = gen_nids(y_pred, 0)
    mapping = {}
    min_mcs = min(len(g1_nodes), len(g2_nodes))
    for i in range(min_mcs):
        mapping[g1_nodes[i]] = g2_nodes[i]
    node_feat_name = 'type'
    dir = join(get_temp_path(), 'analysis')
    mcs_correct = int(100 * result['mcs_size_frac'])
    fn = '{}-{}_{}_{}'.format(mcs_correct, FLAGS.dataset, g1.graph['gid'], g2.graph['gid'])
    plot_node_mapping(g1, g2, mapping, node_feat_name, False, dir, fn, True)
'''


def lsap(mat):
    row_ind, col_ind = linear_sum_assignment(-mat)  # TODO: check the negative sign
    rtn = np.zeros(mat.shape)
    rtn[row_ind, col_ind] = 1
    return rtn


def threshold(mat, value):
    rtn = mat >= value
    return rtn


def rc_thres_lsap(mat, value):
    # print(mat)
    sum_rows = mat.sum(axis=1)
    sum_cols = mat.sum(axis=0)
    # print('sum_rows', sum_rows)
    # print('sum_cols', sum_cols)
    select_rows = sum_rows >= value
    select_cols = sum_cols >= value
    # print(1 - select_rows)
    # print('select_rows', select_rows)
    # print('select_cols', select_cols)
    mask_0 = _new_rc_select(mat, select_rows, select_cols)
    # print(rtn)
    clean_lsap = lsap(mask_0)
    rtn = _new_rc_select(clean_lsap, select_rows, select_cols)
    return rtn


def _new_rc_select(mat, select_rows, select_cols):
    rtn = np.zeros(mat.shape)
    rtn[select_rows,] = mat[select_rows,]
    rtn[:, select_cols] = mat[:, select_cols]
    return rtn


def _match_result_single_mat_pair(
        y_pred_mat, y_true_mat, mode, pair, MCS_eval, reward_calculator,
        timestamp, timestamp_real_time, FLAGS):
    rtn = OrderedDict()
    assert MCS_eval is not None
    if y_true_mat is not None:
        assert y_pred_mat.shape == y_true_mat.shape
        y_pred_arr = y_pred_mat.flatten()
        y_true_arr = y_true_mat.flatten()
        diff = y_pred_arr - y_true_arr
        # rtn = OrderedDict()
        rtn['L1_dist'] = LA.norm(diff, 1) / len(diff)
        rtn['L2_dist'] = LA.norm(diff, 2) / len(diff)
        # if mode == 'lsap': # TODO: may need to fix it later since some models do not produce 0-1, i.e. binary node-node matching prediction matrix
        # rtn['p'] = precision_score(y_true_arr, y_pred_arr)
        # rtn['r'] = recall_score(y_true_arr, y_pred_arr)
        # rtn['f1'] = f1_score(y_true_arr, y_pred_arr)
        # rtn['avg_p'] = average_precision_score(y_true_arr, y_pred_arr)
        # (sim_left,sim_right) = max_emb_sim(y_pred_mat, y_true_mat, pair)  # (sim_left, sim_right)
        # rtn['max_sim_left'] = sim_left
        # rtn['max_sim_right'] = sim_right
    if mode == 'regular':
        if timestamp == 'final':
            # For the 'final' mode, always use the time collected by model forward().
            # timestamp_real_time is ignored.
            rtn['time_pred(msec)'] = pair.get_pred_time()
            try:
                t = pair.get_true_time()
                if t is None:
                    t = -1
                rtn['time_true(msec)'] = t
            except:
                rtn['time_true(msec)'] = -1
        else:
            rtn['time_pred(msec)'] = timestamp_real_time
            rtn['num_iters_pred'] = timestamp

    if FLAGS.align_metric in ['mcs', 'random']:
        # rtn = OrderedDict()
        rtn['mcs_iso'], rtn['mcs_size_frac'], rtn['mcs_size_dev'], \
        rtn['mcs_exact'], rtn['mcs_size'], rtn['reward'] = \
            MCS_eval.eval(y_pred_mat, y_true_mat, pair, reward_calculator, FLAGS)
        # print(rtn['mcs_size_frac'])
        # 0 <= frac <= 1 if not debug
    # rtn['auc'] = roc_auc_score(y_true_arr, y_pred_arr)
    return rtn


def _gen_avg_best(result_list, global_result):
    metric_nums_dict = _collect_into_metric_nums_dict(result_list)
    rtn = OrderedDict()
    rtn['avg'] = OrderedDict()
    rtn['bestid'] = OrderedDict()
    rtn['best'] = OrderedDict()
    '''
    # commented out code does not work!
    num_iso = None
    '''
    for metric, num_list in metric_nums_dict.items():
        assert len(result_list) == len(num_list)
        mean_num = np.mean(num_list)
        '''
        if metric == 'mcs_iso':
            num_iso = mean_num
        if metric == 'mcs_size_frac':
            assert num_iso != None
            if num_iso != 0:
                mean_num /= num_iso
        '''
        rtn['avg'][metric] = mean_num
        if metric not in global_result['avg']:
            global_result['avg'][metric] = []
        global_result['avg'][metric].append(mean_num)
        getter = METRIC_BEST_GETTER_MAP.get(metric)
        if not getter:
            raise ValueError('Metric {} need to have a best getter defined in '
                             'METRIC_BEST_GETTER_MAP {}'.
                             format(metric, METRIC_BEST_GETTER_MAP))
        bestid = getter(num_list)
        rtn['bestid'][metric] = bestid
        assert num_list, '{]={}, {}, {}'.format(metric, num_list, bestid,
                                                metric_nums_dict)
        try:
            best_num = num_list[bestid]
        except:
            print(num_list)
            print(bestid)
            print('Weird exception caught')
            exit(-1)
        rtn['best'][metric] = best_num
        if metric not in global_result['best']:
            global_result['best'][metric] = []
        global_result['best'][metric].append(best_num)
    return rtn


def _collect_into_metric_nums_dict(result_list):
    if not result_list:
        raise ValueError('Must pass a non-empty list of result dicts')
    metric_nums_dict = OrderedDict()
    for result in result_list:
        assert type(result) is OrderedDict
        for metric, number in result.items():
            if metric not in metric_nums_dict:
                metric_nums_dict[metric] = []
            metric_nums_dict[metric].append(number)
    return metric_nums_dict


def rerun_from_loaded_logs(dataset_name, log_folder, theta):
    from utils import get_model_path, load
    from load_data import load_dataset
    from pprint import pprint
    print('theta {}'.format(theta))

    log_folder = join(get_model_path(), 'OurMCS', 'logs', log_folder)
    ld = load(join(log_folder, 'final_test_pairs.klepto'))
    pairs = ld['test_data_pairs']

    dataset = load_dataset(dataset_name, 'all', 'mcs', 'bfs')

    # regenerate y_true_dict_list
    for gids in pairs.keys():
        gid1, gid2 = gids
        g1 = dataset.look_up_graph_by_gid(gid1)
        g2 = dataset.look_up_graph_by_gid(gid2)
        pair_true = dataset.look_up_pair_by_gids(gid1, gid2)
        pair = pairs[gids]
        pair.assign_g1_g2(g1, g2)
        pair.assign_y_true_dict_list(pair_true.get_y_true_list_dict_view())

    # construct flags
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--only_iters_for_debug', type=int, default=None)
    parser.add_argument('--dataset', default=dataset_name)
    parser.add_argument('--align_metric', default='mcs')
    parser.add_argument('--theta', type=float, default=theta)
    parser.add_argument('--debug', type=bool, default='debug' in dataset_name)
    FLAGS = parser.parse_args()

    # call prediction code
    pair_list = [pairs[gids] for gids in pairs.keys()]
    global_result = eval_pair_list(pair_list, FLAGS)

    pprint(global_result)
    fn = join(log_folder, 'updated_results_theta_{}.txt'.format(theta))
    with open(fn, 'w') as f:
        pprint(global_result, stream=f)


if __name__ == '__main__':
    dataset_name = 'imdbmulti'
    log_folder = 'GMN-BCE_imdbmulti_2019-08-31T16-15-04.134997'
    theta = 0.7

    from utils import slack_notify
    import traceback

    try:
        rerun_from_loaded_logs(dataset_name, log_folder, theta)
    except:
        traceback.print_exc()
        slack_notify('rerun_from_loaded_logs error: {}'.format(log_folder))
    else:
        slack_notify('rerun_from_loaded_logs {} complete'.format(log_folder))
