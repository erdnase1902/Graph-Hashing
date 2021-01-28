class MultiPath():
    def __init__(self,
                 dqn_fp, mcsp_fp, mcsppy_fp, mcsprl_fp,
                 kdown_fp, sgw_fp, sgwregret_fp, pca_fp,
                 neuralmcs_fp, dqnother_fp_list=None):
        self.fp_dict = {
            'dqn':dqn_fp,
            'mcsp':mcsp_fp,
            'mcsppy':mcsppy_fp,
            'mcsprl':mcsprl_fp,
            'kdown':kdown_fp,
            'sgw':sgw_fp,
            'sgwregret':sgwregret_fp,
            'pca':pca_fp,
            'neuralmcs':neuralmcs_fp,
        }
        if dqnother_fp_list is not None:
            for i, fp in enumerate(dqnother_fp_list):
                self.fp_dict[f'dqnother[{i}]'] = fp

# f_list = {
# 	'dataset': multipath(
# 		dqn_fp = '',
# 		mcsp_fp = '',
# 		mcsprl_fp = '',
# 		sgw_fp = '',
# 		pca_fp = '',
# 		neuralmcs_fp = '',
# 	),

import os
import csv
import operator
import matplotlib.pyplot as plt
import numpy as np
from utils import load_klepto
from os.path import join
from collections import defaultdict

def generate_plots():
    model_config = {
        'McSp':('MCSRL_backtrack_cocktail_2020-09-27T01-18-51.866868', 'py_dir'),
        # 'McSp+RL':('mcsp+rl', 'cpp_dir'),
        # 'GW-QAP':('MCSRL_backtrack_cocktail_2020-09-28T00-56-23.738291', 'py_dir'),
        # 'I-PCA':('MCSRL_backtrack_cocktail_2020-09-29T01-02-48.401424', 'py_dir'),
        # 'NMcs':('MCSRL_backtrack_cocktail_2020-09-29T10-51-52.487130', 'py_dir'),
        # 'GLSearch-Rand':('MCSRL_backtrack_cocktail_2020-10-01T13-56-37.258537', 'py_dir'),
        # 'Rand':('MCSRL_backtrack_cocktail_2020-09-28T13-28-29.819996', 'py_dir'),
        # 'GLSearch-0 Iter':('MCSRL_backtrack_cocktail_2020-09-28T13-31-53.259840', 'py_dir'),
        'GLSearch':('MCSRL_backtrack_cocktail_2020-09-28T11-34-39.628606', 'py_dir'),
    }
    dataset_config = {
        'Road': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/0_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_roadNet-CA_rw_1957_1_roadNet-CA_rw_1957_2',
        },
        'DbEn': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/1_val.klepto',
            'cpp_dir': '/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_DBP15K_en_zh_1_rw_1945_1_DBP15K_en_zh_1_rw_1945_2',
        },
        'DbZh': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/2_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_DBP15K_en_zh_2_rw_1907_1_DBP15K_en_zh_2_rw_1907_2',
        },
        'Dbpd': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/3_val.klepto',
            'cpp_dir': '/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_DBP15K_en_zh_1_rw_1945_1_DBP15K_en_zh_2_rw_1907_1',
        },
        'MuFb': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/4_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_musae_facebook_edges_rw_2247_1_musae_facebook_edges_rw_2247_2',
        },
        'Enro': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/5_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_Email-Enron_rw_3369_1_Email-Enron_rw_3369_2',
        },
        'CoPr': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/6_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_Amazon_Computers_rw_3518_1_Amazon_Computers_rw_3518_2',
        },
        'Circ': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/7_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_circuit_graph',
        },
        'HPpi': {
            'py_dir':'/home/yba/Documents/GraphMatching/model/OurMCS/logs/*/obj/8_val.klepto',
            'cpp_dir':'/home/yba/Documents/GraphMatching/model/mcs/data/temp_*_iters_PP-Pathways_ppi_rw_2152_1_PP-Pathways_ppi_rw_2152_2',
        }
    }
    x_type = 'iters_7500'
    x_type = 'time_600'
    # ss = ['.r-', 'xb-', '*g-', '+k-', 'oc-', '+m-', '^y-',
    #       '*g:', '+k:', 'oc:', '+m:', '^y:']
    # cs = [s[1] for s in ss]
    runtime_dict = defaultdict(list)
    iters_dict = defaultdict(list)
    cmap = plt.get_cmap('rainbow')#tab10,plasma,gist_ncar,jet
    cs = cmap(np.linspace(0, 1, len(model_config)))
    ms = ['o', '*', '^', 'P', 'v', 'X', 'd', '.']
    if 'time' in x_type:
        ls = ['-']
    else:
        ls = ['-', '--', '-.']
    for dataset_name, dir_map in dataset_config.items():
        i=0
        plt.figure(figsize=(9, 6))
        for model_name, (path, dir_type) in model_config.items():
            fp = dir_map[dir_type].replace('*', path)
            if model_name == 'McSp+RL':
                x_iters, x_runtime, y = parse_mcsprl_dict(fp, x_type)
            elif model_name == 'McSp':
                x_iters, x_runtime, y = parse_mcsppy_dict(fp, x_type)
            else:
                x_iters, x_runtime, y = parse_dqn_dict(fp, x_type)

            [x_iters, x_runtime], y = get_inflection_points([x_iters, x_runtime], y)

            if 'iters' in x_type:
                x, y = x_iters, y
            elif 'time' in x_type:
                x,y = x_runtime, y
            else:
                assert False

            if model_name == 'McSp+RL':
                print(dataset_name, y[-1])
            # a, b = -3, 10
            # iters_dict[model_name].append(float(x_iters[a]))
            # runtime_dict[model_name].append(float(x_runtime[a]))


            # plt.scatter(np.array(x_li), np.array(y), label=model_name, color=cs[i % len(cs)])
            plt.plot(np.array(x), np.array(y),
                     markersize=7, marker=ms[i % len(ms)], linestyle=ls[i % len(ls)],
                     linewidth=2, color=cs[i % len(cs)], label=model_name, alpha=0.9)
            i += 1

        from utils import get_temp_path
        # plt.title(dataset_name)
        plt.ylabel('Size of Best Solution Found So Far', fontsize=14)
        if 'time' in x_type:
            plt.xlabel('Runtime (sec)', fontsize=14)
        else:
            plt.xlabel('# Search Iterations', fontsize=14)
        ax = plt.gca()
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        plt.grid(True, linestyle='--')
        plt.legend(prop={'size': 12}, loc='best', ncol=2)
        plt.axis('on')

        if 'time' in x_type:
            fn = dataset_name + '_time'
        else:
            fn = dataset_name
        plt.savefig(f'{get_temp_path()}/{fn}.png')
        plt.close()

    # for model_name in iters_dict.keys():
    #     # print(iters_dict[model_name])
    #     # print(runtime_dict[model_name])
    #     total_iters = np.sum(np.array(iters_dict[model_name]))
    #     total_runtime = np.sum(np.array(runtime_dict[model_name]))
    #     print(f'per_iteration_runtime of {model_name}\t {total_runtime/total_iters}')

def get_inflection_points(xs,y):
    x_procs, y_proc = [], []
    len_y = len(y)
    for x in xs:
        assert len(x) == len_y
        x_procs.append([])
    last_y = None
    for i, y_elt in enumerate(y):
        if y_elt != last_y:
            for j, x in enumerate(xs):
                x_procs[j].append(x[i])
            # x_proc.append(x_elt)
            y_proc.append(y_elt)
            last_y = y_elt
    for j, x in enumerate(xs):
        x_procs[j].append(x[-1])
    # x_proc.append(x[-1])
    y_proc.append(y[-1])
    return x_procs, y_proc

def get_thresh(x_type):
    ver, val = x_type.split('_')
    rec_thresh = float('inf')
    time_thresh = float('inf')
    if 'iters' == ver:
        rec_thresh = float(val)
    elif 'time' == ver:
        time_thresh = float(val)
    return rec_thresh, time_thresh

def parse_dqn_dict(fp, x_type):
    result_d = load_result_dict_from_val_klepto(fp)
    dqn_data = result_d['dqn']
    x_iters, x_runtime, y = [0], [0], [0]
    rec_thresh, time_thresh = get_thresh(x_type)
    for x in dqn_data['incumbent_data']:
        if x[1] < rec_thresh and x[2] < time_thresh:
            x_iters.append(x[1])
            x_runtime.append(x[2])
            y.append(x[0])
    if 'time' not in x_type:
        x_iters.append(rec_thresh)
        x_runtime.append(x_runtime[-1])
        y.append(y[-1])
    x_iters.append(x_iters[-1])
    x_runtime.append(time_thresh)
    y.append(y[-1])
    return x_iters, x_runtime, y

def parse_mcsppy_dict(fp, x_type):
    result_d = load_result_dict_from_val_klepto(fp)
    mcsp_data = result_d['mcsp']
    x_iters, x_runtime, y = [0], [0], [0]
    rec_thresh, time_thresh = get_thresh(x_type)
    for x in  mcsp_data['incumbent_data']:
        if x[1] < rec_thresh and x[2] < time_thresh:
            x_iters.append(x[1])
            x_runtime.append(x[2])
            y.append(x[0])
    x_iters.append(rec_thresh)
    x_runtime.append(x_runtime[-1])
    y.append(y[-1])
    x_iters.append(x_iters[-1])
    x_runtime.append(time_thresh)
    y.append(y[-1])
    return x_iters, x_runtime, y


def parse_mcsp_dict(fp, x_type):
    csv_file = open(fp)
    rows = [row for row in csv_file.readlines()]
    assert len(rows) == 3
    mcs_size, _, runtime = rows
    x_iters = None
    x_runtime = [0, float(runtime)/10000.]
    y = [0, float(mcs_size)]
    return x_iters, x_runtime, y


def parse_mcsprl_dict(dir_path, x_type):
    x_iters = [0]
    x_runtime = [0]
    y = [0]
    rec_thresh, time_thresh = get_thresh(x_type)
    for fp in sorted(os.listdir(dir_path)):
        if '_' in fp and 'csv' in fp:
            # print(fp)
            _, iter_num, runtime_num = fp.split('.')[0].split('_')
            csv_file = open(join(dir_path,fp))
            rows = [row for row in csv_file.readlines()]
            assert len(rows) == 3
            mcs_size, _, _ = rows
            mcs_size = float(mcs_size)
            iter_num = int(iter_num)
            runtime_num = float(runtime_num) / 10000.
            if iter_num < rec_thresh and runtime_num < time_thresh:
                x_iters.append(iter_num)
                x_runtime.append(runtime_num)
                y.append(mcs_size)
        else:
            continue
    L = sorted(zip(x_iters, x_runtime, y), key=operator.itemgetter(0))
    x_iters, x_runtime, y = zip(*L)
    x_iters, x_runtime, y = [list(a) for a in [x_iters, x_runtime, y]]
    x_iters.append(rec_thresh)
    x_runtime.append(x_runtime[-1])
    y.append(y[-1])
    x_iters.append(x_iters[-1])
    x_runtime.append(time_thresh)
    y.append(y[-1])
    return x_iters, x_runtime, y


def parse_kdown_dict(fp, x_type):
    csv_file = open(fp)
    rows = [row for row in csv_file.readlines()]
    assert len(rows) == 3
    mcs_size, _, runtime = rows
    x_iters = None
    x_runtime = [0, float(runtime)/10000.]
    y = [0, float(mcs_size)]
    return x_iters, x_runtime, y


def load_result_dict_from_val_klepto(fp):
    dict_klepto = load_klepto(fp, '')
    if len(dict_klepto) != 1:
        print(dict_klepto.values())
        print(fp)
        assert False
    result_d = list(dict_klepto.values())[0]['result']
    return result_d


# main()
generate_plots()



