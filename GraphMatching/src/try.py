# # # from scipy.stats import wasserstein_distance
# # #
# # # d = wasserstein_distance([[0, 1], [0, 1]], [[0, 1], [0, 1]])
# # #
# # # print(d)
# #
# #
# # # from utils import load
# # #
# # # loaded = load('/home/yba/Documents/GraphMatching/model/OurMCS/logs/prototype_transformer_linux_2019-07-03T10-42-50.985102/final_test_pairs.klepto')
# # #
# # # print(loaded)
# # #
# # #
# # # from eval_ranking import eval_ranking
# # #
# # # result_dict, true_m, pred_m = eval_ranking(
# # #     true_ds_mat, pred_ds_mat, FLAGS.dos_pred, time_mat)
# #
# # # import numpy as np
# # #
# # # from eval_pairs import _rc_thres_lsap
# # # # threshold = 0.5
# # # mat = np.array([[0, 0.45, 0, 0, 0], [0, 0, 0.8, 0.1, 0], [0, 0.45, 0, 0.3, 0], [0, 0, 0, 0, 0]])
# # # print(mat)
# # # # sum_rows = mat.sum(axis=1)
# # # # sum_cols = mat.sum(axis=0)
# # # # print('sum_rows', sum_rows)
# # # # print('sum_cols', sum_cols)
# # # # select_rows = sum_rows > threshold
# # # # select_cols = sum_cols > threshold
# # # # print('select_rows', select_rows)
# # # # print('select_cols', select_cols)
# # # # rtn = np.zeros(mat.shape)
# # # # rtn[select_rows,] = mat[select_rows,]
# # # # rtn[:, select_cols] = mat[:, select_cols]
# # # print(_rc_thres_lsap(mat, 0.5))
# # # # print(mat)
# #
# # # import numpy as np
# # #
# # # mat = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 0]])
# # # print(mat)
# # # ind = np.where(mat == 0.5)
# # # print(ind)
# #
# #
# # # from functools import wraps
# # # import errno
# # # import os
# # # import signal
# # #
# # # class TimeoutError(Exception):
# # #     pass
# # #
# # # def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
# # #     def decorator(func):
# # #         def _handle_timeout(signum, frame):
# # #             raise TimeoutError(error_message)
# # #
# # #         def wrapper(*args, **kwargs):
# # #             signal.signal(signal.SIGALRM, _handle_timeout)
# # #             signal.alarm(seconds)
# # #             try:
# # #                 result = func(*args, **kwargs)
# # #             finally:
# # #                 signal.alarm(0)
# # #             return result
# # #
# # #         return wraps(func)(wrapper)
# # #
# # #     return decorator
# # #
# # # from time import sleep
# # #
# # #
# # # #
# # # #
# # # # @timeout(1)
# # # def long_running_function2():
# # #     sum = 0
# # #     for i in range(1000000):
# # #         sum += i / 5031
# # #     sleep(10)
# # #     return sum
# # #
# # #
# # # # result = long_running_function2()
# # # #
# # # # print(result)
# # # import signal
# # #
# # #
# # # class timeout:
# # #     def __init__(self, seconds=1, error_message='Timeout'):
# # #         self.seconds = seconds
# # #         self.error_message = error_message
# # #
# # #     def handle_timeout(self, signum, frame):
# # #         raise TimeoutError(self.error_message)
# # #
# # #     def __enter__(self):
# # #         signal.signal(signal.SIGALRM, self.handle_timeout)
# # #         signal.alarm(self.seconds)
# # #
# # #     def __exit__(self, type, value, traceback):
# # #         signal.alarm(0)
# # #
# # # try:
# # #     with timeout(seconds=3):
# # #         result = long_running_function2()
# # #         print(result)
# # # except TimeoutError as e:
# # #     print(e)
# #
# #
# # # from utils import load
# # #
# # # x = load('/home/yba/Documents/GraphMatching/model/OurMCS/logs/gmn_icml_mlp_mcs_debug_2019-09-01T23-09-37.641091/FLAGS.klepto')
# # # print(x)
# # # FLAGS = x['FLAGS']
# # # print(FLAGS.theta)
# #
# # # import torch
# # #
# # # index = torch.Tensor([[0, 1], [1, 2], [2, 3]]).type(torch.long)
# # # A2 = torch.Tensor(
# # #     [[0, 1, 1, 1, 1], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]).type(torch.long)
# # #
# # # print(index)
# # # print(A2)
# # #
# # # index_A2 = torch.Tensor([1, 2, 3]).type(torch.long)
# # # q = torch.index_select(A2, 1, index_A2)
# # # print(q)
# #
# #
# # # import numpy as np
# # #
# # # y_mat = np.array([[1,0,0],[0,0,1],[0,1,0]])
# # # rtn = np.where(y_mat == 1)[1]
# # # print(rtn)
# #
# # # class Node(object):
# # #     def __init__(self, d):
# # #         self.d = d
# # #
# # #
# # # n = Node({})
# # #
# # # nn_map = n.d.copy()
# # # nn_map[6] = 4
# # # new_n = Node(nn_map)
# # #
# # # nn_map = new_n.d.copy()
# # # nn_map[7] = 5
# # # new_new_n = Node(nn_map)
# # #
# # #
# # # nn_map = new_n.d.copy()
# # # nn_map[7] = 5
# # # new_new_n = Node(nn_map)
# # #
# # #
# # # print(n.d)
# # # print(new_n.d)
# # # print(new_new_n.d)
# # #
# #
# #
# # # class VtxPair(object):
# # #     def __init__(self, v, w):
# # #         assert type(v) is int
# # #         assert type(w) is int
# # #         self.v = v
# # #         self.w = w
# # #
# # #
# # # def _check_list_of(li_to_check, type_str):
# # #     assert type(li_to_check) is list
# # #     if len(li_to_check) != 0:
# # #         z = li_to_check[0].__class__.__name__
# # #         assert li_to_check[0].__class__.__name__ == type_str
# # #
# # # x = [VtxPair(1,2)]
# # # _check_list_of(x, 'VtxPair')
# # #
# # # x = [1]
# # # _check_list_of(x, 'int')
# #
# # # import torch
# # #
# # # a = torch.tensor([1, 2, 3])
# # # b = torch.tensor([4, 5, 6])
# # # c = torch.tensor([7, 8, 9])
# # # stuff = [a, b, c]
# # #
# # # for i, x in enumerate(stuff):
# # #     if i == 0:
# # #         cat_tensor = torch.unsqueeze(x, dim=0)
# # #     else:
# # #         v = torch.unsqueeze(x, dim=0)
# # #         cat_tensor = torch.cat([cat_tensor, v], dim=0)
# # # print(cat_tensor)
# #
# # '''
# # from data_model import load_train_test_data
# # from utils import load
# # from config import FLAGS
# # train_data, test_data = load_train_test_data()
# #
# # # p_mcsrl = '/home/derek/Documents/Research/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1000,np_te=100,nn_core=8,nn_tot=15,ed=3,gen_type=BA_2020-07-22T18-25-33.161285/final_test_pairs.klepto'
# # # p_mcsplit = '/home/derek/Documents/Research/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1000,np_te=100,nn_core=8,nn_tot=15,ed=3,gen_type=BA_2020-07-22T18-26-24.824275/final_test_pairs.klepto'
# #
# # if 'BA' in FLAGS.dataset:
# #     # BA dataset
# #     fout='BA'
# #     p_mcsrl = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=1,num_feat=500,gen_type=BA_2020-07-19T21-49-41.370092/final_test_pairs.klepto'
# #     p_mcsplit = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=1,num_feat=500,gen_type=BA_2020-07-19T20-42-25.056747/final_test_pairs.klepto'
# #     # fout = 'BA_less_labels'
# #     # p_mcsrl = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=1,num_feat=50,gen_type=BA_2020-07-23T00-15-28.311890/final_test_pairs.klepto'
# #     # p_mcsplit = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=1,num_feat=50,gen_type=BA_2020-07-23T08-22-37.295504/final_test_pairs.klepto'
# # elif 'ER' in FLAGS.dataset:
# #     # ER dataset
# #     fout='ER'
# #     p_mcsrl = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=0.05,num_feat=500,gen_type=ER_2020-07-19T21-49-57.481972/final_test_pairs.klepto'
# #     p_mcsplit = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=0.05,num_feat=500,gen_type=ER_2020-07-19T20-45-39.462021/final_test_pairs.klepto'
# # elif 'WS' in FLAGS.dataset:
# #     # WS dataset
# #     fout='WS'
# #     p_mcsrl = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=0.2|4,num_feat=500,gen_type=WS_2020-07-19T21-50-16.471536/final_test_pairs.klepto'
# #     p_mcsplit = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_syn:np_tr=1,np_te=1,nn_core=2500,nn_tot=5000,ed=0.2|4,num_feat=500,gen_type=WS_2020-07-19T20-46-37.587515/final_test_pairs.klepto'
# # elif 'circuit' in FLAGS.dataset:
# #     # circuit dataset
# #     # fout='ckt'
# #     # p_mcsrl = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_circuit_graph_2020-07-15T21-34-51.179578/final_test_pairs.klepto'
# #     # p_mcsplit = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_circuit_graph_2020-07-17T19-00-40.858857/final_test_pairs.klepto'
# #
# #     fout='ckt-less'
# #     p_mcsrl = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_circuit_graph_2020-07-25T00-58-47.236226/final_test_pairs.klepto'
# #     p_mcsplit = '/home/yba/Documents/GraphMatching/model/OurMCS/logs/MCSRL_backtrack_circuit_graph_2020-07-25T00-57-51.651285/final_test_pairs.klepto'
# #
# # else:
# #     assert False
# #
# # def get_ith(mat_list, i):
# #     if i == -1:
# #         return mat_list[-1]
# #     elif len(mat_list) < i+1:
# #         return mat_list[-1]
# #     else:
# #         return mat_list[i]
# #
# #
# # if 'CA-GrQc' in FLAGS.dataset:
# #     fnrl = 'MCSRL_backtrack_isogexf:/CA-GrQc_2020-08-09T09-43-53.890191'
# #     fnmcsp = 'MCSRL_backtrack_isogexf:/CA-GrQc_2020-08-09T09-44-39.249824'
# #     fout = FLAGS.dataset
# # elif 'ChG-Miner_miner-chem-gene' in FLAGS.dataset:
# #     fnrl = 'MCSRL_backtrack_isogexf:/ChG-Miner_miner-chem-gene_2020-08-08T23-45-12.173636'
# #     fnmcsp = 'MCSRL_backtrack_isogexf:/ChG-Miner_miner-chem-gene_2020-08-08T23-44-09.570445'
# #     fout = FLAGS.dataset
# # elif 'p2p-Gnutella08' in FLAGS.dataset:
# #     fnrl = 'MCSRL_backtrack_isogexf:/p2p-Gnutella08_2020-08-09T17-07-43.210293'
# #     fnmcsp = 'MCSRL_backtrack_isogexf:/p2p-Gnutella08_2020-08-09T16-48-49.438602'
# #     fout = FLAGS.dataset
# # elif 'circuit' in FLAGS.dataset:
# #     nolab = False
# #     if nolab:
# #         fnrl = 'MCSRL_backtrack_circuit_graph_2020-08-09T09-09-27.228486'
# #         fnmcsp = 'MCSRL_backtrack_circuit_graph_2020-08-09T11-10-18.039144'
# #         fout = FLAGS.dataset + 'nolab'
# #     else:
# #         fnrl = 'MCSRL_backtrack_circuit_graph_2020-08-09T00-40-49.131211'
# #         fnmcsp = 'MCSRL_backtrack_circuit_graph_2020-08-09T09-09-27.228486'
# #         fout = FLAGS.dataset + 'full_duration'
# # else:
# #     assert False
# #
# # p_mcsrl = f'/home/yba/Documents/GraphMatching/model/OurMCS/logs/{fnrl}/final_test_pairs.klepto'
# # p_mcsplit = f'/home/yba/Documents/GraphMatching/model/OurMCS/logs/{fnmcsp}/final_test_pairs.klepto'
# # add_edges = 'dqn'
# #
# # import torch
# # import networkx as nx
# # import numpy as np
# # def get_found_by(nid, y_vec_mcsrl, y_vec_mcsplit):
# #     is_sel_mcsrl = y_vec_mcsrl[nid] > 0.5
# #     is_sel_mcsplit = y_vec_mcsplit[nid] > 0.5
# #     if is_sel_mcsrl and is_sel_mcsplit:
# #         return 'both'
# #     elif is_sel_mcsrl:
# #         return 'dqn'
# #     elif is_sel_mcsplit:
# #         return 'mcsp'
# #     else:
# #         return 'neither'
# #
# # def manual_disjnt_union(g1, g2):
# #     nid_new = 0
# #     relabel_dict_g1, relabel_dict_g2 = {}, {}
# #     for node in g1.nodes():
# #         relabel_dict_g1[node] = nid_new
# #         nid_new += 1
# #     for node in g2.nodes():
# #         relabel_dict_g2[node] = nid_new
# #         nid_new += 1
# #     g1 = nx.relabel_nodes(g1, relabel_dict_g1)
# #     g2 = nx.relabel_nodes(g2, relabel_dict_g2)
# #     g = nx.union(g1, g2)
# #     return g, relabel_dict_g1, relabel_dict_g2
# #
# # def get_edges(relabel_dict_g1, relabel_dict_g2, y_pred_mat_mcsrl, y_pred_mat_mcsplit):
# #     edges_mcsrl = get_edges_single_mat(relabel_dict_g1, relabel_dict_g2, y_pred_mat_mcsrl)
# #     edges_mcsp = get_edges_single_mat(relabel_dict_g1, relabel_dict_g2, y_pred_mat_mcsplit)
# #     return edges_mcsrl, edges_mcsp
# #
# # def mat2dict(mat):
# #     dictionary = {}
# #     row_indices, col_indices = np.where(mat==1)
# #     assert len(row_indices) == len(col_indices)
# #     for i, row_idx in enumerate(row_indices):
# #         dictionary[row_idx] = col_indices[i]
# #     return dictionary
# #
# # def get_edges_single_mat(relabel_dict_g1, relabel_dict_g2, matching_mat):
# #     matching_dict = mat2dict(matching_mat)
# #     edges = []
# #     for v,w in matching_dict.items():
# #         edges.append((relabel_dict_g1[v], relabel_dict_g2[w]))
# #     return edges
# #
# # def label_g(g, y_pred_mat_mcsrl, y_pred_mat_mcsplit):
# #     y_vec_mcsrl = np.sum(y_pred_mat_mcsrl, axis=1)
# #     y_vec_mcsplit = np.sum(y_pred_mat_mcsplit, axis=1)
# #     for nid in range(g.number_of_nodes()):
# #         g.nodes[nid]['found_by'] = get_found_by(nid, y_vec_mcsrl, y_vec_mcsplit)
# #         if 'circuit' in FLAGS.dataset:
# #             actual_label = f"{g.nodes[nid]['is_device']}-{g.nodes[nid]['name']}-{g.nodes[nid]['type']}-{g.nodes[nid]['port']}"
# #             g.nodes[nid]['actual_label'] = actual_label
# #
# # def give_all_edges_in_g_label(g, label):
# #     for edge in g.edges:
# #         g.edges[edge]['label'] = label
# #
# # def write_graphs(p_mcsrl, p_mcsplit, fout, add_edges):
# #     f_mcsrl = load(p_mcsrl)['test_data_pairs']
# #     f_mcsplit = load(p_mcsplit)['test_data_pairs']
# #     get_graph = lambda k: test_data.dataset.look_up_graph_by_gid(k).nxgraph
# #     for key in f_mcsplit.keys():
# #         assert len(key) == 2
# #         g1, g2 = get_graph(key[0]), get_graph(key[1])
# #         y_pred_mat_mcsrl = get_ith(f_mcsrl[key].y_pred_mat_list,-1)
# #         y_pred_mat_mcsplit = get_ith(f_mcsplit[key].y_pred_mat_list,-1)
# #
# #         label_g(g1, y_pred_mat_mcsrl, y_pred_mat_mcsplit)
# #         label_g(g2, np.transpose(y_pred_mat_mcsrl), np.transpose(y_pred_mat_mcsplit))
# #
# #         if add_edges is not None:
# #             assert add_edges in ['mcsp', 'dqn']
# #             u, relabel_dict_g1, relabel_dict_g2 = manual_disjnt_union(g1, g2)
# #             edges_mcsrl, edges_mcsp = get_edges(relabel_dict_g1, relabel_dict_g2, y_pred_mat_mcsrl, y_pred_mat_mcsplit)
# #             give_all_edges_in_g_label(u, label='graph')
# #             if add_edges == 'dqn':
# #                 u.add_edges_from(edges_mcsrl, label='mcsrl')
# #             elif add_edges == 'mcsp':
# #                 u.add_edges_from(edges_mcsp, label='mcsp')
# #             else:
# #                 assert False
# #             fout += f'_{add_edges}'
# #         else:
# #             u = nx.disjoint_union(g1, g2)
# #         nx.write_gexf(u, '/home/yba/Documents/GraphMatching/file/{}.gexf'.format(fout))
# #         exit(-1)
# #
# # write_graphs(p_mcsrl, p_mcsplit, fout, add_edges)
#
#
# # num_features=50
# # gaussian = np.random.normal(0, 0.1, 100*(num_features*2))
# # reward_vec, _ = np.histogram(gaussian, bins=num_features*2)
# # reward_matrix = []
# # for i in range(num_features-1, -1, -1):
# #     reward_matrix.append(reward_vec[i:i+num_features])
# # reward_matrix = np.stack(tuple(reward_matrix))
# # reward_matrix = (reward_matrix + np.transpose(reward_matrix)) / (2*np.max(reward_matrix))
# # x =0
# # # from utils import get_temp_path
# # # # from torch_geometric.datasets import ZINC
# # #
# # #
# # # import os
# # # import os.path as osp
# # # import shutil
# # # import pickle
# # #
# # # import torch
# # # from tqdm import tqdm
# # # from torch_geometric.data import (InMemoryDataset, Data, download_url,
# # #                                   extract_zip)
# # #
# # #
# # # class ZINC(InMemoryDataset):
# # #     r"""The ZINC dataset from the `"Grammar Variational Autoencoder"
# # #     <https://arxiv.org/abs/1703.01925>`_ paper, containing about 250,000
# # #     molecular graphs with up to 38 heavy atoms.
# # #     The task is to regress a molecular property known as the constrained
# # #     solubility.
# # #
# # #     Args:
# # #         root (string): Root directory where the dataset should be saved.
# # #         subset (boolean, optional): If set to :obj:`True`, will only load a
# # #             subset of the dataset (13,000 molecular graphs), following the
# # #             `"Benchmarking Graph Neural Networks"
# # #             <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
# # #         split (string, optional): If :obj:`"train"`, loads the training
# # #             dataset.
# # #             If :obj:`"val"`, loads the validation dataset.
# # #             If :obj:`"test"`, loads the test dataset.
# # #             (default: :obj:`"train"`)
# # #         transform (callable, optional): A function/transform that takes in an
# # #             :obj:`torch_geometric.data.Data` object and returns a transformed
# # #             version. The data object will be transformed before every access.
# # #             (default: :obj:`None`)
# # #         pre_transform (callable, optional): A function/transform that takes in
# # #             an :obj:`torch_geometric.data.Data` object and returns a
# # #             transformed version. The data object will be transformed before
# # #             being saved to disk. (default: :obj:`None`)
# # #         pre_filter (callable, optional): A function that takes in an
# # #             :obj:`torch_geometric.data.Data` object and returns a boolean
# # #             value, indicating whether the data object should be included in the
# # #             final dataset. (default: :obj:`None`)
# # #     """
# # #
# # #     url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
# # #     split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
# # #                  'benchmarking-gnns/master/data/molecules/{}.index')
# # #
# # #     def __init__(self, root, subset=False, split='train', transform=None,
# # #                  pre_transform=None, pre_filter=None):
# # #         self.subset = subset
# # #         assert split in ['train', 'val', 'test']
# # #         super(ZINC, self).__init__(root, transform, pre_transform, pre_filter)
# # #         path = osp.join(self.processed_dir, f'{split}.pt')
# # #         self.data, self.slices = torch.load(path)
# # #
# # #     @property
# # #     def raw_file_names(self):
# # #         return [
# # #             'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
# # #             'val.index', 'test.index'
# # #         ]
# # #
# # #     @property
# # #     def processed_dir(self):
# # #         name = 'subset' if self.subset else 'full'
# # #         return osp.join(self.root, name, 'processed')
# # #
# # #     @property
# # #     def processed_file_names(self):
# # #         return ['train.pt', 'val.pt', 'test.pt']
# # #
# # #     def download(self):
# # #         shutil.rmtree(self.raw_dir)
# # #         path = download_url(self.url, self.root)
# # #         extract_zip(path, self.root)
# # #         os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
# # #         os.unlink(path)
# # #
# # #         for split in ['train', 'val', 'test']:
# # #             download_url(self.split_url.format(split), self.raw_dir)
# # #
# # #     def process(self):
# # #         for split in ['train', 'val', 'test']:
# # #             with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
# # #                 mols = pickle.load(f)
# # #
# # #             indices = range(len(mols))
# # #
# # #             if self.subset:
# # #                 with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
# # #                     indices = [int(x) for x in f.read()[:-1].split(',')]
# # #
# # #             pbar = tqdm(total=len(indices))
# # #             pbar.set_description(f'Processing {split} dataset')
# # #
# # #             data_list = []
# # #             for idx in indices:
# # #                 mol = mols[idx]
# # #
# # #                 x = mol['atom_type'].to(torch.long).view(-1, 1)
# # #                 y = mol['logP_SA_cycle_normalized'].to(torch.float)
# # #
# # #                 adj = mol['bond_type']
# # #                 edge_index = adj.nonzero().t().contiguous()
# # #                 edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)
# # #
# # #                 data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
# # #                             y=y)
# # #
# # #                 if self.pre_filter is not None and not self.pre_filter(data):
# # #                     continue
# # #
# # #                 if self.pre_transform is not None:
# # #                     data = self.pre_transform(data)
# # #
# # #                 data_list.append(data)
# # #                 pbar.update(1)
# # #
# # #             pbar.close()
# # #
# # #             torch.save(self.collate(data_list),
# # #                        osp.join(self.processed_dir, f'{split}.pt'))
# # #
# # #
# # # x = ZINC(get_temp_path())
# # # print(x)
# # '''
# #
# # import pickle
# # import networkx as nx
# # import numpy as np
# # import ast, csv
# # from time import time
# # import matplotlib.pyplot as plt
# # from os.path import join, basename, exists
# # import os, sys, datetime
# # from glob import glob
# # # from pdb_format import parseLoc
# #
# # path = os.path.join(os.path.dirname(__file__), os.pardir)
# # sys.path.append(path)
# #
# # from utils import get_data_path, create_dir_if_not_exists
# # import argparse
# # import traceback
# # from tqdm import tqdm
# #
# # protein_dir = join('/home/yba/Documents/GraphMatching/data/PPI_Datasets/protein_identifier/protein_identifier.pickle')
# # with open(protein_dir, "rb") as f:
# #     protein_identifer = pickle.load(f)
# #
# # print(type(protein_dir))
# # print(protein_dir)
# #
# #
# # #
# # #
# # import matplotlib.pyplot as plt
# # from collections import Counter
# #
# # # Create your list
# # # BA
# # # x_li1 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 4, 1, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 1, 2, 3, 3, 3, 6, 2, 6, 2, 3, 3, 4, 0, 3, 0, 2, 0, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 4, 1, 1, 1, 1, 2, 2, 2, 4, 1, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 1, 2, 3, 3, 3, 6, 2, 6, 2, 3, 3, 4, 0, 3, 0, 2, 0, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 0, 3, 0, 2, 1, 0, 0, 1, 1, 0, 0, 0, 2, 4, 0, 3, 0, 2, 1, 0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 4, 0, 3, 0, 2, 1, 0, 0, 1, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 4, 1, 2, 2, 2, 2, 2, 2, 4, 1, 4, 1, 2, 2, 2, 1, 2, 9, 4, 1, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 6, 1, 2, 3, 3, 6, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 5, 2, 3, 0, 2, 0, 1, 1, 2, 2, 4, 1, 4, 1, 2, 2, 2, 1, 2, 9, 4, 1, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 4, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 394, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 2, 2, 2, 1, 4, 1, 2, 6, 1, 1, 2, 1, 1, 6, 1, 1, 2, 4, 1, 9, 4, 1, 3, 1, 3, 1, 4, 1, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 3, 1, 3, 0, 2, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 6, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 393, 1, 1, 1, 2, 2, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # # x_li2 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 4, 1, 2, 1, 6, 2, 3, 15, 1, 6, 2, 1, 5, 10, 4, 0, 3, 0, 2, 0, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 0, 3, 0, 2, 0, 1, 0, 0, 2, 4, 0, 9, 4, 0, 3, 0, 2, 0, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 1, 0, 0, 2, 0, 1, 0, 0, 1, 3, 1, 0, 0, 2, 0, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 3, 3, 6, 2, 3, 4, 0, 3, 0, 2, 0, 1, 0, 0, 2, 4, 0, 3, 0, 2, 0, 1, 0, 0, 1, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 4, 1, 1, 3, 3, 0, 2, 0, 1, 0, 0, 2, 1, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 4, 1, 2, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 394, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 393, 1, 1, 2, 1, 1, 1, 4, 1, 2, 2, 2, 1, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 2, 2, 0, 1, 0, 0, 1, 2, 0, 3, 1, 1, 1, 1, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 392, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # # x_li3 =[400, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 4, 1, 2, 4, 1, 2, 1, 2, 4, 1, 1, 2, 3, 1, 6, 2, 0, 1, 0, 0, 5, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 2, 4, 1, 1, 2, 2, 2, 4, 1, 2, 2, 4, 1, 2, 3, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 1, 1, 2, 2, 2, 2, 4, 1, 2, 4, 1, 2, 1, 2, 4, 1, 1, 2, 3, 1, 6, 2, 0, 1, 0, 0, 5, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 2, 0, 1, 0, 3, 1, 2, 2, 4, 1, 2, 3, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 397, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 4, 1, 2, 2, 2, 2, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 2, 2, 0, 1, 0, 0, 1, 2, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 0, 1, 0, 0, 5, 2, 0, 1, 0, 0, 4, 1, 0, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0, 1, 6, 2, 0, 1, 0, 0, 5, 2, 0, 1, 395, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 4, 1, 1, 1, 1, 4, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 4, 1, 2, 2, 2, 2, 2, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 1, 3, 3, 3, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 1, 2, 1, 2, 2, 2, 2, 3, 1, 2, 2, 3, 3, 6, 2, 4, 0, 3, 0, 2, 0, 1, 0, 0, 1, 4, 0, 3, 0, 2, 0, 1, 0, 0, 4, 0, 3, 0]
# # # x_li4 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 2, 1, 4, 1, 1, 1, 2, 2, 4, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 4, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 6, 2, 0, 1, 0, 0, 5, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 4, 1, 4, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 4, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 3, 1, 4, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 4, 1, 0, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 1, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1, 0, 0, 0, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 1, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 4, 1, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 8, 3, 0, 2, 0, 1, 0, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 1, 2, 2, 2, 1, 2, 2, 2, 2, 4, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 3, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 2, 4, 1, 1, 2, 1, 2, 2, 1, 4, 1, 2, 2, 2, 2, 2, 1, 5, 0, 4, 0, 3, 0, 2, 0, 1, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 2, 0, 1, 0, 392, 1, 1, 1, 1, 1, 1]
# # # x_li5 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 4, 1, 2, 2, 4, 1, 4, 1, 1, 2, 2, 2, 2, 2, 2, 6, 2, 6, 2, 3, 0, 2, 0, 1, 0, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 1, 2, 6, 1, 2, 3, 3, 3, 3, 0, 2, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 398, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 4, 1, 1, 2, 4, 1, 1, 4, 1, 2, 2, 2, 4, 1, 4, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 3, 6, 2, 3, 3, 3, 3, 4, 0, 3, 0, 2, 0, 397, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 4, 1, 4, 1, 2, 4, 1, 4, 1, 2, 3, 1, 3, 8, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 6, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 3, 1, 1, 4, 1, 2, 4, 1, 4, 1, 2, 3, 1, 3, 8, 1, 2, 0, 1, 1, 0, 0, 0, 2, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 4, 1, 1, 2, 3, 3, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 1, 2, 2, 2, 2, 2, 3, 6, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 394, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 3, 3, 3, 8, 3, 0, 2, 0, 1, 0, 0, 7, 3, 0, 2, 0, 1, 0, 0, 6, 3, 0, 2, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 4, 1, 1, 2, 3, 3, 3, 3, 0, 2, 0, 1, 0, 0, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 4, 1, 2, 2, 2, 2, 4, 1, 2]
# #
# # # #ER
# # x_li1 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 0, 1, 0, 0, 2, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 392, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 1, 2, 391, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 390, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 389, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 388, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 387, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 386, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 385, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # x_li2 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 1, 0, 0, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 392, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 391, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 1, 0, 390, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 389, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 388, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 387, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 386, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 385, 1, 1, 1, 1, 1, 1, 1]
# # x_li3 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 2, 0, 1, 1, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 2, 0, 1, 0, 0, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 2, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 0, 0, 2, 1, 0, 0, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 392, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 391, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 390, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 389, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 2, 2, 0, 1, 388, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 2, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 2, 0, 2, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 1, 0, 0, 387, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2, 0, 1, 0, 386, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # x_li4 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 392, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 391, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 390, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 389, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 388, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 387, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 0, 0, 0, 0, 1, 2, 386, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0]
# # x_li5 =[400, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 399, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0, 0, 3, 1, 0, 0, 398, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 397, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 396, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 395, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 394, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 393, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 392, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 391, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 0, 0, 0, 2, 390, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 2, 0, 1, 0, 389, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 388, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2, 387, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 386, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 385, 1, 1, 1, 1, 1]
# #
# # #WS
# # # x_li1 =[400, 0, 399, 0, 398, 0, 397, 0, 396, 1, 0, 0, 395, 1, 0, 394, 0, 393, 0, 392, 0, 391, 390, 0, 389, 0, 388, 0, 387, 0, 0, 386, 0, 385, 3, 2, 2, 0, 1, 0, 0, 1, 1, 0, 0, 2, 0, 1, 384, 2, 1, 0, 0, 1, 1, 0, 383, 4, 0, 3, 0, 2, 0, 1, 382, 0, 381, 1, 1, 2, 2, 0, 1, 0, 0, 1, 380, 0, 379, 0, 378, 0, 377, 1, 2, 0, 376, 0, 375, 0, 374, 0, 373, 0, 372, 0, 371, 0, 370, 1, 0, 0, 369, 0, 368, 0, 367, 0, 366, 1, 1, 0, 0, 0, 365, 0, 364, 0, 363, 2, 0, 1, 0, 0, 362, 1, 361, 0, 360, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0, 0, 359, 2, 0, 1, 358, 0, 357, 0, 356, 0, 355, 0, 354, 0, 353, 0, 352, 0, 351, 1, 1, 0, 0, 0, 350, 1, 0, 0, 349, 0, 348, 0, 347, 0, 346, 1, 1, 2, 2, 0, 1, 0, 0, 1, 345, 0, 344, 0, 343, 4, 0, 3, 0, 2, 1, 0, 0, 0, 342, 0, 341, 0, 340, 339, 0, 338, 0, 337, 0, 336, 1, 0, 1, 335, 1, 1, 0, 0, 1, 0, 1, 0, 0, 334, 2, 0, 1, 0, 0, 333, 0, 332, 0, 0, 331, 0, 330, 1, 0, 0, 329, 1, 328, 1, 2, 0, 1, 0, 0, 2, 0, 1, 327, 1, 1, 1, 0, 0, 1, 0, 0, 1, 326, 0, 325, 0, 324, 0, 323, 1, 0, 0, 322, 0, 321, 0, 320, 1, 0, 0, 319, 0, 318, 0, 317, 0, 316, 0, 315, 2, 0, 1, 314, 0, 313, 0, 312, 0, 311, 0, 310, 0, 309, 0, 308, 2, 0, 1, 2, 0, 1, 0, 307, 2, 0, 1, 0, 0, 306, 0, 305, 0, 304, 0, 303, 1, 1, 2, 1, 0, 0, 1, 302, 0, 301, 0, 300, 2, 0, 1, 0, 0, 299, 4, 1, 3, 0, 2, 0, 1, 0, 0, 3, 1, 3, 0, 2, 0, 1, 0, 0, 298, 1, 1, 0, 0, 0, 297, 2, 0, 1, 296, 1, 0, 0, 295, 0, 294, 0, 293, 0, 292, 1, 0, 0, 291, 2, 0, 1, 0, 0, 290, 0, 289, 1, 0, 0, 288, 0, 287, 0, 286, 1, 1, 0, 0, 2, 0, 1, 0, 0, 285, 0, 284, 0, 283, 0, 282, 0, 281, 0, 1, 0, 0, 280, 0, 279, 0, 278, 277, 0, 276, 0, 275, 0, 274, 0, 273, 0, 272, 0, 271, 1, 0, 0, 270, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 269, 0, 268, 0, 267, 2, 0, 1, 0, 0, 266, 1, 0, 0, 265, 0, 264, 0, 263, 1, 262, 0, 261, 0, 260, 3, 0, 2, 1, 0, 259, 0, 258, 0, 257, 1, 0, 0, 256, 0, 255, 0, 254, 0, 253, 1, 0, 0, 252, 1, 251, 0, 250, 1, 0, 0, 249, 0, 248, 0, 247, 0, 246, 1, 2, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 0, 245, 244, 0, 243, 0, 242, 0, 241, 0, 240, 0, 239, 0, 238, 0, 237, 0, 236, 0, 235, 0, 234, 0, 233, 0, 232, 4, 0, 3, 0, 2, 1, 0, 0, 0, 231, 0, 230, 0, 229, 228, 1, 0, 0, 227, 0, 226, 0, 225, 0, 0, 224, 2, 0, 1, 0, 0, 223, 0, 0, 222, 1, 1, 0, 0, 0, 221, 2, 220, 1, 0, 0, 219, 0, 218, 1, 1, 0, 217, 2, 0, 1, 1, 0, 0, 0, 216, 1, 215, 0, 214, 0, 213, 0, 212, 6, 1, 0, 5, 1, 0, 0, 4, 0, 3, 0, 2, 1, 0, 0, 0, 0, 211, 1, 0, 0, 210, 0, 209, 0, 208, 0, 207, 0, 206, 0, 205, 0, 204, 3, 0, 2, 0, 1, 0, 0, 203, 0, 202, 0, 201, 3, 2, 0, 1, 0, 200, 0, 199, 0, 198, 3, 0, 2, 0, 1, 197, 1, 1, 1, 0, 0, 0, 0, 196, 0, 195, 0, 194, 1, 0, 0, 193, 0, 192, 1, 191, 1, 0, 0, 190, 1, 0, 0, 189, 1, 188, 0, 187, 0, 186, 3, 0, 2, 0, 1, 185, 0, 184, 1, 1, 1, 0, 0, 1, 0, 183, 0, 182, 0, 181, 0, 180, 0, 179, 0, 178, 0, 177, 3, 1, 0, 0, 2, 1, 0, 176, 0, 175, 0, 174, 0, 173, 0, 172, 1, 171, 1, 0, 0, 170, 0, 169, 0, 168, 0, 167, 0, 166, 0, 165, 0, 164, 0, 163, 0, 162, 3, 1, 0, 0, 2, 1, 0, 0, 1, 161, 0, 160, 1, 2, 1, 0, 0, 1, 1, 159, 6, 1, 0, 0, 5, 1, 0, 0, 4, 158, 1, 0, 0, 157, 0, 156, 0, 155, 1, 154, 0, 153, 0, 152, 2, 0, 1, 0, 0, 151, 4, 0, 3, 0, 2, 0, 1, 0, 0, 150, 1, 0, 0, 149, 2, 1, 0, 0, 1, 148, 1, 0, 0, 147, 0, 146, 4, 1, 0, 145, 0, 144, 0, 143, 0, 142, 0, 141, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0, 140, 0, 139, 3, 1, 0, 0, 2, 1, 0, 138, 1, 0, 0, 137, 0, 136, 1, 4, 0, 135, 1, 1, 0, 1, 0, 0, 1, 2, 1, 134, 2, 2, 0, 1, 0, 0, 1, 0, 0, 133, 0, 132, 1, 1, 0, 0, 0, 131, 2, 1, 1, 1, 0, 0, 0, 0, 0, 130, 129, 1, 0, 0, 128, 1, 2, 0, 1, 0, 127, 1, 0, 0, 126, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 125, 0, 124, 0, 1, 0, 0, 123, 0, 122, 0, 121, 0, 3, 0, 2, 0, 1, 0, 0, 120, 119, 0, 118, 3, 1, 0]
# # # x_li2 =[400, 0, 399, 0, 398, 0, 397, 0, 396, 0, 395, 0, 394, 0, 393, 0, 392, 0, 391, 0, 390, 0, 389, 0, 388, 0, 387, 0, 386, 0, 385, 0, 384, 0, 383, 0, 382, 0, 381, 0, 380, 1, 379, 1, 0, 0, 378, 1, 0, 377, 0, 376, 0, 375, 0, 374, 373, 1, 1, 1, 0, 0, 0, 0, 372, 371, 0, 370, 0, 369, 1, 2, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 368, 0, 367, 0, 366, 365, 0, 364, 0, 363, 0, 362, 0, 361, 0, 0, 360, 0, 359, 1, 0, 0, 358, 0, 357, 0, 356, 0, 355, 0, 354, 353, 6, 0, 5, 1, 1, 0, 0, 0, 352, 6, 1, 0, 0, 5, 1, 1, 0, 351, 6, 1, 1, 0, 0, 0, 5, 1, 350, 1, 6, 1, 1, 0, 0, 0, 5, 0, 349, 3, 1, 1, 0, 0, 0, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 348, 3, 1, 0, 0, 2, 1, 0, 0, 1, 1, 0, 0, 347, 3, 1, 0, 0, 2, 1, 0, 0, 1, 1, 0, 0, 346, 6, 0, 5, 0, 4, 1, 0, 0, 3, 1, 0, 0, 345, 0, 344, 3, 0, 2, 1, 0, 0, 1, 0, 0, 343, 342, 0, 341, 0, 340, 0, 339, 0, 338, 0, 337, 0, 336, 3, 0, 2, 0, 1, 0, 0, 0, 335, 0, 334, 3, 333, 6, 1, 1, 0, 0, 0, 5, 1, 1, 0, 0, 0, 332, 6, 1, 1, 0, 0, 0, 5, 1, 1, 0, 0, 0, 331, 6, 1, 1, 0, 0, 0, 5, 1, 1, 0, 0, 0, 330, 6, 1, 0, 0, 5, 1, 1, 0, 0, 0, 4, 1, 329, 3, 0, 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, 3, 1, 1, 0, 0, 0, 2, 1, 0, 0, 1, 0, 328, 3, 0, 2, 1, 0, 0, 1, 1, 0, 0, 0, 327, 326, 6, 1, 0, 0, 5, 1, 0, 0, 4, 1, 0, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0, 325, 0, 324, 323, 3, 0, 2, 0, 1, 0, 0, 322, 0, 321, 0, 320, 319, 0, 318, 0, 317, 0, 316, 0, 315, 0, 314, 1, 1, 313, 2, 0, 1, 1, 2, 1, 1, 1, 0, 0, 0, 1, 312, 2, 1, 0, 0, 1, 1, 0, 0, 0, 311, 2, 1, 310, 2, 1, 0, 0, 1, 0, 0, 309, 1, 1, 2, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 308, 1, 1, 0, 0, 0, 307, 1, 1, 0, 0, 0, 306, 3, 0, 2, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 0, 0, 0, 305, 0, 304, 1, 0, 0, 303, 1, 3, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 302, 0, 301, 0, 300, 0, 299, 0, 298, 0, 297, 0, 296, 295, 0, 294, 1, 0, 0, 293, 2, 0, 1, 0, 0, 292, 291, 2, 0, 1, 0, 0, 290, 2, 2, 1, 1, 1, 0, 289, 1, 0, 0, 288, 1, 0, 0, 287, 1, 0, 0, 286, 2, 0, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 285, 0, 284, 1, 283, 1, 0, 1, 0, 0, 282, 0, 281, 0, 280, 0, 279, 278, 0, 277, 0, 276, 2, 1, 1, 1, 0, 0, 0, 1, 275, 0, 274, 0, 273, 0, 272, 0, 271, 2, 1, 1, 1, 270, 0, 269, 0, 268, 0, 267, 0, 266, 0, 265, 2, 0, 264, 0, 263, 1, 0, 0, 262, 0, 261, 0, 260, 0, 259, 258, 0, 257, 0, 256, 0, 255, 0, 254, 1, 1, 0, 0, 253, 2, 0, 1, 0, 0, 252, 2, 0, 1, 0, 0, 251, 250, 2, 0, 1, 0, 0, 249, 1, 0, 0, 248, 1, 0, 247, 1, 0, 0, 246, 2, 0, 1, 0, 0, 245, 1, 0, 2, 0, 1, 0, 0, 0, 244, 1, 0, 0, 243, 1, 242, 0, 241, 0, 240, 1, 1, 0, 0, 1, 0, 0, 239, 238, 1, 0, 0, 237, 0, 236, 0, 235, 0, 234, 1, 0, 233, 1, 0, 0, 232, 0, 231, 0, 230, 0, 229, 1, 1, 228, 0, 227, 0, 226, 0, 225, 0, 224, 0, 223, 0, 222, 221, 0, 220, 2, 0, 1, 0, 0, 219, 0, 218, 2, 0, 217, 0, 216, 0, 215, 0, 214, 1, 1, 0, 0, 0, 213, 212, 2, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 211, 2, 1, 0, 0, 1, 1, 0, 0, 0, 210, 2, 1, 209, 1, 1, 0, 0, 0, 208, 1, 1, 0, 0, 0, 207, 2, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 206, 2, 0, 1, 0, 0, 205, 204, 1, 0, 0, 203, 1, 1, 0, 0, 1, 0, 0, 202, 0, 0, 201, 0, 200, 1, 0, 0, 199, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 198, 1, 0, 0, 197, 0, 196, 0, 195, 0, 194, 1, 0, 193, 1, 1, 0, 0, 2, 0, 1, 0, 0, 192, 2, 0, 191, 2, 0, 1, 1, 1, 0, 0, 0, 0, 190, 2, 0, 189, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 188, 1, 0, 0, 187, 1, 1, 0, 0, 0, 0, 186, 2, 0, 1, 0, 0, 185, 184, 1, 0, 0, 183, 1, 0, 0, 182, 0, 181, 0, 180, 179, 1, 0, 0, 178, 2, 0, 1, 0, 0, 177]
# # # x_li3 =[400, 0, 399, 0, 398, 0, 397, 0, 396, 0, 395, 0, 394, 0, 393, 0, 392, 0, 391, 0, 390, 0, 389, 0, 388, 0, 387, 1, 0, 0, 386, 0, 385, 384, 0, 383, 0, 382, 0, 381, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 380, 379, 1, 0, 0, 378, 3, 0, 2, 0, 377, 1, 0, 0, 376, 0, 375, 2, 0, 1, 0, 0, 1, 0, 0, 374, 2, 1, 0, 0, 0, 373, 0, 372, 1, 371, 1, 0, 0, 370, 1, 0, 1, 0, 1, 0, 0, 0, 369, 2, 1, 0, 1, 0, 0, 0, 368, 2, 0, 1, 367, 1, 0, 0, 366, 2, 0, 1, 0, 365, 2, 0, 1, 0, 0, 364, 1, 0, 363, 2, 0, 1, 0, 0, 362, 2, 1, 361, 0, 360, 0, 359, 0, 358, 0, 357, 356, 0, 355, 0, 354, 0, 353, 0, 352, 351, 0, 350, 1, 0, 0, 349, 0, 348, 347, 1, 0, 0, 346, 0, 345, 1, 0, 344, 0, 343, 0, 342, 0, 341, 1, 0, 0, 0, 340, 0, 339, 0, 338, 0, 337, 0, 336, 1, 2, 0, 1, 1, 0, 335, 0, 334, 0, 333, 0, 332, 0, 331, 0, 0, 330, 0, 329, 0, 328, 1, 327, 0, 326, 0, 325, 0, 324, 0, 323, 322, 0, 321, 0, 320, 0, 319, 0, 318, 317, 1, 0, 0, 316, 0, 315, 0, 314, 0, 1, 0, 0, 313, 1, 0, 0, 1, 0, 0, 312, 0, 311, 2, 0, 1, 0, 0, 310, 0, 309, 1, 0, 1, 0, 0, 0, 308, 0, 307, 1, 306, 0, 305, 1, 0, 0, 304, 1, 0, 303, 2, 0, 1, 0, 0, 302, 1, 0, 301, 1, 1, 0, 0, 0, 300, 1, 0, 299, 1, 0, 0, 298, 3, 0, 2, 0, 297, 1, 2, 0, 1, 0, 0, 0, 296, 295, 2, 1, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 294, 2, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 293, 0, 292, 291, 1, 1, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 1, 2, 0, 1, 0, 0, 0, 0, 290, 1, 0, 0, 1, 1, 0, 0, 0, 289, 1, 2, 1, 0, 0, 1, 1, 0, 288, 2, 1, 0, 0, 1, 1, 0, 0, 287, 1, 1, 0, 0, 0, 286, 2, 0, 285, 2, 1, 0, 0, 1, 0, 0, 284, 283, 2, 1, 2, 0, 1, 0, 0, 2, 282, 1, 1, 1, 0, 0, 0, 2, 1, 281, 0, 280, 1, 0, 0, 279, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 278, 3, 2, 0, 1, 0, 0, 2, 0, 277, 1, 0, 0, 276, 0, 275, 2, 0, 1, 0, 0, 1, 0, 0, 274, 2, 1, 0, 0, 0, 273, 1, 1, 0, 1, 0, 0, 0, 0, 272, 2, 271, 1, 0, 0, 270, 1, 0, 0, 269, 1, 0, 0, 2, 0, 1, 0, 0, 0, 268, 2, 0, 1, 0, 0, 267, 1, 0, 0, 266, 2, 0, 1, 0, 265, 2, 0, 1, 0, 0, 264, 1, 0, 263, 2, 0, 1, 0, 0, 262, 2, 0, 261, 0, 260, 2, 0, 1, 0, 0, 259, 1, 0, 0, 1, 0, 0, 258, 0, 257, 2, 0, 1, 0, 0, 256, 0, 255, 254, 0, 253, 1, 0, 0, 252, 0, 251, 250, 0, 249, 0, 248, 0, 247, 0, 246, 245, 0, 244, 0, 243, 0, 242, 0, 241, 0, 0, 240, 1, 1, 1, 0, 0, 239, 1, 0, 0, 238, 3, 0, 2, 0, 237, 1, 0, 0, 236, 0, 235, 2, 0, 0, 1, 0, 0, 234, 2, 0, 1, 233, 0, 232, 2, 0, 1, 0, 0, 231, 230, 1, 0, 0, 229, 2, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 228, 2, 0, 1, 0, 0, 227, 1, 1, 226, 2, 0, 1, 0, 0, 225, 2, 1, 224, 1, 0, 0, 223, 2, 0, 1, 0, 222, 2, 1, 1, 0, 0, 0, 1, 0, 221, 0, 220, 0, 219, 1, 1, 0, 0, 1, 0, 0, 0, 218, 0, 217, 0, 216, 1, 0, 0, 215, 0, 214, 0, 213, 1, 0, 0, 0, 212, 1, 0, 0, 211, 1, 0, 0, 210, 1, 0, 0, 209, 208, 1, 1, 0, 0, 0, 207, 0, 206, 205, 0, 204, 0, 203, 0, 202, 0, 201, 1, 2, 0, 1, 0, 0, 0, 0, 0, 200, 0, 199, 0, 198, 0, 197, 196, 0, 195, 0, 194, 0, 193, 1, 0, 0, 0, 192, 0, 191, 0, 190, 0, 0, 189, 0, 188, 1, 0, 0, 187, 186, 0, 185, 0, 184, 0, 183, 0, 182, 181, 0, 180, 1, 0, 0, 179, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 178, 3, 2, 177, 1, 0, 0, 176, 0, 175, 2, 0, 1, 0, 0, 1, 0, 0, 174, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 173, 1, 1, 0, 1, 0, 0, 1, 0, 172, 2, 0, 1, 0, 0, 171, 1, 0, 170, 1, 0, 0, 169, 2, 0, 1, 0, 168, 1, 2, 0, 1, 0, 0, 2, 0, 167, 1, 0, 0, 166, 2, 0, 1, 0, 165, 2, 0, 1, 0, 0, 164, 1, 1, 163, 2, 0, 1, 0, 0, 162, 2, 0, 161, 0, 160, 3, 1, 0, 0, 2, 2, 159, 3, 0, 2, 0, 1, 0, 0, 158, 157, 3, 0, 2, 0, 1, 0, 0, 156, 9, 4, 1, 0, 0, 3, 1, 0, 155, 6, 0, 5, 0, 4, 1, 0, 0, 0, 3, 1, 0, 0, 2, 0, 1, 8, 2, 0]
# # # x_li4 =[400, 1, 0, 0, 399, 0, 398, 397, 1, 0, 0, 396, 1, 0, 395, 0, 394, 0, 393, 1, 0, 0, 0, 392, 0, 391, 1, 390, 2, 0, 1, 0, 0, 389, 388, 0, 387, 2, 0, 1, 0, 386, 1, 1, 0, 0, 1, 0, 0, 385, 1, 0, 0, 384, 0, 383, 1, 382, 1, 0, 2, 0, 1, 0, 0, 381, 1, 0, 0, 380, 1, 0, 0, 379, 2, 0, 1, 0, 0, 378, 1, 377, 1, 0, 0, 376, 0, 375, 1, 374, 0, 373, 2, 0, 1, 0, 0, 372, 2, 0, 1, 0, 0, 371, 1, 370, 1, 0, 0, 369, 0, 368, 0, 367, 1, 0, 0, 366, 0, 365, 0, 364, 0, 363, 0, 362, 1, 0, 0, 361, 0, 360, 1, 1, 0, 0, 1, 359, 1, 0, 0, 358, 0, 357, 1, 356, 3, 1, 1, 0, 0, 0, 2, 1, 355, 0, 354, 0, 353, 0, 352, 0, 351, 350, 1, 0, 0, 349, 0, 348, 0, 347, 346, 0, 345, 0, 344, 9, 1, 1, 2, 343, 3, 2, 1, 0, 0, 1, 0, 0, 342, 1, 0, 0, 341, 9, 1, 2, 1, 8, 1, 0, 0, 7, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 1, 3, 1, 2, 0, 1, 0, 0, 0, 2, 8, 1, 0, 0, 7, 1, 2, 1, 1, 340, 3, 1, 1, 1, 0, 0, 0, 0, 2, 339, 2, 1, 0, 0, 1, 0, 0, 338, 1, 337, 0, 336, 0, 335, 1, 1, 0, 0, 0, 334, 0, 333, 0, 332, 0, 331, 0, 330, 1, 329, 1, 1, 1, 1, 0, 0, 0, 0, 0, 328, 0, 327, 1, 1, 0, 0, 0, 326, 0, 325, 0, 324, 2, 0, 1, 0, 0, 323, 0, 322, 0, 321, 1, 1, 1, 0, 0, 0, 0, 320, 0, 319, 0, 318, 1, 1, 0, 0, 1, 317, 1, 0, 0, 316, 1, 1, 0, 2, 0, 315, 0, 314, 1, 0, 0, 313, 0, 312, 1, 311, 0, 310, 0, 309, 0, 308, 1, 0, 0, 307, 0, 306, 1, 2, 6, 0, 5, 0, 4, 305, 0, 304, 0, 303, 1, 1, 0, 0, 1, 302, 1, 0, 0, 301, 1, 0, 0, 300, 0, 299, 0, 298, 1, 3, 2, 0, 1, 0, 0, 297, 1, 1, 0, 0, 1, 0, 0, 296, 1, 295, 0, 294, 1, 0, 0, 293, 0, 292, 0, 3, 1, 0, 1, 0, 0, 2, 291, 1, 0, 0, 290, 2, 0, 1, 0, 0, 289, 0, 288, 0, 287, 1, 1, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 286, 285, 2, 0, 1, 0, 0, 284, 3, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 1, 283, 1, 1, 2, 1, 0, 0, 1, 0, 0, 282, 1, 0, 0, 281, 3, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 280, 1, 279, 0, 278, 1, 0, 0, 277, 1, 0, 0, 276, 1, 0, 0, 275, 0, 274, 1, 0, 0, 273, 2, 1, 1, 0, 0, 0, 1, 0, 0, 272, 0, 271, 0, 270, 0, 269, 1, 0, 2, 268, 2, 0, 1, 0, 0, 267, 2, 0, 1, 266, 0, 265, 2, 0, 1, 0, 0, 264, 0, 263, 0, 262, 0, 261, 1, 2, 0, 1, 0, 260, 1, 1, 0, 0, 1, 0, 0, 259, 1, 258, 1, 1, 0, 0, 1, 0, 0, 257, 0, 256, 1, 0, 0, 255, 1, 1, 0, 0, 1, 254, 0, 253, 2, 0, 1, 0, 0, 252, 0, 251, 1, 1, 0, 0, 1, 0, 0, 250, 1, 249, 1, 1, 0, 0, 0, 248, 1, 0, 0, 247, 0, 246, 0, 245, 1, 1, 0, 0, 1, 244, 0, 243, 1, 0, 0, 242, 1, 0, 0, 241, 0, 240, 0, 239, 0, 238, 1, 0, 0, 237, 1, 1, 1, 1, 0, 0, 1, 0, 0, 236, 1, 3, 3, 1, 2, 0, 1, 0, 0, 0, 235, 0, 234, 1, 0, 0, 233, 0, 232, 0, 231, 2, 3, 1, 2, 0, 1, 0, 0, 0, 230, 0, 229, 1, 1, 2, 0, 1, 0, 0, 0, 228, 0, 227, 0, 226, 0, 225, 2, 0, 1, 0, 224, 0, 223, 1, 1, 1, 1, 0, 0, 0, 1, 222, 1, 1, 0, 0, 0, 221, 3, 0, 2, 0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 1, 3, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 3, 1, 2, 0, 1, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 1, 0, 0, 220, 219, 0, 218, 0, 217, 0, 216, 0, 215, 0, 214, 213, 0, 212, 2, 0, 1, 0, 0, 211, 0, 210, 209, 0, 208, 0, 207, 1, 0, 0, 206, 1, 0, 205, 0, 204, 0, 203, 0, 202, 0, 201, 0, 200, 199, 3, 0, 2, 0, 1, 0, 0, 198, 0, 197, 196, 0, 195, 0, 194, 0, 193, 1, 0, 0, 192, 1, 0, 0, 0, 191, 0, 190, 0, 189, 0, 0, 188, 0, 187, 0, 186, 0, 185, 2, 0, 184, 0, 183, 0, 182, 0, 181, 0, 180, 0, 179, 178, 3, 0, 2, 0, 1, 0, 0, 177, 0, 176, 175, 2, 6, 2, 1, 4, 1, 1, 4, 1, 0, 0, 3, 1, 0, 174, 0, 173, 0, 172, 0, 171, 1, 0, 0, 170, 0, 169, 3, 0, 0, 2, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 168, 0, 0, 167, 0, 166, 0, 165, 1, 0, 0, 164, 0, 163, 0, 162, 161, 0, 160, 0, 159, 0, 158, 1, 0, 0, 157, 1, 1, 0, 0, 156, 0, 155, 0]
# # # x_li5 =[400, 2, 0, 1, 0, 0, 399, 398, 0, 397, 0, 396, 1, 0, 395, 0, 394, 0, 393, 0, 392, 391, 0, 390, 0, 389, 0, 388, 387, 0, 386, 1, 1, 0, 0, 0, 385, 0, 1, 0, 0, 384, 0, 383, 0, 0, 382, 0, 381, 2, 0, 1, 0, 0, 0, 380, 0, 379, 3, 378, 3, 4, 1, 1, 0, 0, 1, 0, 0, 377, 0, 376, 3, 0, 2, 0, 1, 0, 0, 3, 1, 1, 0, 0, 1, 0, 0, 375, 1, 3, 3, 0, 2, 0, 1, 0, 0, 374, 0, 373, 0, 372, 0, 371, 1, 2, 0, 370, 1, 2, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 369, 0, 368, 1, 0, 0, 367, 0, 366, 0, 365, 364, 3, 2, 1, 0, 0, 1, 1, 0, 0, 0, 363, 3, 0, 2, 0, 1, 0, 0, 362, 3, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 361, 0, 360, 0, 359, 3, 0, 2, 1, 1, 1, 358, 2, 1, 0, 1, 0, 0, 1, 1, 2, 0, 357, 8, 1, 0, 0, 7, 2, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 6, 356, 0, 355, 0, 354, 0, 353, 0, 352, 0, 351, 5, 1, 0, 0, 4, 0, 3, 0, 2, 0, 350, 1, 0, 0, 349, 1, 0, 0, 348, 2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 347, 0, 346, 0, 345, 0, 344, 4, 2, 0, 1, 2, 0, 1, 0, 0, 0, 0, 3, 1, 1, 0, 0, 0, 0, 2, 2, 0, 1, 0, 343, 6, 1, 1, 0, 0, 1, 0, 0, 5, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 4, 1, 0, 0, 3, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 342, 4, 3, 1, 1, 0, 0, 1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 341, 0, 340, 0, 339, 0, 338, 1, 0, 0, 337, 336, 1, 0, 4, 2, 0, 1, 0, 0, 3, 1, 335, 1, 2, 0, 1, 0, 0, 0, 334, 1, 1, 333, 2, 0, 1, 2, 1, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 332, 0, 331, 1, 330, 1, 1, 0, 0, 1, 0, 0, 329, 1, 4, 328, 0, 327, 0, 326, 1, 0, 0, 325, 1, 0, 3, 1, 1, 2, 0, 1, 0, 0, 2, 1, 1, 2, 0, 1, 0, 0, 2, 324, 1, 0, 0, 323, 1, 0, 0, 322, 1, 1, 321, 0, 320, 0, 319, 0, 318, 0, 317, 0, 316, 315, 0, 314, 0, 313, 0, 312, 1, 0, 0, 311, 310, 0, 309, 6, 1, 0, 0, 5, 1, 0, 0, 308, 0, 307, 0, 306, 0, 305, 0, 304, 0, 303, 6, 1, 0, 0, 5, 1, 0, 0, 4, 0, 3, 0, 2, 2, 0, 1, 0, 0, 1, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 302, 301, 0, 300, 0, 299, 0, 298, 0, 297, 0, 296, 295, 0, 294, 0, 293, 0, 292, 2, 0, 1, 0, 291, 1, 0, 0, 290, 1, 1, 0, 0, 0, 289, 288, 0, 287, 0, 286, 0, 285, 0, 284, 0, 283, 6, 2, 0, 1, 0, 0, 5, 1, 1, 0, 6, 0, 5, 0, 4, 1, 0, 0, 3, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 4, 1, 0, 0, 3, 1, 0, 282, 0, 281, 0, 280, 0, 279, 0, 278, 1, 1, 277, 0, 276, 1, 0, 0, 275, 1, 0, 0, 274, 273, 2, 1, 1, 2, 2, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 272, 271, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 1, 270, 1, 1, 1, 2, 1, 0, 0, 1, 0, 0, 2, 269, 1, 0, 0, 268, 0, 267, 0, 266, 0, 265, 1, 1, 1, 0, 0, 0, 0, 0, 264, 1, 1, 1, 4, 1, 1, 0, 0, 1, 0, 0, 3, 1, 1, 263, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0, 262, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0, 261, 0, 260, 0, 259, 0, 258, 1, 1, 1, 2, 0, 257, 4, 0, 3, 0, 2, 0, 1, 0, 0, 256, 0, 255, 0, 254, 1, 0, 0, 253, 2, 0, 1, 1, 0, 252, 2, 1, 0, 0, 1, 0, 0, 251, 1, 1, 0, 250, 1, 0, 0, 249, 1, 2, 0, 1, 0, 0, 0, 248, 1, 1, 0, 0, 0, 247, 0, 246, 0, 245, 1, 244, 2, 2, 0, 1, 0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 243, 1, 3, 1, 0, 0, 2, 1, 0, 0, 1, 1, 242, 1, 2, 1, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 241, 1, 0, 0, 0, 240, 0, 239, 0, 238, 0, 237, 236, 0, 235, 0, 234, 1, 0, 0, 233, 2, 0, 1, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 232, 0, 231, 1, 1, 1, 0, 0, 1, 0, 0, 1, 230, 0, 229, 1, 2, 1, 1, 0, 0, 0, 1, 1, 228, 1, 1, 0, 0, 1, 0, 0, 227, 2, 0, 1, 226, 0, 225, 1, 0, 1, 0, 0, 224, 0, 223, 0, 222, 1, 0, 0, 221, 0, 220, 0, 219, 0, 218, 0, 217, 0, 216, 0, 215, 2, 0, 1, 0, 0, 214, 0, 213, 0, 212, 0, 211, 0, 210, 0, 209, 0, 208, 2, 207, 0, 206, 0, 205, 0, 204, 0, 203, 0, 202, 0, 201, 1, 1, 0, 1, 0]
# #
# # #BA2
# # # x_li1 = [1000000, 12978, 204, 12, 2, 1, 1, 6, 1, 1, 2, 9, 1, 1, 4, 1, 3, 3, 2, 1, 1, 2, 2, 2, 6, 2, 1, 2, 4, 1, 1, 1, 2, 2, 6, 2, 9, 4, 1, 1, 3, 1, 1, 1, 3, 3, 3, 9, 1, 1, 1, 1, 4, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 6, 2, 6, 2, 9, 4, 1, 1, 1, 2, 2, 6, 2, 1, 1, 4, 1, 4, 1, 2, 4, 1, 2, 1, 2, 6, 2, 3, 2, 1, 3, 2, 3, 6, 2, 6, 1, 2, 1, 2, 1, 2, 2, 4, 1, 3, 3, 3, 1, 2, 4, 1, 1, 1, 3, 2, 3, 2, 2, 3, 9, 4, 1, 3, 12, 1, 6, 2, 4, 1, 2, 6, 2, 2, 1, 3, 3, 8, 3, 2, 4, 1, 2, 2, 2, 4, 1, 1, 1, 1, 2, 2, 1, 3, 16, 9, 4, 1, 2, 3, 4, 1, 2, 1, 1, 4, 1, 3, 8, 3, 3, 1, 1, 3, 3, 8, 3, 1, 3, 3, 12, 1, 2, 1, 6, 2, 2, 2, 3, 4, 1, 8, 3, 6, 1, 2, 4, 8, 3, 3, 1, 4, 4, 10, 2, 2, 1, 3, 4, 2, 5, 1, 3, 10, 4, 2, 24, 1, 2, 15, 1, 8, 3, 4, 4, 5, 12, 3, 1, 5, 18, 1, 1, 2, 10, 4, 18, 10, 2, 4, 1, 21, 3, 12, 2, 5, 28, 1, 15, 8, 3, 8, 27, 20, 12, 6, 2, 16, 3, 1, 4, 1, 6, 1, 16, 1, 7, 3, 2, 36, 3, 24, 14, 6, 180, 4, 1, 153, 1, 128, 105, 84, 65, 48, 33, 20, 8, 320, 2, 240, 1, 210, 2, 2, 182, 156, 1, 132, 110, 90, 72, 56, 42, 30, 20, 12, 6, 2, 0, 1, 0, 0, 5, 999999, 11214, 204, 12, 1, 1, 1, 1, 1, 1, 2, 6, 1, 1, 2, 3, 1, 6, 1, 1, 2, 4, 1, 1, 1, 2, 2, 1, 1, 1, 4, 1, 1, 2, 6, 2, 6, 2, 2, 3, 1, 6, 2, 2, 2, 4, 1, 4, 1, 2, 6, 2, 6, 2, 6, 2, 4, 1, 2, 3, 6, 1, 1, 2, 2, 2, 4, 1, 4, 1, 3, 2, 6, 2, 2, 1, 3, 6, 2, 2, 3, 3, 4, 1, 3, 6, 2, 3, 6, 2, 8, 3, 1, 2, 3, 16, 9, 4, 1, 1, 6, 2, 3, 3, 4, 4, 2, 1, 8, 1, 4, 1, 1, 3, 6, 2, 2, 1, 3, 4, 4, 6, 2, 1, 4, 8, 2, 2, 3, 4, 1, 8, 3, 6, 2, 1, 8, 3, 6, 2, 4, 12, 6, 2, 3, 4, 8, 1, 3, 2, 4, 5, 4, 1, 2, 5, 15, 3, 8, 4, 1, 3, 8, 3, 6, 2, 3, 8, 3, 8, 3, 4, 4, 1, 8, 3, 1, 3, 8, 3, 10, 1, 2, 1, 12, 6, 2, 4, 1, 4, 1, 25, 16, 9, 4, 1, 2, 6, 2, 1, 4, 1, 4, 3, 5, 6, 6, 7, 6, 2, 4, 1, 12, 6, 2, 4, 6, 54, 10, 4, 4, 40, 6, 2, 28, 2, 18, 10, 4, 18, 8, 18, 1, 8, 70, 2, 54, 40, 28, 18, 10, 4, 224, 2, 4, 1, 195, 1, 2, 2, 4, 2, 1, 154, 130, 99, 80, 3, 63, 48, 35, 24, 15, 8, 3, 225, 196, 169, 132, 110, 90, 1, 72, 56, 42, 1, 30, 20, 12, 6, 2, 54, 34, 1, 16, 0, 15, 0, 14, 0, 13, 0, 12, 0, 11, 0, 10, 0, 9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 12977, 272, 12, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 4, 1, 1, 1, 4, 1, 1, 4, 1, 1, 2, 1, 2, 4, 1, 1, 4, 1, 1, 1, 4, 1, 2, 2, 1, 2, 6, 2, 4, 1, 1, 1, 2, 2, 1, 1, 2, 9, 4, 1, 4, 1, 1, 2, 4, 1, 1, 1, 4, 1, 1, 9, 1, 4, 1, 6, 2, 2, 6, 2, 3, 2, 4, 1, 2, 3, 3, 3, 2, 1, 1, 3, 3, 9, 4, 1, 2, 1, 4, 1, 2, 1, 2, 1, 3, 6, 2, 2, 4, 1, 6, 2, 1, 2, 1, 2, 4, 9, 4, 1, 4, 1, 1, 4, 1, 1, 1, 2, 9, 4, 1, 3, 2, 2, 1, 3, 3, 9, 1, 4, 1, 4, 1, 1, 1, 2, 2, 4, 1, 3, 3, 6, 2, 3, 2, 9, 4, 1, 4, 1, 6, 2, 1, 4, 1, 1, 3, 2, 3, 4, 1, 2, 4, 1, 9, 1, 1, 4, 1, 9, 4, 1, 2, 2, 2, 2, 3, 9, 4, 1, 2, 1, 6, 2, 2, 6, 2, 2, 1, 9, 1, 1, 4, 1, 1, 3, 12, 6, 2, 4, 8, 3, 8, 3, 8, 3, 5, 5, 10, 1, 4, 2, 5, 1, 24, 15, 8, 1, 3, 10, 6, 2, 4, 5, 14, 6, 14, 15, 2, 8, 3, 6, 48, 1, 35, 24, 15, 8, 3, 72, 2, 56, 1, 3, 1, 42, 30, 20, 12, 6, 2, 80, 4, 63, 2, 48, 3, 35, 24, 12, 6, 2, 66, 50, 4, 1, 1, 36, 24, 1, 14, 6, 36, 24, 2, 2, 14, 6, 192, 165, 140, 117, 88, 70, 54, 40, 2, 28, 18, 10, 1, 4, 198, 2, 1, 160, 135, 1, 112, 91, 72, 55, 40, 27, 16, 6, 0, 5, 0, 4, 0, 209, 182, 156, 1, 132, 110, 90, 72, 56, 42, 30, 20, 12, 6, 2, 0, 1, 0, 0, 5, 2, 0, 1, 999998, 9828, 272, 12, 1, 1, 1, 2, 2, 2, 2, 4, 1, 4, 1, 4, 1, 2, 6, 1, 2, 3, 6, 1, 2, 6, 1, 2, 1]
# # # x_li2 = [1000000, 16632, 345, 24, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 6, 1, 2, 2, 1, 6, 2, 2, 3, 1, 3, 2, 1, 3, 3, 1, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 6, 2, 2, 1, 1, 2, 3, 1, 2, 1, 2, 3, 1, 1, 2, 2, 2, 3, 2, 3, 4, 1, 1, 2, 9, 4, 1, 2, 3, 2, 6, 2, 2, 1, 1, 4, 1, 3, 1, 1, 3, 6, 1, 2, 4, 1, 3, 3, 2, 1, 8, 3, 1, 2, 3, 6, 2, 3, 6, 2, 6, 2, 3, 1, 1, 1, 1, 2, 2, 6, 1, 2, 2, 6, 2, 4, 16, 1, 9, 4, 1, 3, 12, 6, 1, 2, 3, 9, 4, 1, 6, 2, 1, 2, 2, 1, 6, 2, 2, 1, 3, 1, 2, 8, 3, 1, 4, 1, 1, 6, 2, 3, 12, 6, 2, 2, 1, 2, 6, 2, 6, 2, 2, 6, 2, 4, 16, 9, 4, 1, 3, 3, 4, 2, 12, 6, 2, 4, 5, 5, 25, 16, 9, 4, 1, 4, 4, 5, 18, 10, 2, 4, 1, 6, 3, 7, 64, 49, 36, 25, 16, 9, 4, 1, 56, 42, 2, 1, 2, 30, 20, 12, 6, 2, 16, 2, 7, 36, 24, 1, 14, 6, 130, 1, 99, 80, 63, 1, 48, 35, 24, 15, 8, 3, 78, 1, 60, 1, 44, 30, 18, 8, 13, 3, 234, 204, 176, 140, 2, 117, 96, 77, 60, 1, 45, 32, 21, 12, 5, 54, 4, 1, 34, 16, 0, 15, 0, 14, 0, 999999, 10206, 345, 24, 1, 4, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 6, 2, 2, 1, 1, 2, 6, 2, 3, 6, 2, 1, 2, 6, 2, 6, 2, 6, 2, 3, 12, 1, 6, 2, 3, 6, 1, 2, 9, 1, 4, 1, 2, 2, 2, 1, 2, 2, 6, 2, 6, 2, 3, 1, 9, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 4, 1, 1, 4, 1, 2, 2, 2, 6, 2, 4, 1, 4, 1, 2, 4, 1, 1, 2, 4, 1, 6, 2, 1, 4, 1, 2, 2, 3, 3, 2, 2, 3, 6, 2, 3, 2, 1, 9, 4, 1, 2, 4, 1, 1, 1, 6, 2, 3, 3, 3, 3, 6, 2, 2, 6, 2, 2, 4, 1, 6, 2, 3, 6, 2, 4, 1, 1, 2, 9, 4, 1, 2, 1, 3, 3, 1, 2, 2, 9, 4, 1, 6, 2, 4, 1, 1, 2, 2, 3, 3, 2, 6, 2, 4, 1, 3, 3, 6, 2, 4, 1, 1, 6, 2, 2, 1, 3, 2, 9, 4, 1, 3, 3, 6, 2, 3, 1, 2, 2, 4, 2, 8, 2, 3, 8, 2, 2, 2, 3, 2, 1, 4, 4, 4, 1, 4, 16, 1, 9, 4, 1, 12, 6, 2, 32, 21, 12, 1, 5, 16, 7, 56, 42, 1, 30, 20, 12, 6, 2, 72, 2, 56, 35, 24, 15, 8, 2, 3, 54, 1, 40, 2, 28, 18, 1, 10, 4, 8, 45, 32, 1, 1, 21, 1, 12, 5, 18, 8, 102, 2, 75, 56, 1, 1, 39, 24, 11, 1, 228, 198, 170, 1, 4, 1, 144, 112, 91, 72, 1, 1, 55, 40, 27, 16, 7, 216, 1, 187, 2, 160, 135, 112, 91, 72, 1, 55, 40, 27, 16, 7, 0, 6, 0, 5, 0, 16631, 368, 24, 1, 4, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 4, 1, 2, 1, 2, 2, 1, 2, 4, 1, 2, 2, 2, 6, 1, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1, 3, 2, 2, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 4, 1, 3, 1, 1, 2, 1, 2, 1, 9, 4, 1, 4, 1, 2, 3, 2, 1, 1, 4, 1, 1, 2, 4, 1, 1, 1, 6, 2, 3, 1, 1, 2, 6, 2, 2, 3, 9, 4, 1, 4, 1, 1, 2, 2, 2, 2, 4, 1, 6, 2, 9, 4, 1, 1, 2, 4, 1, 2, 1, 2, 2, 2, 4, 1, 2, 4, 1, 3, 2, 4, 1, 3, 1, 4, 1, 1, 2, 2, 1, 2, 6, 2, 2, 2, 2, 9, 4, 1, 3, 3, 9, 4, 1, 1, 1, 2, 2, 6, 1, 2, 1, 1, 2, 3, 9, 4, 1, 1, 6, 2, 2, 3, 1, 3, 16, 6, 2, 4, 1, 8, 3, 4, 2, 2, 6, 2, 3, 8, 3, 3, 1, 3, 6, 2, 3, 1, 9, 4, 1, 2, 2, 12, 4, 1, 6, 2, 4, 4, 1, 2, 2, 4, 8, 3, 4, 2, 1, 10, 2, 4, 15, 1, 8, 3, 5, 5, 2, 5, 6, 2, 24, 1, 15, 8, 3, 6, 2, 35, 24, 15, 8, 3, 7, 77, 6, 2, 54, 1, 40, 28, 18, 1, 10, 4, 120, 3, 99, 80, 63, 48, 35, 24, 15, 8, 3, 66, 1, 50, 36, 24, 14, 6, 48, 33, 20, 9, 228, 5, 1, 180, 1, 153, 1, 112, 90, 70, 52, 33, 20, 9, 300, 252, 221, 1, 192, 165, 140, 117, 1, 96, 77, 60, 1, 3, 45, 32, 21, 12, 5, 0, 4, 0, 3, 0, 2, 999998, 9072, 368, 24, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 1, 1, 1, 1, 2, 4, 1, 2, 2, 2, 4, 1, 1, 2, 1, 2, 1, 4, 1, 1, 4, 1, 1, 2, 2, 2, 1, 1, 4, 1, 2, 2, 1, 2, 9, 1, 4, 1, 3, 3, 3, 3, 6, 2, 1, 2, 4, 1, 2, 1, 4, 1, 2, 1, 2, 1, 3, 1, 3, 3, 9, 4, 1, 1, 2]
# # # x_li3 = [1000000, 15252, 182, 12, 6, 1, 2, 4, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 4, 1, 1, 2, 2, 2, 4, 1, 4, 1, 2, 4, 1, 6, 2, 3, 9, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 6, 2, 9, 1, 4, 1, 1, 2, 3, 3, 2, 1, 6, 2, 3, 6, 2, 1, 6, 2, 4, 1, 2, 2, 2, 6, 2, 3, 3, 1, 3, 1, 3, 3, 1, 2, 3, 6, 1, 1, 1, 2, 9, 1, 4, 1, 1, 2, 3, 3, 4, 1, 9, 4, 1, 2, 1, 3, 1, 1, 3, 3, 6, 2, 6, 2, 3, 2, 2, 4, 1, 3, 2, 9, 4, 1, 1, 6, 2, 2, 1, 2, 6, 2, 3, 3, 6, 2, 12, 1, 6, 2, 3, 1, 2, 1, 3, 3, 1, 3, 4, 3, 8, 3, 4, 6, 2, 3, 2, 4, 6, 2, 1, 3, 4, 3, 2, 1, 2, 1, 3, 3, 1, 1, 3, 12, 4, 1, 1, 6, 2, 2, 4, 1, 4, 6, 2, 1, 2, 4, 1, 3, 1, 3, 6, 2, 6, 1, 2, 3, 2, 6, 2, 8, 3, 12, 1, 6, 1, 2, 12, 1, 6, 1, 1, 2, 2, 2, 4, 4, 20, 9, 1, 4, 1, 5, 12, 6, 6, 6, 28, 3, 12, 5, 2, 7, 45, 2, 32, 21, 1, 12, 5, 72, 2, 56, 3, 42, 30, 1, 20, 12, 2, 6, 2, 49, 2, 36, 1, 1, 1, 25, 16, 9, 4, 1, 9, 10, 4, 110, 80, 2, 63, 48, 35, 24, 1, 15, 8, 3, 154, 130, 108, 1, 80, 63, 1, 1, 48, 35, 24, 15, 8, 3, 294, 2, 260, 4, 1, 2, 2, 216, 176, 150, 117, 1, 96, 77, 60, 1, 45, 32, 21, 12, 5, 0, 4, 0, 3, 0, 999999, 12177, 182, 12, 6, 1, 2, 4, 1, 1, 2, 2, 1, 3, 2, 1, 1, 4, 1, 1, 3, 1, 3, 1, 3, 12, 2, 6, 2, 2, 6, 2, 3, 3, 6, 2, 2, 3, 1, 8, 3, 12, 4, 1, 4, 1, 2, 8, 3, 9, 4, 1, 1, 9, 4, 1, 1, 3, 12, 6, 2, 2, 1, 9, 4, 1, 1, 2, 4, 1, 6, 1, 4, 1, 2, 2, 3, 3, 6, 1, 2, 2, 2, 2, 9, 1, 1, 4, 1, 3, 1, 6, 2, 9, 4, 1, 1, 4, 1, 2, 1, 2, 4, 1, 1, 2, 2, 4, 1, 3, 3, 3, 2, 1, 3, 2, 1, 4, 1, 2, 3, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 2, 3, 4, 4, 8, 3, 4, 1, 6, 1, 2, 3, 6, 2, 1, 4, 1, 2, 4, 8, 3, 1, 4, 1, 3, 1, 8, 2, 2, 3, 8, 1, 1, 3, 3, 2, 4, 1, 1, 3, 4, 1, 8, 2, 3, 3, 8, 3, 4, 2, 4, 20, 12, 6, 2, 10, 6, 2, 4, 1, 4, 6, 2, 4, 10, 1, 4, 2, 6, 1, 3, 1, 10, 4, 6, 6, 2, 6, 35, 5, 24, 15, 3, 1, 8, 3, 6, 6, 35, 2, 24, 15, 2, 1, 8, 3, 21, 12, 5, 40, 1, 28, 1, 18, 10, 1, 4, 54, 3, 40, 28, 2, 18, 10, 4, 10, 1, 182, 1, 156, 132, 110, 2, 90, 1, 72, 56, 42, 1, 1, 6, 2, 30, 20, 12, 6, 2, 156, 132, 110, 90, 1, 72, 56, 42, 30, 20, 12, 6, 2, 220, 4, 189, 4, 1, 160, 1, 126, 1, 90, 1, 70, 52, 36, 22, 10, 0, 9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 21, 10, 0, 9, 0, 15251, 195, 12, 2, 1, 1, 1, 1, 4, 1, 1, 2, 4, 1, 1, 1, 2, 2, 4, 1, 2, 2, 3, 1, 1, 2, 2, 2, 6, 1, 1, 1, 4, 1, 1, 2, 2, 3, 3, 2, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 4, 1, 1, 4, 1, 2, 2, 2, 4, 1, 1, 4, 1, 4, 1, 2, 4, 1, 3, 3, 2, 3, 6, 2, 2, 3, 6, 2, 3, 1, 3, 1, 2, 9, 1, 4, 1, 2, 2, 2, 9, 1, 4, 1, 2, 1, 2, 3, 6, 2, 9, 4, 1, 1, 2, 2, 1, 1, 4, 1, 2, 2, 4, 1, 1, 2, 6, 2, 2, 6, 2, 4, 1, 6, 2, 1, 2, 2, 1, 3, 1, 1, 1, 1, 3, 2, 6, 2, 4, 1, 9, 4, 2, 3, 3, 2, 6, 1, 2, 1, 3, 2, 1, 2, 3, 3, 6, 2, 4, 1, 3, 6, 2, 1, 2, 1, 3, 6, 2, 2, 9, 4, 1, 1, 4, 1, 1, 9, 1, 1, 4, 1, 9, 4, 1, 6, 1, 2, 3, 3, 3, 6, 2, 2, 3, 3, 3, 1, 6, 1, 2, 6, 1, 2, 12, 1, 6, 2, 2, 2, 8, 3, 4, 4, 4, 5, 20, 2, 12, 6, 2, 5, 18, 1, 10, 1, 1, 4, 8, 3, 2, 5, 6, 6, 6, 7, 64, 49, 1, 36, 25, 16, 9, 4, 1, 54, 2, 40, 28, 18, 10, 4, 27, 16, 7, 150, 5, 4, 1, 117, 3, 96, 77, 60, 45, 24, 14, 6, 144, 120, 98, 78, 60, 1, 44, 2, 30, 18, 8, 195, 1, 168, 143, 1, 120, 99, 80, 63, 48, 35, 1, 24, 15, 8, 3, 0, 2, 0, 1, 0, 0, 7, 1, 2, 3, 0, 2, 0, 1, 0, 0, 1, 3, 0, 2, 0, 1, 0, 0, 3, 0, 999998, 10455, 195, 12, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 4, 1, 1, 2, 2, 2, 2, 4, 1, 1, 2, 3, 2, 3, 3, 3]
# # # x_li4 =[1000000, 13923, 256, 10, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 4, 1, 2, 4, 1, 1, 1, 2, 9, 4, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 9, 1, 4, 1, 1, 1, 3, 6, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 4, 1, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 4, 1, 2, 6, 1, 2, 1, 4, 1, 1, 4, 1, 4, 1, 1, 1, 6, 2, 4, 1, 4, 1, 6, 2, 4, 1, 2, 4, 1, 1, 3, 3, 3, 1, 3, 1, 3, 6, 2, 1, 1, 2, 3, 3, 2, 6, 1, 2, 2, 3, 2, 1, 1, 3, 1, 2, 6, 2, 1, 2, 6, 2, 2, 3, 1, 2, 12, 1, 1, 1, 1, 6, 2, 6, 2, 4, 1, 1, 2, 3, 3, 4, 2, 1, 12, 6, 2, 6, 2, 4, 1, 2, 2, 8, 3, 8, 1, 3, 3, 1, 3, 16, 9, 4, 1, 1, 2, 1, 2, 3, 8, 3, 8, 3, 4, 4, 15, 8, 3, 1, 20, 12, 6, 2, 2, 5, 10, 1, 4, 1, 5, 18, 1, 8, 3, 6, 5, 14, 6, 2, 6, 8, 77, 1, 2, 2, 60, 4, 35, 24, 15, 1, 8, 3, 33, 20, 1, 9, 108, 88, 70, 54, 40, 28, 18, 10, 4, 126, 6, 2, 104, 2, 84, 66, 50, 36, 24, 14, 6, 140, 117, 96, 77, 1, 45, 32, 21, 12, 5, 135, 2, 98, 78, 55, 36, 24, 14, 6, 30, 13, 323, 288, 255, 2, 2, 224, 1, 195, 168, 143, 120, 99, 80, 63, 48, 35, 24, 15, 8, 3, 0, 2, 0, 1, 0, 999999, 11662, 256, 10, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 6, 2, 3, 3, 6, 1, 9, 1, 1, 1, 1, 4, 1, 2, 9, 4, 1, 6, 2, 6, 2, 4, 1, 1, 6, 2, 6, 1, 3, 6, 1, 1, 2, 3, 6, 2, 1, 1, 3, 6, 2, 1, 2, 6, 2, 3, 4, 9, 1, 4, 1, 1, 4, 1, 2, 6, 2, 2, 9, 1, 4, 1, 4, 1, 2, 6, 2, 9, 4, 1, 2, 4, 1, 4, 1, 2, 2, 2, 2, 6, 2, 6, 2, 4, 1, 4, 1, 1, 2, 4, 1, 1, 2, 1, 2, 2, 3, 6, 2, 6, 1, 2, 9, 4, 1, 2, 6, 2, 4, 1, 1, 2, 2, 9, 4, 1, 3, 2, 3, 2, 2, 3, 2, 3, 6, 2, 3, 3, 1, 1, 3, 3, 2, 4, 1, 6, 2, 3, 16, 2, 9, 4, 1, 4, 1, 3, 1, 12, 4, 1, 2, 2, 6, 2, 8, 3, 8, 3, 3, 8, 1, 3, 4, 4, 15, 3, 8, 2, 3, 5, 5, 5, 10, 4, 5, 1, 18, 4, 1, 1, 1, 2, 10, 4, 6, 21, 12, 5, 7, 3, 48, 2, 35, 1, 3, 2, 2, 1, 24, 15, 8, 3, 7, 56, 4, 1, 2, 42, 2, 30, 20, 12, 6, 2, 24, 14, 6, 40, 1, 28, 18, 10, 4, 1, 10, 210, 4, 1, 1, 182, 2, 143, 120, 99, 70, 54, 1, 40, 28, 18, 10, 4, 208, 1, 165, 130, 108, 88, 70, 54, 40, 28, 18, 10, 4, 243, 200, 168, 3, 2, 1, 2, 138, 2, 110, 80, 57, 36, 1, 17, 0, 16, 0, 15, 0, 14, 0, 13, 0, 12, 13922, 208, 10, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 4, 1, 1, 2, 1, 2, 4, 1, 2, 2, 2, 4, 1, 2, 4, 1, 2, 2, 2, 2, 3, 3, 9, 2, 2, 4, 1, 1, 1, 1, 2, 6, 2, 6, 2, 3, 6, 1, 1, 1, 2, 2, 2, 3, 2, 3, 9, 4, 1, 1, 3, 2, 6, 2, 6, 2, 1, 4, 1, 3, 4, 1, 1, 1, 6, 2, 4, 1, 1, 2, 2, 2, 3, 9, 1, 4, 1, 3, 6, 2, 3, 4, 1, 4, 6, 1, 2, 6, 2, 4, 4, 1, 1, 1, 8, 1, 2, 3, 3, 9, 4, 1, 1, 3, 3, 6, 2, 2, 2, 6, 2, 2, 2, 1, 3, 3, 3, 4, 1, 2, 1, 1, 9, 4, 1, 1, 1, 2, 2, 2, 6, 2, 2, 2, 6, 2, 3, 2, 6, 2, 3, 3, 6, 2, 2, 3, 3, 1, 3, 12, 6, 2, 2, 1, 6, 2, 2, 1, 1, 1, 4, 8, 1, 3, 12, 6, 1, 2, 4, 1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 9, 4, 1, 1, 4, 1, 2, 2, 3, 1, 1, 2, 3, 3, 4, 1, 4, 1, 3, 2, 4, 8, 3, 4, 8, 3, 4, 8, 3, 10, 3, 4, 10, 4, 5, 1, 6, 4, 1, 7, 64, 1, 49, 2, 36, 2, 1, 20, 12, 1, 4, 1, 1, 6, 2, 6, 11, 104, 84, 66, 45, 32, 21, 12, 2, 5, 168, 4, 143, 120, 1, 99, 80, 63, 48, 35, 24, 15, 8, 3, 117, 2, 6, 2, 96, 70, 1, 2, 54, 40, 1, 28, 18, 10, 4, 26, 1, 12, 90, 2, 70, 52, 36, 22, 10, 0, 9, 0, 8, 0, 255, 15, 2, 2, 2, 2, 6, 1, 2, 2, 6, 1, 1, 1, 1, 1, 1, 2, 2, 2, 6, 2, 3, 1, 3, 1, 2, 1, 2, 6, 2, 2, 2, 6, 2, 3, 1, 3, 1, 9, 4, 1, 1, 6, 2, 1, 1, 3, 4, 1, 4, 1, 1, 6, 2, 2, 3, 8, 2, 3, 6, 2, 4, 1, 2]
# # # x_li5 =[1000000, 19596, 288, 48, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 4, 1, 4, 1, 1, 1, 2, 2, 2, 2, 6, 2, 3, 9, 4, 1, 3, 2, 3, 1, 1, 6, 1, 2, 2, 6, 1, 1, 1, 2, 2, 6, 2, 4, 1, 1, 1, 2, 6, 2, 8, 3, 2, 9, 1, 4, 1, 1, 2, 2, 1, 1, 9, 4, 1, 2, 4, 1, 6, 2, 1, 2, 6, 2, 9, 4, 1, 2, 9, 4, 1, 6, 1, 1, 2, 1, 2, 3, 9, 1, 4, 1, 1, 2, 6, 2, 3, 8, 3, 8, 1, 1, 3, 1, 6, 2, 4, 6, 1, 2, 9, 4, 1, 4, 1, 2, 3, 1, 8, 1, 1, 3, 2, 6, 2, 4, 1, 1, 1, 1, 4, 1, 1, 6, 2, 2, 1, 12, 6, 2, 6, 2, 6, 1, 2, 3, 6, 2, 3, 3, 4, 1, 1, 1, 3, 1, 1, 3, 3, 1, 2, 2, 6, 2, 9, 4, 1, 3, 1, 1, 2, 2, 2, 12, 6, 2, 8, 3, 4, 5, 10, 4, 4, 5, 10, 1, 4, 5, 5, 1, 5, 5, 3, 2, 3, 4, 6, 2, 24, 4, 1, 1, 1, 15, 8, 3, 42, 1, 30, 20, 4, 1, 2, 12, 6, 2, 27, 2, 16, 7, 2, 2, 22, 6, 2, 10, 144, 3, 110, 2, 90, 72, 56, 42, 30, 20, 12, 6, 2, 169, 144, 121, 1, 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 42, 1, 26, 12, 64, 45, 28, 13, 204, 2, 2, 165, 130, 3, 1, 108, 2, 80, 63, 42, 30, 20, 12, 6, 2, 0, 1, 0, 0, 5, 999999, 12696, 288, 54, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 4, 1, 1, 1, 2, 1, 1, 2, 4, 1, 1, 4, 1, 1, 4, 1, 1, 2, 4, 1, 1, 4, 1, 2, 2, 3, 2, 3, 1, 3, 1, 6, 2, 3, 4, 1, 4, 1, 2, 1, 2, 1, 4, 1, 1, 2, 3, 1, 4, 1, 4, 1, 1, 2, 1, 1, 1, 3, 6, 2, 2, 2, 1, 3, 6, 2, 1, 6, 2, 9, 4, 1, 2, 3, 3, 3, 6, 2, 2, 2, 6, 2, 1, 3, 3, 6, 2, 9, 4, 1, 2, 6, 1, 1, 2, 2, 2, 6, 2, 1, 1, 3, 9, 4, 1, 1, 2, 6, 2, 6, 2, 2, 6, 2, 4, 1, 3, 3, 6, 2, 1, 1, 4, 1, 3, 3, 2, 6, 2, 8, 3, 4, 3, 8, 1, 2, 3, 4, 4, 4, 15, 3, 8, 1, 3, 3, 1, 4, 1, 4, 2, 1, 4, 2, 25, 16, 1, 9, 4, 1, 4, 5, 8, 3, 5, 15, 8, 3, 2, 24, 15, 4, 1, 8, 2, 3, 6, 35, 24, 15, 1, 8, 1, 3, 12, 5, 7, 3, 9, 4, 1, 2, 6, 1, 8, 66, 1, 2, 20, 1, 12, 6, 2, 50, 5, 1, 36, 4, 1, 24, 2, 1, 14, 6, 33, 20, 9, 22, 10, 224, 195, 2, 168, 1, 143, 1, 108, 88, 70, 54, 1, 40, 28, 18, 1, 6, 2, 10, 4, 108, 80, 60, 3, 42, 26, 1, 11, 231, 200, 1, 171, 144, 119, 96, 75, 56, 2, 2, 39, 24, 11, 264, 2, 231, 200, 1, 171, 144, 119, 90, 70, 52, 36, 22, 1, 10, 0, 9, 0, 8, 0, 19595, 176, 48, 3, 1, 1, 1, 1, 1, 4, 1, 2, 2, 1, 1, 2, 2, 3, 1, 1, 1, 4, 1, 2, 2, 1, 1, 4, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 4, 1, 1, 9, 2, 4, 1, 2, 1, 2, 2, 4, 1, 1, 2, 2, 2, 2, 1, 2, 4, 1, 1, 2, 2, 2, 1, 9, 1, 4, 1, 4, 1, 1, 2, 1, 2, 2, 2, 4, 1, 1, 1, 1, 1, 2, 1, 4, 1, 2, 2, 1, 1, 1, 2, 2, 6, 1, 2, 3, 4, 1, 1, 1, 4, 1, 2, 1, 1, 2, 2, 1, 6, 2, 4, 1, 2, 3, 2, 2, 1, 1, 9, 4, 1, 1, 1, 2, 3, 3, 3, 1, 2, 6, 2, 1, 3, 6, 2, 9, 4, 1, 2, 2, 1, 2, 4, 1, 6, 2, 6, 2, 3, 3, 3, 2, 3, 6, 2, 1, 1, 6, 1, 1, 2, 6, 2, 6, 2, 3, 2, 4, 1, 2, 4, 4, 1, 3, 6, 2, 9, 1, 1, 4, 1, 1, 1, 6, 2, 8, 3, 8, 3, 3, 4, 4, 4, 1, 16, 1, 9, 4, 1, 4, 8, 3, 8, 2, 2, 3, 4, 4, 15, 8, 3, 5, 15, 8, 2, 3, 12, 5, 45, 2, 32, 1, 21, 12, 5, 1, 15, 8, 3, 30, 6, 1, 2, 9, 90, 2, 72, 1, 56, 42, 30, 20, 12, 6, 2, 10, 110, 90, 72, 48, 35, 24, 15, 8, 3, 84, 2, 66, 50, 36, 1, 24, 14, 6, 171, 10, 4, 136, 96, 75, 56, 1, 39, 24, 1, 11, 228, 4, 180, 153, 128, 105, 78, 1, 2, 60, 44, 30, 18, 8, 0, 7, 0, 6, 0, 1, 2, 4, 1, 4, 1, 1, 1, 2, 2, 2, 2, 3, 6, 2, 3, 9, 4, 1, 4, 1, 1, 3, 3, 1, 1, 1, 6, 1, 2, 2, 2, 1, 3, 8, 1, 2, 2, 3, 4, 1, 2, 6, 2, 4, 3, 8, 3, 9, 2, 4, 1, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 4, 1, 1, 3, 2, 3, 1, 2, 3, 8, 3, 6, 2, 3, 3, 9, 1, 4, 1, 1, 1, 1, 2, 1, 4, 1, 6, 2, 3, 6, 2, 4, 1, 1, 2, 4, 1, 6, 2, 3, 6, 1, 2, 12, 6, 2, 8, 1, 1, 3, 6, 2, 4, 1]
# #
# # #ER2
# # # x_li1 =[1000000, 64, 15, 9, 2, 6, 9, 2, 6, 1, 8, 1, 4, 3, 8, 3, 8, 3, 12, 4, 6, 1, 9, 1, 1, 1, 1, 4, 1, 4, 3, 6, 1, 2, 6, 1, 6, 2, 4, 1, 6, 1, 9, 2, 4, 6, 1, 2, 9, 2, 2, 9, 2, 2, 3, 9, 2, 6, 2, 6, 2, 3, 9, 2, 6, 2, 9, 2, 3, 9, 4, 1, 1, 2, 2, 3, 3, 6, 6, 1, 2, 6, 1, 4, 9, 2, 4, 1, 2, 2, 4, 2, 6, 1, 2, 9, 4, 1, 9, 2, 1, 6, 2, 4, 1, 1, 1, 6, 1, 2, 4, 1, 1, 2, 4, 1, 4, 2, 9, 2, 6, 9, 2, 6, 6, 2, 6, 1, 3, 6, 2, 2, 2, 6, 1, 2, 3, 2, 6, 2, 4, 1, 3, 3, 6, 1, 2, 4, 1, 2, 4, 9, 3, 6, 1, 4, 1, 3, 6, 1, 2, 6, 2, 3, 9, 1, 4, 4, 1, 9, 2, 2, 2, 3, 6, 1, 2, 4, 6, 2, 3, 9, 4, 1, 1, 1, 6, 2, 2, 2, 2, 6, 1, 4, 1, 1, 2, 6, 1, 4, 1, 4, 2, 4, 1, 3, 2, 2, 6, 2, 6, 1, 2, 9, 2, 9, 4, 6, 2, 3, 9, 4, 1, 6, 1, 4, 1, 9, 1, 2, 3, 6, 1, 4, 2, 2, 6, 2, 2, 3, 3, 4, 9, 2, 6, 4, 3, 4, 4, 1, 1, 2, 2, 6, 1, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 6, 2, 4, 1, 4, 4, 9, 2, 1, 2, 2, 4, 1, 4, 2, 4, 1, 1, 1, 2, 2, 9, 4, 3, 6, 1, 3, 4, 1, 1, 2, 4, 1, 2, 4, 1, 2, 4, 4, 2, 2, 4, 1, 2, 2, 3, 2, 2, 6, 1, 2, 2, 2, 9, 2, 6, 1, 1, 3, 3, 6, 2, 4, 1, 1, 4, 6, 2, 4, 1, 1, 6, 2, 2, 1, 2, 9, 2, 2, 3, 1, 6, 1, 1, 2, 4, 1, 1, 2, 6, 1, 1, 6, 2, 3, 2, 2, 6, 1, 1, 2, 1, 2, 9, 1, 1, 1, 2, 3, 3, 6, 1, 1, 6, 2, 6, 1, 3, 6, 1, 3, 2, 3, 2, 6, 1, 3, 2, 2, 6, 2, 2, 6, 1, 1, 2, 2, 3, 3, 2, 3, 2, 2, 9, 2, 1, 4, 1, 3, 2, 2, 3, 2, 2, 2, 3, 3, 6, 2, 2, 6, 1, 3, 6, 1, 1, 8, 2, 3, 8, 1, 4, 1, 12, 1, 2, 4, 4, 1, 0, 0, 3, 1, 0, 999999, 56, 15, 8, 1, 4, 12, 4, 6, 6, 1, 1, 1, 6, 1, 9, 4, 1, 3, 3, 9, 1, 1, 1, 1, 1, 1, 1, 2, 9, 1, 4, 1, 6, 1, 1, 2, 9, 2, 6, 2, 2, 6, 2, 2, 3, 6, 1, 2, 9, 1, 1, 4, 1, 4, 9, 4, 6, 2, 6, 2, 6, 2, 1, 9, 2, 6, 9, 4, 1, 9, 4, 1, 1, 2, 6, 2, 6, 2, 4, 1, 3, 6, 2, 2, 6, 2, 2, 9, 2, 4, 9, 2, 2, 3, 6, 1, 1, 6, 1, 6, 6, 2, 2, 6, 2, 4, 3, 9, 1, 1, 1, 2, 9, 2, 9, 1, 3, 6, 1, 6, 6, 2, 9, 4, 6, 1, 4, 1, 6, 1, 4, 4, 4, 6, 1, 6, 1, 4, 4, 1, 2, 6, 2, 2, 2, 4, 1, 6, 6, 2, 1, 4, 6, 2, 2, 6, 6, 1, 9, 2, 3, 4, 1, 4, 1, 6, 2, 2, 4, 2, 4, 4, 4, 1, 3, 3, 6, 1, 2, 2, 9, 4, 6, 9, 2, 2, 2, 4, 1, 1, 1, 1, 3, 1, 4, 6, 1, 4, 1, 3, 9, 2, 4, 1, 9, 4, 1, 1, 6, 1, 3, 9, 1, 1, 1, 2, 2, 4, 3, 2, 2, 6, 3, 3, 6, 2, 2, 4, 2, 4, 1, 3, 1, 6, 2, 2, 2, 2, 6, 1, 1, 4, 4, 1, 4, 2, 2, 6, 6, 2, 2, 2, 4, 1, 2, 4, 2, 2, 2, 3, 6, 2, 6, 3, 6, 1, 6, 1, 4, 1, 2, 6, 1, 1, 2, 6, 1, 3, 6, 2, 2, 3, 6, 1, 1, 1, 2, 4, 1, 4, 1, 4, 2, 4, 1, 9, 2, 6, 1, 4, 1, 9, 4, 1, 1, 1, 1, 2, 3, 4, 6, 4, 6, 1, 3, 3, 3, 6, 1, 6, 1, 1, 4, 1, 6, 1, 3, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 6, 1, 1, 1, 1, 3, 2, 2, 6, 1, 2, 6, 1, 1, 6, 1, 2, 3, 4, 1, 9, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 4, 2, 2, 4, 1, 1, 4, 3, 6, 1, 1, 6, 2, 3, 6, 2, 6, 1, 3, 6, 1, 1, 3, 4, 3, 3, 9, 1, 4, 1, 3, 6, 1, 3, 1, 3, 9, 2, 2, 4, 2, 2, 1, 4, 1, 1, 3, 1, 1, 3, 2, 6, 1, 2, 6, 1, 4, 1, 1, 1, 3, 3, 2, 2, 9, 2, 3, 3, 6, 1, 4, 1, 12, 6, 3, 6, 1, 1, 12, 3, 2, 1, 3, 2, 10, 2, 2, 4, 1, 15, 2, 1, 1, 1, 2, 1, 4, 0, 3, 0, 2, 0, 2, 12, 4, 6, 1, 9, 1, 1, 1, 1, 4, 1, 4, 3, 6, 1, 2, 6, 1, 6, 2, 4, 1, 6, 1, 9, 2, 4, 3, 9, 2, 2, 9, 2, 2, 3, 3, 6, 2, 6, 2, 3, 3, 9, 2, 6, 2, 9, 2, 1, 2, 9, 2, 3, 9, 4, 1, 1, 2, 2, 3, 6, 1, 1, 9, 2, 3, 6, 1, 1, 9, 2, 9, 4, 1]
# # # x_li2 =[1000000, 56, 12, 1, 2, 2, 6, 1, 3, 3, 9, 2, 12, 4, 9, 4, 1, 6, 3, 16, 2, 2, 2, 4, 1, 4, 6, 1, 3, 4, 2, 6, 1, 6, 3, 6, 2, 4, 1, 2, 3, 6, 2, 6, 1, 9, 1, 1, 2, 6, 2, 2, 4, 4, 1, 6, 2, 6, 1, 1, 6, 9, 4, 1, 4, 4, 1, 2, 4, 1, 4, 1, 6, 2, 2, 6, 1, 2, 2, 6, 2, 3, 3, 2, 6, 1, 6, 1, 2, 4, 3, 3, 6, 1, 1, 4, 9, 2, 4, 6, 1, 2, 4, 1, 1, 1, 6, 2, 6, 2, 3, 6, 3, 6, 6, 1, 2, 6, 1, 4, 6, 2, 2, 3, 6, 1, 6, 1, 4, 1, 3, 3, 6, 1, 2, 6, 6, 2, 4, 1, 4, 1, 2, 4, 2, 2, 6, 1, 1, 1, 1, 4, 1, 4, 2, 6, 6, 6, 6, 2, 6, 2, 6, 2, 1, 2, 6, 1, 2, 2, 6, 1, 4, 1, 9, 1, 1, 1, 4, 1, 6, 1, 2, 6, 2, 6, 1, 4, 4, 2, 4, 1, 3, 6, 1, 2, 6, 3, 6, 1, 6, 2, 2, 6, 1, 1, 6, 1, 1, 1, 1, 1, 9, 1, 1, 4, 6, 1, 2, 3, 4, 1, 1, 3, 3, 4, 6, 2, 1, 4, 1, 3, 6, 1, 4, 1, 1, 1, 4, 1, 4, 1, 4, 2, 2, 4, 1, 4, 1, 4, 1, 2, 2, 1, 9, 4, 1, 3, 4, 3, 2, 2, 4, 3, 6, 1, 2, 2, 2, 1, 2, 3, 3, 2, 2, 6, 1, 2, 4, 6, 2, 2, 6, 6, 1, 2, 4, 9, 4, 4, 6, 1, 4, 1, 3, 6, 1, 1, 1, 1, 1, 2, 3, 2, 2, 1, 4, 1, 2, 4, 2, 4, 1, 4, 1, 2, 1, 4, 1, 4, 1, 2, 2, 3, 6, 1, 4, 1, 1, 2, 4, 1, 4, 3, 4, 1, 6, 2, 4, 2, 1, 3, 1, 6, 1, 1, 2, 6, 2, 1, 2, 2, 4, 1, 1, 1, 1, 2, 3, 4, 1, 1, 4, 1, 2, 4, 1, 6, 1, 1, 6, 1, 1, 1, 1, 2, 4, 6, 1, 1, 2, 2, 4, 1, 2, 1, 1, 4, 4, 4, 2, 2, 2, 2, 1, 2, 6, 1, 9, 2, 2, 2, 3, 3, 1, 2, 3, 2, 6, 6, 4, 6, 2, 2, 2, 6, 2, 6, 6, 4, 6, 1, 3, 3, 1, 3, 1, 4, 6, 1, 4, 1, 1, 4, 8, 1, 8, 8, 2, 1, 4, 2, 5, 0, 4, 0, 3, 0, 999999, 49, 12, 3, 6, 1, 2, 6, 12, 2, 9, 2, 9, 2, 8, 6, 2, 2, 8, 2, 9, 1, 1, 2, 6, 2, 9, 2, 6, 2, 2, 9, 1, 1, 3, 9, 2, 2, 9, 2, 2, 6, 1, 2, 6, 2, 4, 8, 1, 2, 12, 6, 9, 4, 1, 4, 1, 2, 4, 6, 6, 2, 4, 1, 4, 1, 4, 2, 8, 6, 1, 1, 9, 1, 1, 4, 3, 8, 2, 6, 6, 2, 6, 6, 2, 6, 1, 4, 6, 16, 3, 8, 2, 2, 9, 4, 1, 6, 6, 1, 4, 1, 9, 1, 1, 1, 6, 3, 3, 8, 8, 2, 2, 6, 1, 1, 9, 4, 1, 6, 2, 6, 1, 4, 1, 2, 2, 4, 1, 1, 9, 2, 9, 2, 2, 1, 4, 6, 9, 1, 1, 2, 6, 1, 1, 3, 6, 1, 4, 6, 1, 1, 4, 6, 6, 2, 6, 9, 2, 4, 3, 2, 9, 2, 4, 1, 3, 6, 2, 6, 1, 9, 2, 4, 1, 9, 2, 4, 1, 3, 9, 6, 2, 1, 6, 2, 6, 1, 2, 3, 9, 4, 4, 1, 2, 2, 4, 1, 4, 3, 9, 2, 3, 9, 2, 6, 1, 6, 9, 4, 1, 3, 6, 6, 6, 1, 4, 9, 1, 1, 4, 1, 1, 6, 1, 1, 2, 3, 9, 4, 1, 2, 2, 4, 2, 6, 1, 4, 1, 2, 4, 1, 1, 4, 3, 1, 9, 2, 6, 1, 6, 2, 2, 1, 1, 2, 4, 2, 6, 6, 3, 3, 4, 3, 3, 2, 6, 2, 3, 3, 3, 4, 1, 1, 6, 2, 2, 6, 1, 2, 2, 4, 1, 1, 4, 3, 2, 1, 2, 2, 3, 2, 4, 1, 4, 1, 1, 4, 1, 4, 1, 2, 4, 1, 6, 2, 2, 3, 9, 2, 6, 2, 2, 2, 4, 1, 1, 6, 3, 6, 2, 1, 9, 2, 6, 1, 6, 1, 3, 9, 2, 2, 2, 3, 2, 6, 1, 4, 1, 2, 2, 2, 9, 1, 1, 4, 2, 4, 2, 2, 2, 2, 2, 1, 1, 3, 6, 2, 2, 6, 2, 1, 6, 2, 3, 4, 6, 1, 6, 2, 3, 3, 3, 6, 1, 3, 4, 1, 4, 4, 1, 2, 6, 2, 1, 4, 2, 2, 4, 1, 2, 4, 1, 2, 2, 6, 1, 2, 3, 4, 2, 3, 6, 1, 1, 3, 2, 4, 1, 2, 6, 2, 6, 2, 8, 3, 6, 1, 3, 2, 8, 2, 1, 4, 2, 2, 3, 3, 4, 4, 3, 4, 8, 1, 2, 2, 4, 1, 0, 0, 3, 1, 2, 2, 0, 1, 3, 2, 11, 1, 2, 2, 6, 1, 3, 3, 9, 2, 12, 4, 9, 4, 1, 6, 3, 16, 2, 2, 2, 4, 1, 4, 6, 1, 3, 4, 4, 1, 1, 6, 1, 6, 3, 6, 2, 4, 1, 2, 3, 6, 2, 6, 1, 9, 1, 1, 2, 6, 2, 2, 6, 2, 6, 1, 1, 6, 6, 9, 4, 1, 4, 4, 1, 2, 4, 1, 4, 1, 6, 2, 2, 6, 1, 2, 2, 6, 2, 6, 2, 3, 3, 2, 6, 1, 6, 1, 2, 4, 3, 3, 2, 4, 3, 6, 1, 1, 4, 9, 2, 6, 2, 6]
# # # x_li3 =[1000000, 56, 20, 12, 2, 1, 4, 1, 4, 12, 3, 4, 3, 4, 8, 2, 2, 6, 1, 4, 1, 3, 3, 9, 2, 1, 4, 6, 2, 6, 2, 2, 2, 16, 9, 2, 2, 6, 2, 9, 4, 1, 1, 1, 4, 1, 2, 3, 6, 6, 1, 2, 4, 4, 9, 2, 6, 1, 3, 6, 6, 6, 1, 6, 6, 3, 8, 2, 12, 6, 2, 2, 4, 1, 4, 4, 9, 2, 9, 2, 6, 1, 6, 2, 6, 3, 2, 4, 8, 2, 2, 2, 3, 4, 1, 1, 3, 9, 4, 3, 8, 2, 1, 1, 1, 4, 4, 1, 3, 9, 2, 9, 2, 2, 2, 2, 6, 2, 3, 6, 1, 3, 6, 2, 9, 4, 4, 6, 2, 4, 6, 2, 4, 2, 6, 2, 3, 6, 1, 2, 4, 3, 3, 1, 6, 2, 6, 1, 2, 6, 1, 2, 9, 2, 1, 4, 1, 1, 2, 2, 4, 6, 6, 1, 3, 6, 2, 4, 2, 4, 1, 3, 9, 4, 4, 1, 1, 9, 2, 9, 4, 9, 2, 4, 3, 3, 9, 2, 4, 6, 2, 1, 6, 1, 1, 2, 6, 1, 3, 6, 1, 3, 3, 2, 3, 3, 3, 9, 2, 2, 3, 6, 1, 3, 2, 2, 4, 2, 9, 1, 2, 6, 1, 6, 3, 9, 2, 3, 4, 3, 6, 1, 4, 4, 3, 1, 4, 6, 9, 4, 1, 9, 1, 4, 1, 2, 3, 2, 3, 3, 6, 1, 4, 4, 2, 3, 9, 2, 4, 1, 4, 1, 1, 6, 2, 3, 4, 1, 1, 6, 2, 3, 2, 3, 3, 6, 1, 4, 1, 9, 2, 9, 2, 1, 3, 3, 9, 2, 9, 1, 1, 4, 6, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 3, 6, 2, 2, 6, 2, 3, 4, 1, 4, 3, 2, 2, 3, 6, 1, 6, 1, 3, 4, 1, 6, 3, 3, 1, 6, 2, 9, 2, 6, 1, 4, 1, 2, 4, 1, 2, 6, 1, 2, 6, 1, 4, 6, 2, 4, 3, 2, 2, 6, 1, 6, 1, 6, 1, 3, 3, 3, 4, 3, 2, 4, 1, 4, 1, 1, 1, 2, 2, 2, 6, 2, 2, 2, 4, 1, 1, 3, 6, 3, 4, 6, 2, 6, 1, 3, 6, 2, 4, 1, 3, 1, 2, 4, 1, 1, 3, 2, 2, 3, 3, 3, 6, 1, 3, 2, 1, 2, 9, 2, 4, 1, 3, 4, 8, 3, 4, 1, 3, 3, 8, 1, 1, 1, 8, 2, 6, 1, 2, 1, 6, 1, 0, 0, 5, 1, 0, 999999, 56, 25, 12, 2, 2, 6, 1, 2, 6, 1, 6, 1, 9, 1, 1, 6, 2, 9, 4, 1, 4, 4, 1, 4, 3, 6, 1, 6, 1, 1, 2, 3, 9, 6, 2, 9, 2, 6, 1, 1, 4, 1, 2, 2, 6, 9, 2, 1, 1, 3, 6, 1, 2, 6, 2, 4, 1, 9, 2, 6, 1, 8, 3, 6, 2, 4, 1, 1, 6, 2, 6, 4, 6, 2, 3, 9, 2, 9, 1, 1, 1, 2, 9, 2, 6, 2, 16, 9, 2, 2, 6, 2, 4, 2, 6, 1, 4, 1, 9, 2, 4, 1, 6, 9, 2, 4, 1, 3, 6, 2, 6, 6, 2, 1, 1, 2, 2, 2, 6, 2, 6, 2, 9, 2, 4, 9, 4, 1, 2, 1, 6, 1, 4, 1, 6, 4, 1, 1, 2, 2, 2, 9, 4, 1, 4, 2, 6, 1, 1, 4, 1, 4, 1, 2, 6, 1, 1, 4, 9, 2, 6, 1, 1, 2, 2, 2, 2, 2, 3, 9, 2, 2, 2, 2, 4, 1, 6, 1, 2, 6, 4, 1, 3, 3, 3, 6, 1, 6, 1, 4, 2, 4, 1, 2, 2, 6, 3, 4, 1, 1, 6, 6, 2, 6, 2, 4, 1, 1, 2, 6, 1, 4, 1, 2, 2, 2, 2, 6, 1, 1, 1, 2, 2, 6, 1, 3, 3, 2, 3, 4, 6, 1, 2, 1, 9, 2, 2, 2, 3, 3, 2, 4, 1, 1, 2, 1, 4, 3, 2, 2, 4, 1, 2, 2, 9, 1, 2, 2, 2, 3, 6, 1, 2, 9, 1, 2, 2, 3, 3, 3, 6, 2, 6, 1, 1, 9, 1, 4, 3, 9, 2, 4, 1, 2, 1, 2, 3, 4, 3, 3, 2, 3, 2, 2, 6, 1, 1, 1, 2, 2, 6, 1, 4, 1, 1, 2, 2, 6, 1, 4, 6, 1, 1, 1, 2, 2, 1, 4, 1, 2, 4, 1, 4, 4, 1, 3, 2, 6, 1, 1, 1, 1, 2, 2, 1, 4, 9, 1, 1, 1, 2, 3, 2, 6, 1, 2, 1, 6, 3, 2, 2, 6, 1, 1, 2, 3, 1, 3, 2, 6, 1, 4, 1, 6, 1, 4, 4, 1, 6, 2, 2, 2, 2, 1, 2, 4, 4, 2, 4, 1, 1, 1, 2, 2, 4, 1, 2, 2, 2, 1, 4, 2, 4, 4, 1, 4, 6, 1, 1, 4, 1, 2, 2, 1, 4, 1, 2, 1, 2, 1, 4, 1, 2, 6, 1, 1, 2, 6, 1, 6, 1, 3, 6, 6, 1, 9, 2, 4, 1, 4, 1, 2, 3, 1, 2, 6, 2, 1, 1, 3, 2, 6, 1, 1, 4, 1, 1, 2, 1, 3, 3, 8, 3, 2, 4, 1, 4, 2, 2, 4, 2, 0, 1, 0, 0, 3, 11, 2, 2, 2, 6, 1, 9, 4, 4, 1, 6, 1, 1, 6, 1, 4, 9, 2, 6, 2, 4, 1, 4, 3, 12, 1, 1, 2, 6, 2, 6, 9, 1, 1, 2, 6, 1, 1, 6, 2, 6, 6, 2, 2, 6, 1, 6, 1, 4, 1, 2, 6, 6, 1, 2, 9, 4, 1, 4, 2, 2, 3, 3, 9, 2, 3, 4, 9, 4, 1, 2, 2, 4, 1, 4, 1, 1, 2, 9, 2, 3, 6, 1, 1, 2, 6, 3, 9, 4, 6, 2, 3, 3]
# # # x_li4 =[1000000, 49, 16, 9, 2, 4, 4, 9, 4, 1, 1, 1, 1, 1, 1, 1, 2, 9, 1, 1, 1, 1, 4, 6, 1, 9, 4, 1, 6, 1, 6, 2, 1, 9, 2, 9, 2, 6, 6, 1, 4, 1, 6, 2, 6, 1, 4, 1, 4, 1, 4, 1, 9, 1, 1, 1, 1, 4, 1, 1, 1, 6, 6, 2, 2, 9, 2, 3, 6, 9, 2, 4, 6, 2, 2, 6, 2, 4, 6, 1, 2, 3, 6, 6, 9, 2, 3, 9, 2, 6, 1, 1, 4, 1, 1, 2, 4, 3, 6, 2, 3, 4, 6, 2, 4, 4, 6, 1, 1, 1, 2, 9, 4, 1, 1, 4, 2, 2, 4, 2, 6, 6, 1, 1, 6, 9, 2, 9, 4, 1, 4, 4, 1, 6, 1, 2, 6, 9, 1, 2, 6, 2, 2, 1, 2, 6, 6, 1, 6, 1, 6, 1, 2, 3, 6, 2, 6, 1, 1, 1, 1, 6, 1, 4, 2, 4, 4, 1, 6, 2, 2, 4, 6, 3, 6, 1, 6, 1, 2, 2, 9, 2, 4, 6, 4, 6, 1, 1, 1, 6, 4, 2, 9, 1, 1, 6, 2, 3, 3, 6, 2, 4, 1, 1, 2, 6, 6, 1, 2, 2, 4, 6, 1, 4, 1, 2, 4, 1, 1, 2, 2, 2, 4, 6, 2, 3, 6, 2, 6, 2, 2, 6, 1, 2, 2, 4, 1, 1, 4, 1, 2, 3, 2, 3, 6, 3, 3, 6, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 6, 2, 6, 1, 4, 3, 1, 4, 6, 1, 6, 2, 2, 4, 4, 1, 4, 2, 4, 2, 4, 1, 6, 3, 9, 2, 2, 3, 2, 3, 1, 1, 2, 9, 2, 2, 9, 1, 1, 1, 4, 1, 4, 2, 1, 4, 9, 1, 4, 2, 1, 3, 2, 6, 3, 3, 4, 2, 4, 1, 1, 2, 3, 4, 1, 1, 6, 1, 3, 4, 3, 3, 3, 2, 4, 1, 6, 1, 2, 3, 2, 6, 3, 6, 1, 2, 3, 6, 2, 6, 1, 4, 6, 1, 2, 2, 3, 2, 2, 3, 4, 2, 3, 2, 3, 9, 2, 4, 1, 1, 1, 3, 9, 4, 1, 1, 3, 6, 1, 2, 2, 2, 2, 3, 4, 1, 2, 2, 2, 2, 1, 2, 4, 1, 1, 1, 3, 6, 1, 3, 4, 1, 1, 4, 1, 1, 2, 4, 1, 4, 4, 1, 1, 3, 3, 3, 3, 3, 2, 3, 6, 1, 2, 1, 8, 1, 2, 4, 2, 4, 12, 2, 2, 4, 2, 2, 3, 1, 4, 1, 1, 6, 1, 6, 2, 6, 1, 3, 8, 1, 1, 3, 2, 3, 8, 1, 12, 1, 1, 1, 1, 8, 0, 7, 2, 0, 1, 0, 0, 6, 999999, 49, 12, 3, 6, 2, 6, 2, 4, 1, 4, 1, 4, 9, 1, 1, 1, 1, 1, 1, 1, 2, 9, 4, 1, 1, 1, 2, 6, 1, 2, 9, 2, 3, 9, 2, 4, 1, 2, 2, 6, 2, 6, 6, 1, 2, 6, 2, 9, 2, 6, 1, 2, 6, 1, 4, 1, 1, 6, 2, 6, 1, 1, 2, 2, 4, 1, 1, 2, 2, 2, 6, 2, 6, 2, 6, 1, 1, 6, 2, 9, 4, 1, 9, 2, 4, 2, 4, 1, 2, 6, 1, 4, 1, 4, 3, 9, 2, 9, 2, 2, 2, 6, 2, 2, 9, 2, 2, 4, 9, 2, 6, 1, 1, 2, 1, 4, 4, 1, 2, 6, 2, 4, 1, 6, 1, 1, 4, 1, 4, 1, 4, 1, 3, 9, 2, 2, 4, 1, 2, 6, 9, 2, 2, 6, 2, 1, 3, 3, 2, 3, 3, 9, 4, 1, 2, 6, 4, 1, 6, 2, 2, 6, 1, 2, 6, 6, 1, 2, 4, 9, 1, 1, 2, 2, 4, 6, 2, 4, 1, 9, 1, 1, 1, 1, 2, 4, 1, 1, 6, 3, 3, 3, 6, 2, 4, 2, 2, 9, 2, 2, 4, 1, 1, 6, 9, 2, 4, 6, 2, 2, 2, 9, 4, 2, 2, 4, 6, 9, 2, 3, 3, 6, 1, 1, 1, 2, 6, 2, 9, 9, 1, 1, 1, 4, 1, 3, 1, 2, 4, 3, 2, 9, 1, 1, 2, 3, 3, 4, 1, 1, 3, 6, 6, 1, 1, 3, 2, 6, 2, 6, 2, 3, 6, 3, 3, 2, 2, 6, 2, 2, 9, 1, 1, 3, 2, 2, 6, 2, 3, 6, 3, 9, 2, 3, 2, 3, 4, 1, 2, 1, 6, 1, 1, 2, 3, 3, 3, 2, 3, 3, 1, 4, 1, 4, 1, 2, 4, 1, 6, 1, 1, 6, 4, 2, 6, 1, 3, 9, 2, 1, 1, 3, 1, 2, 4, 1, 1, 3, 4, 6, 1, 2, 1, 9, 1, 2, 2, 3, 2, 6, 1, 2, 6, 6, 1, 1, 1, 1, 2, 1, 4, 1, 2, 6, 2, 3, 4, 4, 9, 6, 1, 2, 1, 3, 2, 2, 1, 1, 6, 3, 1, 4, 4, 4, 1, 6, 1, 3, 3, 3, 3, 3, 1, 6, 1, 3, 6, 1, 3, 3, 1, 4, 6, 1, 2, 2, 6, 1, 2, 4, 1, 3, 3, 6, 2, 3, 4, 1, 1, 2, 4, 2, 4, 8, 1, 1, 3, 3, 8, 1, 1, 1, 6, 1, 2, 2, 2, 6, 1, 6, 1, 4, 8, 2, 6, 1, 1, 1, 2, 4, 12, 4, 1, 2, 2, 3, 8, 1, 1, 2, 2, 2, 6, 4, 1, 0, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0, 8, 9, 4, 1, 1, 1, 1, 1, 1, 1, 2, 3, 6, 1, 9, 4, 1, 9, 2, 9, 2, 6, 6, 1, 4, 1, 4, 1, 4, 1, 9, 1, 1, 1, 1, 4, 1, 1, 1, 6, 6, 2, 2, 9, 2, 9, 2, 4, 6, 2, 2, 6, 2, 4, 6, 1, 2, 3, 6, 6, 9, 2, 6, 1]
# # # x_li5 =[1000000, 64, 16, 6, 1, 2, 2, 6, 1, 8, 2, 6, 12, 6, 1, 4, 6, 1, 4, 2, 6, 2, 9, 4, 1, 9, 2, 9, 4, 1, 3, 6, 1, 9, 4, 6, 2, 4, 9, 4, 1, 6, 6, 9, 2, 2, 6, 1, 4, 6, 1, 2, 2, 2, 6, 6, 9, 4, 1, 9, 2, 9, 2, 9, 2, 9, 4, 1, 4, 2, 12, 3, 6, 2, 4, 1, 2, 3, 4, 1, 1, 9, 1, 1, 2, 9, 2, 4, 2, 4, 1, 1, 6, 2, 1, 9, 2, 4, 1, 4, 1, 3, 4, 1, 3, 9, 2, 3, 6, 1, 9, 1, 1, 1, 1, 2, 2, 2, 6, 2, 4, 1, 3, 6, 1, 3, 4, 1, 1, 3, 8, 2, 8, 2, 2, 3, 3, 3, 3, 6, 9, 2, 4, 1, 6, 2, 4, 4, 1, 2, 2, 4, 1, 1, 1, 1, 6, 1, 9, 1, 1, 2, 2, 9, 2, 6, 1, 4, 4, 3, 3, 6, 2, 3, 9, 2, 2, 9, 2, 2, 4, 2, 2, 6, 4, 6, 1, 1, 1, 6, 2, 2, 6, 1, 1, 1, 9, 1, 1, 4, 1, 2, 2, 4, 4, 6, 1, 6, 3, 3, 1, 2, 2, 9, 2, 6, 1, 2, 6, 1, 4, 2, 1, 2, 2, 2, 4, 2, 4, 1, 2, 4, 1, 4, 4, 2, 2, 4, 2, 4, 1, 4, 2, 4, 1, 1, 1, 1, 4, 2, 3, 6, 2, 4, 1, 6, 3, 6, 2, 1, 1, 9, 2, 4, 1, 2, 2, 2, 4, 2, 3, 1, 3, 3, 3, 9, 1, 1, 1, 6, 1, 4, 1, 2, 1, 2, 2, 3, 6, 1, 3, 2, 3, 6, 2, 2, 2, 3, 4, 1, 1, 4, 2, 4, 1, 6, 1, 3, 3, 2, 6, 1, 9, 2, 2, 3, 4, 1, 2, 2, 4, 1, 2, 2, 9, 1, 1, 1, 3, 6, 1, 3, 4, 1, 2, 4, 1, 2, 2, 1, 9, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 2, 2, 2, 6, 1, 4, 1, 1, 4, 4, 1, 2, 6, 3, 6, 9, 1, 1, 2, 4, 2, 6, 3, 1, 6, 2, 1, 3, 2, 6, 2, 3, 3, 9, 2, 2, 2, 3, 3, 3, 3, 1, 3, 9, 4, 3, 9, 1, 1, 3, 4, 1, 2, 3, 3, 4, 1, 2, 2, 3, 3, 3, 4, 4, 1, 2, 3, 3, 3, 6, 8, 1, 1, 1, 1, 4, 1, 4, 6, 12, 3, 4, 3, 8, 2, 4, 4, 4, 1, 1, 4, 0, 3, 0, 2, 0, 999999, 56, 20, 6, 1, 4, 1, 2, 3, 6, 2, 4, 9, 4, 1, 9, 2, 6, 2, 6, 1, 1, 1, 2, 4, 6, 6, 1, 9, 4, 2, 6, 3, 2, 3, 6, 1, 1, 4, 6, 1, 9, 2, 2, 6, 1, 4, 1, 1, 1, 1, 2, 6, 2, 6, 2, 9, 4, 4, 1, 2, 3, 6, 1, 6, 1, 9, 1, 1, 2, 3, 6, 1, 1, 1, 4, 1, 1, 2, 2, 2, 4, 3, 6, 3, 6, 1, 2, 4, 1, 6, 6, 2, 9, 2, 3, 6, 6, 2, 4, 2, 9, 4, 4, 1, 6, 1, 3, 6, 2, 3, 4, 1, 9, 2, 3, 3, 6, 2, 6, 2, 6, 6, 1, 9, 2, 2, 3, 6, 1, 6, 1, 4, 1, 4, 1, 2, 9, 2, 9, 4, 3, 9, 2, 2, 9, 2, 1, 6, 1, 2, 4, 1, 6, 9, 1, 1, 1, 2, 4, 1, 2, 3, 2, 3, 4, 1, 2, 2, 4, 1, 6, 2, 3, 6, 6, 1, 1, 2, 6, 1, 9, 2, 3, 3, 6, 2, 9, 2, 2, 6, 2, 6, 1, 3, 9, 2, 2, 2, 6, 1, 1, 4, 1, 3, 3, 4, 1, 1, 1, 3, 6, 6, 6, 6, 1, 1, 1, 1, 3, 4, 1, 6, 1, 4, 1, 3, 3, 2, 2, 9, 4, 1, 2, 1, 9, 1, 9, 4, 9, 2, 2, 1, 6, 1, 3, 3, 2, 3, 3, 6, 1, 6, 1, 2, 1, 6, 1, 2, 3, 4, 3, 4, 3, 3, 1, 3, 3, 6, 1, 1, 6, 1, 2, 2, 4, 1, 1, 6, 2, 2, 9, 2, 6, 3, 3, 1, 9, 4, 4, 2, 4, 1, 6, 2, 4, 1, 2, 1, 2, 6, 2, 6, 1, 6, 1, 2, 9, 4, 1, 1, 2, 2, 1, 2, 3, 4, 4, 6, 1, 6, 2, 2, 2, 3, 2, 6, 1, 2, 6, 2, 9, 2, 3, 2, 4, 1, 9, 1, 1, 1, 2, 6, 1, 4, 1, 1, 1, 4, 2, 2, 2, 2, 6, 1, 2, 3, 6, 1, 1, 9, 1, 1, 6, 9, 1, 1, 4, 1, 6, 1, 2, 2, 9, 1, 1, 3, 9, 1, 6, 1, 2, 2, 6, 1, 6, 2, 6, 1, 2, 2, 9, 1, 1, 2, 2, 6, 1, 6, 1, 2, 3, 9, 4, 2, 1, 4, 6, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 6, 1, 3, 2, 3, 3, 6, 1, 2, 2, 2, 6, 2, 4, 1, 2, 2, 6, 1, 1, 3, 6, 1, 2, 1, 2, 6, 1, 3, 9, 2, 6, 1, 2, 1, 2, 6, 1, 9, 2, 2, 3, 3, 2, 6, 1, 3, 1, 2, 0, 1, 3, 0, 2, 4, 1, 0, 0, 3, 1, 0, 5, 1, 4, 2, 6, 2, 9, 4, 12, 3, 4, 1, 9, 4, 1, 1, 3, 6, 2, 6, 2, 4, 4, 1, 3, 1, 1, 2, 2, 6, 1, 3, 3, 8, 2, 6, 2, 9, 2, 9, 2, 3, 6, 2, 8, 3, 1, 6, 1, 2, 4, 1, 6, 2, 9, 4, 9, 1, 1, 2, 4, 1, 2, 6, 2, 4, 1, 3, 3]
# #
# # #WS2
# # # x_li1 =[1000000, 64, 15, 9, 2, 6, 9, 2, 6, 1, 8, 1, 4, 3, 8, 3, 8, 3, 12, 4, 6, 1, 9, 1, 1, 1, 1, 4, 1, 4, 3, 6, 1, 2, 6, 1, 6, 2, 4, 1, 6, 1, 9, 2, 4, 6, 1, 2, 9, 2, 2, 9, 2, 2, 3, 9, 2, 6, 2, 6, 2, 3, 9, 2, 6, 2, 9, 2, 3, 9, 4, 1, 1, 2, 2, 3, 3, 6, 6, 1, 2, 6, 1, 4, 9, 2, 4, 1, 2, 2, 4, 2, 6, 1, 2, 9, 4, 1, 9, 2, 1, 6, 2, 4, 1, 1, 1, 6, 1, 2, 4, 1, 1, 2, 4, 1, 4, 2, 9, 2, 6, 9, 2, 6, 6, 2, 6, 1, 3, 6, 2, 2, 2, 6, 1, 2, 3, 2, 6, 2, 4, 1, 3, 3, 6, 1, 2, 4, 1, 2, 4, 9, 3, 6, 1, 4, 1, 3, 6, 1, 2, 6, 2, 3, 9, 1, 4, 4, 1, 9, 2, 2, 2, 3, 6, 1, 2, 4, 6, 2, 3, 9, 4, 1, 1, 1, 6, 2, 2, 2, 2, 6, 1, 4, 1, 1, 2, 6, 1, 4, 1, 4, 2, 4, 1, 3, 2, 2, 6, 2, 6, 1, 2, 9, 2, 9, 4, 6, 2, 3, 9, 4, 1, 6, 1, 4, 1, 9, 1, 2, 3, 6, 1, 4, 2, 2, 6, 2, 2, 3, 3, 4, 9, 2, 6, 4, 3, 4, 4, 1, 1, 2, 2, 6, 1, 4, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2, 6, 2, 4, 1, 4, 4, 9, 2, 1, 2, 2, 4, 1, 4, 2, 4, 1, 1, 1, 2, 2, 9, 4, 3, 6, 1, 3, 4, 1, 1, 2, 4, 1, 2, 4, 1, 2, 4, 4, 2, 2, 4, 1, 2, 2, 3, 2, 2, 6, 1, 2, 2, 2, 9, 2, 6, 1, 1, 3, 3, 6, 2, 4, 1, 1, 4, 6, 2, 4, 1, 1, 6, 2, 2, 1, 2, 9, 2, 2, 3, 1, 6, 1, 1, 2, 4, 1, 1, 2, 6, 1, 1, 6, 2, 3, 2, 2, 6, 1, 1, 2, 1, 2, 9, 1, 1, 1, 2, 3, 3, 6, 1, 1, 6, 2, 6, 1, 3, 6, 1, 3, 2, 3, 2, 6, 1, 3, 2, 2, 6, 2, 2, 6, 1, 1, 2, 2, 3, 3, 2, 3, 2, 2, 9, 2, 1, 4, 1, 3, 2, 2, 3, 2, 2, 2, 3, 3, 6, 2, 2, 6, 1, 3, 6, 1, 1, 8, 2, 3, 8, 1, 4, 1, 12, 1, 2, 4, 4, 1, 0, 0, 3, 1, 0, 999999, 56, 15, 8, 1, 4, 12, 4, 6, 6, 1, 1, 1, 6, 1, 9, 4, 1, 3, 3, 9, 1, 1, 1, 1, 1, 1, 1, 2, 9, 1, 4, 1, 6, 1, 1, 2, 9, 2, 6, 2, 2, 6, 2, 2, 3, 6, 1, 2, 9, 1, 1, 4, 1, 4, 9, 4, 6, 2, 6, 2, 6, 2, 1, 9, 2, 6, 9, 4, 1, 9, 4, 1, 1, 2, 6, 2, 6, 2, 4, 1, 3, 6, 2, 2, 6, 2, 2, 9, 2, 4, 9, 2, 2, 3, 6, 1, 1, 6, 1, 6, 6, 2, 2, 6, 2, 4, 3, 9, 1, 1, 1, 2, 9, 2, 9, 1, 3, 6, 1, 6, 6, 2, 9, 4, 6, 1, 4, 1, 6, 1, 4, 4, 4, 6, 1, 6, 1, 4, 4, 1, 2, 6, 2, 2, 2, 4, 1, 6, 6, 2, 1, 4, 6, 2, 2, 6, 6, 1, 9, 2, 3, 4, 1, 4, 1, 6, 2, 2, 4, 2, 4, 4, 4, 1, 3, 3, 6, 1, 2, 2, 9, 4, 6, 9, 2, 2, 2, 4, 1, 1, 1, 1, 3, 1, 4, 6, 1, 4, 1, 3, 9, 2, 4, 1, 9, 4, 1, 1, 6, 1, 3, 9, 1, 1, 1, 2, 2, 4, 3, 2, 2, 6, 3, 3, 6, 2, 2, 4, 2, 4, 1, 3, 1, 6, 2, 2, 2, 2, 6, 1, 1, 4, 4, 1, 4, 2, 2, 6, 6, 2, 2, 2, 4, 1, 2, 4, 2, 2, 2, 3, 6, 2, 6, 3, 6, 1, 6, 1, 4, 1, 2, 6, 1, 1, 2, 6, 1, 3, 6, 2, 2, 3, 6, 1, 1, 1, 2, 4, 1, 4, 1, 4, 2, 4, 1, 9, 2, 6, 1, 4, 1, 9, 4, 1, 1, 1, 1, 2, 3, 4, 6, 4, 6, 1, 3, 3, 3, 6, 1, 6, 1, 1, 4, 1, 6, 1, 3, 2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 1, 6, 1, 1, 1, 1, 3, 2, 2, 6, 1, 2, 6, 1, 1, 6, 1, 2, 3, 4, 1, 9, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 4, 2, 2, 4, 1, 1, 4, 3, 6, 1, 1, 6, 2, 3, 6, 2, 6, 1, 3, 6, 1, 1, 3, 4, 3, 3, 9, 1, 4, 1, 3, 6, 1, 3, 1, 3, 9, 2, 2, 4, 2, 2, 1, 4, 1, 1, 3, 1, 1, 3, 2, 6, 1, 2, 6, 1, 4, 1, 1, 1, 3, 3, 2, 2, 9, 2, 3, 3, 6, 1, 4, 1, 12, 6, 3, 6, 1, 1, 12, 3, 2, 1, 3, 2, 10, 2, 2, 4, 1, 15, 2, 1, 1, 1, 2, 1, 4, 0, 3, 0, 2, 0, 2, 12, 4, 6, 1, 9, 1, 1, 1, 1, 4, 1, 4, 3, 6, 1, 2, 6, 1, 6, 2, 4, 1, 6, 1, 9, 2, 4, 3, 9, 2, 2, 9, 2, 2, 3, 3, 6, 2, 6, 2, 3, 3, 9, 2, 6, 2, 9, 2, 1, 2, 9, 2, 3, 9, 4, 1, 1, 2, 2, 3, 6, 1, 1, 9, 2, 3, 6, 1, 1, 9, 2, 9, 4, 1]
# # # x_li2 =[1000000, 56, 12, 1, 2, 2, 6, 1, 3, 3, 9, 2, 12, 4, 9, 4, 1, 6, 3, 16, 2, 2, 2, 4, 1, 4, 6, 1, 3, 4, 2, 6, 1, 6, 3, 6, 2, 4, 1, 2, 3, 6, 2, 6, 1, 9, 1, 1, 2, 6, 2, 2, 4, 4, 1, 6, 2, 6, 1, 1, 6, 9, 4, 1, 4, 4, 1, 2, 4, 1, 4, 1, 6, 2, 2, 6, 1, 2, 2, 6, 2, 3, 3, 2, 6, 1, 6, 1, 2, 4, 3, 3, 6, 1, 1, 4, 9, 2, 4, 6, 1, 2, 4, 1, 1, 1, 6, 2, 6, 2, 3, 6, 3, 6, 6, 1, 2, 6, 1, 4, 6, 2, 2, 3, 6, 1, 6, 1, 4, 1, 3, 3, 6, 1, 2, 6, 6, 2, 4, 1, 4, 1, 2, 4, 2, 2, 6, 1, 1, 1, 1, 4, 1, 4, 2, 6, 6, 6, 6, 2, 6, 2, 6, 2, 1, 2, 6, 1, 2, 2, 6, 1, 4, 1, 9, 1, 1, 1, 4, 1, 6, 1, 2, 6, 2, 6, 1, 4, 4, 2, 4, 1, 3, 6, 1, 2, 6, 3, 6, 1, 6, 2, 2, 6, 1, 1, 6, 1, 1, 1, 1, 1, 9, 1, 1, 4, 6, 1, 2, 3, 4, 1, 1, 3, 3, 4, 6, 2, 1, 4, 1, 3, 6, 1, 4, 1, 1, 1, 4, 1, 4, 1, 4, 2, 2, 4, 1, 4, 1, 4, 1, 2, 2, 1, 9, 4, 1, 3, 4, 3, 2, 2, 4, 3, 6, 1, 2, 2, 2, 1, 2, 3, 3, 2, 2, 6, 1, 2, 4, 6, 2, 2, 6, 6, 1, 2, 4, 9, 4, 4, 6, 1, 4, 1, 3, 6, 1, 1, 1, 1, 1, 2, 3, 2, 2, 1, 4, 1, 2, 4, 2, 4, 1, 4, 1, 2, 1, 4, 1, 4, 1, 2, 2, 3, 6, 1, 4, 1, 1, 2, 4, 1, 4, 3, 4, 1, 6, 2, 4, 2, 1, 3, 1, 6, 1, 1, 2, 6, 2, 1, 2, 2, 4, 1, 1, 1, 1, 2, 3, 4, 1, 1, 4, 1, 2, 4, 1, 6, 1, 1, 6, 1, 1, 1, 1, 2, 4, 6, 1, 1, 2, 2, 4, 1, 2, 1, 1, 4, 4, 4, 2, 2, 2, 2, 1, 2, 6, 1, 9, 2, 2, 2, 3, 3, 1, 2, 3, 2, 6, 6, 4, 6, 2, 2, 2, 6, 2, 6, 6, 4, 6, 1, 3, 3, 1, 3, 1, 4, 6, 1, 4, 1, 1, 4, 8, 1, 8, 8, 2, 1, 4, 2, 5, 0, 4, 0, 3, 0, 999999, 49, 12, 3, 6, 1, 2, 6, 12, 2, 9, 2, 9, 2, 8, 6, 2, 2, 8, 2, 9, 1, 1, 2, 6, 2, 9, 2, 6, 2, 2, 9, 1, 1, 3, 9, 2, 2, 9, 2, 2, 6, 1, 2, 6, 2, 4, 8, 1, 2, 12, 6, 9, 4, 1, 4, 1, 2, 4, 6, 6, 2, 4, 1, 4, 1, 4, 2, 8, 6, 1, 1, 9, 1, 1, 4, 3, 8, 2, 6, 6, 2, 6, 6, 2, 6, 1, 4, 6, 16, 3, 8, 2, 2, 9, 4, 1, 6, 6, 1, 4, 1, 9, 1, 1, 1, 6, 3, 3, 8, 8, 2, 2, 6, 1, 1, 9, 4, 1, 6, 2, 6, 1, 4, 1, 2, 2, 4, 1, 1, 9, 2, 9, 2, 2, 1, 4, 6, 9, 1, 1, 2, 6, 1, 1, 3, 6, 1, 4, 6, 1, 1, 4, 6, 6, 2, 6, 9, 2, 4, 3, 2, 9, 2, 4, 1, 3, 6, 2, 6, 1, 9, 2, 4, 1, 9, 2, 4, 1, 3, 9, 6, 2, 1, 6, 2, 6, 1, 2, 3, 9, 4, 4, 1, 2, 2, 4, 1, 4, 3, 9, 2, 3, 9, 2, 6, 1, 6, 9, 4, 1, 3, 6, 6, 6, 1, 4, 9, 1, 1, 4, 1, 1, 6, 1, 1, 2, 3, 9, 4, 1, 2, 2, 4, 2, 6, 1, 4, 1, 2, 4, 1, 1, 4, 3, 1, 9, 2, 6, 1, 6, 2, 2, 1, 1, 2, 4, 2, 6, 6, 3, 3, 4, 3, 3, 2, 6, 2, 3, 3, 3, 4, 1, 1, 6, 2, 2, 6, 1, 2, 2, 4, 1, 1, 4, 3, 2, 1, 2, 2, 3, 2, 4, 1, 4, 1, 1, 4, 1, 4, 1, 2, 4, 1, 6, 2, 2, 3, 9, 2, 6, 2, 2, 2, 4, 1, 1, 6, 3, 6, 2, 1, 9, 2, 6, 1, 6, 1, 3, 9, 2, 2, 2, 3, 2, 6, 1, 4, 1, 2, 2, 2, 9, 1, 1, 4, 2, 4, 2, 2, 2, 2, 2, 1, 1, 3, 6, 2, 2, 6, 2, 1, 6, 2, 3, 4, 6, 1, 6, 2, 3, 3, 3, 6, 1, 3, 4, 1, 4, 4, 1, 2, 6, 2, 1, 4, 2, 2, 4, 1, 2, 4, 1, 2, 2, 6, 1, 2, 3, 4, 2, 3, 6, 1, 1, 3, 2, 4, 1, 2, 6, 2, 6, 2, 8, 3, 6, 1, 3, 2, 8, 2, 1, 4, 2, 2, 3, 3, 4, 4, 3, 4, 8, 1, 2, 2, 4, 1, 0, 0, 3, 1, 2, 2, 0, 1, 3, 2, 11, 1, 2, 2, 6, 1, 3, 3, 9, 2, 12, 4, 9, 4, 1, 6, 3, 16, 2, 2, 2, 4, 1, 4, 6, 1, 3, 4, 4, 1, 1, 6, 1, 6, 3, 6, 2, 4, 1, 2, 3, 6, 2, 6, 1, 9, 1, 1, 2, 6, 2, 2, 6, 2, 6, 1, 1, 6, 6, 9, 4, 1, 4, 4, 1, 2, 4, 1, 4, 1, 6, 2, 2, 6, 1, 2, 2, 6, 2, 6, 2, 3, 3, 2, 6, 1, 6, 1, 2, 4, 3, 3, 2, 4, 3, 6, 1, 1, 4, 9, 2, 6, 2, 6]
# # # x_li3 =[1000000, 56, 20, 12, 2, 1, 4, 1, 4, 12, 3, 4, 3, 4, 8, 2, 2, 6, 1, 4, 1, 3, 3, 9, 2, 1, 4, 6, 2, 6, 2, 2, 2, 16, 9, 2, 2, 6, 2, 9, 4, 1, 1, 1, 4, 1, 2, 3, 6, 6, 1, 2, 4, 4, 9, 2, 6, 1, 3, 6, 6, 6, 1, 6, 6, 3, 8, 2, 12, 6, 2, 2, 4, 1, 4, 4, 9, 2, 9, 2, 6, 1, 6, 2, 6, 3, 2, 4, 8, 2, 2, 2, 3, 4, 1, 1, 3, 9, 4, 3, 8, 2, 1, 1, 1, 4, 4, 1, 3, 9, 2, 9, 2, 2, 2, 2, 6, 2, 3, 6, 1, 3, 6, 2, 9, 4, 4, 6, 2, 4, 6, 2, 4, 2, 6, 2, 3, 6, 1, 2, 4, 3, 3, 1, 6, 2, 6, 1, 2, 6, 1, 2, 9, 2, 1, 4, 1, 1, 2, 2, 4, 6, 6, 1, 3, 6, 2, 4, 2, 4, 1, 3, 9, 4, 4, 1, 1, 9, 2, 9, 4, 9, 2, 4, 3, 3, 9, 2, 4, 6, 2, 1, 6, 1, 1, 2, 6, 1, 3, 6, 1, 3, 3, 2, 3, 3, 3, 9, 2, 2, 3, 6, 1, 3, 2, 2, 4, 2, 9, 1, 2, 6, 1, 6, 3, 9, 2, 3, 4, 3, 6, 1, 4, 4, 3, 1, 4, 6, 9, 4, 1, 9, 1, 4, 1, 2, 3, 2, 3, 3, 6, 1, 4, 4, 2, 3, 9, 2, 4, 1, 4, 1, 1, 6, 2, 3, 4, 1, 1, 6, 2, 3, 2, 3, 3, 6, 1, 4, 1, 9, 2, 9, 2, 1, 3, 3, 9, 2, 9, 1, 1, 4, 6, 2, 2, 2, 4, 1, 2, 2, 2, 2, 2, 3, 6, 2, 2, 6, 2, 3, 4, 1, 4, 3, 2, 2, 3, 6, 1, 6, 1, 3, 4, 1, 6, 3, 3, 1, 6, 2, 9, 2, 6, 1, 4, 1, 2, 4, 1, 2, 6, 1, 2, 6, 1, 4, 6, 2, 4, 3, 2, 2, 6, 1, 6, 1, 6, 1, 3, 3, 3, 4, 3, 2, 4, 1, 4, 1, 1, 1, 2, 2, 2, 6, 2, 2, 2, 4, 1, 1, 3, 6, 3, 4, 6, 2, 6, 1, 3, 6, 2, 4, 1, 3, 1, 2, 4, 1, 1, 3, 2, 2, 3, 3, 3, 6, 1, 3, 2, 1, 2, 9, 2, 4, 1, 3, 4, 8, 3, 4, 1, 3, 3, 8, 1, 1, 1, 8, 2, 6, 1, 2, 1, 6, 1, 0, 0, 5, 1, 0, 999999, 56, 25, 12, 2, 2, 6, 1, 2, 6, 1, 6, 1, 9, 1, 1, 6, 2, 9, 4, 1, 4, 4, 1, 4, 3, 6, 1, 6, 1, 1, 2, 3, 9, 6, 2, 9, 2, 6, 1, 1, 4, 1, 2, 2, 6, 9, 2, 1, 1, 3, 6, 1, 2, 6, 2, 4, 1, 9, 2, 6, 1, 8, 3, 6, 2, 4, 1, 1, 6, 2, 6, 4, 6, 2, 3, 9, 2, 9, 1, 1, 1, 2, 9, 2, 6, 2, 16, 9, 2, 2, 6, 2, 4, 2, 6, 1, 4, 1, 9, 2, 4, 1, 6, 9, 2, 4, 1, 3, 6, 2, 6, 6, 2, 1, 1, 2, 2, 2, 6, 2, 6, 2, 9, 2, 4, 9, 4, 1, 2, 1, 6, 1, 4, 1, 6, 4, 1, 1, 2, 2, 2, 9, 4, 1, 4, 2, 6, 1, 1, 4, 1, 4, 1, 2, 6, 1, 1, 4, 9, 2, 6, 1, 1, 2, 2, 2, 2, 2, 3, 9, 2, 2, 2, 2, 4, 1, 6, 1, 2, 6, 4, 1, 3, 3, 3, 6, 1, 6, 1, 4, 2, 4, 1, 2, 2, 6, 3, 4, 1, 1, 6, 6, 2, 6, 2, 4, 1, 1, 2, 6, 1, 4, 1, 2, 2, 2, 2, 6, 1, 1, 1, 2, 2, 6, 1, 3, 3, 2, 3, 4, 6, 1, 2, 1, 9, 2, 2, 2, 3, 3, 2, 4, 1, 1, 2, 1, 4, 3, 2, 2, 4, 1, 2, 2, 9, 1, 2, 2, 2, 3, 6, 1, 2, 9, 1, 2, 2, 3, 3, 3, 6, 2, 6, 1, 1, 9, 1, 4, 3, 9, 2, 4, 1, 2, 1, 2, 3, 4, 3, 3, 2, 3, 2, 2, 6, 1, 1, 1, 2, 2, 6, 1, 4, 1, 1, 2, 2, 6, 1, 4, 6, 1, 1, 1, 2, 2, 1, 4, 1, 2, 4, 1, 4, 4, 1, 3, 2, 6, 1, 1, 1, 1, 2, 2, 1, 4, 9, 1, 1, 1, 2, 3, 2, 6, 1, 2, 1, 6, 3, 2, 2, 6, 1, 1, 2, 3, 1, 3, 2, 6, 1, 4, 1, 6, 1, 4, 4, 1, 6, 2, 2, 2, 2, 1, 2, 4, 4, 2, 4, 1, 1, 1, 2, 2, 4, 1, 2, 2, 2, 1, 4, 2, 4, 4, 1, 4, 6, 1, 1, 4, 1, 2, 2, 1, 4, 1, 2, 1, 2, 1, 4, 1, 2, 6, 1, 1, 2, 6, 1, 6, 1, 3, 6, 6, 1, 9, 2, 4, 1, 4, 1, 2, 3, 1, 2, 6, 2, 1, 1, 3, 2, 6, 1, 1, 4, 1, 1, 2, 1, 3, 3, 8, 3, 2, 4, 1, 4, 2, 2, 4, 2, 0, 1, 0, 0, 3, 11, 2, 2, 2, 6, 1, 9, 4, 4, 1, 6, 1, 1, 6, 1, 4, 9, 2, 6, 2, 4, 1, 4, 3, 12, 1, 1, 2, 6, 2, 6, 9, 1, 1, 2, 6, 1, 1, 6, 2, 6, 6, 2, 2, 6, 1, 6, 1, 4, 1, 2, 6, 6, 1, 2, 9, 4, 1, 4, 2, 2, 3, 3, 9, 2, 3, 4, 9, 4, 1, 2, 2, 4, 1, 4, 1, 1, 2, 9, 2, 3, 6, 1, 1, 2, 6, 3, 9, 4, 6, 2, 3, 3]
# # # x_li4 =[1000000, 49, 16, 9, 2, 4, 4, 9, 4, 1, 1, 1, 1, 1, 1, 1, 2, 9, 1, 1, 1, 1, 4, 6, 1, 9, 4, 1, 6, 1, 6, 2, 1, 9, 2, 9, 2, 6, 6, 1, 4, 1, 6, 2, 6, 1, 4, 1, 4, 1, 4, 1, 9, 1, 1, 1, 1, 4, 1, 1, 1, 6, 6, 2, 2, 9, 2, 3, 6, 9, 2, 4, 6, 2, 2, 6, 2, 4, 6, 1, 2, 3, 6, 6, 9, 2, 3, 9, 2, 6, 1, 1, 4, 1, 1, 2, 4, 3, 6, 2, 3, 4, 6, 2, 4, 4, 6, 1, 1, 1, 2, 9, 4, 1, 1, 4, 2, 2, 4, 2, 6, 6, 1, 1, 6, 9, 2, 9, 4, 1, 4, 4, 1, 6, 1, 2, 6, 9, 1, 2, 6, 2, 2, 1, 2, 6, 6, 1, 6, 1, 6, 1, 2, 3, 6, 2, 6, 1, 1, 1, 1, 6, 1, 4, 2, 4, 4, 1, 6, 2, 2, 4, 6, 3, 6, 1, 6, 1, 2, 2, 9, 2, 4, 6, 4, 6, 1, 1, 1, 6, 4, 2, 9, 1, 1, 6, 2, 3, 3, 6, 2, 4, 1, 1, 2, 6, 6, 1, 2, 2, 4, 6, 1, 4, 1, 2, 4, 1, 1, 2, 2, 2, 4, 6, 2, 3, 6, 2, 6, 2, 2, 6, 1, 2, 2, 4, 1, 1, 4, 1, 2, 3, 2, 3, 6, 3, 3, 6, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 6, 2, 6, 1, 4, 3, 1, 4, 6, 1, 6, 2, 2, 4, 4, 1, 4, 2, 4, 2, 4, 1, 6, 3, 9, 2, 2, 3, 2, 3, 1, 1, 2, 9, 2, 2, 9, 1, 1, 1, 4, 1, 4, 2, 1, 4, 9, 1, 4, 2, 1, 3, 2, 6, 3, 3, 4, 2, 4, 1, 1, 2, 3, 4, 1, 1, 6, 1, 3, 4, 3, 3, 3, 2, 4, 1, 6, 1, 2, 3, 2, 6, 3, 6, 1, 2, 3, 6, 2, 6, 1, 4, 6, 1, 2, 2, 3, 2, 2, 3, 4, 2, 3, 2, 3, 9, 2, 4, 1, 1, 1, 3, 9, 4, 1, 1, 3, 6, 1, 2, 2, 2, 2, 3, 4, 1, 2, 2, 2, 2, 1, 2, 4, 1, 1, 1, 3, 6, 1, 3, 4, 1, 1, 4, 1, 1, 2, 4, 1, 4, 4, 1, 1, 3, 3, 3, 3, 3, 2, 3, 6, 1, 2, 1, 8, 1, 2, 4, 2, 4, 12, 2, 2, 4, 2, 2, 3, 1, 4, 1, 1, 6, 1, 6, 2, 6, 1, 3, 8, 1, 1, 3, 2, 3, 8, 1, 12, 1, 1, 1, 1, 8, 0, 7, 2, 0, 1, 0, 0, 6, 999999, 49, 12, 3, 6, 2, 6, 2, 4, 1, 4, 1, 4, 9, 1, 1, 1, 1, 1, 1, 1, 2, 9, 4, 1, 1, 1, 2, 6, 1, 2, 9, 2, 3, 9, 2, 4, 1, 2, 2, 6, 2, 6, 6, 1, 2, 6, 2, 9, 2, 6, 1, 2, 6, 1, 4, 1, 1, 6, 2, 6, 1, 1, 2, 2, 4, 1, 1, 2, 2, 2, 6, 2, 6, 2, 6, 1, 1, 6, 2, 9, 4, 1, 9, 2, 4, 2, 4, 1, 2, 6, 1, 4, 1, 4, 3, 9, 2, 9, 2, 2, 2, 6, 2, 2, 9, 2, 2, 4, 9, 2, 6, 1, 1, 2, 1, 4, 4, 1, 2, 6, 2, 4, 1, 6, 1, 1, 4, 1, 4, 1, 4, 1, 3, 9, 2, 2, 4, 1, 2, 6, 9, 2, 2, 6, 2, 1, 3, 3, 2, 3, 3, 9, 4, 1, 2, 6, 4, 1, 6, 2, 2, 6, 1, 2, 6, 6, 1, 2, 4, 9, 1, 1, 2, 2, 4, 6, 2, 4, 1, 9, 1, 1, 1, 1, 2, 4, 1, 1, 6, 3, 3, 3, 6, 2, 4, 2, 2, 9, 2, 2, 4, 1, 1, 6, 9, 2, 4, 6, 2, 2, 2, 9, 4, 2, 2, 4, 6, 9, 2, 3, 3, 6, 1, 1, 1, 2, 6, 2, 9, 9, 1, 1, 1, 4, 1, 3, 1, 2, 4, 3, 2, 9, 1, 1, 2, 3, 3, 4, 1, 1, 3, 6, 6, 1, 1, 3, 2, 6, 2, 6, 2, 3, 6, 3, 3, 2, 2, 6, 2, 2, 9, 1, 1, 3, 2, 2, 6, 2, 3, 6, 3, 9, 2, 3, 2, 3, 4, 1, 2, 1, 6, 1, 1, 2, 3, 3, 3, 2, 3, 3, 1, 4, 1, 4, 1, 2, 4, 1, 6, 1, 1, 6, 4, 2, 6, 1, 3, 9, 2, 1, 1, 3, 1, 2, 4, 1, 1, 3, 4, 6, 1, 2, 1, 9, 1, 2, 2, 3, 2, 6, 1, 2, 6, 6, 1, 1, 1, 1, 2, 1, 4, 1, 2, 6, 2, 3, 4, 4, 9, 6, 1, 2, 1, 3, 2, 2, 1, 1, 6, 3, 1, 4, 4, 4, 1, 6, 1, 3, 3, 3, 3, 3, 1, 6, 1, 3, 6, 1, 3, 3, 1, 4, 6, 1, 2, 2, 6, 1, 2, 4, 1, 3, 3, 6, 2, 3, 4, 1, 1, 2, 4, 2, 4, 8, 1, 1, 3, 3, 8, 1, 1, 1, 6, 1, 2, 2, 2, 6, 1, 6, 1, 4, 8, 2, 6, 1, 1, 1, 2, 4, 12, 4, 1, 2, 2, 3, 8, 1, 1, 2, 2, 2, 6, 4, 1, 0, 0, 3, 1, 0, 0, 2, 0, 1, 0, 0, 8, 9, 4, 1, 1, 1, 1, 1, 1, 1, 2, 3, 6, 1, 9, 4, 1, 9, 2, 9, 2, 6, 6, 1, 4, 1, 4, 1, 4, 1, 9, 1, 1, 1, 1, 4, 1, 1, 1, 6, 6, 2, 2, 9, 2, 9, 2, 4, 6, 2, 2, 6, 2, 4, 6, 1, 2, 3, 6, 6, 9, 2, 6, 1]
# # # x_li5 =[1000000, 64, 16, 6, 1, 2, 2, 6, 1, 8, 2, 6, 12, 6, 1, 4, 6, 1, 4, 2, 6, 2, 9, 4, 1, 9, 2, 9, 4, 1, 3, 6, 1, 9, 4, 6, 2, 4, 9, 4, 1, 6, 6, 9, 2, 2, 6, 1, 4, 6, 1, 2, 2, 2, 6, 6, 9, 4, 1, 9, 2, 9, 2, 9, 2, 9, 4, 1, 4, 2, 12, 3, 6, 2, 4, 1, 2, 3, 4, 1, 1, 9, 1, 1, 2, 9, 2, 4, 2, 4, 1, 1, 6, 2, 1, 9, 2, 4, 1, 4, 1, 3, 4, 1, 3, 9, 2, 3, 6, 1, 9, 1, 1, 1, 1, 2, 2, 2, 6, 2, 4, 1, 3, 6, 1, 3, 4, 1, 1, 3, 8, 2, 8, 2, 2, 3, 3, 3, 3, 6, 9, 2, 4, 1, 6, 2, 4, 4, 1, 2, 2, 4, 1, 1, 1, 1, 6, 1, 9, 1, 1, 2, 2, 9, 2, 6, 1, 4, 4, 3, 3, 6, 2, 3, 9, 2, 2, 9, 2, 2, 4, 2, 2, 6, 4, 6, 1, 1, 1, 6, 2, 2, 6, 1, 1, 1, 9, 1, 1, 4, 1, 2, 2, 4, 4, 6, 1, 6, 3, 3, 1, 2, 2, 9, 2, 6, 1, 2, 6, 1, 4, 2, 1, 2, 2, 2, 4, 2, 4, 1, 2, 4, 1, 4, 4, 2, 2, 4, 2, 4, 1, 4, 2, 4, 1, 1, 1, 1, 4, 2, 3, 6, 2, 4, 1, 6, 3, 6, 2, 1, 1, 9, 2, 4, 1, 2, 2, 2, 4, 2, 3, 1, 3, 3, 3, 9, 1, 1, 1, 6, 1, 4, 1, 2, 1, 2, 2, 3, 6, 1, 3, 2, 3, 6, 2, 2, 2, 3, 4, 1, 1, 4, 2, 4, 1, 6, 1, 3, 3, 2, 6, 1, 9, 2, 2, 3, 4, 1, 2, 2, 4, 1, 2, 2, 9, 1, 1, 1, 3, 6, 1, 3, 4, 1, 2, 4, 1, 2, 2, 1, 9, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 2, 2, 2, 6, 1, 4, 1, 1, 4, 4, 1, 2, 6, 3, 6, 9, 1, 1, 2, 4, 2, 6, 3, 1, 6, 2, 1, 3, 2, 6, 2, 3, 3, 9, 2, 2, 2, 3, 3, 3, 3, 1, 3, 9, 4, 3, 9, 1, 1, 3, 4, 1, 2, 3, 3, 4, 1, 2, 2, 3, 3, 3, 4, 4, 1, 2, 3, 3, 3, 6, 8, 1, 1, 1, 1, 4, 1, 4, 6, 12, 3, 4, 3, 8, 2, 4, 4, 4, 1, 1, 4, 0, 3, 0, 2, 0, 999999, 56, 20, 6, 1, 4, 1, 2, 3, 6, 2, 4, 9, 4, 1, 9, 2, 6, 2, 6, 1, 1, 1, 2, 4, 6, 6, 1, 9, 4, 2, 6, 3, 2, 3, 6, 1, 1, 4, 6, 1, 9, 2, 2, 6, 1, 4, 1, 1, 1, 1, 2, 6, 2, 6, 2, 9, 4, 4, 1, 2, 3, 6, 1, 6, 1, 9, 1, 1, 2, 3, 6, 1, 1, 1, 4, 1, 1, 2, 2, 2, 4, 3, 6, 3, 6, 1, 2, 4, 1, 6, 6, 2, 9, 2, 3, 6, 6, 2, 4, 2, 9, 4, 4, 1, 6, 1, 3, 6, 2, 3, 4, 1, 9, 2, 3, 3, 6, 2, 6, 2, 6, 6, 1, 9, 2, 2, 3, 6, 1, 6, 1, 4, 1, 4, 1, 2, 9, 2, 9, 4, 3, 9, 2, 2, 9, 2, 1, 6, 1, 2, 4, 1, 6, 9, 1, 1, 1, 2, 4, 1, 2, 3, 2, 3, 4, 1, 2, 2, 4, 1, 6, 2, 3, 6, 6, 1, 1, 2, 6, 1, 9, 2, 3, 3, 6, 2, 9, 2, 2, 6, 2, 6, 1, 3, 9, 2, 2, 2, 6, 1, 1, 4, 1, 3, 3, 4, 1, 1, 1, 3, 6, 6, 6, 6, 1, 1, 1, 1, 3, 4, 1, 6, 1, 4, 1, 3, 3, 2, 2, 9, 4, 1, 2, 1, 9, 1, 9, 4, 9, 2, 2, 1, 6, 1, 3, 3, 2, 3, 3, 6, 1, 6, 1, 2, 1, 6, 1, 2, 3, 4, 3, 4, 3, 3, 1, 3, 3, 6, 1, 1, 6, 1, 2, 2, 4, 1, 1, 6, 2, 2, 9, 2, 6, 3, 3, 1, 9, 4, 4, 2, 4, 1, 6, 2, 4, 1, 2, 1, 2, 6, 2, 6, 1, 6, 1, 2, 9, 4, 1, 1, 2, 2, 1, 2, 3, 4, 4, 6, 1, 6, 2, 2, 2, 3, 2, 6, 1, 2, 6, 2, 9, 2, 3, 2, 4, 1, 9, 1, 1, 1, 2, 6, 1, 4, 1, 1, 1, 4, 2, 2, 2, 2, 6, 1, 2, 3, 6, 1, 1, 9, 1, 1, 6, 9, 1, 1, 4, 1, 6, 1, 2, 2, 9, 1, 1, 3, 9, 1, 6, 1, 2, 2, 6, 1, 6, 2, 6, 1, 2, 2, 9, 1, 1, 2, 2, 6, 1, 6, 1, 2, 3, 9, 4, 2, 1, 4, 6, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 6, 1, 3, 2, 3, 3, 6, 1, 2, 2, 2, 6, 2, 4, 1, 2, 2, 6, 1, 1, 3, 6, 1, 2, 1, 2, 6, 1, 3, 9, 2, 6, 1, 2, 1, 2, 6, 1, 9, 2, 2, 3, 3, 2, 6, 1, 3, 1, 2, 0, 1, 3, 0, 2, 4, 1, 0, 0, 3, 1, 0, 5, 1, 4, 2, 6, 2, 9, 4, 12, 3, 4, 1, 9, 4, 1, 1, 3, 6, 2, 6, 2, 4, 4, 1, 3, 1, 1, 2, 2, 6, 1, 3, 3, 8, 2, 6, 2, 9, 2, 9, 2, 3, 6, 2, 8, 3, 1, 6, 1, 2, 4, 1, 6, 2, 9, 4, 9, 1, 1, 2, 4, 1, 2, 6, 2, 4, 1, 3, 3]
# #
# #
# # # Use a Counter to count the number of instances in x
# # for x in [x_li1, x_li2, x_li3, x_li4, x_li5]:
# #     c = Counter(x)
# #     print(dict(c))
# # plt.bar(c.keys(), c.values())
# # plt.show()
#
# import torch
# import torch.nn as nn
#
# # cat_outputs = []
# # D = 8
# # N = 10
# # num_channels = 1
# # for i, input in enumerate([torch.randn(N, D), torch.randn(N, D)]):
# #     print('i', i)
# #     input = torch.unsqueeze(input, 1)
# #
# #     outputs = []
# #     for kern in [1, 3, D]:
# #         m = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=kern, stride=1)
# #         output = m(input)
# #         outputs.append(output)
# #
# #         print('kern', kern)
# #         print(input.shape)
# #         print(output.shape)
# #
# #     output = torch.cat(outputs, 2)
# #     cat_outputs.append(output)
# #     print('cat output', output.shape)
# #
# #
# # out_dim = D-2+1+D
# # print(out_dim)
# #
# # print('max pool')
# # cat_outputs = torch.stack(cat_outputs)
# # print('cat_outputs', cat_outputs.shape)
# # final, _ = torch.max(cat_outputs, 0)
# # final = final.view(N,-1)
# # print('final', final.shape)
# # # print(id.shape)
#
#
# # from torch_scatter import scatter_add
# #
# # src = torch.Tensor([[2, 99], [-1, 29], [-10, 8]])
# # index = torch.tensor([1.0, 1.0, 0.0])
# # out = src.new_zeros((2, 2))
# #
# # out = scatter_add(src, index, out=out, dim=0)
# #
# # print(out)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # fn = f'scatter_{label}.png'
# plt.figure()
#
# data = np.random.randint(0, 10, (1000))
#
# fig, ax = plt.subplots()
# ax.hist(data, bins=10, edgecolor='black', label="Entry 1")
#
# ax.legend()
#
# plt.show()


# from copy import deepcopy
#
# class A:
#     def __init__(self, li):
#         self.li = deepcopy(li)
#
# class B:
#     def __init__(self, li=None):
#         self.li = 5 if li is None else deepcopy(li)
#
# li = [3,4]
# li2 = [5,6]
# li3 = [1]
# a = A(li)
# b = A(li2)
# s = A(li3)
# s.li = a.li
# s.li = b.li
# print(a.li)
# print(b.li)
# s.li = a.li



#
# li = []
# li.extend([2,3])
# a = A(li)
# b = B(li)
# ad = A(deepcopy(li))
# bd = B(deepcopy(li)pip install torch torchvision
# )
#
# c = b.li
# c.append(5)
#
# li.append(4)
#
# print(a.li)
# print(b.li)
# print(ad.li)
# print(bd.li) pip install torch-scatter==1.1.2+${CUDA}

# 1.1.2
from torch_geometric.datasets import DBP15K
from utils import get_temp_path

dataset = DBP15K(root=get_temp_path(), pair='en_zh')
print(dataset)


