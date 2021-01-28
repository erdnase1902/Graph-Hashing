import pickle
import networkx as nx
import numpy as np
import ast, csv
from time import time
import matplotlib.pyplot as plt
from os.path import join, basename, exists
import os, sys, datetime
from glob import glob
from pdb_format import parseLoc

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from utils import get_data_path, create_dir_if_not_exists
import argparse
import traceback
from tqdm import tqdm


def get_pdb_data_path():
    return '/media/yba/HDD/data'


def parse_pdb_files(split_chain=True, list_of_pdb=None, raw_dir=None, graph_dir=None):
    pdbank_graph_dir = None
    pdbank_raw_dir = None
    PDB_CONECT_ERROR = 1
    RUNTIME_ERROR = 2
    if raw_dir == None or graph_dir == None:
        print("Please specify raw dir and graph dir")
        exit(1)
    else:
        pdbank_raw_dir = join(get_pdb_data_path(), raw_dir)
        pdbank_graph_dir = join(get_pdb_data_path(), graph_dir)
    if not os.path.exists(pdbank_graph_dir):
        os.mkdir(pdbank_graph_dir)
    i = 0
    t = time()
    if list_of_pdb is None:
        all_files = sorted(glob(join(pdbank_raw_dir, '*.ent')))
    else:
        all_files = sorted([join(pdbank_raw_dir, list_of_pdb)])  # [join(pdbank_raw_dir, n) for n in list_of_pdb]
    # if list_of_pdb is None:  # parse all files

    #process_time = str(datetime.datetime.now()).replace(' ', '_')
    log_path = join(get_data_path(), "PPI_Datasets", "log")  # , process_time)
    if processed_exists(log_path):
        loaded_num = read_processed_log(log_path)
    else:
        loaded_num = 0
    processed_num = 0
    for file in tqdm(all_files):
        model_list = []
        if processed_num < loaded_num:
            processed_num += 1
            print("less than loaded - ", file)
            continue
        elif loaded_num <= processed_num < int(loaded_num + 0.05 * loaded_num):
            filename = basename(file)[3:7] + '*.gexf'
            num_files = len(glob(join(pdbank_graph_dir, filename)))
            if num_files > 0 and num_files % 3 == 0:
                processed_num += 1
                print("greater than loaded and intact - ", file)
                continue


        ss_g = {} if split_chain else nx.Graph()
        aa_g = {} if split_chain else nx.Graph()
        c_g = {} if split_chain else nx.Graph()
        ss_node = {} if split_chain else []
        aa_node = {} if split_chain else []
        aa_edge_lists = {} if split_chain else []
        c_node = {} if split_chain else []
        atom_to_aa = {}
        mod_residue = {}
        num_strand = 0
        range_strings = []
        chain_start = None
        chain_end = None
        current_chain = None
        skip_lines = False
        skip_file = False
        mod_res = False
        ring_locs = []
        pdb_name = basename(file).split(".")[0]
        primes = 0
        prev_dict = {"label": None, "type": None}
        chain_finish = {}
        try:
            with open(file, "r") as f:
                for line in f:

                    line_name = line[0:6].rstrip()
                    attr_dict = {}
                    attr_dict_c = {}

                    ## TODO HETNAM entries 6anm case, how to deal with them?

                    if not skip_lines and line_name == "MODRES":
                        mod_residue_name = eval("line[" + parseLoc['MODRES']['mod_name'] + "]").strip()
                        chain_id = eval("line[" + parseLoc['MODRES']['chain_id'] + "]").strip()
                        seq = eval("line[" + parseLoc['MODRES']['seq_num'] + "]").strip()
                        aa_seq = chain_id + seq
                        if mod_residue_name not in mod_residue.keys():
                            mod_residue[mod_residue_name] = [aa_seq]
                        else:
                            mod_residue[mod_residue_name].append(aa_seq)

                    if not skip_lines and split_chain and line_name == "SEQRES":
                        chain_id = eval("line[" + parseLoc['SEQRES']['chain_id'] + "]").strip()
                        if chain_id not in ss_g.keys():
                            ss_g[chain_id] = nx.Graph()
                            ss_node[chain_id] = []
                        if chain_id not in aa_g.keys():
                            aa_g[chain_id] = nx.Graph()
                            aa_node[chain_id] = []
                            aa_edge_lists[chain_id] = []
                        if chain_id not in c_g.keys():
                            c_g[chain_id] = nx.Graph()
                            c_node[chain_id] = []
                            chain_finish[chain_id] = False

                    elif not skip_lines and line_name == "HELIX" or line_name == "SHEET":
                        # if split_chain and len(ss_g) < 2:  # meaning there is only one chain, skip it
                        #     skip_file = True
                        #     break
                        # try:
                        num_strand, range_strings = parse_ss_section(line, ss_node, attr_dict, line_name,
                                                                     num_strand, range_strings, pdb_name,
                                                                     split_chain, log_path)
                        # except ValueError:
                        #     pdb_name = basename(file).split(".")[0]
                        #     msg = "SS: ss mixed between two chains"
                        #     pdb_parse_log(pdb_name, msg)
                        #     skip_file = True

                    elif not skip_lines and line_name == "MODEL":
                        if len(model_list) == 0:
                            model_list.append(int(line.rstrip().split()[1]))
                        else:  # only choose the first model to parse
                            skip_lines = True

                    elif not skip_lines and line_name == "HETATM":
                        res_name = eval("line[" + parseLoc['HETATM']['res_name'] + "]").strip()
                        chain_id = eval("line[" + parseLoc['HETATM']['chain_id'] + "]").strip()
                        if chain_id in chain_finish.keys() and chain_finish[chain_id]:
                            # skip_lines = True
                            continue

                        elif res_name in mod_residue.keys():
                            mod_res = True
                            chain_start, chain_end, current_chain, ring_locs, \
                            primes, prev_dict, mod_res = parse_aa_section(line, aa_node, attr_dict,
                                                                          atom_to_aa,
                                                                          chain_start,
                                                                          current_chain, ring_locs,
                                                                          pdb_name,
                                                                          split_chain,
                                                                          primes,
                                                                          prev_dict, mod_res,
                                                                          mod_residue)

                        elif chain_id in chain_finish.keys() and not chain_finish[chain_id]:
                            chain_start, chain_end, current_chain, ring_locs, \
                            primes, prev_dict, mod_res = parse_aa_section(line,
                                                                          aa_node,
                                                                          attr_dict,
                                                                          atom_to_aa,
                                                                          chain_start,
                                                                          current_chain,
                                                                          ring_locs,
                                                                          pdb_name,
                                                                          split_chain,
                                                                          primes,
                                                                          prev_dict,
                                                                          mod_res,
                                                                          mod_residue)


                    elif not skip_lines and line_name == "ATOM":

                        chain_start, chain_end, current_chain, ring_locs, \
                        primes, prev_dict, mod_res = parse_aa_section(line, aa_node, attr_dict, atom_to_aa,
                                                                      chain_start,
                                                                      current_chain, ring_locs, pdb_name,
                                                                      split_chain, primes,
                                                                      prev_dict, mod_res, mod_residue)


                    elif not skip_lines and line_name == "TER":
                        ring_locs, chain_finish, mod_res, primes = parse_ter_section(line, aa_node, c_node, attr_dict_c,
                                                                                     chain_start,
                                                                                     chain_end,
                                                                                     current_chain, pdb_name,
                                                                                     split_chain, primes, ring_locs,
                                                                                     chain_finish,
                                                                                     mod_res, prev_dict)
                        prev_dict = {"label": None, "type": None}
                    elif line_name == "ENDMDL":
                        skip_lines = False
                    elif line_name == "CONECT":
                        try:
                            parse_connect_section(line, aa_node, aa_edge_lists, atom_to_aa, pdb_name,
                                                  split_chain, log_path)
                        except ValueError:
                            msg = "CONECT: label not in list AND {}".format(format(traceback.format_exc()))
                            pdb_parse_log(pdb_name, msg, log_path, PDB_CONECT_ERROR)
                            skip_file = True
            if not skip_file:
                print(file)
                i += 1
                save_path = join(pdbank_graph_dir, basename(file)[3:7])
                if split_chain:
                    for ss, aa, c in zip(ss_g.items(), aa_g.items(), c_g.items()):
                        ss_node_data = list(zip(range(0, len(ss_node[ss[0]])), ss_node[ss[0]]))
                        ss[1].add_nodes_from(ss_node_data)
                        aa_node_data = list(zip(range(0, len(aa_node[aa[0]])), aa_node[aa[0]]))
                        aa[1].add_nodes_from(aa_node_data)
                        # aa_edge_data = list(zip(range(0, len(aa_edge_lists[aa[0]])), aa_edge_lists[aa[0]]))
                        aa[1].add_edges_from(aa_edge_lists[aa[0]])
                        c_node_data = list(zip(range(0, len(c_node[c[0]])), c_node[c[0]]))
                        c[1].add_nodes_from(c_node_data)
                    for ss, aa, c in zip(ss_g.items(), aa_g.items(), c_g.items()):
                        nx.write_gexf(ss[1], "{}_{}_ss.gexf".format(save_path, ss[0]))
                        nx.write_gexf(aa[1], "{}_{}_aa.gexf".format(save_path, aa[0]))
                        nx.write_gexf(c[1], "{}_{}_c.gexf".format(save_path, c[0]))
                else:
                    ss_node_data = list(zip(range(0, len(ss_node)), ss_node))
                    ss_g.add_nodes_from(ss_node_data)
                    aa_node_data = list(zip(range(0, len(aa_node)), aa_node))
                    aa_g.add_nodes_from(aa_node_data)
                    # aa_edge_data = list(zip(range(0, len(aa_edge_lists[aa[0]])), aa_edge_lists[aa[0]]))
                    aa_g.add_edges_from(aa_edge_lists)
                    c_node_data = list(zip(range(0, len(c_node)), c_node))
                    c_g.add_nodes_from(c_node_data)
                    nx.write_gexf(ss_g, "{}_ss.gexf".format(save_path))
                    nx.write_gexf(aa_g, "{}_aa.gexf".format(save_path))
                    nx.write_gexf(c_g, "{}_c.gexf".format(save_path))


            if i % 10000 == 1:
                msg = "Processed {} files using {} mins".format(i, (time() - t) / 60)
                pdb_parse_log(None, msg, log_path)

        except Exception as e:
            print(traceback.format_exc())
            msg = "Some Runtime Error occurred: {}".format(traceback.format_exc())
            pdb_parse_log(pdb_name, msg, log_path, RUNTIME_ERROR)
            pass

        processed_num += 1
        pdb_processed_log(processed_num, log_path)

    processed_file = join(log_path, "processed.txt")
    os.remove(processed_file)
    msg = "Processed {} pdb files total using {} mins".format(i, (time() - t) / 60)
    pdb_parse_log(None, msg, log_path)


def processed_exists(log_path):
    filename = join(log_path, "processed.txt")
    return exists(filename)

def read_processed_log(log_dir):
    create_dir_if_not_exists(log_dir)
    processed_file = join(log_dir, "processed.txt")
    with open(processed_file, "r") as f:
        num = f.readline()

    return int(num)

def pdb_processed_log(num, log_dir):
    create_dir_if_not_exists(log_dir)
    processed_file = join(log_dir, "processed.txt")
    with open(processed_file, "w") as f:
        f.write(str(num))

def pdb_parse_log(pdb_name, msg, log_dir, error_type=0):

    create_dir_if_not_exists(log_dir)
    log_file = join(log_dir, "log.txt")
    pdb_error_file = join(log_dir, "pdb_error_list.txt")
    runtime_error_file = join(log_dir, "runtime_error.txt")

    if error_type == 1:
        with open(pdb_error_file, "a+") as f:
            f.write("{} {}\n".format(pdb_name, msg))
    elif error_type == 2:
        with open(runtime_error_file, "a+") as f:
            f.write("{} {}\n".format(pdb_name, msg))
    else:
        with open(log_file, "a+") as f:
            if pdb_name is None:
                f.write("{}\n".format(msg))
            else:
                f.write("{}: {}\n".format(pdb_name, msg))


def parse_ss_section(line, ss_node, attr_dict, ss_type, num_strand, range_strings, pdb_name, split_chain, log_path):
    new_num_strand = num_strand
    new_range_strings = range_strings
    if ss_type == "HELIX":
        attr_dict["type"] = eval("line[" + parseLoc['HELIX']['name'] + "]").strip()
        ssid = attr_dict["type"][0] + eval("line[" + parseLoc['HELIX']['hid'] + "]").strip()
        attr_dict["label"] = ssid
        chain_start = eval("line[" + parseLoc['HELIX']['chain_start'] + "]").strip()
        start = chain_start + eval("line[" + parseLoc['HELIX']['startaa'] + "]").strip()
        chain_end = eval("line[" + parseLoc['HELIX']['chain_end'] + "]").strip()
        end = chain_end + eval("line[" + parseLoc['HELIX']['endaa'] + "]").strip()
        hclass = eval("line[" + parseLoc['HELIX']['hclass'] + "]").strip()

        attr_dict["range"] = "{},{},{}".format(start, end, hclass)
        if chain_start != chain_end:
            msg = "chain_start = {}, chain_end = {}".format(chain_start, chain_end)
            pdb_parse_log(pdb_name, msg, log_path)
        if split_chain:
            ss_node[chain_start].append(attr_dict)
        else:
            ss_node.append(attr_dict)

    elif ss_type == 'SHEET':
        if new_num_strand == 0:
            new_num_strand = int(eval("line[" + parseLoc['SHEET']['num_strand'] + "]").strip())
        attr_dict["type"] = eval("line[" + parseLoc['SHEET']['name'] + "]").strip()
        ssid = attr_dict["type"][0] + eval("line[" + parseLoc['SHEET']['sid'] + "]").strip()
        attr_dict["label"] = ssid
        chain_start = eval("line[" + parseLoc['SHEET']['chain_start'] + "]").strip()
        start = chain_start + eval("line[" + parseLoc['SHEET']['startaa'] + "]").strip()
        chain_end = eval("line[" + parseLoc['SHEET']['chain_end'] + "]").strip()
        end = chain_end + eval("line[" + parseLoc['SHEET']['endaa'] + "]").strip()
        sense = eval("line[" + parseLoc['SHEET']['sense'] + "]").strip()

        if chain_start != chain_end:
            msg = "chain_start = {}, chain_end = {}".format(chain_start, chain_end)
            pdb_parse_log(pdb_name, msg, log_path)
        if new_num_strand > 0:
            new_num_strand -= 1
            new_range_strings.append("{},{},{}".format(start, end, sense))
        if new_num_strand == 0:
            attr_dict["range"] = ",".join(new_range_strings)

            if split_chain:
                ss_node[chain_start].append(attr_dict)
            else:
                ss_node.append(attr_dict)

            new_range_strings = []
    return new_num_strand, new_range_strings


def parse_ter_section(line, aa_node, c_node, attr_dict_c, chain_start, chain_end, current_chain, pdb_name, split_chain,
                      primes, ring_locs, chain_finish, mod_res, prev_dict):
    attr_dict_c["range"] = "{},{}".format(chain_start, chain_end)
    attr_dict_c["label"] = current_chain
    new_ring_locs = ring_locs
    new_chain_finish = chain_finish
    new_mod_res = mod_res
    new_primes = primes
    if split_chain:
        c_node[current_chain].append(attr_dict_c)
    else:
        c_node.append(attr_dict_c)

    attr_dict = {}
    chain = eval("line[" + parseLoc['TER']['chain_id'] + "]").strip()
    aa_seq = chain + eval("line[" + parseLoc['TER']['aa_seq'] + "]").strip()

    if mod_res and len(new_ring_locs) > 0:

        average_loc = np.mean(np.array(new_ring_locs), axis=0)
        attr_dict["label"] = prev_dict["label"]
        attr_dict["type"] = prev_dict["type"]
        attr_dict["x"] = round(float(average_loc[0]), 3)
        attr_dict["y"] = round(float(average_loc[1]), 3)
        attr_dict["z"] = round(float(average_loc[2]), 3)

        if split_chain:
            aa_node[current_chain].append(attr_dict)
        else:
            aa_node.append(attr_dict)

        new_mod_res = False
        new_ring_locs = []
        attr_dict = {}


    elif primes == 1:

        attr_dict["label"] = aa_seq
        attr_dict["type"] = eval("line[" + parseLoc['TER']['aa_name'] + "]").strip()
        average_loc = np.mean(np.array(new_ring_locs), axis=0)
        attr_dict["x"] = round(float(average_loc[0]), 3)
        attr_dict["y"] = round(float(average_loc[1]), 3)
        attr_dict["z"] = round(float(average_loc[2]), 3)
        if split_chain:
            aa_node[current_chain].append(attr_dict)
        else:
            aa_node.append(attr_dict)
        new_primes = 0
        new_ring_locs = []
    new_chain_finish[chain] = True
    return new_ring_locs, new_chain_finish, new_mod_res, new_primes


def parse_connect_section(line, aa_node, aa_edge_lists, atom_to_aa, pdb_name, split_chain, log_path):
    source = eval("line[" + parseLoc['CONECT']['source'] + "]").strip()
    if source not in atom_to_aa.keys():
        return
    else:
        source_aa = atom_to_aa[source]

    source_chain = source_aa[0]
    for i in range(1, 5):
        name = "target{}".format(i)
        target = eval("line[" + parseLoc['CONECT'][name] + "]").strip()
        if target == "" or target not in atom_to_aa.keys():
            break
        target_aa = atom_to_aa[target]
        if source_aa == target_aa:
            break
        target_chain = target_aa[0]
        if source_chain != target_chain:
            msg = "{}: conect {} and {} not in the same chain".format(pdb_name, source_aa, target_aa)
            pdb_parse_log(None, msg, log_path)
        else:
            if split_chain:
                source_idx = find_node_idx(aa_node[source_chain], source_aa)
                starget_idx = find_node_idx(aa_node[source_chain], target_aa)
                aa_edge_lists[source_chain].append((source_idx, starget_idx))
            else:
                source_idx = find_node_idx(aa_node, source_aa)
                starget_idx = find_node_idx(aa_node, target_aa)
                aa_edge_lists.append((source_idx, starget_idx))
    return


def find_node_idx(label_list, aa_name):

    for i, k in enumerate(label_list):
        if k['label'] == aa_name:
            return i
    msg = "label not found in node list"
    raise ValueError(msg)


def find_aa_seq(nodes_in_chain, aa_seq):
    for node in nodes_in_chain:
        if node["label"] == aa_seq:
            return True
    return False


def parse_aa_section(line, aa_node, attr_dict, atom_to_aa, chain_start, current_chain, ring_locs, pdb_name,
                     split_chain, primes, prev_dict, mod_res, mod_residue):
    serial = eval("line[" + parseLoc['ATOM']['serial_num'] + "]").strip()
    name = eval("line[" + parseLoc['ATOM']['name'] + "]").strip()
    aa_name = eval("line[" + parseLoc['ATOM']['aa_name'] + "]").strip()
    chain = eval("line[" + parseLoc['ATOM']['chain_id'] + "]").strip()
    aa_seq = chain + eval("line[" + parseLoc['ATOM']['aa_seq'] + "]").strip()
    atom_to_aa[serial] = aa_seq
    x = float(eval("line[" + parseLoc['ATOM']['x'] + "]").strip())
    y = float(eval("line[" + parseLoc['ATOM']['y'] + "]").strip())
    z = float(eval("line[" + parseLoc['ATOM']['z'] + "]").strip())
    new_ring_locs = ring_locs
    new_prev_dict = prev_dict
    new_mod_res = mod_res
    new_primes = primes
    if chain_start == None:
        new_chain_start = chain + aa_seq
    else:
        new_chain_start = chain_start
    new_current_chain = current_chain

    #  TODO Parse ring shape atoms in nucleic acids

    # Information for chains
    if new_current_chain != chain:
        new_chain_start = aa_seq
        new_current_chain = chain
    new_chain_end = aa_seq

    # Information for 3D locations
    attr_dict = {}
    if mod_res and aa_seq != new_prev_dict["label"] and len(new_ring_locs) > 0:

        average_loc = np.mean(np.array(new_ring_locs), axis=0)
        attr_dict["label"] = new_prev_dict["label"]
        attr_dict["type"] = new_prev_dict["type"]
        attr_dict["x"] = round(float(average_loc[0]), 3)
        attr_dict["y"] = round(float(average_loc[1]), 3)
        attr_dict["z"] = round(float(average_loc[2]), 3)

        if split_chain:
            aa_node[new_current_chain].append(attr_dict)
        else:
            aa_node.append(attr_dict)

        new_mod_res = False
        new_ring_locs = []
        attr_dict = {}
    elif primes == 1 and new_prev_dict["label"] is not None and aa_seq != new_prev_dict["label"]:
        average_loc = np.mean(np.array(new_ring_locs), axis=0)
        attr_dict["label"] = new_prev_dict["label"]
        attr_dict["type"] = new_prev_dict["type"]
        attr_dict["x"] = round(float(average_loc[0]), 3)
        attr_dict["y"] = round(float(average_loc[1]), 3)
        attr_dict["z"] = round(float(average_loc[2]), 3)
        if split_chain:
            aa_node[new_current_chain].append(attr_dict)
        else:
            aa_node.append(attr_dict)
        new_primes = 0
        new_ring_locs = []

    if mod_res:
        new_ring_locs.append([x, y, z])

    elif name == "CA":
        if find_aa_seq(aa_node[chain], aa_seq):
            return new_chain_start, new_chain_end, new_current_chain, new_ring_locs, new_primes, new_prev_dict, mod_res

        attr_dict["label"] = aa_seq
        attr_dict["type"] = aa_name
        attr_dict["x"] = float(x)
        attr_dict["y"] = float(y)
        attr_dict["z"] = float(z)
        if split_chain:
            aa_node[new_current_chain].append(attr_dict)
        else:
            aa_node.append(attr_dict)

        new_ring_locs = []
    elif "'" in name:
        new_ring_locs.append([x, y, z])
        new_primes = 1
    new_prev_dict["label"] = aa_seq
    new_prev_dict["type"] = aa_name

    return new_chain_start, new_chain_end, new_current_chain, new_ring_locs, new_primes, new_prev_dict, new_mod_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_chain")
    parser.add_argument("--raw_dir")
    parser.add_argument("--graph_dir")
    parser.add_argument("--pdb_file")
    # if len(sys.argv) < 6:
    #     parser.print_help()
    #     exit(1)
    args = parser.parse_args()

    if args.split_chain == '1':
        split_chain = True
    else:
        split_chain = False

    pdb_file = args.pdb_file

    parse_pdb_files(split_chain=split_chain, raw_dir=args.raw_dir, graph_dir=args.graph_dir, list_of_pdb=pdb_file)


    # parse_pdb_files(split_chain=True, raw_dir="/media/yba/HDD/data/test2", graph_dir="/media/yba/HDD/data/test2")

