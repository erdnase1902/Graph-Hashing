from os.path import join, basename
from glob import glob
import shutil
import csv, pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from load_pdb_data import _get_string_pdb_mapping
from utils import create_dir_if_not_exists
import argparse
import networkx as nx
from utils import get_data_path

def plot_hist(counts, labeling, filename, standard_bins=None):
    if standard_bins is not None:
        plt.figure(figsize=(30, 15))
    n, bins, patches = plt.hist(x=counts,
                                bins=50 if standard_bins is None else standard_bins,
                                color='#0504aa',
                                alpha=0.7, rwidth=2.0)
    if standard_bins is not None:
        for i in range(len(patches)):
            if n[i] != 0:
                plt.text(bins[i], n[i], str(n[i]))


    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(labeling["xlabel"])  # Number of PDB structures Per STRING protein')
    plt.ylabel(labeling["ylabel"])
    raw_title = labeling["title"] #'STRING-PDB mapping (counts)'
    final_title = raw_title + ' (All)' if 'all' in filename else raw_title
    plt.title(final_title)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(
        ymax=np.ceil(maxfreq / 10) * 10 + 100 if maxfreq % 10 else maxfreq + 50)
    if "PDB-STRING" in labeling["title"]:
        plt.xticks(np.arange(min(counts), max(counts) + 1, 1.0))
    plt.savefig(filename)
    plt.close()


def show_stats(protein_pdb, species, target_dir):
    protein_known_pdb_stats = {}
    protein_all_pdb_stats = {}
    with_pdb = 0
    no_pdb = 0
    for p, v in protein_pdb.items():
        if len(v['pdb']) == 0:
            no_pdb = no_pdb + 1
        else:
            with_pdb = with_pdb + 1
            protein_known_pdb_stats[p] = len(v['pdb'])

        protein_all_pdb_stats[p] = len(v['pdb'])
    labeling = {}
    labeling["xlabel"] = "Number of PDB structures Per STRING protein"
    labeling["ylabel"] = "Frequency"
    labeling["title"] = "STRING-PDB mapping (counts)"
    print("There are {} ({}) proteins with pdb info.".format(with_pdb,
                                                             with_pdb / len(
                                                                 protein_pdb)))
    print("There are {} ({}) proteins with no pdb info".format(no_pdb,
                                                               no_pdb / len(
                                                                   protein_pdb)))
    print("============== Stats for proteins with pdbs =============")
    print("Mean STRING-PDBs: {}".format(
        np.mean(list(protein_known_pdb_stats.values()))))
    print("Std STRING-PDBs: {}".format(
        np.std(list(protein_known_pdb_stats.values()))))
    print("Min STRING-PDBs: {}".format(
        np.min(list(protein_known_pdb_stats.values()))))
    print("Max STRING-PDBs: {}".format(
        np.max(list(protein_known_pdb_stats.values()))))
    filename = join(target_dir, species, "plots/known_hist.png")
    plot_hist(list(protein_known_pdb_stats.values()), labeling, filename, standard_bins=None)

    print("============== Stats for all proteins ==============")
    print("Mean STRING-PDBs: {}".format(
        np.mean(list(protein_all_pdb_stats.values()))))
    print("Std STRING-PDBs: {}".format(
        np.std(list(protein_all_pdb_stats.values()))))
    print("Min STRING-PDBs: {}".format(
        np.min(list(protein_all_pdb_stats.values()))))
    print("Max STRING-PDBs: {}".format(
        np.max(list(protein_all_pdb_stats.values()))))
    filename = join(target_dir, species, "plots/all_hist.png")
    plot_hist(list(protein_all_pdb_stats.values()), labeling, filename, standard_bins=None)


def sequence_preprocessing(species, common_dir):
    protein_seq = defaultdict(str)
    filename = species + ".protein.sequences*.fa"
    filepath = glob(join(common_dir, species+"_raw", filename))[0]
    with open(filepath, 'r') as f:
        lines = f.readlines()
        flag = 0
        for line in lines:
            if ">" in line:
                if flag == 0:
                    flag = 1
                    protein_name = line.strip().strip(">")
                else:
                    protein_name = line.strip().strip(">")
            else:
                if flag == 1:
                    protein_seq[protein_name] = protein_seq[
                                                    protein_name] + line.strip()

    filename = species + "_sequence.tsv"
    filepath = join(common_dir, species, filename)
    with open(filepath, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for protein, sequence in protein_seq.items():
            tsv_writer.writerow([protein, sequence])


def show_link_type_stats(df, protein_pdb, species, target_dir):
    all_raw = {k: df[k].values.tolist() for k in list(df)[2:10]}
    standard_bin = np.arange(50, 1001, 50).tolist()
    standard_bin.insert(0, 0)
    standard_bin.insert(1, 1)

    binary_raw = {k: len(df[df[k] > 0]) / len(df.index) for k in list(df)[2:10]}
    sorted_binary = sorted(binary_raw.items(), key=lambda x: x[1], reverse=True)

    print("============== Link Type Distributions ===============")
    for x in sorted_binary:
        print("{}: {:.3f}".format(x[0], x[1]))


    for k, v in all_raw.items():
        print("============== Stats for {} type =============".format(k))
        print("There are {} PPI interactions.".format(len(v)))
        print("Mean score: {}".format(np.mean(v)))
        print("Std score: {}".format(np.std(v)))
        print("Min score: {}".format(np.min(v)))
        print("Max score: {}".format(np.max(v)))
        labeling = {}
        labeling["xlabel"] = "Number of {} links".format(k)
        labeling["ylabel"] = "Frequency"
        labeling["title"] = "Confidence Scores"
        filename = join(target_dir, species, "plots/{}_{}.png".format(species, k))
        plot_hist(v, labeling, filename, standard_bin)


def show_string_conf_and_length_stats(species, common_dir):

    filename = join(common_dir, species, "{}_protein_map.pickle".format(species))
    with open(filename, "rb") as f:
        pdb_species = pickle.load(f)

    filename = join(common_dir, "pdb_seq_processed.pickle")
    with open(filename, "rb") as f:
        pdb_seq = pickle.load(f)


    pdb_lengths = []
    percent_length_string_larger = []
    string_larger_count = 0
    pdb_larger_count = 0
    no_info_count = 0
    total_count = 0
    percent_length_pdb_larger = []
    conf_scores = []

    for sid, info in pdb_species.items():
        for pdb in info['pdb']:
            pdb_id = pdb['pdb_id'][0:4] + '_' + pdb['pdb_id'][4]
            matched_length = pdb['range'][1] - pdb['range'][0] + 1
            if 'length' in pdb_seq[pdb_id]:
                total_length = pdb_seq[pdb_id]['length']
            else:
                no_info_count += 1
                total_length = matched_length
            total_count += 1
            if matched_length >= total_length:
                percent_length_string_larger.append(
                    total_length / matched_length)
                string_larger_count += 1
            else:
                percent_length_pdb_larger.append(matched_length / total_length)
                pdb_larger_count += 1

            #         if matched_length > total_length:
            #             print(sid, pdb_id)
            conf_scores.append(pdb['identity'])
            pdb_lengths.append(pdb['range'][1] - pdb['range'][0] + 1)

    print("# Total Mapping = {}".format(total_count))
    print("# Mapping (STRING segment larger) = {}".format(string_larger_count))
    print("# Mapping (PDB segment larger) = {}".format(pdb_larger_count))
    print("# Mapping (no PDB info) = {}".format(no_info_count))

    plot_list = {"Matched_PDB Lengths": pdb_lengths,
                 "Links of Identity": conf_scores,
                 "Percentage of Mapping (STRING segment larger)": percent_length_string_larger,
                 "Percentage of Mapping (PDB segment larger)": percent_length_pdb_larger}

    for k, v in plot_list.items():

        print("================= {} ===================".format(k))
        print("Mean score: {}".format(np.mean(v)))
        print("Std score: {}".format(np.std(v)))
        print("Min score: {}".format(np.min(v)))
        print("Max score: {}".format(np.max(v)))
        labeling = {}
        labeling["xlabel"] = "Measure (conf scores or percentage)"
        labeling["ylabel"] = "Frequency"
        labeling["title"] = k
        filename = join(common_dir, species, "plots/{}_{}.png".format(species, k))
        num_bins = 20 if "Percentage" in k else None
        plot_hist(v, labeling, filename, num_bins)



def species_preprocessing(species, common_dir):

    print("Reading all_string_pdb mapping dictionary...")
    protein_identifier = _get_string_pdb_mapping() # key: mapping from string to pdb

    filename = species + ".protein.links*.txt"
    #print(glob(join(raw_dir, species+"_raw", filename)))
    links_dir = glob(join(common_dir, species+"_raw", filename))[0]
    df = pd.read_csv(links_dir, delimiter=' ', header=0)

    source = set(df['protein1'].values.tolist())
    target = set(df['protein2'].values.tolist())
    combined = source.union(target)
    print("Total Number of STRING proteins in this species: {}".format(len(combined)))

    protein_pdb = {}
    for protein in sorted(combined): # strinh id
        protein_pdb[protein] = protein_identifier[protein]

    # Removing pdb duplicates between Homology Model and PDB
    for sid, info in protein_pdb.items():
        temp = info
        check = []
        for ele in temp['pdb']:
            single_pdb = (ele['pdb_id'], ele['identity'], ele['range'])
            if single_pdb not in check:
                check.append(single_pdb)
            else:
                protein_pdb[sid]['pdb'].remove(ele)

    create_dir_if_not_exists(join(common_dir, species, "plots"))
    fname = species + "_protein_map.pickle"
    target_path = join(common_dir, species, fname)
    with open(target_path, "wb") as f:
        pickle.dump(protein_pdb, f)

    show_stats(protein_pdb, species, common_dir)
    show_link_type_stats(df, protein_pdb, species, common_dir)


    copy_dir = join(common_dir, species, species+"_links.tsv")
    shutil.copy(links_dir, copy_dir)

def show_pdb_to_string_stats(species, common_dir):
    filename = join(common_dir, species, "{}_protein_map.pickle".format(species))
    with open(filename, "rb") as f:
        pdb_species = pickle.load(f)

    pdb_to_string_map = defaultdict(list)
    pdb_to_num_string = {}
    for sid, info in pdb_species.items():

        for pdb in info['pdb']:
            pid = pdb['pdb_id']
            pdb_to_string_map[pid].append(sid)

    for k, v in pdb_to_string_map.items():
        pdb_to_num_string[k] = len(v)

    labeling = {}
    labeling["xlabel"] = "Number of Mapped STRING proteins"
    labeling["ylabel"] = "Frequency"
    labeling["title"] = 'Distribution of PDB-STRING Mapping'
    filename = join(common_dir, species, "plots/{}_{}.png".format(species, "PDB_STRING_Mapping"))
    #num_bins = 20 if "Percentage" in k else None
    plot_hist(list(pdb_to_num_string.values()), labeling, filename, None)


def show_species_network_stats(species, common_dir):
    links_dir = join(common_dir, species, "{}_links.tsv".format(species))
    df = pd.read_csv(links_dir, delimiter=' ', header=0)
    G = nx.from_pandas_edgelist(df, 'protein1', 'protein2',
                                ['neighborhood', 'fusion', 'cooccurence',
                                 'coexpression', 'experimental', 'database',
                                 'textmining'])

    print("================== PPI Network Stats ====================")
    print("# Connected Component = {}".format(nx.number_connected_components(G)))
    print("# Nodes = {}".format(G.number_of_nodes()))
    print("# Edges = {}".format(G.number_of_edges()))
    degrees = list(dict(G.degree()).values())
    print("Avg Degree = {}".format(np.mean(degrees)))
    print("Std Degree = {}".format(np.std(degrees)))
    print("Min Degree = {}".format(np.min(degrees)))
    print("Max Degree = {}".format(np.max(degrees)))
    print("Density = {}".format(nx.density(G)))
    labeling = {}
    labeling["xlabel"] = "Node Degrees"
    labeling["ylabel"] = "Frequency"
    labeling["title"] = "Degree Distribution of {}".format(species)
    filename = join(common_dir, species,
                    "plots/{}_{}.png".format(species, "Degrees"))
    plot_hist(degrees, labeling, filename, None)


def show_types_per_edge(species, common_dir):
    links_dir = join(common_dir, species, "{}_links.tsv".format(species))
    df = pd.read_csv(links_dir, delimiter=' ', header=0)

    cols = df.columns.difference(['protein1', 'protein2', 'combined_score'])

    df['> zero'] = df[cols].gt(0).sum(axis=1)
    types_per_edge = df['> zero'].values.tolist()
    print("================== Link Types Stats ====================")
    print("Mean # Types per edge: {}".format(np.mean(types_per_edge)))
    print("Std # Types per edge: {}".format(np.std(types_per_edge)))
    print("Min # Types per edge: {}".format(np.min(types_per_edge)))
    print("Max # Types per edge: {}".format(np.max(types_per_edge)))
    labeling = {}
    labeling["xlabel"] = "Number of Types"
    labeling["ylabel"] = "Frequency"
    labeling["title"] = "Histogram of Types per Edge - {}".format(species)
    filename = join(common_dir, species,
                    "plots/{}_{}.png".format(species, "TypePerEdge"))
    plot_hist(types_per_edge, labeling, filename, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--species")
    parser.add_argument("--common_dir")
    #parser.add_argument("--target_dir")

    args = parser.parse_args()

    #species_preprocessing(species=args.species, common_dir=args.common_dir)

    # common_dir = "../../data/PPI_Datasets/preprocess/ecoli/"
    common_dir = join(get_data_path(), "PPI_Datasets", "preprocess")
    species = "511145"
    species_preprocessing(species=species, common_dir=common_dir)

    sequence_preprocessing(species=species, common_dir=common_dir)

    show_string_conf_and_length_stats(species=species, common_dir=common_dir)

    show_types_per_edge(species=species, common_dir=common_dir)

    show_pdb_to_string_stats(species=species, common_dir=common_dir)

    show_species_network_stats(species=species, common_dir=common_dir)

