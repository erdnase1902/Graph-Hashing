from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import re
import os
import utils
from collections import defaultdict
import deepchem as dc
import pandas as pd

hybridization_to_int = {
    1: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.UNSPECIFIED:6
}

DRUG_FEATS = {"16": dc.feat.CircularFingerprint(size=16),
              "32": dc.feat.CircularFingerprint(size=32),
              "64": dc.feat.CircularFingerprint(size=64),
              "128": dc.feat.CircularFingerprint(size=128),
              "512": dc.feat.CircularFingerprint(size=512),
              "1024": dc.feat.CircularFingerprint(size=1024)}


def pubchem_to_stitch_stereo(pid):
    return 'CID' + '{0:0>9}'.format(pid)

def mol_to_nxgraph(drug_mol, edge_atts=False):
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = chem_feature_factory.GetFeaturesForMol(drug_mol)
    # assert len(drug_mol.GetConformers()) == 1
    # geom = drug_mol.GetConformers()[0].GetPositions()

    nx_graph = nx.Graph()
    for i in range(drug_mol.GetNumAtoms()):
        atom = drug_mol.GetAtomWithIdx(i)

        nx_graph.add_node(i, atom_type=atom.GetSymbol(), aromatic=atom.GetIsAromatic(), acceptor=0,
                          donor=0, hybridization=hybridization_to_int[atom.GetHybridization()],
                          num_h=atom.GetTotalNumHs())
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                nx_graph.node[i]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                nx_graph.node[i]['acceptor'] = 1
    # Read Edges
    for i in range(drug_mol.GetNumAtoms()):
        for j in range(drug_mol.GetNumAtoms()):
            e_ij = drug_mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                if not edge_atts:
                    nx_graph.add_edge(i, j)
                else:
                    btype = e_ij.GetBondType()
                    nx_graph.add_edge(i, j, b_type=btype.name)
    return nx_graph


def generate_gexf_from_drugcombo(drug_csv_file, dataset_name, edge_atts):
    klepto_path = os.path.join(utils.get_data_path(), dataset_name, "klepto")
    utils.create_dir_if_not_exists(klepto_path)
    df = pd.read_csv(drug_csv_file)
    cids = list(df['cIds'])
    drug_names = list(df['drugName'])
    smile_strings = list(df['smilesString'])
    drug_mols = [Chem.MolFromSmiles(smile) for smile in smile_strings]
    skipped = []
    drug_to_cid_dict = {}
    drug_name_to_cid = {}
    pubchem_results = utils.load_pickle(os.path.join(klepto_path, 'pubchem_results'), True)
    drug_names_pubchem = []
    cids_pubchem = []
    drug_mols_pubchem = []
    for k, compounds in pubchem_results.items():
        if len(compounds) == 0 or type(compounds[0]) is list:
            continue
        else:
            drug_names_pubchem.append(str(k))
            cids_pubchem.append('CIDs' + str(compounds[0].cid))
            drug_mols_pubchem.append(Chem.MolFromSmiles(compounds[0].isomeric_smiles))
    drug_mols = drug_mols + drug_mols_pubchem
    cids = cids + cids_pubchem
    drug_names = drug_names + drug_names_pubchem
    for i, drug_mol in enumerate(drug_mols):
        try:
            if i % 25 == 0:
                print("processed {} graphs".format(i))
            if drug_mol is None:
                print("rdkit did not parse drug ", i)
                skipped.append(i)
                continue
            nx_graph = mol_to_nxgraph(drug_mol, edge_atts)
            if '-' in cids[i]:
                print("here")

            if cids[i] == 'CAS1202884-94-3':
                cids[i] = 'CIDs44599690'
            elif cids[i] == 'CAS311795-38-7':
                cids[i] = 'CIDs3099980'
            nx_graph.graph["cId"] = cids[i]
            nx_graph.graph["drug_name"] = drug_names[i]
            drug_name_to_cid[drug_names[i].lower()] = cids[i]
            graph_id = cids[i]
            drug_feat = {key: feat.featurize([drug_mol])[0] for key, feat in DRUG_FEATS.items()}
            drug_to_cid_dict[graph_id] = {"cId": cids[i], "drug_name": drug_names[i], "drug_feat": drug_feat}
            nx.readwrite.write_gexf(nx_graph, os.path.join(utils.get_data_path(), dataset_name, graph_id + ".gexf"))
        except Exception as e:
            print(e)
            skipped.append(i)
    utils.save(drug_to_cid_dict, os.path.join(klepto_path, "graph_data.klepto"))
    utils.save(drug_name_to_cid, os.path.join(klepto_path, "drug_name_to_cid.klepto"))

    print("skipped: ", skipped)


def generate_gexf_from_sdf(sdf_file, dataset_name=None, nodes=None, edge_atts=False):
    if not dataset_name:
        dataset_path =  os.path.join(utils.get_data_path(), "DrugBank")
    elif dataset_name == "drugbank_snap":
        dataset_path = os.path.join(utils.get_data_path(), "DrugBank", "ddi_data", "drugs_snap")
    elif dataset_name == "drugbank_deepddi":
        dataset_path = os.path.join(utils.get_data_path(), "DrugBank", "ddi_data", "drugs_deepddi")
    elif dataset_name == "decagon":
        dataset_path = os.path.join(utils.get_data_path(), "Decagon")
    else:
        raise ValueError("{} not supported".format(dataset_name))
    klepto_path = os.path.join(dataset_path, "klepto")
    utils.create_dir_if_not_exists(dataset_path)
    utils.create_dir_if_not_exists(klepto_path)
    drug_mols = Chem.SDMolSupplier(sdf_file)
    skipped = []
    if "drugbank" in dataset_name:
        drug_grp_count = defaultdict(int)
        drug_to_grp_dict = {}
    elif "decagon" in dataset_name:
        drug_to_stitch_id_dict = {}
    for i, drug_mol in enumerate(drug_mols):
        try:
            if i % 25 == 0:
                print("processed {} graphs".format(i))
            if drug_mol is None:
                print("rdkit did not parse drug ", i)
                skipped.append(i)
                continue
            if "drugbank" in dataset_name:
                if not drug_mol.HasProp("DRUGBANK_ID"):
                    raise AssertionError("DRUGBANK_ID attribute not found")
                graph_id = drug_mol.GetProp("DRUGBANK_ID")

                if nodes and graph_id not in nodes:
                    continue
                nx_graph = mol_to_nxgraph(drug_mol, edge_atts)


                nx_graph.graph["db_id"] = graph_id

                if not drug_mol.HasProp("DRUG_GROUPS"):
                    raise AssertionError("DRUG_GROUPS attribute not found")
                db_grp = drug_mol.GetProp("DRUG_GROUPS")
                drug_grp_count[db_grp] += 1
                nx_graph.graph["db_grp"] = db_grp.split(';')
                drug_feat = {key: feat.featurize([drug_mol])[0] for key, feat in DRUG_FEATS.items()}
                drug_to_grp_dict[graph_id] = {"db_grp": db_grp.split(';'), "db_id": graph_id, "drug_feat": drug_feat}

            elif "decagon" in dataset_name:
                if not drug_mol.HasProp("PUBCHEM_COMPOUND_CID"):
                    raise AssertionError("PUBCHEM_COMPOUND_CID attribute not found")
                graph_id = drug_mol.GetProp("PUBCHEM_COMPOUND_CID")
                nx_graph = mol_to_nxgraph(drug_mol, edge_atts)
                nx_graph.graph["stitch_id"] = pubchem_to_stitch_stereo(int(graph_id))
                drug_to_stitch_id_dict[graph_id] ={"stitch_id": pubchem_to_stitch_stereo(int(graph_id))}
            nx.readwrite.write_gexf(nx_graph, os.path.join(dataset_path, graph_id + ".gexf"))
        except Exception as e:
            print(e)
            skipped.append(i)
    if "drugbank" in dataset_name:
        utils.save(dict(drug_grp_count), os.path.join(klepto_path, "drug_group_count.klepto"))
        utils.save(drug_to_grp_dict, os.path.join(klepto_path, "graph_data.klepto"))
    elif "decagon" in dataset_name:
        utils.save(drug_to_stitch_id_dict, os.path.join(klepto_path, "graph_data.klepto"))

    return skipped


def check_unqiue_nodes(file_path, parse_edges_func):
    drug_ids = set()
    edge_list = []
    edge_types = set()
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):

            edges, edge_type = parse_edges_func(line)
            if edges:
                drug_ids.add(edges[0])
                drug_ids.add(edges[1])
                edge_list.append((edges[0], edges[1]))
                edge_types.add(edge_type)

    return drug_ids, edge_types, edge_list

def parse_edges_biosnap(line):
    return line.rstrip('\n').split('\t'), None

def parse_edges_pnas(line):
    line = line.split(',')[0]
    pattern = 'DB[0-9]{5}'
    drugs = list(set(re.findall(pattern, line)))
    edge = None
    if drugs:
        if line.find(drugs[0]) < line.find(drugs[1]):
            drug1, drug2 = drugs[0], drugs[1]
        else:
            drug1, drug2 = drugs[1], drugs[0]
        temp = re.sub(drug1, 'D1', line)
        edge = re.sub(drug2, 'D2', temp)
    return drugs, edge

if __name__ == "__main__":
    path = os.path.join(utils.get_data_path(), "DrugCombo", "ddi_data", "drug_chemical_info.csv")
    generate_gexf_from_drugcombo(path, "DrugCombo", edge_atts=True)




    # decagon_file_path = os.path.join(utils.get_data_path(), "Decagon", "ddi_data", "decagon_structures.sdf")
    # file_path = "/data/DrugBank/ddi_data/ddi_snap.tsv"
    # ddi_drugs_biosnap, edge_types, edge_list = check_unqiue_nodes(file_path, parse_edges_biosnap)
    # # file_path = r"D:\Ken\University\Research Please\Datasets\DDI_Datasets\pnas_ddi.csv"
    # # ddi_drugs_pnas, edge_types, edge_list = check_unqiue_nodes(file_path, parse_edges_pnas)
    # #
    # file_path = "/home/kengu13/PycharmProjects/GraphMatching/data/DrugBank/structures.sdf"
    # skipped = generate_gexf_from_sdf(file_path, dataset_name="drugbank_snap", nodes=ddi_drugs_biosnap)
    #
    # print("skipped: ", skipped)
    # skipped_decagon = generate_gexf_from_sdf(decagon_file_path, dataset_name="decagon")
    # print("skipped decagon: ", skipped_decagon)

    print("Done")
