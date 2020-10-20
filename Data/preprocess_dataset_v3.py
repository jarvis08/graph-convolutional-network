import os
import json
from math import sqrt

import networkx as nx
import numpy as np


def load_embedding_from_txt(file_name):
    names = []
    embeddings = []
    with open(file_name, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            splitted = line.split()
            names.append(splitted[0])
            embeddings.append([float(value) for value in splitted[1:]])
    print(len(names)," words loaded.")
    return names, embeddings


BASE = "./results"
if not os.path.exists(BASE):
    os.mkdir(BASE)
BASE = BASE + "/v3"
if not os.path.exists(BASE):
    os.mkdir(BASE)


# Read 981 used cell-lines
print(">>> Read cell-lines' names")
with open("./gdp/cell_list.txt", 'r') as f:
    cell_names = f.readlines()
n_cells = len(cell_names)
print(f"Number of cell-lines = {n_cells}")
for i in range(n_cells):
    cell_names[i] = cell_names[i].replace("\n",'')
cells = set()
cells.update(cell_names)
    
PROTEIN_PROTEIN_FILE = './gdp/IRefindex_protein_protein.txt'
CELL_PROTEIN_FILE = './gdp/IRefindex_cell_protein.txt'

protein_protein_file = open(PROTEIN_PROTEIN_FILE, 'rb')
protein_protein_graph = nx.read_edgelist(protein_protein_file, data=(('weight', float),))

cell_protein_file = open(CELL_PROTEIN_FILE, 'rb')
cell_protein_graph = nx.read_edgelist(cell_protein_file, data=(('weight', float),))

base = nx.compose(protein_protein_graph, cell_protein_graph)

n_nodes = 19283
n_drugs = 265
labels = np.zeros((n_nodes, n_drugs), dtype=np.float64)

CROSS_VALIDATION = 10
for cv_num in range(1, CROSS_VALIDATION + 1):
    print(f"\n>>> Start making CV-{cv_num} datasets..")

    SAVE_PATH = f"./results/v3/FOLD-{cv_num}"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    CELL_DRUG_FILE = f'./gdp/fold/Cell-Drug-{cv_num}.txt'

    cell_drug_file = open(CELL_DRUG_FILE, 'rb')
    cell_drug_edges = nx.read_edgelist(cell_drug_file, data=(('weight', int),))
    graph = nx.compose(base, cell_drug_edges)
    

    # Load Embedding
    # { real_name : id }
    print(">>> Load Embeddings form txt files")
    embed_names, embed_values = load_embedding_from_txt(f"./gdp/EmbeddingData/ORI/total_embedding-{cv_num}.txt")
    n_nodes = len(embed_names)
    print(f"Number of Nodes = {n_nodes}")


    # Make feats.npy
    print(">>> Make feats.npy")
    with open(f"{SAVE_PATH}/feats.npy", "wb") as f:
        features = np.asarray(embed_values, dtype=np.float64)
        np.save(f, features)


    # Make id_map.json
    print(">>> Make id_map.json")
    name_to_id = dict()
    id_to_name = dict()
    idx = 0
    for node_name in embed_names:
        name_to_id[node_name] = idx
        id_to_name[idx] = node_name
        idx += 1
    with open(f"{SAVE_PATH}/id_map.json", 'w') as f:
        id_map = {}
        id_map["name_to_id"] = name_to_id
        id_map["id_to_name"] = id_to_name
        json.dump(id_map, f)


    # Relabel graph's ids with name_to_id
    print(">>> Make graph.json")
    with open(f"{SAVE_PATH}/graph.json", "w") as f:
        id_graph = nx.relabel_nodes(graph, name_to_id)
        data = nx.json_graph.node_link_data(id_graph)
        json.dump(data, f)


    # Make labels.npy & masks.json
    with open(CELL_DRUG_FILE, 'r') as f:
        c_d_edges = f.readlines()

    idx = 0
    drugs = dict()
    connected_cells = dict()
    for i in range(len(c_d_edges)):
        if i % 2 == 0:
            edge = c_d_edges[i].split("\t")
            c = edge[0]
            d = edge[1]
            if d not in drugs.keys():
                drugs[d] = idx
                idx += 1
            if not connected_cells.get(c):
                connected_cells[c] = [drugs[d]]
                continue
            connected_cells[c].append(drugs[d])

    print(">>> Save dictionary for drugs and corresponding label's IDs")
    with open(f"{SAVE_PATH}/drug_ids.json", "w") as f:
        json.dump(drugs, f)

    train_mask = np.zeros(n_nodes, dtype=np.bool)
    for c in connected_cells.keys():
        node_id = name_to_id[c]
        labels[node_id][connected_cells[c]] = 1
        train_mask[node_id] = True
    print(f"Number of connected cell-lines = {len(connected_cells.keys())}")
    print(">>> Save train_mask.npy")
    with open(f"{SAVE_PATH}/train_mask.npy", "wb") as f:
        np.save(f, train_mask)

    valid_mask = np.zeros(n_nodes, dtype=np.bool)
    connected = set(connected_cells.keys())
    disconnected = cells - connected
    print(f"Number of disconnected cell-lines = {len(disconnected)}")
    for c in disconnected:
        node_id = name_to_id[c]
        valid_mask[node_id] = True
    print(">>> Save valid_mask.npy")
    with open(f"{SAVE_PATH}/valid_mask.npy", "wb") as f:
        np.save(f, valid_mask)

print(">>> Save labels.npy")
with open(f"{BASE}/labels.npy", "wb") as f:
    np.save(f, labels)
