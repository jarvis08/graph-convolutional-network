import os
import sys
import json
import networkx as nx
import numpy as np

from spektral.layers import GraphConv, ops

import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from utils import load_train_dataset, save_embedding


def save_embedding(embedding, p_embed, p_dataset, cross_val_num):
    print(f">>> Save Embedding of FOLD-{cross_val_num}..")
    n_nodes = embedding.shape[0]
    dim = embedding.shape[1]

    cells = set()
    with open("./Data/gdp/cell_list.txt", 'r') as f:
        cell_names = f.readlines()
        n_cells = len(cell_names)
        for i in range(n_cells):
            cells.add(cell_names[i].replace("\n", ''))

    with open(f"{p_dataset}/id_map.json", 'r') as imf:
        id_map = json.load(imf)
        id_to_name = id_map["id_to_name"]
        name_to_id = id_map["name_to_id"]

    cell_indices = []
    for i in range(len(id_to_name.keys())):
        if id_to_name[str(i)] in cells:
            cell_indices.append(i)

    with open(f"{p_embed}/cell_embedding-{cross_val_num}.txt", 'w') as ef:
        ef.write(f"{n_nodes} {dim}\n")
        for i in range(n_nodes):
            if i not in cell_indices:
                continue

            ef.write(f"{id_to_name[str(i)]} ")
            for j in range(dim - 1):
                ef.write(f"{embedding[i][j]} ")
            ef.write(f"{embedding[i][dim - 1]}\n")


def load_dataset(label_path, folded_path):
    labels = np.load(label_path + "/labels.npy")

    with open(folded_path + "/graph.json", 'r') as f:
        graph_json = json.load(f)
    graph = nx.json_graph.node_link_graph(graph_json)
    adjacency_mat = nx.adjacency_matrix(graph)

    features = np.load(folded_path + "/feats.npy")
    return adjacency_mat, features, labels


def make_embedding(CV, MODEL, DATA, EMBED):
    DATA_FOLD = DATA + f"/FOLD-{CV}"
    if not os.path.exists(EMBED):
        os.mkdir(EMBED)

    graph, features, labels = load_dataset(DATA, DATA_FOLD)
    fltr = GraphConv.preprocess(graph).astype('f4')
    fltr = ops.sp_matrix_to_sp_tensor(fltr)

    X_in = Input((features.shape[1],))
    fltr_in = Input((features.shape[0],), sparse=True)
    X_1 = GraphConv(512, 'relu', True, kernel_regularizer=l2(5e-4))([X_in, fltr_in])
    X_1 = Dropout(0.5)(X_1)
    X_2 = GraphConv(256, 'relu', True, kernel_regularizer=l2(5e-4))([X_1, fltr_in])
    X_2 = Dropout(0.5)(X_2)
    X_3 = GraphConv(128, 'relu', True, kernel_regularizer=l2(5e-4))([X_2, fltr_in])
    X_3 = Dropout(0.5)(X_3)
    X_4 = GraphConv(64, 'linear', True, kernel_regularizer=l2(5e-4))([X_3, fltr_in])
    X_5 = Dense(labels.shape[1], use_bias=True)(X_4)

    loaded_model = load_model(f"{MODEL}")
    model_without_task = Model(inputs=[X_in, fltr_in], outputs=X_4)
    model_without_task.set_weights(loaded_model.get_weights()[:8])

    final_node_representations = model_without_task([features, fltr], training=False)
    save_embedding(final_node_representations, EMBED, DATA_FOLD, CV)


if __name__=="__main__":
    #cross_validation_num = sys.argv[1]
    model_path = sys.argv[1]
    data_path = f"./Data/results/v3"
    embed_path = model_path + "/Embed"
    if not os.path.exists(embed_path):
        os.mkdir(embed_path)

    for cv in range(1, 11):
        make_embedding(cv, model_path + f"/FOLD-{str(cv)}", data_path, embed_path)
