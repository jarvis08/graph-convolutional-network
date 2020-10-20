import sys
import os
import json
import time
import logging

import numpy as np
import networkx as nx
from spektral.layers import GraphConv, ops

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
tf.keras.backend.set_floatx('float64')


class GCN:
    now = None
    max_epochs = 10000
    min_epochs = 7500
    early_stop = 0.005
    patience = 500
    lr = 0.01


    def __init__(self):
        self.fury = 0
        self.best = 0

    def set_path(self, cv):
        from datetime import datetime
        if not GCN.now:
            GCN.now = datetime.now().strftime("%m-%d-%H%M")

        base = f"./Model_v3"
        models = base + f"/GCN_{GCN.now}"
        model = models + f"/FOLD-{cv}"

        if not os.path.exists(base):
            os.mkdir(base)
        if not os.path.exists(models):
            os.mkdir(models)
        if not os.path.exists(model):
            os.mkdir(model)

        data = f"./Data/results/v3/FOLD-{cv}"

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=f"{model}/train.log", level=logging.DEBUG)
        logger = logging.getLogger()
        logger.info(f"Path for Dataset = {data}")
        logger.info(f"Path for Models = {model}")
        self.logger = logger
        return data, model


    @classmethod
    def load_labels(cls, path):
        GCN.labels = np.load(path + "/labels.npy")


    def load_folded_dataset(self, path):
        with open(path + "/graph.json", 'r') as f:
            graph_json = json.load(f)
        graph = nx.json_graph.node_link_graph(graph_json)
        adjacency_mat = nx.adjacency_matrix(graph)
        fltr = GraphConv.preprocess(adjacency_mat).astype('f4')

        self.fltr = ops.sp_matrix_to_sp_tensor(fltr)
        self.features = np.load(path + "/feats.npy")
        self.train_mask = np.load(path + "/train_mask.npy")
        self.valid_mask = np.load(path + "/valid_mask.npy")
        self.train_labels = GCN.labels[self.train_mask]
        self.valid_labels = GCN.labels[self.valid_mask]


    def create_model(self):
        X_in = Input((self.features.shape[1],))
        fltr_in = Input((self.features.shape[0],), sparse=True)
        X_1 = GraphConv(512, 'relu', True, kernel_regularizer=l2(5e-4))([X_in, fltr_in])
        X_1 = Dropout(0.5)(X_1)
        X_2 = GraphConv(256, 'relu', True, kernel_regularizer=l2(5e-4))([X_1, fltr_in])
        X_2 = Dropout(0.5)(X_2)
        X_3 = GraphConv(128, 'relu', True, kernel_regularizer=l2(5e-4))([X_2, fltr_in])
        X_3 = Dropout(0.5)(X_3)
        X_4 = GraphConv(64, 'linear', True, kernel_regularizer=l2(5e-4))([X_3, fltr_in])
        X_5 = Dense(GCN.labels.shape[1], use_bias=True)(X_4)
        return Model(inputs=[X_in, fltr_in], outputs=X_5)


    def micro_f1(self, labels, logits):
        predicted = tf.math.round(tf.nn.sigmoid(logits))
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)

        true_pos = tf.math.count_nonzero(predicted * labels)
        false_pos = tf.math.count_nonzero(predicted * (labels - 1))
        false_neg = tf.math.count_nonzero((predicted - 1) * labels)

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fmeasure = (2 * precision * recall) / (precision + recall)
        return tf.cast(fmeasure, tf.float32)


    def compute_loss(self, labels, logits):
        per_node_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        return tf.reduce_mean(tf.reduce_sum(per_node_losses, axis=-1))  # Compute mean loss _per node_


    def check_early_stopping(self, current_score):
        # Early stopping
        diff =  current_score - self.best
        if diff >= GCN.early_stop:
            self.best = current_score
            self.fury = 0

            self.model.save(f"{self.MODEL}")
            log = f"Save the best model, so far."
            self.logger.debug(log)
        else:
            if self.fury == GCN.patience:
                log = f"Stop training: Ran out of patience({GCN.patience})"
                print(log)
                self.logger.debug(log)
                return True
            else:
                self.fury += 1
        return False


    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            predictions = self.model([self.features, self.fltr], training=True)
            loss = self.compute_loss(self.train_labels, predictions[self.train_mask])
            loss += sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        train_f1_score = self.micro_f1(self.train_labels, predictions[self.train_mask])
        valid_f1_score = self.micro_f1(self.valid_labels, predictions[self.valid_mask])
        return loss, train_f1_score * 100, valid_f1_score * 100


    def train_model(self, CV):
        DATA, self.MODEL = self.set_path(CV)

        # Load data
        self.load_folded_dataset(DATA)

        # Params
        self.model = self.create_model()
        self.optimizer = Adam(lr=GCN.lr)
        self.model.summary(print_fn=self.logger.info)

        train_time = time.time()
        ema_loss = 0
        for step in range(1, GCN.max_epochs+1):
            step_time = time.time()
            loss, train_score, valid_score = self.train_step()
            if step == 1:
                ema_loss = loss
            ema_loss = ema_loss * 0.99 + loss * 0.01

            log = "step: {}/{}  loss: {:.2f}  ema_loss: {:.2f}  train: {:.4f}  valid: {:.4f}  time: {:.2f}".format(step, GCN.max_epochs, loss, ema_loss, train_score, valid_score, time.time()-step_time)
            if step > GCN.min_epochs:
                log = log + "  best: {}, {:.4f}".format(step - self.fury, self.best)
            print(log)
            self.logger.info(log)
            
            if step <= GCN.min_epochs:
                continue
            if self.check_early_stopping(valid_score):
                break

        log = "Training time: {:.4f}".format(time.time()-train_time)
        print(log)
        self.logger.info(log)


if __name__=="__main__":
    GCN.load_labels("./Data/results/v3")
    for CROSS_VAL in range(1, 2):
        gcn = GCN()
        gcn.train_model(CROSS_VAL)
        del gcn
