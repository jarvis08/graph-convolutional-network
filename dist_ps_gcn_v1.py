import sys
import os
import json
import logging

# Set distributed environment
# os.environ.pop('TF_CONFIG', None)
tf_config = dict()
tf_config["cluster"] = {
    "chief": ["10.20.18.215:25000"],
    "worker": ["10.20.18.216:25001", "10.20.18.217:25002"],
    "ps": ["10.20.18.218:25003"]
}
if sys.argv[1] == "0":
    tf_config["task"] = {"type": "chief", "index": 0}
elif sys.argv[1] in ['1', '2']:
    tf_config["task"] = {"type": "worker", "index": int(sys.argv[1]) - 1}
else:
    tf_config["task"] = {"type": "ps", "index": 0}
os.environ["TF_CONFIG"] = json.dumps(tf_config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["GRPC_FAIL_FAST"] = "use_caller"

import tensorflow as tf

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

if cluster_resolver.task_type in ("worker", "ps"):
    print("\nI'm {}\n".format(cluster_resolver.task_type))
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    server.join()

import time
from datetime import datetime
import numpy as np
import networkx as nx
from spektral.layers import GraphConv, ops
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.regularizers import l2

chief = True if not int(sys.argv[1]) else False
n_workers = len(tf_config["cluster"]["worker"])
# n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))

class GCN:
    now = None
    orig_max_epochs = 14000
    orig_min_epochs = 12000
    # max_epochs_per_worker = int(orig_max_epochs / n_workers / n_gpu)
    # min_epochs_per_worker = int(orig_min_epochs / n_workers / n_gpu)
    max_epochs_per_worker = int(orig_max_epochs / n_workers)
    min_epochs_per_worker = int(orig_min_epochs / n_workers)
    early_stop = 0.005
    patience = 500


    def __init__(self):
        self.fury = 0
        self.best = 0


    def set_path(self, cv):
        from datetime import datetime
        if not GCN.now:
            GCN.now = datetime.now().strftime("%m-%d-%H%M")

        base = f"./Model_dist_v3"
        models = base + f"/GCN_{GCN.now}"
        model = models + f"/FOLD-{cv}"

        if chief:
            if not os.path.exists(base):
                os.mkdir(base)
            if not os.path.exists(models):
                os.mkdir(models)
            if not os.path.exists(model):
                os.mkdir(model)

        data = f"./Data/results/v3/FOLD-{cv}"

        if chief:
            logging.basicConfig(filename=f"{model}/train.log", level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.logger.info(f"Path for Dataset = {data}")
        self.logger.info(f"Path for Models = {model}")
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

    def check_early_stopping(self, cur_score):
        # Early stopping
        diff =  cur_score - self.best
        if diff >= GCN.early_stop:
            self.best = cur_score
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


    def distributed_training(self, CV):
        DATA, self.MODEL = self.set_path(CV)
        self.load_folded_dataset(DATA)

        with strategy.scope():
            def compute_loss(labels, predictions):
                loss = self.loss_object(labels, predictions)
                return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))  # Compute mean loss _per node_

            def micro_f1(labels, logits):
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

            @tf.function
            def replica_fn():
                with tf.GradientTape() as tape:
                    predictions = self.model([self.features, self.fltr], training=True)
                    loss = compute_loss(self.train_labels, predictions[self.train_mask])
                    loss += sum(self.model.losses)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                train_f1_score = micro_f1(self.train_labels, predictions[self.train_mask])
                valid_f1_score = micro_f1(self.valid_labels, predictions[self.valid_mask])
                return loss, train_f1_score * 100, valid_f1_score * 100

            @tf.function
            def step_fn():
                per_replica_losses, per_replica_train_scores, per_replica_valid_scores = strategy.run(replica_fn)
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), \
                       per_replica_train_scores, per_replica_valid_scores

            self.loss_object = tf.nn.sigmoid_cross_entropy_with_logits
            self.model = self.create_model()
            self.optimizer = Adam(lr=1e-2)



        print("\nI'm {}\n".format(cluster_resolver.task_type))
        coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
        train_time = time.time()
        ema_loss = 0
        for step in range(1, GCN.max_epochs_per_worker+1):
            step_time = time.time()
            loss, train_score, valid_score = coordinator.schedule(step_fn)
            coordinator.join()

            # loss /= n_workers * n_gpu
            loss /= n_workers
            if not ema_loss:
                ema_loss = loss
            ema_loss = ema_loss * 0.99 + loss * 0.01
            # if n_gpu > 1:
            #     train_score = tf.reduce_mean(train_score.values)
            #     valid_score = tf.reduce_mean(valid_score.values)

            if step < GCN.min_epochs_per_worker:
                log = "step: {}/{}  loss: {:.2f}  ema_loss: {:.2f}  train: {:.3f} %  valid: {:.3f} %  time: {:.1f} sec".format(step, GCN.max_epochs_per_worker, loss, ema_loss, train_score, valid_score, time.time()-step_time)
                print(log)
                self.logger.info(log)
            else:
                log = "step: {}/{}  loss: {:.2f}  ema_loss: {:.2f}  train: {:.3f} %  valid: {:.3f} %  best: {}, {:.4f} %  time: {:.1f} sec".format(step, GCN.max_epochs_per_worker, loss, ema_loss, train_score, valid_score, step - self.fury, self.best, time.time()-step_time)
                print(log)
                self.logger.info(log)

            if step < GCN.min_epochs_per_worker:
                continue
            if self.check_early_stopping(valid_score):
                break

        log = "Training time: {:.4f}".format(time.time()-train_time)
        print(log)
        self.logger.info(log)


if __name__=="__main__":
    GCN.load_labels("./Data/results/v3")
    gcn = GCN()
    gcn.distributed_training(1)
