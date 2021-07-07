import sys
import os
import json

# Set distributed environment
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["GRPC_FAIL_FAST"] = "use_caller"

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.regularizers import l2

from spektral.layers import GraphConv, ops
import networkx as nx
import numpy as np

import time
from datetime import datetime
import logging

tf.keras.backend.set_floatx('float64')
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

if cluster_resolver.task_type in ("worker", "ps"):
    print("\nI'm {}\n".format(cluster_resolver.task_type))
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    server.join()
else:
    chief = True if not int(sys.argv[1]) else False
    n_workers = len(tf_config["cluster"]["worker"])
    n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    now = datetime.now().strftime("%m-%d-%H%M")
    orig_max_epochs = 14000
    orig_min_epochs = 12000
    max_epochs_per_worker = int(orig_max_epochs / n_workers)
    min_epochs_per_worker = int(orig_min_epochs / n_workers)

    base = f"./Model_dist_v3"
    models = base + f"/GCN_{now}"
    MODEL = models + f"/FOLD-{1}"
    if chief:
        if not os.path.exists(base):
            os.mkdir(base)
        if not os.path.exists(models):
            os.mkdir(models)
        if not os.path.exists(MODEL):
            os.mkdir(MODEL)
    DATA = f"./Data/results/v3/FOLD-{1}"

    logging.basicConfig(filename=f"{MODEL}/train.log", level=logging.DEBUG)
    logger = logging.getLogger()
    logger.info(f"Path for Dataset = {DATA}")
    logger.info(f"Path for Models = {MODEL}")

    labels = np.load("./Data/results/v3/labels.npy")
    with open(DATA + "/graph.json", 'r') as f:
        graph_json = json.load(f)
    graph = nx.json_graph.node_link_graph(graph_json)
    adjacency_mat = nx.adjacency_matrix(graph)
    fltr = GraphConv.preprocess(adjacency_mat).astype('f4')

    fltr = ops.sp_matrix_to_sp_tensor(fltr)
    features = np.load(DATA + "/feats.npy")
    train_mask = np.load(DATA + "/train_mask.npy")
    valid_mask = np.load(DATA + "/valid_mask.npy")
    train_labels = labels[train_mask]
    valid_labels = labels[valid_mask]

    strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
    with strategy.scope():
        def create_model(features_shape, labels_shape):
            X_in = Input((features_shape[1],))
            fltr_in = Input((features_shape[0],), sparse=True)
            X_1 = GraphConv(512, 'relu', True, kernel_regularizer=l2(5e-4))([X_in, fltr_in])
            X_1 = Dropout(0.5)(X_1)
            X_2 = GraphConv(256, 'relu', True, kernel_regularizer=l2(5e-4))([X_1, fltr_in])
            X_2 = Dropout(0.5)(X_2)
            X_3 = GraphConv(128, 'relu', True, kernel_regularizer=l2(5e-4))([X_2, fltr_in])
            X_3 = Dropout(0.5)(X_3)
            X_4 = GraphConv(64, 'linear', True, kernel_regularizer=l2(5e-4))([X_3, fltr_in])
            X_5 = Dense(labels_shape[1], use_bias=True)(X_4)
            return Model(inputs=[X_in, fltr_in], outputs=X_5)

        def compute_loss(labels, predictions):
            loss = loss_object(labels, predictions)
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
                predictions = model([features, fltr], training=True)
                loss = compute_loss(train_labels, predictions[train_mask])
                loss += sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def step_fn():
            per_replica_losses, per_replica_train_scores, per_replica_valid_scores = strategy.run(replica_fn)
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        loss_object = tf.nn.sigmoid_cross_entropy_with_logits
        model = create_model(features.shape, labels.shape)
        optimizer = Adam(lr=1e-2)

    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
    train_time = time.time()
    ema_loss = 0
    for step in range(1, max_epochs_per_worker+1):
        step_time = time.time()
        remote_value = coordinator.schedule(step_fn)
        coordinator.join()

        loss = remote_value.fetch()
        if not ema_loss:
            ema_loss = loss
        ema_loss = ema_loss * 0.99 + loss * 0.01

        log = "step: {}/{}  loss: {:.2f}  ema_loss: {:.2f}  time: {:.1f} sec".format(step, max_epochs_per_worker, loss, ema_loss, time.time()-step_time)
        print(log)
        logger.info(log)
    log = "Training time: {:.4f}".format(time.time()-train_time)
    print(log)
    logger.info(log)
