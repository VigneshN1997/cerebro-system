import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
 
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import dask.dataframe as dd
import numpy as np

import os
import time

def get_data_reader(train_file_path):
    petastorm_dataset_url = "file://" + train_file_path
    reader = make_batch_reader(petastorm_dataset_url)

# def train_model(model_checkpoint_file, train_file_path, estimator_gen_fn, model_config, log_files, model_name, worker):
#     worker_log_file = log_files[0]
#     time_log_file = log_files[1]
#     start = time.time()
#     logs = []
#     model = estimator_gen_fn(model_config)
#     # logs.append("Model name: " + str(model_name) +" Worker: " + str(worker) + " Config: " + str(model.optimizer.get_config()))
#     if(os.path.isfile(model_checkpoint_file)):
#         model = tf.keras.models.load_model(model_checkpoint_file)

#     petastorm_dataset_url = "file://" + train_file_path
#     res = None
#     with make_batch_reader(petastorm_dataset_url, num_epochs=1) as reader:
#       dataset = make_petastorm_dataset(reader)
#       res = model.fit(dataset, steps_per_epoch=10, epochs=1, verbose=1, batch_size=model_config["batch_size"])
#       model.save(model_checkpoint_file)

#     # res = model.fit(dataset, epochs=1)
#     # logs.append(str(res.history))
    
#     finish = time.time()
#     log = [worker, model_name, start, finish]
#     stats_log = [worker, model_name, str(res.history)]

#     with open(time_log_file, 'a') as f:
#         for param in log:
#             f.write("%s, " % str(param))
#         f.write("\n")

#     with open(worker_log_file, 'a') as f:
#         for param in stats_log:
#             f.write("%s, " % str(param))
#         f.write("\n")

def train_model(model_checkpoint_file, train_data, estimator_gen_fn, model_config, log_files, model_name, worker):
    # train_data.persist()
    # train_data_df = train_data.compute()
    np_train_feat_vals = train_data[:, :-1].astype(np.float32)
    np_labels = train_data[:, -1].astype(int)
    np_labels = np.eye(2)[np_labels]

    worker_log_file = log_files[0]
    time_log_file = log_files[1]
    start = time.time()
    logs = []
    model = estimator_gen_fn(model_config)

    if(os.path.isfile(model_checkpoint_file)):
        model = tf.keras.models.load_model(model_checkpoint_file)

    res = model.fit(np_train_feat_vals, np_labels, epochs=1, verbose=1, batch_size=model_config["batch_size"])
    finish = time.time()
    log = [worker, model_name, start, finish]
    stats_log = [worker, model_name, str(res.history)]
    with open(time_log_file, 'a') as f:
        for param in log:
            f.write("%s, " % str(param))
        f.write("\n")

    with open(worker_log_file, 'a') as f:
        for param in stats_log:
            f.write("%s, " % str(param))
        f.write("\n")

def evaluate_model(model_cpkt_file, validation_data_path):
    model = tf.keras.models.load_model(model_cpkt_file)
    petastorm_dataset_url = "file://" + validation_data_path
    
    with make_batch_reader(petastorm_dataset_url) as reader:
        dataset = make_petastorm_dataset(reader)
        results = model.evaluate(dataset)
        
    return results
    