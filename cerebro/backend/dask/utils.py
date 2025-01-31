# Copyright 2022 Vignesh Nanda Kumar, Pratik Ratadiya and Arun Kumar. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
 
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import dask.dataframe as dd

import os
import time

def get_data_reader(train_file_path):
    petastorm_dataset_url = "file://" + train_file_path
    reader = make_batch_reader(petastorm_dataset_url)

def train_model(model_checkpoint_file, train_data, estimator_gen_fn, model_config, log_files, model_name, worker):
    """
    trail model function called for training one sub epoch (one model run on one data shard)
    :param model_checkpoint_file : model checkpoint file path to load model from
    :param train_file_path : for parquet, the file path is passed, for dask data structures: the dask future of the dataset can be passed
    :param estimator_gen_fn: the base model (containing the model structure)
    :param model_config: model configuration
    :param log_files: worker log file paths for logging
    :param model_name: model number being trained (needed for logging and testing)
    :param worker: worker number (needed for logging and testing)
    """
    worker_log_file = log_files[0]
    time_log_file = log_files[1]
    start = time.time()
    logs = []
    model = estimator_gen_fn(model_config)
    print("Model name: " + str(model_name) +" Worker: " + str(worker) + " Config: " + str(model.optimizer.get_config()))
    # print("train_file:" + str(train_data))
    if(os.path.isfile(model_checkpoint_file)):
        model = tf.keras.models.load_model(model_checkpoint_file)
    numeric_feature_names = list(data_ddf.columns)[:784]
    pd_df = data_ddf.compute()
    target = pd_df.pop(list(data_ddf.columns)[-1])
    numeric_features = pd_df[numeric_feature_names].astype(float)
    tf.convert_to_tensor(numeric_features)

    res = model.fit(numeric_features, target, epochs=1)

    # petastorm_dataset_url = "file://" + train_data
    # res = None
    # etl_log = ""
    # etl_st = time.time()
    # with make_batch_reader(petastorm_dataset_url, num_epochs=1) as reader:
    #     dataset = make_petastorm_dataset(reader)
    #     etl_en = time.time()
    #     # print("etl st:" + str(etl_st) + " etl_en:" + str(etl_en))
    #     etl_log = "etl st:" + str(etl_st) + " etl_en:" + str(etl_en)
    #     print("batch_size:" + str(model_config["batch_size"]) + " for config:" + str(model_config))
    #     res = model.fit(dataset, epochs=1, verbose=1, batch_size=model_config["batch_size"])
    #     model.save(model_checkpoint_file)

    # res = model.fit(dataset, epochs=1)
    # logs.append(str(res.history))
    
    finish = time.time()
    log = [worker, model_name, start, finish]
    stats_log = [worker, model_name, str(res.history), str(etl_log)]

    with open(time_log_file, 'a') as f:
        for param in log:
            f.write("%s, " % str(param))
        f.write("\n")

    with open(worker_log_file, 'a') as f:
        for param in stats_log:
            f.write("%s, " % str(param))
        f.write("\n")

def evaluate_model(model_cpkt_file, validation_data, model_log_file):
    """
    validate model function called for evaluating model on complete validation dataset
    :param model_checkpoint_file : model checkpoint file path to load model from
    :param train_file_path : for parquet, the file path is passed, for dask data structures: the dask future of the dataset can be passed
    :param estimator_gen_fn: the base model (containing the model structure)
    :param model_config: model configuration
    :param log_files: worker log file paths for logging
    :param model_name: model number being trained (needed for logging and testing)
    :param worker: worker number (needed for logging and testing)
    """
    model = tf.keras.models.load_model(model_cpkt_file)
    petastorm_dataset_urls = ["file://" + vp for vp in validation_data]
    val_pd_df = validation_data_ddf.compute()
    numeric_feature_names = list(val_pd_df.columns)[:784]
    target = val_pd_df.pop(list(val_pd_df.columns)[-1])
    numeric_features = val_pd_df[numeric_feature_names].astype(float)
    results = model.evaluate(val_pd_df, target)

    # with make_batch_reader(petastorm_dataset_urls) as reader:
    #     dataset = make_petastorm_dataset(reader)
    #     results = model.evaluate(dataset)
    with open(model_log_file, 'a') as f:
        for l in results:
            f.write("%s, " % str(l))
        f.write("\n")
    return results
    