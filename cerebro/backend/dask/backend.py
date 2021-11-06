
from __future__ import absolute_import

import datetime
import dask
from dask.distributed import Client
import time
import random
import numpy as np
import dask.dataframe as dd
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import shutil




from .. import constants
from ..backend import Backend




class DaskBackend(Backend):
    """Dask backend implementing Cerebro model hopping

        :param spark_context: Spark context
        :param num_workers: Number of Cerebro workers.
        :param start_timeout: Timeout for Spark tasks to spawn, register and start running the code, in seconds.
                   If it is not set as well, defaults to 600 seconds.
        :param disk_cache_size_gb: Size of the disk data cache in GBs (default 10GB).
        :param data_readers_pool_type: Data readers pool type ('process' or 'thread') (default 'thread')
        :param num_data_readers: Number of data readers (default 10)
        :param nics: List of NIC names, will only use these for communications. If None is specified, use any
            available networking interfaces (default None)
        :param verbose: Debug output verbosity (0-2). Defaults to 1.
    """

    def __init__(self, spark_context=None, num_workers=None, start_timeout=600, disk_cache_size_gb=10,
                 data_readers_pool_type='thread', num_data_readers=10,
                 nics=None, verbose=1):

        '''
        tmout = timeout.Timeout(start_timeout,
                                message='Timed out waiting for {activity}. Please check that you have '
                                        'enough resources to run all Cerebro processes. Each Cerebro '
                                        'process runs in a Spark task. You may need to increase the '
                                        'start_timeout parameter to a larger value if your Spark resources '
                                        'are allocated on-demand.')
        settings = spark_settings.Settings(verbose=verbose,
                                           key=secret.make_secret_key(),
                                           timeout=tmout,
                                           disk_cache_size_bytes=disk_cache_size_gb * constants.BYTES_PER_GIB,
                                           data_readers_pool_type=data_readers_pool_type,
                                           num_data_readers=num_data_readers,
                                           nics=nics)
        '''
        # if spark_context is None:
        #     spark_context = pyspark.SparkContext._active_spark_context
        #     if spark_context is None:
        #         raise Exception('Could not find an active SparkContext, are you '
        #                         'running in a PySpark session?')
        # self.spark_context = spark_context
        self.dask_client = Client(n_workers=num_workers)
        self.num_workers = num_workers
        # if num_workers is None:
        #     num_workers = spark_context.defaultParallelism
        #     if settings.verbose >= 1:
        #         print('CEREBRO => Time: {}, Running {} Workers (inferred from spark.default.parallelism)'.format(
        #             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), num_workers))
        # else:
        #     if settings.verbose >= 1:
        #         print('CEREBRO => Time: {}, Running {} Workers'.format(datetime.datetime.now().strftime(
        #             "%Y-%m-%d %H:%M:%S"), num_workers))

        # settings.num_workers = num_workers
        # self.settings = settings
        self.verbose = verbose
        if self.verbose >= 1:
                print('CEREBRO-Dask => Time: {}, Running {} Workers'.format(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"), self.num_workers))
        self.workers_initialized = False
        self.task_clients = None
        # self.driver = None
        # self.driver_client = None
        # self.spark_job_group = None
        self.data_loaders_initialized = False
        self.worker_id_ip_dict = {}
        self.rand = np.random.RandomState(constants.RANDOM_SEED)
        self.data_mapping = {}

    def _num_workers(self):
        """Returns the number of workers to use for training."""
        return self.num_workers

    def initialize_workers(self):
        """Initialize workers (get worker IPs)"""
        all_worker_details = self.client.scheduler_info()['workers']
        for ip in all_worker_details:
            self.worker_id_ip_dict[all_worker_details[ip]['id']] = str(ip)

    def initialize_data_loaders(self, store, schema_fields):
        """Initialize data loaders"""
        print('Workers are initialized')

    def get_basic_model(self, numeric_features):
#     normalizer = tf.keras.layers.Normalization(axis=-1)
#     normalizer.adapt(numeric_features)
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        return model

    def train_model(self, model_checkpoint_file, data_ddf):
        numeric_feature_names = ['age', 'thalach', 'trestbps',  'chol', 'oldpeak']
        pd_df = data_ddf.compute()
        target = pd_df.pop('target')
        numeric_features = pd_df[numeric_feature_names]
        tf.convert_to_tensor(numeric_features)
        model = self.get_basic_model(numeric_features)
        if(os.path.isfile(model_checkpoint_file)):
            model.load_weights(model_checkpoint_file)
        model.fit(numeric_features, target, epochs=1, batch_size=2)
        model.save_weights(model_checkpoint_file)

    def create_model_checkpoint_paths(self, n_models):
        checkpoint_base_path = '/Users/vignesh/Desktop/data/checkpoints/'
        model_checkpoint_paths = []
        for i in range(n_models):
            model_path = checkpoint_base_path + 'model_' + str(i)
            if os.path.exists(model_path) and os.path.isdir(model_path):
                shutil.rmtree(model_path)
            os.mkdir(model_path)
            checkpoint_path = model_path + "/" + "cp.ckpt"
    #         print(model_path)
            model_checkpoint_paths.append(checkpoint_path)
        return model_checkpoint_paths


    def train_for_one_epoch(self, models, store, feature_col, label_col, is_train=True):
        """
        Takes a set of Keras models and trains for one epoch. If is_train is False, validation is performed
         instead of training.
        :param models:
        :param store: single store object common for all models or a dictionary of store objects indexed by model id.
        :param feature_col: single list of feature columns common for all models or a dictionary of feature lists indexed by model id.
        :param label_col: single list of label columns common for all models or a dictionary of label lists indexed by model id.
        :param is_train:
        """
        print('Yet to implement train_for_one_epoch')

    def teardown_workers(self):
        """Teardown workers"""
        print('Yet to implement teardown_workers')

    def send_data(self, partitioned_dfs):
        for d in range(self.n_workers):
            self.data_mapping["data_w{0}".format(d)] = self.client.scatter(partitioned_dfs[d], workers=self.worker_id_ip_dict[d])

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        """
        Prepare data by writing out into persistent storage
        :param store:
        :param dataset:
        :param validation:
        :param compress_sparse:
        :param verbose:
        """
        # print('Yet to implement prepare_data')
        df = dd.read_csv(dataset)
        part_fracs = [1/self._num_workers() for i in range(self._num_workers())]
        partitioned_dfs = df.random_split(part_fracs, random_state=0)
        self.send_data(partitioned_dfs)




    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        """
        Get metadata from existing data in the persistent storage
        :param store:
        :param label_columns:
        :param feature_columns:
        """
        print('Yet to implement get_metadata_from_parquet')