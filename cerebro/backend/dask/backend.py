
from __future__ import absolute_import

import datetime
import dask
from dask.distributed import Client
import time
import random
import numpy as np
import dask.dataframe as dd
import os
# import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import shutil

from .. import constants
from ..backend import Backend
from .utils import train_model

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

    def __init__(self, num_workers=None, num_models=None, checkpoint_base_path=None, verbose=1, dask_cluster=None, estimator_gen_fn=None):
        if(dask_cluster is not None):
            self.client = Client(dask_cluster)
        else:
            self.client = Client(n_workers=num_workers)
        print("Client dashboard: ",self.client.dashboard_link)
        self.num_workers = num_workers
        self.num_models = num_models
        self.checkpoint_base_path = checkpoint_base_path
    
        self.verbose = verbose
        if self.verbose >= 1:
                print('CEREBRO-Dask => Time: {}, Running {} Workers'.format(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"), self.num_workers))
        self.workers_initialized = False
        self.task_clients = None
        self.data_loaders_initialized = False
        self.worker_id_ip_dict = {}
        self.rand = np.random.RandomState(constants.RANDOM_SEED)
        self.data_mapping = {}
        self.estimator_gen_fn = estimator_gen_fn

    def _num_workers(self):
        """Returns the number of workers to use for training."""
        return self.num_workers

    def initialize_workers(self):
        """Initialize workers (get worker IPs)"""
        all_worker_details = self.client.scheduler_info()['workers']
#         print("helloo" + str(all_worker_details))
        for ip in all_worker_details:
            self.worker_id_ip_dict[all_worker_details[ip]['id']] = str(ip)
            
        self.workers_initialized = True

    def initialize_data_loaders(self, store, schema_fields):
        """Initialize data loaders"""
        
        if self.workers_initialized:
            self.model_worker_stat_dict = [[False for i in range(self.num_models)] for j in range(self.num_workers)]
            self.models_to_build = set()
            for i in range(self.num_models):
                self.models_to_build.add(i)

            self.model_worker_run_dict = {}
            self.worker_model_run_dict = {}

            for i in range(self.num_models): # mapping from model number to worker number and its future
                self.model_worker_run_dict[i] = [None, None]

            for i in range(self.num_workers):# mapping from worker number to model number and its future
                self.worker_model_run_dict[i] = [None, None]
    
            self.data_loaders_initialized = True
            print('Workers are initialized')

    def create_model_checkpoint_paths(self, n_models):
        checkpoint_base_path = self.checkpoint_base_path
        model_checkpoint_paths = []
        for i in range(n_models):
            model_path = checkpoint_base_path + 'model_' + str(i)
            if os.path.exists(model_path) and os.path.isdir(model_path):
                shutil.rmtree(model_path)
            os.mkdir(model_path)
            checkpoint_path = model_path + "/" + "cp.ckpt"
            model_checkpoint_paths.append(checkpoint_path)
        return model_checkpoint_paths
    
    # for a worker get a runnable model
    def get_runnable_model(self, models, model_worker_run_dict, model_worker_stat_dict, w):
        runnable_model = -1
        for i in range(len(models)):
            if((not self.model_worker_stat_dict[w][i])):
                if(self.model_worker_run_dict[i][1] is None):
                    runnable_model = i
                    break        
        return runnable_model

    def train_for_one_epoch(self, model_configs, store, feature_col, label_col, is_train=True):
        """
        Takes a set of Keras models and trains for one epoch. If is_train is False, validation is performed
         instead of training.
        :param models:
        :param store: single store object common for all models or a dictionary of store objects indexed by model id.
        :param feature_col: single list of feature columns common for all models or a dictionary of feature lists indexed by model id.
        :param label_col: single list of label columns common for all models or a dictionary of label lists indexed by model id.
        :param is_train:
        """
        print("Model config length: ",len(model_configs))
        self.num_models = len(model_configs)
        print(model_configs)
        self.initialize_data_loaders('','')
        self.model_checkpoint_paths = self.create_model_checkpoint_paths(self.num_models)
        
        while(len(self.models_to_build) > 0):
            for w in range(self.num_workers):
                if(self.worker_model_run_dict[w][1] == None):
                    m = self.get_runnable_model(self.model_checkpoint_paths, self.model_worker_run_dict, self.model_worker_stat_dict, w)
                    if (m != -1):
                        print('running model:' + self.model_checkpoint_paths[m] + ' on worker:' + str(w))
                        future = self.client.submit(train_model, self.model_checkpoint_paths[m], self.estimator_gen_fn, model_configs[m], self.data_mapping['data_w'+str(w)], str(m), str(w), workers=self.worker_id_ip_dict[w])
                        self.model_worker_run_dict[m] = [w, future]
                        self.worker_model_run_dict[w] = [m, future]
                        print('model assigned:' + str(m) + ' on worker:' + str(w) + ' status:' + future.status)
                else:
                    m = self.worker_model_run_dict[w][0]
                    fut = self.worker_model_run_dict[w][1]
                    if(fut.status == 'finished'):
                        print('done model:' + str(m) + ' on worker:' + str(w))
                        self.model_worker_stat_dict[w][m] = True
                        print('m:' + str(m) + ' val:' + str(self.model_checkpoint_paths[m]))
                        self.worker_model_run_dict[w] = [None, None]
                        self.model_worker_run_dict[m] = [None, None]
                        model_done = True
                        for i in range(self.num_workers):
                            if(not self.model_worker_stat_dict[i][m]):
                                model_done = False
                                break
                        if(model_done):
                            self.models_to_build.remove(m)        
        print('Implemented train_for_one_epoch')
        return []

    def teardown_workers(self):
        """Teardown workers"""
#         self.client.shutdown()
        print('Yet to implement teardown_workers')

    def send_data(self, partitioned_dfs):
        print(self.worker_id_ip_dict)
        for d in range(self.num_workers):
        #    print("D: ",d," N_workers: ",self.num_workers," Partition: ",partitioned_dfs[d]," IP Dict: ",self.worker_id_ip_dict[d]) 
            self.data_mapping["data_w{0}".format(d)] = self.client.scatter(partitioned_dfs[d], workers=self.worker_id_ip_dict[d])
        #print(self.data_mapping)

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        """
        Prepare data by writing out into persistent storage
        :param store:
        :param dataset:
        :param validation:
        :param compress_sparse:
        :param verbose:
        """
        part_fracs = [1/self._num_workers() for i in range(self._num_workers())]
        partitioned_dfs = dataset.random_split(part_fracs, random_state=0)
        self.send_data(partitioned_dfs)
        self.features = list(dataset.columns)[:-1]
        self.target = list(dataset.columns)[-1]
        return {}, {}, {}, {}

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        """
        Get metadata from existing data in the persistent storage
        :param store:
        :param label_columns:
        :param feature_columns:
        """
        print('Yet to implement get_metadata_from_parquet')