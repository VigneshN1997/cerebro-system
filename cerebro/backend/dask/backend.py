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
from .utils import train_model, evaluate_model

class DaskBackend(Backend):
    """Dask backend implementing Cerebro model hopping

        :param scheduler_address: if terminal is used to initialize scheduler and dask workers, need to give scheduler IP
        :param dask_cluster: if python API is used to initialize SSH cluster (https://docs.dask.org/en/stable/how-to/deploy-dask/ssh.html)
        :param checkpoint_base_path: path where model checkpoints will be stored 
        :param logs_path: path where logs will be stored
        :param estimator_gen_fn: model building function
        :param num_workers: for single node execution, can provide number of workers as input
    """
    def __init__(self, scheduler_address=None, dask_cluster=None, checkpoint_base_path=None, logs_path=None, verbose=1, estimator_gen_fn=None, num_workers=None):
        # initialize dask client
        if(scheduler_address is not None):
            self.client = Client(scheduler_address)
        elif(dask_cluster is not None):
            self.client = Client(dask_cluster)
        else:
            self.client = Client(n_workers=num_workers)
        # get the dask dashboard link
        print("Client dashboard: ",self.client.dashboard_link)
        # get the number of workers
        self.num_workers = len(self.client.scheduler_info()['workers'])

        # set the models and log paths
        self.checkpoint_base_path = checkpoint_base_path
        self.logs_base_path = logs_path
        self.verbose = verbose
        if self.verbose >= 1:
                print('CEREBRO-Dask => Time: {}, Running {} Workers'.format(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"), self.num_workers))
        self.workers_initialized = False
        self.data_loaders_initialized = False
        # clear the worker id, worker ip dictionary
        self.worker_id_ip_dict = {}

        self.rand = np.random.RandomState(constants.RANDOM_SEED)
        
        # used when data has to be scattered
        self.data_mapping = {}
        self.val_data_fut = None
        self.estimator_gen_fn = estimator_gen_fn
        self.train_data_paths = []
        self.valid_data_paths = []

    def _num_workers(self):
        """Returns the number of workers to use for training."""
        return self.num_workers

    def initialize_workers(self):
        """Initialize workers (get worker IPs)"""
        all_worker_details = self.client.scheduler_info()['workers']
        i = 0
        for ip in all_worker_details:
            # set the mapping between worker ID and worker IP
            self.worker_id_ip_dict[i] = str(ip)
            i += 1
            
        self.workers_initialized = True

    def initialize_data_loaders(self, store, schema_fields):
        """Initialize data loaders: in dask context, this function is used to initializing the data structures for the random scheduler"""
        
        if self.workers_initialized:
            # if a model m has been trained on worker w
            self.model_worker_stat_dict = [[False for i in range(self.num_models)] for j in range(self.num_workers)]
            # list of models to build
            self.models_to_build = set()
            for i in range(self.num_models):
                self.models_to_build.add(i)

            self.model_worker_run_dict = {}
            self.worker_model_run_dict = {}

            for i in range(self.num_models): # mapping from model number to worker number and its training process future object
                self.model_worker_run_dict[i] = [None, None]

            for i in range(self.num_workers):# mapping from worker number to model number and its training process future object
                self.worker_model_run_dict[i] = [None, None]
    
            self.data_loaders_initialized = True
            # print('Workers are initialized')

    def create_model_checkpoint_paths(self, n_models):
        """Initialize model checkpoint file paths"""
        checkpoint_base_path = self.checkpoint_base_path
        model_checkpoint_paths = []
        # for all the model configs (will use the config map)
        for i in range(n_models): 
            model_path = checkpoint_base_path + 'model_' + str(i)
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            checkpoint_path = model_path + "/" + "model.h5"
            model_checkpoint_paths.append(checkpoint_path)
        self.model_checkpoint_paths = model_checkpoint_paths
    

    def get_runnable_model(self, models, model_worker_run_dict, model_worker_stat_dict, w, shuffled_model_list):
        """
            for a worker get a runnable model (idle model)
            :param model_worker_run_dict: mapping from model number to worker number and its training process future object (to check if there is some model that is free)
            :param model_worker_stat_dict: if a model m has been trained on worker w
            :param w: idle worker
            :param shuffled_model_list: randomly shuffled list of models

        """
        runnable_model = -1
        for m in shuffled_model_list:
            if((not self.model_worker_stat_dict[w][m])):
                if(self.model_worker_run_dict[m][1] is None):
                    runnable_model = m
                    break        
        return runnable_model


    def init_log_files(self):
        """
            initialize the log file paths for logging model execution times on workers, sub epoch losses, accuracies
        """
        self.log_file_paths = []
        for i in range(self.num_workers):
            lp = self.logs_base_path + 'worker_' + str(i) + '.logs'
            wp = self.logs_base_path + 'worker_times_' + str(i) + '.logs'
            self.log_file_paths.append([lp, wp])

    def get_model_log_file(self):
        """
            initialize the model log file paths for logging model validation losses and accuracies
        """
        self.model_log_file_paths = []
        self.epoch_times_path = self.logs_base_path + 'epoch_times.logs'
        for i in range(self.num_models):
            mp = self.logs_base_path + 'model_logs_' + str(i) + '.logs'
            self.model_log_file_paths.append(mp)

    def validate_models_one_epoch(self, model_configs):
        """
            model validation performed using task parallelism
            :param model_configs: model configs to find out number of models
        """
        num_models_to_validate = len(model_configs)

        model_lis = [i for i in range(self.num_models)]

        self.initialize_data_loaders('','')
        validate_models = [False for i in range(self.num_models)]
        while(len(self.models_to_build) > 0): # models to validate**
            for w in range(self.num_workers): # iterate over all workers to find an idle worker
                if(self.worker_model_run_dict[w][1] == None):
                    m = -1
                    for i in range(self.num_models): # iterate over all models to find an idle model (not yet validated nor alloted to a worker)
                        if(validate_models[i] == False and self.model_worker_run_dict[i][1] == None):
                            m = i
                            break
                    if (m != -1):
                        # print('evaluating model:' + self.model_checkpoint_paths[m] + ' on worker:' + str(w))
                        future = self.client.submit(evaluate_model, self.model_checkpoint_paths[m], self.valid_data_paths, self.model_log_file_paths[m], workers=self.worker_id_ip_dict[w])
                        self.model_worker_run_dict[m] = [w, future]
                        self.worker_model_run_dict[w] = [m, future]
                        # print('model assigned:' + str(m) + ' on worker:' + str(w) + ' status:' + future.status)
                else: # if a worker is not idle, check its model status
                    m = self.worker_model_run_dict[w][0]
                    fut = self.worker_model_run_dict[w][1]
                    if(fut.status == 'finished'):
                        # print('evaluated model:' + str(m) + ' on worker:' + str(w))
                        validate_models[m] = True
                        self.worker_model_run_dict[w] = [None, None]
                        self.model_worker_run_dict[m] = [None, None]
                        eval_done = True
                        self.models_to_build.remove(m)
                        for i in range(self.num_models): # check if all models are validated
                            if(not validate_models[i]):
                                eval_done = False
                                break
                        if(eval_done):
                            break
        return []
                    


    def train_for_one_epoch(self, model_configs, store, feature_col, label_col, is_train=True):
        """
        Takes a set of Keras model configs and trains for one epoch using Model Hopping Parallelism
         instead of training.
        :param models: model_configs to train on
        :param is_train:
        """

        print("Model config length: ",len(model_configs))
        self.num_models = len(model_configs)
        # print(model_configs)
        model_lis = [i for i in range(self.num_models)]
        random.shuffle(model_lis) # randomly shuffle the list of models

        self.initialize_data_loaders('','')
        # self.model_checkpoint_paths = self.create_model_checkpoint_paths(self.num_models)
        
        while(len(self.models_to_build) > 0): # loop until all models are built once on all workers
            for w in range(self.num_workers):
                if(self.worker_model_run_dict[w][1] == None):
                    m = self.get_runnable_model(self.model_checkpoint_paths, self.model_worker_run_dict, self.model_worker_stat_dict, w, model_lis)
                    if (m != -1):
                        # print('running model:' + self.model_checkpoint_paths[m] + ' on worker:' + str(w))
                        if(not os.path.isfile(self.model_checkpoint_paths[m])):
                            print("training the model file for first time:" + self.model_checkpoint_paths[m])
                        # model sub epoch training submitted to a worker
                        future = self.client.submit(train_model, self.model_checkpoint_paths[m], self.data_mapping['data_w'+str(w)],self.estimator_gen_fn, model_configs[m], self.log_file_paths[w], str(m), str(w), workers=self.worker_id_ip_dict[w])
                        self.model_worker_run_dict[m] = [w, future]
                        self.worker_model_run_dict[w] = [m, future]
                        # print('model assigned:' + str(m) + ' on worker:' + str(w) + ' status:' + future.status)
                else: # check status of model training
                    m = self.worker_model_run_dict[w][0]
                    fut = self.worker_model_run_dict[w][1]
                    if(fut.status == 'finished'):
                        # print('done model:' + str(m) + ' on worker:' + str(w))
                        self.model_worker_stat_dict[w][m] = True
                        # print('m:' + str(m) + ' val:' + str(self.model_checkpoint_paths[m]))
                        self.worker_model_run_dict[w] = [None, None]
                        self.model_worker_run_dict[m] = [None, None]
                        model_done = True # check if a model is trained on all workers
                        for i in range(self.num_workers):
                            if(not self.model_worker_stat_dict[i][m]):
                                model_done = False
                                break
                        if(model_done):
                            self.models_to_build.remove(m)        
        # print('Implemented train_for_one_epoch')
        return []

    def teardown_workers(self):
        """Teardown workers"""
        self.client.shutdown()

    def send_data(self, partitioned_dfs):
        # print(self.worker_id_ip_dict)
        for d in range(self.num_workers):
        #    print("D: ",d," N_workers: ",self.num_workers," Partition: ",partitioned_dfs[d]," IP Dict: ",self.worker_id_ip_dict[d]) 
            self.data_mapping["data_w{0}".format(d)] = self.client.scatter(partitioned_dfs[d], workers=self.worker_id_ip_dict[d])
        #print(self.data_mapping)

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        """
        Prepare data by writing out into persistent storage
        :param store:
        :param dataset: path to prepared parquet training dataset 
        :param validation: path to prepared parquet validation dataset
        :param compress_sparse:
        :param verbose:
        """
        # using dask data frames
        part_fracs = [1/self._num_workers() for i in range(self._num_workers())]
        partitioned_dfs = dataset.random_split(part_fracs, random_state=0)
        self.send_data(partitioned_dfs)
        self.val_data_fut = self.client.scatter(validation, broadcast=True)
        self.features = list(dataset.columns)[:-1]
        self.target = list(dataset.columns)[-1]
        

        """
        # Using dask arrays
                def load_file(file_name):
            return np.load(file_name)


        def npz_headers(npz):
            with zipfile.ZipFile(npz) as archive:
                for name in archive.namelist():
                    if not name.endswith('.npy'):
                        continue

                    npy = archive.open(name)
                    version = np.lib.format.read_magic(npy)
                    shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                    yield name[:-4], shape, dtype

        def read_npz_file(npz_file):
            npz_ptr = np.load(npz_file)
            return npz_ptr['dataset_mat']

        npz_read = dask.delayed(read_npz_file)
        lazy_train_nps = [[npz_read(path), list(npz_headers(path))[0][1], list(npz_headers(path))[0][2]] for path in train_all_paths]

        lazy_val_nps = [[npz_read(path), list(npz_headers(path))[0][1], list(npz_headers(path))[0][2]] for path in val_all_paths]   # Lazily evaluate imread on each path
        train_dataset = [da.from_delayed(lazy_da_val[0],           # Construct a small Dask array
                          dtype=lazy_da_val[2],   # for every lazy value
                          shape=lazy_da_val[1])
                        for lazy_da_val in lazy_train_nps]

        val_arrays = [da.from_delayed(lazy_da_val[0],           # Construct a small Dask array
                          dtype=lazy_da_val[2],   # for every lazy value
                          shape=lazy_da_val[1])
                        for lazy_da_val in lazy_val_nps]


        self.send_data(train_dataset) 
        """
        """
        # Using processed parquet files (for the purpose of testing)
        self.base_train_path = dataset
        self.base_val_path = validation
        for i in range(self.num_workers):
            self.train_data_paths.append(self.base_train_path + 'train_' + str(i) + '.parquet')
        for i in range(self.num_workers):
            self.valid_data_paths.append(self.base_val_path + 'valid_' + str(i) + '.parquet')
        """
        return {}, {}, {}, {}

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        """
        not using this function
        """
        print('Yet to implement get_metadata_from_parquet')