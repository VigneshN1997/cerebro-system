
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

import numpy as np
import dask.array as da
from numpy import load
from dask.delayed import delayed
import dask
import zipfile

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

    def __init__(self, scheduler_address=None, dask_cluster=None, checkpoint_base_path=None, logs_path=None, verbose=1, estimator_gen_fn=None, num_workers=None, num_models=None):
        if(scheduler_address is not None):
            self.client = Client(scheduler_address)
        elif(dask_cluster is not None):
            self.client = Client(dask_cluster)
        else:
            self.client = Client(n_workers=num_workers)
        print("Client dashboard: ",self.client.dashboard_link)
        self.num_workers = len(self.client.scheduler_info()['workers'])
        self.num_models = num_models
        self.checkpoint_base_path = checkpoint_base_path
        self.logs_base_path = logs_path
        self.verbose = verbose
        if self.verbose >= 1:
                print('CEREBRO-Dask => Time: {}, Running {} Workers'.format(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"), self.num_workers))
        self.workers_initialized = False
        self.data_loaders_initialized = False
        self.worker_id_ip_dict = {}
        self.rand = np.random.RandomState(constants.RANDOM_SEED)
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
            self.worker_id_ip_dict[i] = str(ip)
            i += 1
            
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
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            checkpoint_path = model_path + "/" + "model.h5"
            model_checkpoint_paths.append(checkpoint_path)
        self.model_checkpoint_paths = model_checkpoint_paths
    
    # for a worker get a runnable model
    def get_runnable_model(self, models, model_worker_run_dict, model_worker_stat_dict, w, shuffled_model_list):
        runnable_model = -1
        for m in shuffled_model_list:
            if((not self.model_worker_stat_dict[w][m])):
                if(self.model_worker_run_dict[m][1] is None):
                    runnable_model = m
                    break        
        return runnable_model


    def init_log_files(self):
        self.log_file_paths = []
        for i in range(self.num_workers):
            lp = self.logs_base_path + 'worker_' + str(i) + '.logs'
            wp = self.logs_base_path + 'worker_times_' + str(i) + '.logs'
            self.log_file_paths.append([lp, wp])

    def get_model_log_file(self):
        self.model_log_file_path = self.logs_base_path + 'model_val.logs'

        # for i in range(self.num_models):
        #     mp = self.logs_base_path + 'model_logs_' + str(i) + '.logs'
        #     self.model_log_file_paths.append(mp)

    def validate_models_one_epoch(self, model_configs):
        num_models_to_validate = len(model_configs)

        # print(model_configs)
        self.num_models = len(model_configs)
        model_lis = [i for i in range(self.num_models)]
        # random.shuffle(model_lis)

        self.initialize_data_loaders('','')

        # self.model_checkpoint_paths = self.create_model_checkpoint_paths(self.num_models)
        model_worker_val_stats = [[[] for i in range(self.num_models)] for j in range(self.num_workers)]
        # print(len(model))
        combined_model_stats = [[0.0,0.0] for i in range(self.num_models)]
        while(len(self.models_to_build) > 0):
            for w in range(self.num_workers):
                if(self.worker_model_run_dict[w][1] == None):
                    m = self.get_runnable_model(self.model_checkpoint_paths, self.model_worker_run_dict, self.model_worker_stat_dict, w, model_lis)
                    if (m != -1):
                        print('evaluating model:' + self.model_checkpoint_paths[m] + ' on worker:' + str(w))
                        future = self.client.submit(evaluate_model, self.model_checkpoint_paths[m], self.valid_data_paths[w], workers=self.worker_id_ip_dict[w])
                        self.model_worker_run_dict[m] = [w, future]
                        self.worker_model_run_dict[w] = [m, future]
                        print('model assigned:' + str(m) + ' on worker:' + str(w) + ' status:' + future.status)
                else:
                    m = self.worker_model_run_dict[w][0]
                    fut = self.worker_model_run_dict[w][1]
                    if(fut.status == 'finished'):
                        print('evaluated model:' + str(m) + ' on worker:' + str(w))
                        self.model_worker_stat_dict[w][m] = True
                        model_worker_val_stats[w][m] = fut.result()
                        print('m: ' + str(m) + ' w:' + str(w) + 'stats:'+ str(model_worker_val_stats[m][w]))
                        print('m done:' + str(m) + ' val:' + str(self.model_checkpoint_paths[m]))
                        self.worker_model_run_dict[w] = [None, None]
                        self.model_worker_run_dict[m] = [None, None]
                        model_done = True
                        for i in range(self.num_workers):
                            if(not self.model_worker_stat_dict[i][m]):
                                model_done = False
                                break
                        if(model_done):
                            self.models_to_build.remove(m)

        for m in range(self.num_models):
            for w in range(self.num_workers):
                combined_model_stats[m][0] += model_worker_val_stats[w][m][0]
                combined_model_stats[m][1] += (model_worker_val_stats[w][m][1] * self.val_data_len_fracs[w])
        with open(self.model_log_file_path, 'a') as f:
            for i in range(len(combined_model_stats)):
                f.write("%s, " % str(i))
                f.write("%s, " % str(combined_model_stats[i][0]))
                f.write("%s" % str(combined_model_stats[i][1]))
                f.write("\n")
        print('Implemented eval_for_one_epoch')
        return []
                    


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
        # TODO: shuffle a list of models and pass to get runnable model (Done) => test it

        print("Model config length: ",len(model_configs))
        self.num_models = len(model_configs)
        print(model_configs)
        model_lis = [i for i in range(self.num_models)]
        random.shuffle(model_lis)

        self.initialize_data_loaders('','')
        # self.model_checkpoint_paths = self.create_model_checkpoint_paths(self.num_models)
        
        while(len(self.models_to_build) > 0):
            for w in range(self.num_workers):
                if(self.worker_model_run_dict[w][1] == None):
                    m = self.get_runnable_model(self.model_checkpoint_paths, self.model_worker_run_dict, self.model_worker_stat_dict, w, model_lis)
                    if (m != -1):
                        print('running model:' + self.model_checkpoint_paths[m] + ' on worker:' + str(w))
                        if(not os.path.isfile(self.model_checkpoint_paths[m])):
                            print("training the model file for first time:" + self.model_checkpoint_paths[m])
                        future = self.client.submit(train_model, self.model_checkpoint_paths[m], self.data_mapping['data_w'+str(w)],self.estimator_gen_fn, model_configs[m], self.log_file_paths[w], str(m), str(w), workers=self.worker_id_ip_dict[w])
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

    def send_data(self, dataset):
        print(self.worker_id_ip_dict)
        for d in range(self.num_workers):
        #    print("D: ",d," N_workers: ",self.num_workers," Partition: ",partitioned_dfs[d]," IP Dict: ",self.worker_id_ip_dict[d]) 
            self.data_mapping["data_w{0}".format(d)] = self.client.scatter(dataset[d], workers=self.worker_id_ip_dict[d])
        print(self.data_mapping)


    # def persist_data_workers(self):
    #     for d in range(self.num_workers):
    #         self.client.persist(self.data_mapping['data_w'+str(d)], workers=self.worker_id_ip_dict[d])
    #     self.client.persist(self.val_data_fut)

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        """
        Prepare data by writing out into persistent storage
        :param store:
        :param dataset:
        :param validation:
        :param compress_sparse:
        :param verbose:
        """
        # part_fracs = [1/self._num_workers() for i in range(self._num_workers())]
        # partitioned_dfs = dataset.random_split(part_fracs, random_state=0)
        # self.send_data(partitioned_dfs)
        # self.val_data_fut = self.client.scatter(validation, broadcast=True)
        # self.features = list(dataset.columns)[:-1]
        # self.target = list(dataset.columns)[-1]
        # num_data_pts = dataset.shape[0]
        # chunk_sz = num_data_pts / self.num_workers
        # if(len(dataset.chunks[0]) != self.num_workers):
        #     dataset = dataset.rechunk({0: chunk_sz, 1: -1})

        train_all_paths = [dataset + 'train_' + str(i) + '_compressed.npz' for i in range(8)]
        val_all_paths = [validation + 'valid_' + str(i) + '_compressed.npz' for i in range(8)]

        def load_file(file_name):
            return np.load(file_name)


        def npz_headers(npz):
            """Takes a path to an .npz file, which is a Zip archive of .npy files.
            Generates a sequence of (name, shape, np.dtype).
            """
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
        # self.val_data_fut = self.client.scatter(validation, broadcast=True)
        # self.base_train_path = dataset
        # self.base_val_path = validation
        # for i in range(self.num_workers):
        #     self.train_data_paths.append(self.base_train_path + 'train_' + str(i) + '.parquet')
        # for i in range(self.num_workers):
        #     self.valid_data_paths.append(self.base_val_path + 'valid_' + str(i) + '.parquet')
        # self.val_data_lens = []
        # self.total_sz_val_data = 0
        # for val_data in self.valid_data_paths:
        #     df = dd.read_parquet(val_data)
        #     labels = df["labels"].compute()
        #     num_pts = len(labels)
        #     self.val_data_lens.append(num_pts)
        #     self.total_sz_val_data += num_pts
        # print("total sz val data:" + str(self.total_sz_val_data))
        # print("diff lens val data:" + str(self.val_data_lens))
        # self.val_data_len_fracs = [float(l)/float(self.total_sz_val_data) for l in self.val_data_lens]
        # print("diff len frac val data:" + str(self.val_data_len_fracs))
        return {}, {}, {}, {}

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        """
        Get metadata from existing data in the persistent storage
        :param store:
        :param label_columns:
        :param feature_columns:
        """
        print('Yet to implement get_metadata_from_parquet')