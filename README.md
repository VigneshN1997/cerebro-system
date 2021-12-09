Cerebro
=======
 
``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more 
resource-efficient manner. Detailed technical information about ``Cerebro`` can be found in our 
[Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf).


Cerebro-Dask
=======

Cerebro is integrated with Dask. This support is extended in the [execution layer](https://github.com/ADALabUCSD/cerebro-system/wiki/Execution-Layer#execution-layer) of Cerebro.

The Dask backend first initializes the dask client (given the scheduler address/dask scheduler). (__init__)


Functions in cerebro/backend/dask/backend.py
- _num_workers() : Returns the number of workers to use for training.
- initialize_workers(): Get the worker IPs using the Dask client.
- initialize_data_loaders(): in dask context, this function is used to initializing the data structures for the random scheduler
- create_model_checkpoint_paths(): Initialize model checkpoint file paths
- get_runnable_model(): for a worker get a runnable model (idle model)
- init_log_files(): initialize the log file paths for logging model execution times on workers, sub epoch losses, accuracies
- get_model_log_file(): initialize the model log file paths for logging model validation losses and accuracies
- validate_models_one_epoch(): model validation performed using task parallelism
- train_for_one_epoch(): Takes a set of Keras model configs and trains for one epoch using Model Hopping Parallelism
- teardown_workers(): shutdown dask client
- send_data(): Given partitioned dataframes, send each partition to one worker
- prepare_data(): Prepare data by writing out into persistent storage

Functions in cerebro/backend/dask/utils.py
- train_model(): called for training one sub epoch on one worker.
- evaluate_model(): called for evaluating model


Install
-------

The best way to install the ``Cerebro`` is via pip.

    pip install -U cerebro-dl

Alternatively, you can git clone and run the provided Makefile script

    git clone https://github.com/ADALabUCSD/cerebro-system.git && cd cerebro-system && make

You MUST be running on **Python >= 3.6** with **Tensorflow >= 2.3** and **Apache Spark >= 2.4**


Documentation
-------------

Detailed documentation about the system can be found [here](https://adalabucsd.github.io/cerebro-system/).


Acknowledgement
---------------
This project was/is supported in part by a Hellman Fellowship, the NIDDK of the NIH under award number R01DK114945, and an NSF CAREER Award.

We used the following projects when building Cerebro.
- [Horovod](https://github.com/horovod/horovod): Cerebro's Apache Spark implementation uses code from the Horovod's
 implementation for Apache Spark.
- [Petastorm](https://github.com/uber/petastorm): We use Petastorm to read Apache Parquet data from remote storage
 (e.g., HDFS)  
 
Publications
------------
If you use this software for research, plase cite the following papers:

```latex
@inproceedings{nakandala2019cerebro,
  title={Cerebro: Efficient and Reproducible Model Selection on Deep Learning Systems},
  author={Nakandala, Supun and Zhang, Yuhao and Kumar, Arun},
  booktitle={Proceedings of the 3rd International Workshop on Data Management for End-to-End Machine Learning},
  pages={1--4},
  year={2019}
}

```
