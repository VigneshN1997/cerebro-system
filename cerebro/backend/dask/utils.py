import tensorflow as tf
import os
import time

# def get_basic_model(numeric_features):
#     print("calling get basic model")
#     model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, input_dim=10, activation='relu'),
#     tf.keras.layers.Dense(10, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#     model.compile(optimizer='adam',
#                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
#     return model

def train_model(model_checkpoint_file, estimator_gen_fn, model_config, log_files, data_ddf, model_name, worker):
    worker_log_file = log_files[0]
    time_log_file = log_files[1]
    start = time.time()
    logs = []
    # logs.append("calling train_model for model:" + str(model_checkpoint_file))
    numeric_feature_names = list(data_ddf.columns)[:784]
    pd_df = data_ddf.compute()
    target = pd_df.pop(list(data_ddf.columns)[-1])
    
    numeric_features = pd_df[numeric_feature_names].astype(float)
    tf.convert_to_tensor(numeric_features)
#     model = get_basic_model(numeric_features)
    model = estimator_gen_fn(model_config)
    # logs.append("Model name: " + str(model_name) +" Worker: " + str(worker) + " Config: " + str(model.optimizer.get_config()))
    if(os.path.isfile(model_checkpoint_file)):
        model = tf.keras.models.load_model(model_checkpoint_file)
    res = model.fit(numeric_features, target, epochs=1)
    # logs.append(str(res.history))
    model.save(model_checkpoint_file)
    finish = time.time()
    log = [worker, model_name, start, finish]
    stats_log = [worker, model_name, res.history["loss"][0], res.history["sparse_categorical_accuracy"][0]]

    with open(time_log_file, 'a') as f:
        for param in log:
            f.write("%s, " % str(param))
        f.write("\n")

    with open(worker_log_file, 'a') as f:
        for param in stats_log:
            f.write("%s, " % str(param))
        f.write("\n")