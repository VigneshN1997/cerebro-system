import tensorflow as tf
import os

def get_basic_model(numeric_features):
    print("calling get basic model")
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def train_model(model_checkpoint_file, data_ddf):
    print("calling train_model for model:" + str(model_checkpoint_file))
    numeric_feature_names = list(data_ddf.columns)[:10]
    pd_df = data_ddf.compute()
    target = pd_df.pop(list(data_ddf.columns)[-1])
    
    numeric_features = pd_df[numeric_feature_names].astype(float)
    tf.convert_to_tensor(numeric_features)
    model = get_basic_model(numeric_features)
    if(os.path.isfile(model_checkpoint_file)):
        model.load_weights(model_checkpoint_file)
    model.fit(numeric_features, target, epochs=1, batch_size=2)
    model.save_weights(model_checkpoint_file)