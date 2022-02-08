import tensorflow as tf
import numpy as np
import pandas as pd

data_path='data/fakedata2980.csv'
num_train=2300
num_val=680

def preprocess_features(wind_farm_dataframe):
    selected_features = wind_farm_dataframe[["WSpeed_1"]] 
    return np.array(selected_features)


def preprocess_targets(wind_farm_dataframe):  
    selected_targets = wind_farm_dataframe[["Power_1"]]
    return np.array(selected_targets)

# Function used to construct the columns used by the program with the data
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

# Function used to load dataset
def input_fn(features, labels, training=True, batch_size=16, num_epochs=1):
        """An input function for training or evaluating"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

# Prepare data------------------------------------------------------------------------------------
wind_farm_dataframe = pd.read_csv(data_path, sep=",")
# Randomization of the data
wind_farm_dataframe = wind_farm_dataframe.reindex(np.random.permutation(wind_farm_dataframe.index))

# Separation of the data into training and validation
training_dataframe = wind_farm_dataframe.head(num_train)
validation_dataframe = wind_farm_dataframe.tail(num_val)

# Definition of the training data input variables and targets, calling the preprocess function
training_targets = preprocess_targets(training_dataframe)

training_examples_1 = np.ones((num_train, 1))
training_example = preprocess_features(training_example)
training_examples = np.concatenate((training_examples_1, training_example), axis=1).astype(np.float32)


# Definition of the validation data input variables and targets, calling the preprocess function
validation_targets = preprocess_targets(validation_dataframe)

validation_examples_1 = np.ones((num_val, 1))
validation_example = preprocess_features(validation_dataframe)
validation_examples = np.concatenate((validation_examples_1, validation_example), axis=1).astype(np.float32)
