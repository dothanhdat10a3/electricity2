import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import genfromtxt
import keras.backend as K
import xlrd
np.random.seed(42)

all_turbine = np.array([ 2, 3, 4, 5])
data_path='data/dataFeb10.xlsx'
num_val=240
# num_test=500

# Function to define the inputs. Different depending on the model and turbine
def preprocess_features(wind_farm_dataframe):
    selected_features = wind_farm_dataframe[1:, all_turbine]
    return np.array(selected_features)

def preprocess_targets(wind_farm_dataframe):  
    selected_targets = wind_farm_dataframe[1:, all_turbine+8]
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
wind_farm_dataframe1 = pd.read_excel(data_path)
wind_farm_dataframe = np.array(wind_farm_dataframe1)

examples = preprocess_features(wind_farm_dataframe)
targets = preprocess_targets(wind_farm_dataframe)
targets = np.where(targets>=0, targets, 0).astype(np.float32)
# max_targets = np.max(targets)
# targets /= max_targets

all_data = np.concatenate((examples, targets), axis=-1).astype(np.float32)
# min_targets = np.abs(np.min(targets))
# max_targets = np.abs(np.max(targets))
# target += min_target
# target /= (max_target+min_target)
# print(f'The min of power is: {min_targets}')
# print(f'The max of power is: {max_targets}\n\n')

test_indices = range(2686,2974)

mid = [i for i in range(len(all_data)) if i not in test_indices]
mid_array = np.array(mid)
val_indices = np.random.choice(mid_array.shape[0], size=num_val, replace=False)

train_indices = [i for i in range(len(mid_array)) if i not in val_indices]

# val_indices = np.random.choice(all_data.shape[0], size=num_val, replace=False)

# mid = [i for i in range(len(all_data)) if i not in val_indices]
# mid_array = np.array(mid)
# test_indices = np.random.choice(mid_array.shape[0], size=num_test, replace=False)

# train_indices = [i for i in range(len(mid_array)) if i not in test_indices]

all_test = all_data[test_indices]
test_examples = all_test[:, :len(all_turbine)]
test_targets = all_test[:, len(all_turbine):]

all_train = all_data[train_indices]
training_examples = all_train[:, :len(all_turbine)]
training_targets = all_train[:, len(all_turbine):]

all_validation = all_data[val_indices]
validation_examples = all_validation[:, :len(all_turbine)]
validation_targets = all_validation[:, len(all_turbine):]


