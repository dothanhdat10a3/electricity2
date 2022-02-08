from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils.data_loader import training_examples, training_targets, validation_examples, validation_targets
from nn.nets import network

# active GPU
tf.debugging.set_log_device_placement(True)
# sparse_categorical_crossentropy
# mean_squared_error

# Load model------------------------------------------------------------------------------------
def train(data=None, labels=None, val_data=None, val_labels=None, network=network, num_epochs=20, batch_size=32, show_metric=True, name_saver=None):
  model = network()

  model.compile(loss="mean_squared_error",
                metrics=[tf.keras.metrics.Precision()],
                optimizer=tf.keras.optimizers.Adam())

  history = model.fit(data, labels, epochs=num_epochs,
                    validation_data=(val_data, val_labels))
  model.save('/content/electricity/save/'+name_saver)

train(training_examples, training_targets, validation_examples, validation_targets, network=network, name_saver='model.h5')
