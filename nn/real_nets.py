import tflearn
from utils.optimizers import rmsprop, adam, adagrad, momentum
import keras
import tensorflow as tf

def network():
  # Build neural network
  input_ = keras.layers.Input(shape=[4,])
  hidden1 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="he_normal")(input_)
  # norm1 = keras.layers.BatchNormalization()(hidden1)
  hidden2 = keras.layers.Dense(100, activation=tf.keras.layers.ReLU(), kernel_regularizer='l1_l2', kernel_initializer="he_normal")(hidden1)
  # norm2 = keras.layers.BatchNormalization()(hidden2)
  concat = keras.layers.concatenate([input_, hidden2])
  output = keras.layers.Dense(4, activation=tf.keras.layers.ReLU())(concat)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model

def another_network():
  input_A = keras.layers.Input(shape=[2,], name="wide_input")
  input_B = keras.layers.Input(shape=[2,], name="deep_input")
  hidden1 = keras.layers.Dense(30, activation=tf.keras.layers.PReLU())(input_B)
  hidden2 = keras.layers.Dense(30, activation=tf.keras.layers.PReLU())(hidden1)
  concat = keras.layers.concatenate([input_A, hidden2])
  output = keras.layers.Dense(4, name="main_output")(concat)
  aux_output = keras.layers.Dense(4, name="aux_output")(hidden2)
  model = keras.models.Model(inputs=[input_A, input_B],
                            outputs=[output, aux_output])
  return model
