import tflearn
from utils.optimizers import rmsprop, adam, adagrad, momentum
import keras

def network():
  # Build neural network
  input_ = keras.layers.Input(shape=[2,])
  hidden1 = keras.layers.Dense(30, activation="relu")(input_)
  hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
  concat = keras.layers.concatenate([input_, hidden2])
  output = keras.layers.Dense(1)(concat)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model
