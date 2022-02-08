import tensorflow as tf
from nn.real_nets import network
import numpy as np
import keras
from utils.data_real_loader import test_examples, test_targets
from keras.models import load_model

model = network()
model.summary()
model.load_weights('/content/drive/Shareddrives/newpro112233/electricity/weights/model_5_tuabin.h5') 

y_pred = model.predict(test_examples).astype(np.float32)

print(f'\nThe prediction powers are \n{y_pred}\n')
print('\nThe true powers are:')
print(test_targets)
