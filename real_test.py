import tensorflow as tf
from nn.real_nets import network
import numpy as np
import keras
import pandas as pd
from utils.data_real_loader import test_examples, test_targets
from keras.models import load_model

model = network()
model.summary()
model.load_weights('/content/drive/MyDrive/weightsFeb12/model_5_tuabin_2.h5') 

y_pred = model.predict(test_examples).astype(np.float32)
real_data = test_targets

print(f'\nThe prediction powers are \n{y_pred}\n')
print('\nThe true powers are:')
print(test_targets)

# Export files
df1 = pd.DataFrame(y_pred)
df1.to_excel('predict.xlsx')

df2 = pd.DataFrame(real_data)
df2.to_excel('realData.xlsx')
