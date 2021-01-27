#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import tensorflow as tf
# import keras 
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras import backend 
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential

from sklearn import metrics


print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

print('load module ok')


# %%
x_train = pd.read_csv('./x_train.csv').to_numpy()
print(f'load data ok x_train , {x_train.shape}')
y_train = pd.read_csv('./y_train.csv').to_numpy()
print(f'load data ok y_train , {y_train.shape}')

x_test = pd.read_csv('./x_test.csv').to_numpy()
print(f'load data ok x_train , {x_test.shape}')
y_test = pd.read_csv('./y_test.csv').to_numpy()
print(f'load data ok y_train , {y_test.shape}')
# %%

_X_train = np.expand_dims(x_train,axis=2)

# %%
model = Sequential()
model.add(Conv1D(filters=64,kernel_size=2,activation='relu',input_shape=(30,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss=tf.keras.losses.mse, metrics=['acc'])
print('compile ok')

# %%
model.fit(_X_train,y_train,epochs=10,verbose=1)
# %%
