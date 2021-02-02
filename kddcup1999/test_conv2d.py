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
# from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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
#%%
x_train.shape
# %%

# _X_train = np.expand_dims(x_train,axis=2)
_X_train = np.reshape(x_train,(330994,5,6))
_X_train = np.expand_dims(_X_train,axis=3)
#%%
_X_train.shape
#%%

# %%
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(5,6,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(2,2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()
# model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
print('compile ok')

# %%
model.fit(_X_train,y_train,epochs=1000,verbose=1)
# %%
