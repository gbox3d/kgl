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

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


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

_X_train = x_train
# _X_train = np.expand_dims(x_train,axis=2)

#%%
_X_train.shape

#%%
# y_train
_y_train =np.squeeze(y_train, axis=1)
# _y_train =y_train


# %%
def create_ann():
    model = Sequential()
    
    #here 30 is output dimension
    model.add(Dense(30,input_dim =30,activation = 'relu',kernel_initializer='random_uniform'))
    
    #in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))
    
    #5 classes-normal,dos,probe,r2l,u2r
    model.add(Dense(5,activation='softmax'))
    
    #loss is categorical_crossentropy which specifies that we have multiple classes
    model.compile(loss = tf.keras.losses.categorical_crossentropy ,optimizer = 'adam',metrics = ['accuracy'])
    # model.compile(optimizer='adam', loss=tf.keras.losses.mse, metrics=['acc'])
    # model.summary()
    return model
#Since,the dataset is very big and we cannot fit complete data at once so we use batch size.
#This divides our data into batches each of size equal to batch_size.
#Now only this number of samples will be loaded into memory and processed. 
#Once we are done with one batch it is flushed from memory and the next batch will be processed.

model = KerasClassifier(build_fn=create_ann,epochs=10,batch_size=64) # 100 -> 10
# model = create_ann()

# %%

model.fit(_X_train,_y_train)

# %%
