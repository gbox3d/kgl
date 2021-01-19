#%%
import pandas as pd
import numpy as np


import tensorflow as tf
# import keras 
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras import backend 
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential

from sklearn import metrics

print('load module ok')

#%% 데이터 로딩 
train_data = pd.read_csv('./_train.csv')
print(train_data.shape)

X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]

#%%
print(X_train)
X = X_train.to_numpy()
y = Y_train.to_numpy()


# %%
# print(X)

# %% 차원 늘리기 

_X = np.expand_dims(X,axis=2)
# print(_X)
# %%

# model = Sequential()
# model.add(Conv1D(filters=64,kernel_size=2,activation='relu',input_shape=(7,1)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(50,activation='relu'))
# model.add(Dense(1))

model = Sequential()
model.add(Conv1D(filters=32,kernel_size=2,activation='relu', input_shape=(7,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPooling1D(pool_size=1))   
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss=tf.keras.losses.mse, metrics=['acc'])
print('compile ok')

# %%
model.fit(_X,y,epochs=500,verbose=1)
# %%
predict = model.predict(_X)
#since we have use sigmoid activation function in output layer
predict = (predict > 0.5).astype(int).ravel()
print(predict)


# %%
print('Precision : ', np.round(metrics.precision_score(Y_train, predict)*100,2))
print('Accuracy : ', np.round(metrics.accuracy_score(Y_train, predict)*100,2))
print('Recall : ', np.round(metrics.recall_score(Y_train, predict)*100,2))
print('F1 score : ', np.round(metrics.f1_score(Y_train, predict)*100,2))
print('AUC : ', np.round(metrics.roc_auc_score(Y_train, predict)*100,2))
# %%

matrix = metrics.confusion_matrix(Y_train, predict)
print(matrix)
# %%
