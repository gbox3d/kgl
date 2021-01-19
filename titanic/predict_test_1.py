#%%
# import tensorflow as tf
# import keras 
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.models import Sequential

import tensorflow as tf
# import keras 
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras import backend 
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential

import numpy as np

print('module load done')


# %%

X = np.array([[10,20,30],[30,40,50],[30,40,50],[40,50,60]])
y = np.array([40,50,60,70])

print(X)
print(X.shape)


# %%
# _X = X.reshape(X.shape[0],X.shape[1],1)
_X = np.expand_dims(X,axis=2)
print(_X)
print(_X.shape)

# %%
model = Sequential()
model.add(Conv1D(filters=64,kernel_size=2,activation='relu',input_shape=(3,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss=tf.keras.losses.mse, metrics=['acc'])
print('compile ok')

model.summary()
# %%

model.fit(_X,y,epochs=100,verbose=1)
# %%
# X_input= np.array([50,60,70])
_prdt = model.predict(_X,verbose=1)
print(_prdt)

# %%
