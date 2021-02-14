# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
from sklearn.model_selection import train_test_split

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices(
    "GPU") else "사용 불가능")

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
x_train.shape[0]
# %%
# _X_train = np.expand_dims(x_train,axis=2)
_X_train = np.reshape(x_train, (x_train.shape[0], 5, 6))
_X_train = np.expand_dims(_X_train, axis=3)
_X_train.shape
# %%
# _X_train.shape
X_train, X_val, Y_train, Y_val = train_test_split(
    _X_train, y_train, test_size=0.1, random_state=42)

# %%
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu', input_shape=(5, 6, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.summary()
# model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
print('compile ok')

# %%
history = model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          epochs=100,
          verbose=1,
          batch_size=512)

#%% histiry
history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
_x_test = np.reshape(x_test, (x_test.shape[0], 5, 6))
_x_test = np.expand_dims(_x_test, axis=3)

# %%
y_pred = model.predict(_x_test)

# y_pred[0].argmax()
_y_pred = [_v.argmax() for _v in y_pred]
# _y_pred
accuracy_score(y_test, _y_pred)
# %%
cm = np.array(confusion_matrix(y_test, _y_pred, labels=[4, 3, 2, 1, 0]))
# %%
sns.heatmap(cm, annot=True)
# %%
cm
# %%
