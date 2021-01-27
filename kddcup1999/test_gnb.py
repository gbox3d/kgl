#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Gaussian Naive Bayes (나이브 베이즈 모델)
from sklearn.naive_bayes import GaussianNB

print('load module ok')


# %%
x_train = pd.read_csv('./x_train.csv')
print(f'load data ok x_train , {x_train.shape}')
y_train = pd.read_csv('./y_train.csv')
print(f'load data ok y_train , {y_train.shape}')

x_test = pd.read_csv('./x_test.csv')
print(f'load data ok x_train , {x_test.shape}')
y_test = pd.read_csv('./y_test.csv')
print(f'load data ok y_train , {y_test.shape}')
# %%

model1 = GaussianNB()

start_time = time.time()
model1.fit(x_train, y_train.values.ravel())
end_time = time.time()
print("Training time: ",end_time-start_time)


# %%
start_time = time.time()
Y_test_pred1 = model1.predict(x_test)
end_time = time.time()
print("Testing time: ",end_time-start_time)
# %%
print("Train score is:", model1.score(x_train, y_train))
print("Test score is:",model1.score(x_test,y_test))
# %%
