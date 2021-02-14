#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

print('load module ok')

#%% 칼럼 만들기 
columns=[] # 빈 리스트
with open("./kddcup.names",'r') as f:
    # print(f.read())
    _str = f.read()
    _lines = _str.split('\n')
    # print(_lines[1:])

    for line in _lines[1:] :
        # print(line)
        if len(line) > 0 :
            columns.append(line.split(':')[0])
    columns.append('target')

print(columns)
print(len(columns))

#%%
file = "./kddcup.data_10_percent.gz"
df = pd.read_csv(file,names=columns)
df.head()

#%% 
print(len(df.target.unique()))
df.target.value_counts()

#%% 
attack_type_mapping = {
    "normal":"normal"
}
with open("./training_attack_types",'r') as f:
    # print(f.read())
    _str = f.read()
    _lines = _str.split('\n')
    # print(_lines.split(' '))
    for line in _lines:
        if len(line) > 0 :
            _line = line.split(' ')
            attack_type_mapping[_line[0]] = _line[1]
print(attack_type_mapping)

# %%
new_col = df.target.apply(lambda r:attack_type_mapping[r[:-1]])
df['Attack Type'] = new_col
df.head()
# %%결측치 제거
print('결측치 제거 ')
df = df.dropna('columns')# 결측치(NaN)가 있는 컬럼들 제거
print(f'reault {len(df.columns)}')
# df['target'].value_counts()

#%% 
for col in df :
    print(f'{col} : { len(df[col].unique()) }')


# %% unique 값이 하나 이상 있는 컬럼들만 추출 
print('drop unique count less than 1')
df = df[[col for col in df if df[col].nunique() > 1]]
print(f'reault {len(df.columns)}')
 
# %% 상관관계 구하기 
corr = df.corr() 

#%% 숫자를 갖는 컬럼 이름들 찾기
numerical_cols = df._get_numeric_data().columns
print(numerical_cols)

# %% 상관관계가 높은것들 추려내기 
_cor_drop_list = []
for key in corr:
    for _k in numerical_cols:
        _c = corr[key][_k]
        if _c > 0.95 and key != _k:
            bf = False
            for drop_item in _cor_drop_list :
                if drop_item[0] == key or drop_item[1]== key :
                    bf = True
                    break
            
            if bf==False:
                print(key)
                _cor_drop_list.append([key,_k,_c])

            # print(key,_k,_c)
print(f'drop {len(_cor_drop_list)} items')

for drop_item in _cor_drop_list :
    df.drop(drop_item[0],axis = 1,inplace = True)

print( len(df.columns) )

# %% 심볼을 숫자로 바꾸기
pmap = {'icmp':0,'tcp':1,'udp':2}
df['protocol_type'] = df['protocol_type'].map(pmap)
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
df['flag'] = df['flag'].map(fmap)

# %%
# service 컬럼 제거: 32개 컬럼이 남음.
df.drop('service',axis = 1, inplace= True)
df.drop('target', axis = 1, inplace= True)
print(df.shape)
# %%
df.head()
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# %%
# Target variable and train set
df_y = df[['Attack Type']]
df_X = df.drop(['Attack Type',], axis=1)
# %%
print(df_X.shape)

for col in df_X :
    print(f'{col}')
# %%
sc = MinMaxScaler()
_X = sc.fit_transform(df_X)
#%%
_df_y = df_y['Attack Type'].map({'normal':0, 'u2r':1, 'dos':2, 'r2l':3, 'probe':4})
X_train = _X
Y_train = _df_y
print(X_train.shape)
print(Y_train.shape)
#%% save
pd.DataFrame(X_train).to_csv('x_train.csv',index=False)
Y_train.to_csv('y_train.csv',index=False)
print('save ok')
# %%
