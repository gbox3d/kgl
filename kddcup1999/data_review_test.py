# %%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

print('load module ok')

# %% 칼럼 만들기
columns = []  # 빈 리스트
with open("./kddcup.names", 'r') as f:
    # print(f.read())
    _str = f.read()
    _lines = _str.split('\n')
    # print(_lines[1:])

    for line in _lines[1:]:
        # print(line)
        if len(line) > 0:
            columns.append(line.split(':')[0])
    columns.append('target')

print(columns)
print(len(columns))

# %%
file = "./corrected.gz"
df = pd.read_csv(file, names=columns)
df.head()

# %%
len(df.target.unique())
# %%
df.target.value_counts()

# %%
attack_type_mapping = {
    "normal": "normal"
}
with open("./training_attack_types", 'r') as f:
    # print(f.read())
    _str = f.read()
    _lines = _str.split('\n')
    # print(_lines.split(' '))
    for line in _lines:
        if len(line) > 0:
            _line = line.split(' ')
            attack_type_mapping[_line[0]] = _line[1]
print(attack_type_mapping)

# %%
#추론 가능한 유형만 골라내기 
df = df[(df.target == 'smurf.') |
   (df.target == 'normal.') |
    (df.target == 'back.') |
    (df.target == 'satan.') |
    (df.target == 'ipsweep.') |
    (df.target == 'portsweep.') |
    (df.target == 'warezclient.') |
    (df.target == 'teardrop.') |
    (df.target == 'pod.') |
    (df.target == 'nmap.') |
    (df.target == 'guess_passwd.') |
    (df.target == 'buffer_overflow.') |
    (df.target == 'land.') |
    (df.target == 'warezmaster.') |
    (df.target == 'imap.') |
    (df.target == 'rootkit.') |
    (df.target == 'loadmodule.') |
    (df.target == 'ftp_write.') |
    (df.target == 'multihop.') |
    (df.target == 'phf.') |
    (df.target == 'perl.') |
    (df.target == 'spy.') 
   ]

#%%
print(len(df.target.unique()))
df.target.value_counts()

# %%
new_col = df.target.apply(lambda r: attack_type_mapping[r[:-1]])
df['Attack Type'] = new_col
df.head()
#%% 
df.drop('num_outbound_cmds',axis=1,inplace=True)
df.drop('is_host_login',axis=1,inplace=True)

df.drop('num_compromised',axis=1,inplace=True)
df.drop('serror_rate',axis=1,inplace=True)
df.drop('rerror_rate',axis=1,inplace=True)
df.drop('dst_host_srv_count',axis=1,inplace=True)
df.drop('dst_host_serror_rate',axis=1,inplace=True)
df.drop('dst_host_srv_serror_rate',axis=1,inplace=True)
df.drop('dst_host_rerror_rate',axis=1,inplace=True)
df.drop('dst_host_srv_rerror_rate',axis=1,inplace=True)

print(f'reault {len(df.columns)}')

# %% 심볼을 숫자로 바꾸기
pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
df['protocol_type'] = df['protocol_type'].map(pmap)
fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4,
        'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
df['flag'] = df['flag'].map(fmap)

# %%
# service 컬럼 제거: 32개 컬럼이 남음.
df.drop('service', axis=1, inplace=True)
df.drop('target', axis=1, inplace=True)
print(df.shape)
# %%
df.head()

# %%
# Target variable and train set
df_y = df[['Attack Type']]
df_X = df.drop(['Attack Type', ], axis=1)
# %%
print(df_X.shape)

for col in df_X :
    print(f'{col}')
# %%
sc = MinMaxScaler()
_X = sc.fit_transform(df_X)


# %%
_df_y = df_y['Attack Type'].map(
    {'normal': 0, 'u2r': 1, 'dos': 2, 'r2l': 3, 'probe': 4})
X_train = _X
Y_train = _df_y

# %% save
pd.DataFrame(X_train).to_csv('x_test.csv', index=False)
Y_train.to_csv('y_test.csv', index=False)

print('save ok')


