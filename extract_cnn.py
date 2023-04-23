import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import manifold

# read feature and label data
#feature_data = pd.read_csv("data/feature_VBL-VA001.csv", header=None)
#label_data = pd.read_csv("data/label_VBL-VA001.csv", header=None)

data_path = 'data/VBL-VA001/'

totalFiles = 0
totalDir = 0

for base, dirs, files in os.walk(data_path):
    print('Searching in : ', base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1

print('Total number of files', totalFiles)
print('Total number of directories', totalDir)

# Collecting number data
dir_path1 = data_path + 'normal'
print('Total data Normal :', len([entry for entry in os.listdir(
    dir_path1) if os.path.isfile(os.path.join(dir_path1, entry))]))
dir_path2 = data_path + 'misalignment'
print('Total data misalignment :', len([entry for entry in os.listdir(
    dir_path2) if os.path.isfile(os.path.join(dir_path2, entry))]))
dir_path3 = data_path + 'unbalance'
print('Total data unbalance :', len([entry for entry in os.listdir(
    dir_path3) if os.path.isfile(os.path.join(dir_path3, entry))]))
dir_path4 = data_path + 'bearing'
print('Total data bearing fault:', len([entry for entry in os.listdir(
    dir_path4) if os.path.isfile(os.path.join(dir_path4, entry))]))

# Collecting file names
normal_file_names = glob.glob(data_path + '/normal/*.csv')
imnormal_misalignment = glob.glob(data_path + '/misalignment/*.csv')
imnormal_unbalance = glob.glob(data_path + '/unbalance/*.csv')
imnormal_bearing = glob.glob(data_path + '/bearing/*.csv')

# Extract features from X, Y, Z axis
def data_1x(normal_file_names):
    data1x = pd.DataFrame()
    for f1x in normal_file_names:
        df1x = pd.read_csv(f1x, usecols=[1], header=None)  # read the csv file
        data1x = pd.concat([data1x, df1x], axis=1, ignore_index=True)
    return data1x


def data_1y(normal_file_names):
    data1y = pd.DataFrame()
    for f1y in normal_file_names:
        df1y = pd.read_csv(f1y, usecols=[2], header=None)  # read the csv file
        data1y = pd.concat([data1y, df1y], axis=1, ignore_index=True)
    return data1y


def data_1z(normal_file_names):
    data1z = pd.DataFrame()
    for f1z in normal_file_names:
        df1z = pd.read_csv(f1z, usecols=[3], header=None)  # read the csv file
        data1z = pd.concat([data1z, df1z], axis=1, ignore_index=True)
    return data1z


def data_2x(imnormal_misalignment):
    data2x = pd.DataFrame()
    for f2x in imnormal_misalignment:
        df2x = pd.read_csv(f2x, usecols=[1], header=None)  # read the csv file
        data2x = pd.concat([data2x, df2x], axis=1, ignore_index=True)
    return data2x


def data_2y(imnormal_misalignment):
    data2y = pd.DataFrame()
    for f2y in imnormal_misalignment:
        df2y = pd.read_csv(f2y, usecols=[2], header=None)  # read the csv file
        data2y = pd.concat([data2y, df2y], axis=1, ignore_index=True)
    return data2y


def data_2z(imnormal_misalignment):
    data2z = pd.DataFrame()
    for f2z in imnormal_misalignment:
        df2z = pd.read_csv(f2z, usecols=[3], header=None)  # read the csv file
        data2z = pd.concat([data2z, df2z], axis=1, ignore_index=True)
    return data2z


def data_3x(imnormal_unbalance):
    data3x = pd.DataFrame()
    for f3x in imnormal_unbalance:
        df3x = pd.read_csv(f3x, usecols=[1], header=None)  # read the csv file
        data3x = pd.concat([data3x, df3x], axis=1, ignore_index=True)
    return data3x


def data_3y(imnormal_unbalance):
    data3y = pd.DataFrame()
    for f3y in imnormal_unbalance:
        df3y = pd.read_csv(f3y, usecols=[2], header=None)  # read the csv file
        data3y = pd.concat([data3y, df3y], axis=1, ignore_index=True)
    return data3y


def data_3z(imnormal_unbalance):
    data3z = pd.DataFrame()
    for f3z in imnormal_unbalance:
        df3z = pd.read_csv(f3z, usecols=[3], header=None)  # read the csv file
        data3z = pd.concat([data3z, df3z], axis=1, ignore_index=True)
    return data3z


def data_4x(imnormal_bearing):
    data4x = pd.DataFrame()
    for f4x in imnormal_bearing:
        df4x = pd.read_csv(f4x, usecols=[1], header=None)  # read the csv file
        data4x = pd.concat([data4x, df4x], axis=1, ignore_index=True)
    return data4x


def data_4y(imnormal_bearing):
    data4y = pd.DataFrame()
    for f4y in imnormal_bearing:
        df4y = pd.read_csv(f4y, usecols=[2], header=None)  # read the csv file
        data4y = pd.concat([data4y, df4y], axis=1, ignore_index=True)
    return data4y


def data_4z(imnormal_bearing):
    data4z = pd.DataFrame()
    for f4z in imnormal_bearing:
        df4z = pd.read_csv(f4z, usecols=[3], header=None)  # read the csv file
        data4z = pd.concat([data4z, df4z], axis=1, ignore_index=True)
    return data4z


# Data normal transpose x y z, remove NaN
data_normal_x = data_1x(normal_file_names).T.dropna(axis=1)
data_normal_y = data_1y(normal_file_names).T.dropna(axis=1)
data_normal_z = data_1z(normal_file_names).T.dropna(axis=1)
data_normal_x = data_normal_x.iloc[:, :90000]
data_normal_y = data_normal_y.iloc[:, :90000]
data_normal_z = data_normal_z.iloc[:, :90000]

# Data misalignment transpose x y z
data_misalignment_x = data_2x(imnormal_misalignment).T.dropna(axis=1)
data_misalignment_y = data_2y(imnormal_misalignment).T.dropna(axis=1)
data_misalignment_z = data_2z(imnormal_misalignment).T.dropna(axis=1)
data_misalignment_x = data_misalignment_x.iloc[:, :90000]
data_misalignment_y = data_misalignment_y.iloc[:, :90000]
data_misalignment_z = data_misalignment_z.iloc[:, :90000]

# Data unbalance transpose x y z
data_unbalance_x = data_3x(imnormal_unbalance).T.dropna(axis=1)
data_unbalance_y = data_3y(imnormal_unbalance).T.dropna(axis=1)
data_unbalance_z = data_3z(imnormal_unbalance).T.dropna(axis=1)
data_unbalance_x = data_unbalance_x.iloc[:, :90000]
data_unbalance_y = data_unbalance_y.iloc[:, :90000]
data_unbalance_z = data_unbalance_z.iloc[:, :90000]

# Data bearing transpose x y z
data_bearing_x = data_4x(imnormal_bearing).T.dropna(axis=1)
data_bearing_y = data_4y(imnormal_bearing).T.dropna(axis=1)
data_bearing_z = data_4z(imnormal_bearing).T.dropna(axis=1)
data_bearing_x = data_bearing_x.iloc[:, :90000]
data_bearing_y = data_bearing_y.iloc[:, :90000]
data_bearing_z = data_bearing_z.iloc[:, :90000]

# # Concatenate data and label for each X, Y, and Z
# print(data_normal_x.shape)
# print(data_misalignment_x.shape)
# print(data_unbalance_x.shape)
# print(data_bearing_x.shape)
data_x = np.concatenate((data_normal_x, data_misalignment_x, data_unbalance_x, data_bearing_x))
y_1 = np.full((int(len(data_normal_x)), 1), 0)
y_2 = np.full((int(len(data_misalignment_x)), 1), 1)
y_3 = np.full((int(len(data_unbalance_x)), 1), 2)
y_4 = np.full((int(len(data_bearing_x)), 1), 3)
label_x = np.concatenate((y_1, y_2, y_3, y_4), axis=None)
print(data_x.shape)
print(label_x.shape)
data_y = np.concatenate((data_normal_y, data_misalignment_y, data_unbalance_y, data_bearing_y))
y_1 = np.full((int(len(data_normal_y)), 1), 0)
y_2 = np.full((int(len(data_misalignment_y)), 1), 1)
y_3 = np.full((int(len(data_unbalance_y)), 1), 2)
y_4 = np.full((int(len(data_bearing_y)), 1), 3)
label_y = np.concatenate((y_1, y_2, y_3, y_4), axis=None)
print(data_y.shape)
print(label_y.shape)
data_z = np.concatenate((data_normal_z, data_misalignment_z, data_unbalance_z, data_bearing_z))
y_1 = np.full((int(len(data_normal_z)), 1), 0)
y_2 = np.full((int(len(data_misalignment_z)), 1), 1)
y_3 = np.full((int(len(data_unbalance_z)), 1), 2)
y_4 = np.full((int(len(data_bearing_z)), 1), 3)
label_z = np.concatenate((y_1, y_2, y_3, y_4), axis=None)
print(data_z.shape)
print(label_z.shape)

X1, y1 = data_x, label_x
X2, y2 = data_y, label_y
X3, y3 = data_z, label_z

data_merged = np.concatenate((X1, X2, X3))
# normalize data
def NormalizeData(data):  # Normalisasi (0-1)
    data_max = np.max(data_merged)
    data_min = np.min(data_merged)
    return (data - np.min(data_min)) / (np.max(data_max) - np.min(data_min))
X1 = NormalizeData(X1)
X2 = NormalizeData(X2)
X3 = NormalizeData(X3)

# 設定參數
n_components=2
init='random'
random_state=5
verbose=1

name = {0: "normal", 1: "misalignment", 2: "unbalance", 3: "bearing"}
cmap = plt.colors.ListedColormap(['red', 'blue', 'green', 'purple'])

# 對 X1 做 t-SNE
X_tsne_1 = manifold.TSNE(n_components=n_components, init=init, random_state=random_state, verbose=verbose).fit_transform(X1)
df_1 = pd.DataFrame(dict(Feature_1=X_tsne_1[:,0], Feature_2=X_tsne_1[:,1], label=y1))

# 繪製圖表
groups = df_1.groupby('label')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['Feature_1'], group['Feature_2'], label=name)
ax.legend()
ax.set_title("t-SNE for x-axis")

# 對 X2 做 t-SNE
X_tsne_2 = manifold.TSNE(n_components=n_components, init=init, random_state=random_state, verbose=verbose).fit_transform(X2)
df_2 = pd.DataFrame(dict(Feature_1=X_tsne_2[:,0], Feature_2=X_tsne_2[:,1], label=y2))

# 繪製圖表
groups = df_2.groupby('label')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['Feature_1'], group['Feature_2'], label=name)
ax.legend()
ax.set_title("t-SNE for y-axis")

# 對 X3 做 t-SNE
X_tsne_3 = manifold.TSNE(n_components=n_components, init=init, random_state=random_state, verbose=verbose).fit_transform(X3)
df_3 = pd.DataFrame(dict(Feature_1=X_tsne_3[:,0], Feature_2=X_tsne_3[:,1], label=y3))

# 繪製圖表
groups = df_3.groupby('label')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['Feature_1'], group['Feature_2'], label=name)
ax.legend()
ax.set_title("t-SNE for z-axis")

# 顯示圖表
plt.show()

df = pd.DataFrame({
    'feature_1_x': X_tsne_1[:, 0],
    'feature_1_y': X_tsne_2[:, 0],
    'feature_1_z': X_tsne_3[:, 0]
})
df.to_csv('data/tnse_feature_1.csv', index=None, header=False)
df = pd.DataFrame({
    'feature_2_x': X_tsne_1[:, 1],
    'feature_2_y': X_tsne_2[:, 1],
    'feature_2_z': X_tsne_3[:, 1]
})
df.to_csv('data/tnse_feature_2.csv', index=None, header=False)

'''
data = np.concatenate([data_x[:, :, np.newaxis], data_y[:, :, np.newaxis], data_z[:, :, np.newaxis]], axis=2)
label = np.concatenate([label_x, label_y, label_z], axis=None)
print(data.shape)
print(label.shape)



# split data into train and test sets
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.Tensor(data.values[:train_size]), torch.Tensor(data.values[train_size:])
train_labels, test_labels = torch.Tensor(label.values[:train_size].astype(int).flatten()), torch.Tensor(label.values[train_size:].astype(int).flatten())

print("START!")
batch_size = 500
train_dataset = Dataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Dataset(test_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
num_classes = 4
print("Start! Model!")
print(train_dataset.shape)
'''
