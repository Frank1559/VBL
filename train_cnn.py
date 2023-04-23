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
batch_size = 400
train_dataset = Dataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Dataset(test_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
num_classes = 4
print("Start! Model!")

'''
# create dataset and dataloaders for train and test sets
class VBLVADataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.features)

train_dataset = VBLVADataset(train_features, train_labels)
test_dataset = VBLVADataset(test_features, test_labels)

# set batch size for DataLoader
batch_size = 400

# create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
num_classes = 4

# Define the 1D CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=160, kernel_size=10)
        self.avg_pool = nn.AvgPool1d(kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=480, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
'''
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=10)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=160, kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=160, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize the model
model = CNNModel(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.unsqueeze(1)  # add channel dimension
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(train_dataloader)))

    # evaluate model performance on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1)  # add channel dimension
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))

