# script to train VBL-VA001

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

# load data hasil ekstraksi fitur fft
x = pd.read_csv('data/feature_VBL-VA001.csv', header=None)

# load label
y = pd.read_csv('data/label_VBL-VA001.csv', header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42, shuffle=True)


print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

# kNN Machine Learning
# import KNeighborsClassifier
# Setup arrays to store training and test accuracies
neighbors = np.arange(1, 100)
#train_accuracy = np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))
#progress = tqdm(total=100)

#for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=5)
    # Fit the model
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
    # Compute accuracy on the training set
    #train_accuracy[i] = knn.score(X_train, y_train)
#train_accuracy = knn.score(X_train, y_train)
    # Compute accuracy on the test set
    #test_accuracy[i] = knn.score(X_test, y_test)
#test_accuracy = knn.score(X_test, y_test)
    #progress.update(1)

# print max acccuracy
#print(f"test acc: {test_accuracy}")

print('------Weighted------')
print('Weighted precision', precision_score(y_test, y_test_pred, average='weighted'))
print('Weighted recall', recall_score(y_test, y_test_pred, average='weighted'))
print('Weighted f1-score', f1_score(y_test, y_test_pred, average='weighted'))
print('------Macro------')
print('Macro precision', precision_score(y_test, y_test_pred, average='macro'))
print('Macro recall', recall_score(y_test, y_test_pred, average='macro'))
print('Macro f1-score', f1_score(y_test, y_test_pred, average='macro'))
print('------Micro------')
print('Micro precision', precision_score(y_test, y_test_pred, average='micro'))
print('Micro recall', recall_score(y_test, y_test_pred, average='micro'))
print('Micro f1-score', f1_score(y_test, y_test_pred, average='micro'))

# Generate plot
# plt.title('k-NN Varying number of neighbors')
#plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label='Training accuracy')
#plt.legend()
#plt.xlabel('Number of neighbors')
#plt.ylabel('Accuracy')
#plt.show()
# np.savetxt('knn_n.txt', test_accuracy)
# plt.savefig('acc_knn.pdf')

# print optimal k and max test accuracy
#print(f"Optimal k: {np.argmax(test_accuracy)}")
#print(f"Max test accuracy: {max(test_accuracy)}")
