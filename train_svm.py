# script to train VBL-VA001

from sklearn.svm import SVC
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

# load data hasil ekstraksi fitur fft
x = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42, shuffle=True
)

print("Shape of Train Data : {}".format(X_train.shape))
print("Shape of Test Data : {}".format(X_test.shape))

# Setup arrays to store training and test accuracies
c_svm = np.arange(1, 100)
#train_accuracy = np.empty(len(c_svm))
#test_accuracy = np.empty(len(c_svm))
#progress = tqdm(total=100)

#for i, k in enumerate(c_svm):
    # Setup a knn classifier with c_svm
svm = SVC(C=98)
    # Fit the model
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
    # Compute accuracy on the training set
    #train_accuracy[i] = svm.score(X_train, y_train)
    #train_accuracy[i] = precision_score(X_train, y_train, average='macro')
    # Compute accuracy on the test set
    #test_accuracy[i] = svm.score(X_test, y_test)
    #test_accuracy[i] = precision_score(X_test, y_test, average='macro')
    #progress.update(1)

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
# plt.title('Varying number of SVM')
#plt.plot(c_svm, test_accuracy, label='Testing Accuracy')
#plt.plot(c_svm, train_accuracy, label='Training accuracy')
#plt.legend()
#plt.xlabel('C')
#plt.ylabel('Accuracy')
#plt.show()
# np.savetxt('scm_c.txt', test_accuracy)
# plt.savefig('acc_svm.pdf')

# print optimal C and max test accuracy
#print(f"Optimal C: {np.argmax(test_accuracy)}")
#print(f"Max test accuracy: {max(test_accuracy)}")
