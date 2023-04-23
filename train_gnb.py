# script to train VBL-VA001

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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

# kNN Machine Learning
# import KNeighborsClassifier

# Setup arrays to store training and test accuracies
# SVM Machine Learning
# Setup arrays to store training and test accuracies
var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
train_accuracy = np.empty(len(var_gnb))
test_accuracy = np.empty(len(var_gnb))

#for i, k in enumerate(var_gnb):
    # Setup a Gaussian Naive Bayes Classifier
model = GaussianNB(var_smoothing=1e-8)
gnb = model.fit(X_train, y_train)
y_test_pred = gnb.predict(X_test)
    # Compute accuracy on the training set
#train_accuracy[i] = gnb.score(X_train, y_train)
    # Compute accuracy on the test set
#test_accuracy[i] = gnb.score(X_test, y_test)

# print max acccuracy
print(f"Max test acc: {np.max(test_accuracy)}")
print(f"X: {X_test.shape}")
print(f"y: {y_test.shape}")

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
# plt.title('Varying var_smoothing in GNB')
var = np.arange(1, 100)
#plt.plot(var, test_accuracy, label='Testing Accuracy')
#plt.plot(var, train_accuracy, label='Training accuracy')
#plt.legend()
#plt.xlabel('var_smoothing')
#plt.ylabel('Accuracy')
# np.savetxt('gnb_var.txt', test_accuracy)
# plt.savefig('acc_GNB.pdf')
#plt.show()

# print optimal var_gnb and max test accuracy
#print(f"Optimal var_gnb: {np.argmax(test_accuracy)}")
#print(f"Max test accuracy: {max(test_accuracy)}")
