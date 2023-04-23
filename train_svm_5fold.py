# Cross validation 5 folds SVM evaluation
# Compare this snippet from train_svm.py:

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import plot_confusion_matrix

# load data hasil ekstraksi fitur fft
X = pd.read_csv("data/feature_VBL-VA001.csv", header=None)

# load label
y = pd.read_csv("data/label_VBL-VA001.csv", header=None)

# make 1D array to avoid warning
y = pd.Series.ravel(y)

# Setup arrays to store training and test accuracies
c_svm = np.arange(80, 100)
#test_accuracy = np.empty(len(c_svm))
test_accuracy = []
#progress = tqdm(total=100)

# finding best c for five folds
#for i, k in enumerate(c_svm):
    # Setup a knn classifier with c_svm
clf_svm = SVC(C=88)
    # Do 5-cv to the model
scores = cross_val_score(clf_svm, X, y, cv=5)
print(scores)
    # Compute accuracy on the test set
    #test_accuracy[i] = np.mean(scores)
    #test_accuracy.append(scores.mean())
test_accuracy = np.mean(scores)
    #progress.update(5)

print(f"{test_accuracy}")
# print max test accuracy (average of 5 folds)
#print(f"Max test acc: {np.max(test_accuracy)}")
#print(f"Best C: {np.argmax(test_accuracy)+81}")

#plt.plot(c_svm, test_accuracy)
#plt.xlabel('Value of C for SVM')
#plt.ylabel('Cross-Validated Accuracy')
#plt.show()
