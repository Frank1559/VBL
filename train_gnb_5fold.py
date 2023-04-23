# script to train VBL-VA001

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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
# SVM Machine Learning
# Setup arrays to store training and test accuracies
var_gnb = [10.0 ** i for i in np.arange(-1, -100, -1)]
test_accuracy = np.empty(len(var_gnb))
#test_accuracy = []
#progress = tqdm(total=100)

#for i, k in enumerate(var_gnb):
    # Setup a Gaussian Naive Bayes Classifier
clf_gnb = GaussianNB(var_smoothing=1e-14)
scores = cross_val_score(clf_gnb, X, y, cv=5)
print(scores)
    # Compute accuracy on the test set
    #test_accuracy[i] = np.mean(scores)
test_accuracy = np.mean(scores)
    #progress.update(1)

print(f"{test_accuracy}")
#print(f"Max test acc: {np.max(test_accuracy)}")
#max_var_gnb = np.argmax(test_accuracy)
#print(f"Best var smoothing: {var_gnb[max_var_gnb]}")

#var = np.arange(1, 100)
#plt.plot(var, test_accuracy)
#plt.xlabel('var_smoothing')
#plt.ylabel('Accuracy')
#plt.show()
