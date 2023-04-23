# code to plot all feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取label_VBL-VA001.csv檔案
label_data = pd.read_csv('data/label_VBL-VA001.csv', header=None)

# 讀取feature_VBL-VA001.csv檔案
feature_data = pd.read_csv('data/feature_VBL-VA001.csv', header=None)

# 根據label的值，將feature分類到不同的DataFrame中
x_norm = feature_data.loc[label_data[0] == 0]
x_mis = feature_data.loc[label_data[0] == 1]
x_unb = feature_data.loc[label_data[0] == 2]
x_bear = feature_data.loc[label_data[0] == 3]

# 印出每個類別有幾個rows
print('normal:', len(x_norm))
print('misalignment:', len(x_mis))
print('unbalance:', len(x_unb))
print('bearing:', len(x_bear))

# plot all nine features
feat_name = ['Shape Factor', 'RMS', 'Impulse Factor', 'Peak to Peak', 'Kurtosis', 'Crest Factor', 'Mean', 'Standard Deviation', 'Skewness']
feat_n = np.arange(2, 27, 3)
fig, ax = plt.subplots(3, 3)
fig.set_size_inches(8, 6)

axs = ax.ravel()
for i, n in enumerate(feat_n):
    y1 = x_norm.iloc[:, n]
    y2 = x_mis.iloc[:, n]
    y3 = x_unb.iloc[:, n]
    y4 = x_bear.iloc[:, n]

    y1 = y1.values.flatten()[:45000]
    y2 = y2.values.flatten()[:45000]
    y3 = y3.values.flatten()[:45000]
    y4 = y4.values.flatten()[:45000]

    # helper for x axis
    x = np.arange(0,len(y1),1)

    def movingaverage(interval, window_size):
        window= np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

    y11 = movingaverage(y1, 30)
    y22 = movingaverage(y2, 30)
    y33 = movingaverage(y3, 30)
    y44 = movingaverage(y4, 30)

    # print(f"i = {i}, n = {n}")
    axs[i].plot(x, y11, x, y22, x, y33, x, y44, linewidth=0.25)
    # Decorate
    axs[i].set_title(feat_name[i])
    axs[i].set_xlim(0,45000)
    if i <= 5:
        # axs[i].get_xaxis().set_visible(False)
        axs[i].axes.xaxis.set_ticklabels([])

plt.show()
# plt.savefig('feature_z_axis.pdf')
# save manually for better quality, then crop
