# Variable
    # Input: log.txt files
    # Output: pdf figures

# Explanations
    # Val Loss: val loss (min and max bound)
    # Grad Pred: l2 norm of the difference between adjacent gradient tensors (min and max bound)
    # Beta Smooth: l1 norm of the difference between adjacent gradient tensors (max bound only)

# Reference
    # https://github.com/AlexeyGB/batch-norm-helps-optimization/blob/master/notebooks/3_VGG_Gradient_Predictiveness_Skuratov.ipynb


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

path_001_w = '/root/autodl-tmp/classification_ScanObjectNN/checkpoints/PointNorm_2_2-20220828105159/log.txt'
path_002_w = '/root/autodl-tmp/classification_ScanObjectNN/checkpoints/PointNorm_2_2-20220828105211/log.txt'
path_0005_w = '/root/autodl-tmp/classification_ScanObjectNN/checkpoints/PointNorm_2_2-20220828105218/log.txt'
path_001_wo = '/root/autodl-tmp/classification_ScanObjectNN/checkpoints/PointNorm_2_2-20220828105247/log.txt'
path_002_wo = '/root/autodl-tmp/classification_ScanObjectNN/checkpoints/PointNorm_2_2-20220828105257/log.txt'
path_0005_wo = '/root/autodl-tmp/classification_ScanObjectNN/checkpoints/PointNorm_2_2-20220828105304/log.txt'


def read_process_csv(path):
    row_data = pd.read_csv(path, sep = '\t')

    train_loss = row_data.iloc[:, 2].values
    val_loss = row_data.iloc[:, 7].values
    grad_pred = row_data.iloc[:, 5].values
    beta_smooth = row_data.iloc[:, 6].values
    val_acc = row_data.iloc[:, 9].values
    val_acc_b = row_data.iloc[:, 8].values
    return train_loss, val_loss, grad_pred, beta_smooth, val_acc, val_acc_b


def plot_loss(ind, vert, name):
    # read csv files
    loss_001_w = read_process_csv(path_001_w)[ind]
    loss_002_w = read_process_csv(path_002_w)[ind]
    loss_0005_w = read_process_csv(path_0005_w)[ind]
    loss_001_wo = read_process_csv(path_001_wo)[ind]
    loss_002_wo = read_process_csv(path_002_wo)[ind]
    loss_0005_wo = read_process_csv(path_0005_wo)[ind]

    # init
    min_curve_w = []
    max_curve_w = []
    min_curve_wo = []
    max_curve_wo = []

    # calculate
    if name == 'grad_pred':
        plot_range = len(loss_001_w) // 4 # only the beginning 1/4 epochs
    else:
        plot_range = len(loss_001_w)

    print("name {}, plot_range {}".format(name, plot_range))

    for i in range(plot_range):
        min_curve_w.append(np.min([loss_001_w[i], loss_002_w[i], loss_0005_w[i]]))
        max_curve_w.append(np.max([loss_001_w[i], loss_002_w[i], loss_0005_w[i]]))
        min_curve_wo.append(np.min([loss_001_wo[i], loss_002_wo[i], loss_0005_wo[i]]))
        max_curve_wo.append(np.max([loss_001_wo[i], loss_002_wo[i], loss_0005_wo[i]]))
    
    steps = np.arange(0, plot_range)

    # w/ DualNorm (blue)
    if name != 'beta_smooth':
        plt.fill_between(steps, min_curve_w, max_curve_w, alpha=0.5, color="#60ACFC", label = 'w/ DualNorm')
        plt.plot(steps, min_curve_w, color = "#60ACFC")
        plt.plot(steps, max_curve_w, color = "#60ACFC")
    else:
        plt.plot(steps, max_curve_w, color = "#60ACFC", label = 'w/ DualNorm')

    # w/o DualNorm (red)
    if name != 'beta_smooth':
        plt.fill_between(steps, min_curve_wo, max_curve_wo, alpha=0.5, color="#FF7C7C", label = 'w/o DualNorm')
        plt.plot(steps, min_curve_wo, color = "#FF7C7C")
        plt.plot(steps, max_curve_wo, color = "#FF7C7C")
    else:
        plt.plot(steps, max_curve_wo, color = "#FF7C7C", label = 'w/o DualNorm')

    # plot the figure
    plt.legend(loc='best', fontsize = 15)
    plt.xlabel('Epochs', fontsize = 20)
    plt.ylabel(vert, fontsize = 20)
    plt.savefig(os.path.join(os.getcwd(), "main_draw", f"{name}.pdf"))
    plt.close()



if __name__ == '__main__':
    plot_loss(ind=1, vert='Testing Loss', name='val_loss') 
    plot_loss(ind=2, vert='Gradient Predictveness', name='grad_pred') 
    # plot_loss(ind=3, vert='Beta-Smoothness', name='beta_smooth') # NO use, skip this!!!
    plot_loss(ind=4, vert='Testing OA', name='val_acc')
    plot_loss(ind=5, vert='Testing mAcc', name='val_acc_b')
