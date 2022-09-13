# NOTE; result appears in each checkpoint's folder
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from numpy import genfromtxt
import torch
import torch.nn.functional as F
import os

# NOTE: use this to avoid Type-3 font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# plt.rcParams["font.family"] = "Times New Roman"
# cwd = os.getcwd()
# print("cwd", cwd)

def main(filename):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    # data = genfromtxt(filename+'.txt', delimiter=',')
    data = genfromtxt(filename, delimiter=',')
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 10000000
    data = np.clip(data, a_min=-99999, a_max=100000000)


    X = np.arange(-1, 1.1, 0.01)
    Y = np.arange(-1, 1.1, 0.01)
    X, Y = np.meshgrid(X, Y)

    Z = data[:,2]  # [dirct1, direct2, loss, acc]
    Z = Z.reshape(21,21)
    Z = torch.from_numpy(Z).unsqueeze(dim=0).unsqueeze(dim=0)
    ZZ = F.interpolate(Z, scale_factor=10, mode="bicubic").squeeze(dim=0).squeeze(dim=0).numpy()

    # Plot the surface.
    surf = ax.plot_surface(X, Y, ZZ, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(1.5, 4)
    ax.grid(linestyle='dashed')

    ax.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    ax.set_zlabel('Testing loss', fontsize=16)
    # plt.axis('off')
    # plt.show()
    fig.savefig(filename.replace('.txt', '.pdf'), bbox_inches='tight', pad_inches=0, transparent=True)


if __name__ == '__main__':
    filelist = ["/root/autodl-tmp/classification_ScanObjectNN/checkpoints/pointMLP_2_2-20220701131108/loss_landscape.txt", # Baseline
                "/root/autodl-tmp/classification_ScanObjectNN/checkpoints/pointMLP_2_2-20220701222849/loss_landscape.txt", # GMGS
                "/root/autodl-tmp/classification_ScanObjectNN/checkpoints/pointMLP_2_2-20220702010543/loss_landscape.txt", # GMLS
                "/root/autodl-tmp/classification_ScanObjectNN/checkpoints/pointMLP_2_2-20220701235837/loss_landscape.txt", # LMLS
                "/root/autodl-tmp/classification_ScanObjectNN/checkpoints/pointMLP_2_2-20220702224320/loss_landscape.txt", # No_PN
                "/root/autodl-tmp/classification_ScanObjectNN/checkpoints/pointMLP_2_2-20220703091821/loss_landscape.txt"] # No_RPN

    for filename in filelist:
        main(filename)
