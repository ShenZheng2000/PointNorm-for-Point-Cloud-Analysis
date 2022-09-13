from __future__ import print_function
import os
import argparse
import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random

# import matplotlib.colors as mcolors
# def_colors = mcolors.CSS4_COLORS
# colrs_list = []
# np.random.seed(2021)
# for k, v in def_colors.items():
#     colrs_list.append(k)
# np.random.shuffle(colrs_list)
colrs_list = [
    "C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet","C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet","C0", "C1","C2","C3","C4","C5","C6","C7","C8","C9","deepskyblue", "m","deeppink","hotpink","lime","c","y",
    "gold","darkorange","g","orangered","tomato","tan","darkorchid","violet"
]

def test(args):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("===> The number of test data is:%d", len(test_data))
    # Try to load models
    print("===> Create model...")
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    # For Complete Version
    # model = models.__dict__[args.model](num_classes=num_part, 
    #                                     embed_dim = 64, 
    #                                     res_expansion = 1.0, 
    #                                     point_norm=True, 
    #                                     reverse_point_norm=True, 
    #                                     local_mean=True,
    #                                     local_std=False,
    #                                     global_mean=False,
    #                                     global_std=True,
    #                                     use_xyz=True).to(device) 

    # For Tiny Version
    model = models.__dict__[args.model](num_classes=num_part, 
                                        embed_dim = 32, 
                                        res_expansion = 0.25, 
                                        point_norm=True, 
                                        reverse_point_norm=True, 
                                        local_mean=True,
                                        local_std=False,
                                        global_mean=False,
                                        global_std=True,
                                        use_xyz=True).to(device) 

    print("===> Load checkpoint...")
    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)
    print("===> Start evaluate...")
    model.eval()
    num_classes = 16
    points, label, target, norm_plt = test_data.__getitem__(args.id)
    points = torch.tensor(points).unsqueeze(dim=0)
    label = torch.tensor(label).unsqueeze(dim=0)
    target = torch.tensor(target).unsqueeze(dim=0)
    norm_plt = torch.tensor(norm_plt).unsqueeze(dim=0)
    points = points.transpose(2, 1)
    norm_plt = norm_plt.transpose(2, 1)
    points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(dim=0).cuda(
        non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
    with torch.no_grad():
            cls_lable = to_categorical(label, num_classes)
            predict = model(points, norm_plt, cls_lable)  # b,n,50
    # up to now, points [1, 3, 2048]  predict [1, 2048, 50] target [1, 2048]
    predict = predict.max(dim=-1)[1]
    predict = predict.squeeze(dim=0).cpu().data.numpy()  # 2048
    target = target.squeeze(dim=0).cpu().data.numpy()   # 2048
    points = points.transpose(2, 1).squeeze(dim=0).cpu().data.numpy() #[2048,3]
    
    # # save txt
    # np.savetxt(f"figures/{args.id}-point.txt", points)
    # np.savetxt(f"figures/{args.id}-target.txt", target)
    # np.savetxt(f"figures/{args.id}-predict.txt", predict)

    # start plot
    print(f"===> stat plotting")
    # cus_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    plot_xyz(points, target, args.angle_one, args.angle_two, name=f"{args.output_folder}/{args.angle_one}-{args.angle_two}-{args.id}-gt.pdf")
    plot_xyz(points, predict, args.angle_one, args.angle_two, name=f"{args.output_folder}/{args.angle_one}-{args.angle_two}-{args.id}-predict.pdf")
    # for angle_one in cus_list:
        # for angle_two in cus_list:
        # plot_xyz(points, predict, angle_one, angle_two, name=f"figures/{angle_one}-{angle_two}-{args.id}-predict.pdf")


def plot_xyz(xyz, target, angle_one, angle_two, name="figures/figure.pdf"):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]
    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)
    for i in range(0,2048):
        col = int(target[i])
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col], marker="o", s=30, alpha=0.7)
    ax.set_axis_off() 
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()

    # NOTE: add these two
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.25, 1, 1]))
    # ax.view_init(-100, 0) 
    # ax.view_init(-30, 90) 

    ax.view_init(angle_one, angle_two)
    ax.set_title(f'仰角：{angle_one}  方位角：{angle_two}')
    fig.savefig(name, bbox_inches='tight', pad_inches=-0.3, transparent=True)
    # fig.savefig(name, bbox_inches='tight', pad_inches=-0.5, transparent=True)
    # fig.set_size_inches(16.5, 9.5, forward=True)

    pyplot.close()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='pointMLP') # pointMLPElite for Tiny
    parser.add_argument('--id', type=int, default='1') 
    parser.add_argument('--exp_name', type=str, default='PointMLP_7_8_v1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False, 
                        help='enables CUDA training')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')
    parser.add_argument('--angle_one', type=int, required=True)
    parser.add_argument('--angle_two', type=int, required=True)
    parser.add_argument('--output_folder', type=str, default='figures_sup') # NOTE: change this for output folder

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test(args)
